#include "mmcore/CoreInstance.h"
#include "mmcore/MegaMolGraph.h"

#include "mmcore/utility/log/DefaultTarget.h"
#include "mmcore/utility/log/Log.h"

#include "CUDA_Service.hpp"
#include "FrameStatistics_Service.hpp"
#include "FrontendServiceCollection.hpp"
#include "GUI_Service.hpp"
#include "Lua_Service_Wrapper.hpp"
#include "OpenGL_GLFW_Service.hpp"
#include "Screenshot_Service.hpp"

#include "mmcore/view/AbstractView_EventConsumption.h"

#include <cxxopts.hpp>
#include "mmcore/LuaAPI.h"
#include "mmcore/utility/graphics/ScreenShotComments.h"

// Filesystem
#if defined(_HAS_CXX17) || ((defined(_MSC_VER) && (_MSC_VER > 1916))) // C++2017 or since VS2019
#include <filesystem>
namespace stdfs = std::filesystem;
#else
// WINDOWS
#ifdef _WIN32
#include <filesystem>
namespace stdfs = std::experimental::filesystem;
#else
// LINUX
#include <experimental/filesystem>
namespace stdfs = std::experimental::filesystem;
#endif
#endif

// make sure that all configuration parameters have sane and useful and EXPLICIT initialization values!
struct CLIConfig {
    std::string program_invocation_string = "";
    std::vector<std::string> project_files = {};
    std::string lua_host_address = "tcp://127.0.0.1:33333";
    bool lua_host_port_retry = true;
    bool load_example_project = false;
    bool opengl_khr_debug = false;
    bool opengl_vsync = false;
    std::vector<unsigned int> window_size = {}; // if not set, GLFW service will open window with 3/4 of monitor resolution 
    std::vector<unsigned int> window_position = {};
    enum WindowMode {
        fullscreen   = 1 << 0,
        nodecoration = 1 << 1,
        topmost      = 1 << 2,
        nocursor     = 1 << 3,
    };
    unsigned int window_mode = 0;
    unsigned int window_monitor = 0;
};

CLIConfig handle_cli_inputs(int argc, char* argv[]);

bool set_up_example_graph(megamol::core::MegaMolGraph& graph);

int main(int argc, char* argv[]) {

    auto config = handle_cli_inputs(argc, argv);

    // setup log
    megamol::core::utility::log::Log::DefaultLog.SetLevel(megamol::core::utility::log::Log::LEVEL_ALL);
    megamol::core::utility::log::Log::DefaultLog.SetEchoLevel(megamol::core::utility::log::Log::LEVEL_ALL);
    megamol::core::utility::log::Log::DefaultLog.SetOfflineMessageBufferSize(100);
    megamol::core::utility::log::Log::DefaultLog.SetMainTarget(
        std::make_shared<megamol::core::utility::log::DefaultTarget>(megamol::core::utility::log::Log::LEVEL_ALL));

    megamol::core::CoreInstance core;
    core.Initialise(false); // false makes core not start his own lua service (else we collide on default port)

    const megamol::core::factories::ModuleDescriptionManager& moduleProvider = core.GetModuleDescriptionManager();
    const megamol::core::factories::CallDescriptionManager& callProvider = core.GetCallDescriptionManager();

    megamol::frontend::OpenGL_GLFW_Service gl_service;
    megamol::frontend::OpenGL_GLFW_Service::Config openglConfig;
    openglConfig.windowTitlePrefix = openglConfig.windowTitlePrefix + " ~ Main3000";
    openglConfig.versionMajor = 4;
    openglConfig.versionMinor = 5;
    openglConfig.enableKHRDebug = config.opengl_khr_debug;
    openglConfig.enableVsync = config.opengl_vsync;
    // pass window size and position
    if (!config.window_size.empty()) {
        assert(config.window_size.size() == 2);
        openglConfig.windowPlacement.size = true;
        openglConfig.windowPlacement.w = config.window_size[0];
        openglConfig.windowPlacement.h = config.window_size[1];
    }
    if (!config.window_position.empty()) {
        assert(config.window_position.size() == 2);
        openglConfig.windowPlacement.pos = true;
        openglConfig.windowPlacement.x = config.window_position[0];
        openglConfig.windowPlacement.y = config.window_position[1];
    }
    openglConfig.windowPlacement.mon = config.window_monitor;
    openglConfig.windowPlacement.fullScreen = config.window_mode & CLIConfig::WindowMode::fullscreen;
    openglConfig.windowPlacement.noDec      = config.window_mode & CLIConfig::WindowMode::nodecoration;
    openglConfig.windowPlacement.topMost    = config.window_mode & CLIConfig::WindowMode::topmost;
    openglConfig.windowPlacement.noCursor   = config.window_mode & CLIConfig::WindowMode::nocursor;
    gl_service.setPriority(2);

    megamol::frontend::GUI_Service gui_service;
    megamol::frontend::GUI_Service::Config guiConfig;
    guiConfig.imgui_api = megamol::frontend::GUI_Service::ImGuiAPI::OPEN_GL;
    guiConfig.core_instance = &core;
    // priority must be higher than priority of gl_service (=1)
    // service callbacks get called in order of priority of the service.
    // postGraphRender() and close() are called in reverse order of priorities.
    gui_service.setPriority(23);

    megamol::frontend::Screenshot_Service screenshot_service;
    megamol::frontend::Screenshot_Service::Config screenshotConfig;
    screenshot_service.setPriority(30);

    megamol::frontend::FrameStatistics_Service framestatistics_service;
    megamol::frontend::FrameStatistics_Service::Config framestatisticsConfig;
    // needs to execute before gl_service at frame start, after gl service at frame end
    framestatistics_service.setPriority(1);

    megamol::core::MegaMolGraph graph(core, moduleProvider, callProvider);

    bool lua_imperative_only = false; // allow mmFlush, mmList* and mmGetParam*
    megamol::core::LuaAPI lua_api(graph, lua_imperative_only);
    megamol::frontend::Lua_Service_Wrapper lua_service_wrapper;
    megamol::frontend::Lua_Service_Wrapper::Config luaConfig;
    luaConfig.lua_api_ptr = &lua_api;
    luaConfig.host_address = config.lua_host_address;
    luaConfig.retry_socket_port = config.lua_host_port_retry;
    lua_service_wrapper.setPriority(0);

#ifdef MM_CUDA_ENABLED
    megamol::frontend::CUDA_Service cuda_service;
    cuda_service.setPriority(24);
#endif

    // clang-format off
    // the main loop is organized around services that can 'do something' in different parts of the main loop.
    // a service is something that implements the AbstractFrontendService interface from 'megamol\frontend_services\include'.
    // a central mechanism that allows services to communicate with each other and with graph modules are _resources_.
    // (see FrontendResource in 'megamol\frontend_resources\include').
    // services may provide resources to the system and they may request resources they need themselves for functioning.
    // think of a resource as a struct (or some type of your choice) that gets wrapped
    // by a helper structure and gets a name attached to it. the fronend makes sure (at least
    // attempts to) to hand each service the resources it requested, or else fail execution of megamol with an error message.
    // resource assignment is done by the name of the resource, so this is a very loose interface based on trust.
    // type safety of resources is ensured in the sense that extracting the wrong type from a FrontendResource will
    // lead to an unhandled bad type cast exception, leading to the shutdown of megamol.
    // clang-format on
    bool run_megamol = true;
    megamol::frontend::FrontendServiceCollection services;
    services.add(gl_service, &openglConfig);
    services.add(gui_service, &guiConfig);
    services.add(lua_service_wrapper, &luaConfig);
    services.add(screenshot_service, &screenshotConfig);
    services.add(framestatistics_service, &framestatisticsConfig);
#ifdef MM_CUDA_ENABLED
    services.add(cuda_service, nullptr);
#endif

    // clang-format off
    // TODO: port cinematic as frontend service
    // TODO: FBO-centered rendering (View redesign)
    // => explicit FBOs!
    // => explicit camera / animation time / FBO resources/modules in graph?
    // => do or dont show GUI in screenshots, depending on ...
    // TODO: ZMQ context as frontend resource
    // TODO: port CLI commands from mmconsole
    // TODO: eliminate the core instance:
    //  => extract module/call description manager into new factories; remove from core
    //  => key/value store for CLI configuration as frontend resource (emulate config params)
    // TODO: main3000 raw hot loop performance vs. mmconsole performance
    // TODO: centralize project loading/saving to/from .lua/.png.
    // => has to collect graph serialization from graph, gui state from gui.
    // clang-format on

    const bool init_ok = services.init(); // runs init(config_ptr) on all services with provided config sructs

    if (!init_ok) {
        std::cout << "ERROR: some frontend service could not be initialized successfully. abort. " << std::endl;
        services.close();
        return 1;
    }

    // graph is also a resource that may be accessed by services
    services.getProvidedResources().push_back({"MegaMolGraph", graph});

    // proof of concept: a resource that returns a list of names of available resources
    // used by Lua Wrapper and LuaAPI to return list of available resources via remoteconsole
    const std::function<std::vector<std::string>()> resource_lister = [&]() -> std::vector<std::string> {
        std::vector<std::string> resources;
        for (auto& resource : services.getProvidedResources()) {
            resources.push_back(resource.getIdentifier());
        }
        resources.push_back("FrontendResourcesList");
        return resources;
    };
    services.getProvidedResources().push_back({"FrontendResourcesList", resource_lister});

    // distribute registered resources among registered services.
    const bool resources_ok = services.assignRequestedResources();
    // for each service we call their resource callbacks here:
    //    std::vector<FrontendResource>& getProvidedResources()
    //    std::vector<std::string> getRequestedResourceNames()
    //    void setRequestedResources(std::vector<FrontendResource>& resources)
    if (!resources_ok) {
        std::cout << "ERROR: frontend could not assign requested service resources. abort. " << std::endl;
        run_megamol = false;
    }

    auto frontend_resources = services.getProvidedResources();
    graph.AddModuleDependencies(frontend_resources);

    uint32_t frameID = 0;
    const auto render_next_frame = [&]() -> bool {
        // set global Frame Counter
        core.SetFrameID(frameID++);

        // services: receive inputs (GLFW poll events [keyboard, mouse, window], network, lua)
        services.updateProvidedResources();

        // aka simulation step
        // services: digest new inputs via ModuleResources (GUI digest user inputs, lua digest inputs, network ?)
        // e.g. graph updates, module and call creation via lua and GUI happen here
        services.digestChangedRequestedResources();

        // services tell us wheter we should shut down megamol
        if (services.shouldShutdown())
            return false;

        // actual rendering
        {
            services.preGraphRender(); // e.g. start frame timer, clear render buffers

            graph.RenderNextFrame(); // executes graph views, those digest input events like keyboard/mouse, then render

            services.postGraphRender(); // render GUI, glfw swap buffers, stop frame timer
        }

        services.resetProvidedResources(); // clear buffers holding glfw keyboard+mouse input

        return true;
    };

    // lua can issue rendering of frames
    lua_api.setFlushCallback(render_next_frame);

    // load project files via lua
    for (auto& file : config.project_files) {
        std::string result;
        if (megamol::core::utility::graphics::ScreenShotComments::EndsWithCaseInsensitive(file, ".png")) {
            if (!lua_api.RunString(
                    megamol::core::utility::graphics::ScreenShotComments::GetProjectFromPNG(file), result)) {
                std::cout << "Project file \"" << file << "\" did not execute correctly: " << result << std::endl;
                run_megamol = false;
            }
        } else {
            if (!lua_api.RunFile(file, result)) {
                std::cout << "Project file \"" << file << "\" did not execute correctly: " << result << std::endl;
                run_megamol = false;
            }
        }
    }
    if (config.load_example_project) {
        const bool graph_ok = set_up_example_graph(graph);
        if (!graph_ok) {
            std::cout << "ERROR: frontend could not build graph. abort. " << std::endl;
            run_megamol = false;
        }
    }

    while (run_megamol) {
        run_megamol = render_next_frame();
    }

    // close glfw context, network connections, other system resources
    services.close();

    // clean up modules, calls in graph
    // TODO: implement graph destructor

    return 0;
}

CLIConfig handle_cli_inputs(int argc, char* argv[]) {
    CLIConfig config;

    cxxopts::Options options(argv[0], "MegaMol Frontend 3000");

    config.program_invocation_string = std::string{argv[0]};

    // clang-format off
    // parse input project files
    options.positional_help("<additional project files>");
    options.add_options()
        ("project-files", "projects to load", cxxopts::value<std::vector<std::string>>())
        ("host", "address of lua host server, default: "+config.lua_host_address, cxxopts::value<std::string>())
        ("example", "load minimal test spheres example project", cxxopts::value<bool>())
        ("khrdebug", "enable OpenGL KHR debug messages", cxxopts::value<bool>())
        ("vsync", "enable VSync in OpenGL window", cxxopts::value<bool>())
        ("window", "set the window size and position, accepted format: WIDTHxHEIGHT[+POSX+POSY]", cxxopts::value<std::string>())
        ("fullscreen", "open maximized window", cxxopts::value<bool>())
        ("nodecoration", "open window without decorations", cxxopts::value<bool>())
        ("topmost", "open window that stays on top of all others", cxxopts::value<bool>())
        ("nocursor", "do not show mouse cursor inside window", cxxopts::value<bool>())
        ("help", "print help")
        ;
    // clang-format on

    options.parse_positional({"project-files"});

    try {
        auto parsed_options = options.parse(argc, argv);
        std::string res;

        if (parsed_options.count("help")) {
            std::cout << options.help({""}) << std::endl;
            exit(0);
        }


        // verify project files exist in file system
        if (parsed_options.count("project-files")) {
            const auto& v = parsed_options["project-files"].as<std::vector<std::string>>();
            for (const auto& p : v) {
                if (!stdfs::exists(p)) {
                    std::cout << "Project file \"" << p << "\" does not exist!" << std::endl;
                    std::exit(1);
                }
            }

            config.project_files = v;
        }

        if (parsed_options.count("host")) {
            config.lua_host_address = parsed_options["host"].as<std::string>();
        }

        config.load_example_project = parsed_options["example"].as<bool>();

        config.opengl_khr_debug = parsed_options["khrdebug"].as<bool>();
        config.opengl_vsync = parsed_options["vsync"].as<bool>();

        config.window_mode |= parsed_options["fullscreen"].as<bool>()   * CLIConfig::WindowMode::fullscreen;
        config.window_mode |= parsed_options["nodecoration"].as<bool>() * CLIConfig::WindowMode::nodecoration;
        config.window_mode |= parsed_options["topmost"].as<bool>()      * CLIConfig::WindowMode::topmost;
        config.window_mode |= parsed_options["nocursor"].as<bool>()     * CLIConfig::WindowMode::nocursor;

        if (parsed_options.count("window")) {
            auto s = parsed_options["window"].as<std::string>();
            // 'WIDTHxHEIGHT[+POSX+POSY]'
            // 'wxh+x+y' with optional '+x+y', e.g. 600x800+0+0 opens window in upper left corner
            std::regex geometry("(\\d+)x(\\d+)(?:\\+(\\d+)\\+(\\d+))?");
            std::smatch match;
            if (std::regex_match(s, match, geometry)) {
                config.window_size.push_back( /*width*/ std::stoul(match[1].str(), nullptr, 10) );
                config.window_size.push_back( /*height*/ std::stoul(match[2].str(), nullptr, 10) );
                if (match[3].matched) {
                    config.window_position.push_back( /*x*/ std::stoul(match[3].str(), nullptr, 10) );
                    config.window_position.push_back( /*y*/ std::stoul(match[4].str(), nullptr, 10) );
                }
            } else {
                std::cout << "window option needs to be in the following format: wxh+x+y or wxh" << std::endl;
                std::exit(1);
            }
        }
    } catch (cxxopts::option_not_exists_exception ex) {
        std::cout << ex.what() << std::endl;
        std::cout << options.help({""}) << std::endl;
        std::exit(1);
    }

    return config;
}

bool set_up_example_graph(megamol::core::MegaMolGraph& graph) {
#define check(X) \
    if (!X)      \
        return false;

    check(graph.CreateModule("View3D_2", "::view"));
    check(graph.CreateModule("SphereRenderer", "::spheres"));
    check(graph.CreateModule("TestSpheresDataSource", "::datasource"));
    check(graph.CreateCall("CallRender3D_2", "::view::rendering", "::spheres::rendering"));
    check(graph.CreateCall("MultiParticleDataCall", "::spheres::getdata", "::datasource::getData"));

    check(graph.SetGraphEntryPoint("::view", megamol::core::view::get_gl_view_runtime_resources_requests(),
        megamol::core::view::view_rendering_execution, megamol::core::view::view_init_rendering_state));

    std::string parameter_name("::datasource::numSpheres");
    auto parameterPtr = graph.FindParameter(parameter_name);
    if (parameterPtr) {
        parameterPtr->ParseValue("23");
    } else {
        std::cout << "ERROR: could not find parameter: " << parameter_name << std::endl;
        return false;
    }

    return true;
}
