#include "mmcore/CoreInstance.h"
#include "mmcore/MegaMolGraph.h"

#include "mmcore/utility/log/Log.h"
#include "mmcore/utility/log/DefaultTarget.h"

#include "RuntimeConfig.h"
#include "FrameStatistics_Service.hpp"
#include "FrontendServiceCollection.hpp"
#include "GUI_Service.hpp"
#include "Lua_Service_Wrapper.hpp"
#include "OpenGL_GLFW_Service.hpp"
#include "Screenshot_Service.hpp"
#include "ProjectLoader_Service.hpp"

#include "mmcore/view/AbstractView_EventConsumption.h"

#include <cxxopts.hpp>
#include "mmcore/LuaAPI.h"

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

using megamol::frontend_resources::RuntimeConfig;
RuntimeConfig handle_cli_and_config(const int argc, const char** argv, megamol::core::LuaAPI& lua);
std::vector<std::string> extract_config_file_paths(const int argc, const char** argv);
RuntimeConfig handle_config(RuntimeConfig config, megamol::core::LuaAPI& lua);
RuntimeConfig handle_cli(RuntimeConfig config, const int argc, const char** argv);

int main(const int argc, const char** argv) {

    bool lua_imperative_only = false; // allow mmFlush, mmList* and mmGetParam*
    megamol::core::LuaAPI lua_api(lua_imperative_only);

    auto config = handle_cli_and_config(argc, argv, lua_api);

    // setup log
    megamol::core::utility::log::Log::DefaultLog.SetLevel(megamol::core::utility::log::Log::LEVEL_ALL);
    megamol::core::utility::log::Log::DefaultLog.SetEchoLevel(megamol::core::utility::log::Log::LEVEL_ALL);
    megamol::core::utility::log::Log::DefaultLog.SetOfflineMessageBufferSize(100);
    megamol::core::utility::log::Log::DefaultLog.SetMainTarget(
        std::make_shared<megamol::core::utility::log::DefaultTarget>(megamol::core::utility::log::Log::LEVEL_ALL));

    megamol::core::CoreInstance core;
    core.Initialise(false); // false makes core not start his own lua service (else we collide on default port)

    megamol::frontend::OpenGL_GLFW_Service gl_service;
    megamol::frontend::OpenGL_GLFW_Service::Config openglConfig;
    openglConfig.windowTitlePrefix = "MegaMol";
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
    openglConfig.windowPlacement.fullScreen = config.window_mode & RuntimeConfig::WindowMode::fullscreen;
    openglConfig.windowPlacement.noDec      = config.window_mode & RuntimeConfig::WindowMode::nodecoration;
    openglConfig.windowPlacement.topMost    = config.window_mode & RuntimeConfig::WindowMode::topmost;
    openglConfig.windowPlacement.noCursor   = config.window_mode & RuntimeConfig::WindowMode::nocursor;
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

    megamol::frontend::Lua_Service_Wrapper lua_service_wrapper;
    megamol::frontend::Lua_Service_Wrapper::Config luaConfig;
    luaConfig.lua_api_ptr = &lua_api;
    luaConfig.host_address = config.lua_host_address;
    luaConfig.retry_socket_port = config.lua_host_port_retry;
    lua_service_wrapper.setPriority(0);

    megamol::frontend::ProjectLoader_Service projectloader_service;
    megamol::frontend::ProjectLoader_Service::Config projectloaderConfig;
    projectloader_service.setPriority(1);


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
    services.add(projectloader_service, &projectloaderConfig);

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

    const megamol::core::factories::ModuleDescriptionManager& moduleProvider = core.GetModuleDescriptionManager();
    const megamol::core::factories::CallDescriptionManager& callProvider = core.GetCallDescriptionManager();

    megamol::core::MegaMolGraph graph(core, moduleProvider, callProvider);

    // graph is also a resource that may be accessed by services
    services.getProvidedResources().push_back({"MegaMolGraph", graph});
    services.getProvidedResources().push_back({"RuntimeConfig", config});

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
        if (!projectloader_service.load_file(file)) {
            std::cout << "Project file \"" << file << "\" did not execute correctly"<< std::endl;
            run_megamol = false;

            // if interactive, continue to run MegaMol
            if (config.interactive) {
                std::cout << "Interactive mode: start MegaMol anyway"<< std::endl;
                run_megamol = true;
            }
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

#define config_option "--config"
static auto config_name = std::string(config_option).substr(2);
std::vector<std::string> extract_config_file_paths(const int argc, const char** argv) {
    // load config files from default paths
    // setting Config files from Lua is not possible
    // multiple Config files can be passed via CLI - then default config file paths are ignored

    // clang-format off
    // find config options in CLI string and overwrite default paths
    cxxopts::Options options(argv[0], "MegaMol Config Parsing");
    options.add_options()
        (config_name, "", cxxopts::value<std::vector<std::string>>());
    // clang-format on

    options.allow_unrecognised_options();

    auto error = [](auto const& what) {
        std::cout << what << std::endl;
        std::exit(1);
    };

    try {
        int _argc = argc;
        auto _argv = const_cast<char**>(argv);
        auto parsed_options = options.parse(_argc, _argv);

        std::vector<std::string> config_files;

        if (parsed_options.count(config_name) == 0) {
            // no config files given
            // use defaults
            RuntimeConfig config;
            config_files = config.configuration_files;
        } else {
            config_files = parsed_options[config_name].as<std::vector<std::string>>();
        }

        // check files exist
        for (const auto& file : config_files) {
            if (!stdfs::exists(file)) {
                std::cout << "Config file \"" << file << "\" does not exist!" << std::endl;
                std::exit(1);
            }
        }

        return config_files;

    } catch (cxxopts::option_not_exists_exception ex) {
        error(ex.what());
    } catch (cxxopts::missing_argument_exception ex) {
        error(ex.what());
    }
}

RuntimeConfig handle_cli_and_config(const int argc, const char** argv, megamol::core::LuaAPI& lua) {
    RuntimeConfig config;

    // config files are already checked to exist in file system
    config.configuration_files = extract_config_file_paths(argc, argv);

    // overwrite default values with values from config file
    config = handle_config(config, lua);

    // overwrite default and config values with CLI inputs
    config = handle_cli(config, argc, argv);

    return config;
}

RuntimeConfig handle_config(RuntimeConfig config, megamol::core::LuaAPI& lua) {

    // load config file
    auto& files = config.configuration_files;
    for (auto& file : files) {
        std::ifstream stream(file);
        std::string file_contents = std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
        std::string file_contents_as_cli;

        // interpret lua config commands as CLI commands
        bool lua_config_ok = lua.RunString(file_contents, file_contents_as_cli);

        if (!lua_config_ok) {
            // TODO: ERROR
        }

        config.configuration_file_contents.push_back(file_contents);
        config.configuration_file_contents_as_cli.push_back(file_contents_as_cli);
    }

    return config;
}


RuntimeConfig handle_cli(RuntimeConfig config, const int argc, const char** argv) {

    cxxopts::Options options(argv[0], "MegaMol Frontend 3000");

    config.program_invocation_string = std::string{argv[0]};

    // option handlers fill config struct with passed options
    auto empty_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
    };

    auto config_files_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        // is already done by first CLI pass which checks config files before running them through Lua
    };

    auto project_files_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        const auto& v = parsed_options[option_name].as<std::vector<std::string>>();
        for (const auto& p : v) {
            if (!stdfs::exists(p)) {
                std::cout << "Project file \"" << p << "\" does not exist!" << std::endl;
                std::exit(1);
            }
        }

        config.project_files = v;
    };

    auto host_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        config.lua_host_address = parsed_options[option_name].as<std::string>();
    };

    auto khrdebug_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        config.opengl_khr_debug = parsed_options[option_name].as<bool>();
    };
    auto vsync_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        config.opengl_vsync = parsed_options[option_name].as<bool>();
    };

    auto fullscreen_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        config.window_mode |= parsed_options[option_name].as<bool>() * RuntimeConfig::WindowMode::fullscreen;
    };
    auto nodecoration_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        config.window_mode |= parsed_options[option_name].as<bool>() * RuntimeConfig::WindowMode::nodecoration;
    };
    auto topmost_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        config.window_mode |= parsed_options[option_name].as<bool>() * RuntimeConfig::WindowMode::topmost;
    };
    auto nocursor_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        config.window_mode |= parsed_options[option_name].as<bool>() * RuntimeConfig::WindowMode::nocursor;
    };

    auto interactive_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        config.interactive = parsed_options[option_name].as<bool>();
    };

    auto window_handler = [&](std::string const& option_name, cxxopts::ParseResult const& parsed_options, RuntimeConfig& config)
    {
        auto s = parsed_options[option_name].as<std::string>();
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
    };

    // clang-format off
    std::vector<std::tuple<std::string, std::string, std::shared_ptr<cxxopts::Value>, std::function<void(std::string const&, cxxopts::ParseResult const&, RuntimeConfig&)>>>
    options_list =
    {
          {config_name,     "provide Lua configuration file",                                              cxxopts::value<std::vector<std::string>>(), config_files_handler }
        , {"project-files", "projects to load",                                                            cxxopts::value<std::vector<std::string>>(), project_files_handler}
        , {"host",          "address of lua host server, default: "+config.lua_host_address,               cxxopts::value<std::string>(),              host_handler         }
        , {"khrdebug",      "enable OpenGL KHR debug messages",                                            cxxopts::value<bool>(),                     khrdebug_handler     }
        , {"vsync",         "enable VSync in OpenGL window",                                               cxxopts::value<bool>(),                     vsync_handler        }
        , {"window",        "set the window size and position, accepted format: WIDTHxHEIGHT[+POSX+POSY]", cxxopts::value<std::string>(),              window_handler       }
        , {"fullscreen",    "open maximized window",                                                       cxxopts::value<bool>(),                     fullscreen_handler   }
        , {"nodecoration",  "open window without decorations",                                             cxxopts::value<bool>(),                     nodecoration_handler }
        , {"topmost",       "open window that stays on top of all others",                                 cxxopts::value<bool>(),                     topmost_handler      }
        , {"nocursor",      "do not show mouse cursor inside window",                                      cxxopts::value<bool>(),                     nocursor_handler     }
        , {"interactive",   "open MegaMol even if project files via CLI could not be loaded",              cxxopts::value<bool>(),                     interactive_handler  }
    };

    #define add_option(X) (std::get<0>(X), std::get<1>(X), std::get<2>(X))

    // parse input project files
    options.positional_help("<additional project files>");
    options.add_options()
        add_option(options_list[0])
        add_option(options_list[1])
        add_option(options_list[2])
        add_option(options_list[3])
        add_option(options_list[4])
        add_option(options_list[5])
        add_option(options_list[6])
        add_option(options_list[7])
        add_option(options_list[8])
        add_option(options_list[9])
        add_option(options_list[10])
        ("help", "print help")
        ;
    // clang-format on

    options.parse_positional({"project-files"});

    // actually process passed options
    try {
        int _argc = argc;
        auto _argv = const_cast<char**>(argv);
        auto parsed_options = options.parse(_argc, _argv);
        std::string res;

        if (parsed_options.count("help")) {
            std::cout << options.help({""}) << std::endl;
            exit(0);
        }

        for (auto& option : options_list) {
            auto& option_name = std::get<0>(option);
            if (parsed_options.count(option_name)) {
                auto& option_handler = std::get<3>(option);
                option_handler(option_name, parsed_options, config);
            }
        }

    } catch (cxxopts::option_not_exists_exception ex) {
        std::cout << ex.what() << std::endl;
        std::cout << options.help({""}) << std::endl;
        std::exit(1);
    }

    return config;
}

