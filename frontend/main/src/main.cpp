#include "CLIConfigParsing.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/MegaMolGraph.h"

#include "mmcore/utility/log/DefaultTarget.h"
#include "mmcore/utility/log/Log.h"

#include "RuntimeConfig.h"
#include "CUDA_Service.hpp"
#include "FrameStatistics_Service.hpp"
#include "FrontendServiceCollection.hpp"
#include "GUI_Service.hpp"
#include "Lua_Service_Wrapper.hpp"
#include "OpenGL_GLFW_Service.hpp"
#include "Screenshot_Service.hpp"
#include "ProjectLoader_Service.hpp"

#include "mmcore/view/AbstractView_EventConsumption.h"

#include "mmcore/LuaAPI.h"

int main(const int argc, const char** argv) {

    bool lua_imperative_only = false; // allow mmFlush, mmList* and mmGetParam*
    megamol::core::LuaAPI lua_api(lua_imperative_only);

    auto config = megamol::frontend::handle_cli_and_config(argc, argv, lua_api);

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
    using megamol::frontend_resources::RuntimeConfig;
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
    services.add(projectloader_service, &projectloaderConfig);

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

    graph.Clear();

    // close glfw context, network connections, other system resources
    services.close();

    return 0;
}

