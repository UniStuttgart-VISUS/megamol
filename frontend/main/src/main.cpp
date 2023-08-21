/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "CLIConfigParsing.h"
#include "CUDA_Service.hpp"
#include "Command_Service.hpp"
#include "FrameStatistics_Service.hpp"
#include "FrontendServiceCollection.hpp"
#include "GUI_Service.hpp"
#include "GlobalValueStore.h"
#include "ImagePresentation_Service.hpp"
#include "Lua_Service_Wrapper.hpp"
#include "OpenGL_GLFW_Service.hpp"
#include "PluginsResource.h"
#include "Profiling_Service.hpp"
#include "ProjectLoader_Service.hpp"
#include "Remote_Service.hpp"
#include "RuntimeConfig.h"
#include "Screenshot_Service.hpp"
#include "VR_Service.hpp"
#include "mmcore/LuaAPI.h"
#include "mmcore/MegaMolGraph.h"
#include "mmcore/factories/PluginRegister.h"
#include "mmcore/utility/log/Log.h"

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>
#endif

#ifdef MEGAMOL_USE_POWER
#include "Power_Service.hpp"
#endif

using megamol::core::utility::log::Log;


static void log(std::string const& text) {
    const std::string msg = "Main: " + text;
    Log::DefaultLog.WriteInfo(msg.c_str());
}

static void log_warning(std::string const& text) {
    const std::string msg = "Main: " + text;
    Log::DefaultLog.WriteWarn(msg.c_str());
}

static void log_error(std::string const& text) {
    const std::string msg = "Main: " + text;
    Log::DefaultLog.WriteError(msg.c_str());
}

void loadPlugins(megamol::frontend_resources::PluginsResource& pluginsRes);

int main(const int argc, const char** argv) {
#if defined(MEGAMOL_DETECT_MEMLEAK) && defined(DEBUG)
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
#ifdef MEGAMOL_USE_TRACY
    tracy::StartupProfiler();
    //ZoneScoped;
    TracyCZone(main, true);
#endif
    megamol::core::LuaAPI lua_api;

    auto [config, global_value_store] = megamol::frontend::handle_cli_and_config(argc, argv, lua_api);

    const bool with_gl = !config.no_opengl;

    // setup log
    Log::DefaultLog.SetLevel(config.log_level);
    Log::DefaultLog.SetEchoLevel(config.echo_level);
    if (!config.log_file.empty())
        Log::DefaultLog.AddFileTarget(config.log_file.data(), false);

    log(config.as_string());
    log(global_value_store.as_string());

    megamol::frontend::OpenGL_GLFW_Service gl_service;
    megamol::frontend::OpenGL_GLFW_Service::Config openglConfig;
    openglConfig.windowTitlePrefix = "MegaMol";
    if (config.opengl_context_version.has_value()) {
        openglConfig.versionMajor = std::get<0>(config.opengl_context_version.value());
        openglConfig.versionMinor = std::get<1>(config.opengl_context_version.value());
        openglConfig.glContextCoreProfile = std::get<2>(config.opengl_context_version.value());
    }
    openglConfig.enableKHRDebug = config.opengl_khr_debug;
    openglConfig.enableVsync = config.opengl_vsync;
    // pass window size and position
    if (config.window_size.has_value()) {
        openglConfig.windowPlacement.size = true;
        openglConfig.windowPlacement.w = config.window_size.value().first;
        openglConfig.windowPlacement.h = config.window_size.value().second;
    }
    if (config.window_position.has_value()) {
        openglConfig.windowPlacement.pos = true;
        openglConfig.windowPlacement.x = config.window_position.value().first;
        openglConfig.windowPlacement.y = config.window_position.value().second;
    }
    openglConfig.windowPlacement.mon = config.window_monitor;
    using megamol::frontend_resources::RuntimeConfig;
    openglConfig.windowPlacement.fullScreen = config.window_mode & RuntimeConfig::WindowMode::fullscreen;
    openglConfig.windowPlacement.noDec = config.window_mode & RuntimeConfig::WindowMode::nodecoration;
    openglConfig.windowPlacement.topMost = config.window_mode & RuntimeConfig::WindowMode::topmost;
    openglConfig.windowPlacement.noCursor = config.window_mode & RuntimeConfig::WindowMode::nocursor;
    openglConfig.windowPlacement.hidden = config.window_mode & RuntimeConfig::WindowMode::hidden;
    openglConfig.forceWindowSize = config.force_window_size;
    gl_service.setPriority(2);

    megamol::frontend::GUI_Service gui_service;
    megamol::frontend::GUI_Service::Config guiConfig;
    guiConfig.backend = (with_gl) ? (megamol::gui::GUIRenderBackend::OPEN_GL) : (megamol::gui::GUIRenderBackend::CPU);
    guiConfig.gui_show = config.gui_show;
    guiConfig.gui_scale = config.gui_scale;
    // priority must be higher than priority of gl_service (=1)
    // service callbacks get called in order of priority of the service.
    // postGraphRender() and close() are called in reverse order of priorities.
    gui_service.setPriority(23);

    megamol::frontend::Screenshot_Service screenshot_service;
    megamol::frontend::Screenshot_Service::Config screenshotConfig;
    screenshotConfig.show_privacy_note = config.screenshot_show_privacy_note;
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
    luaConfig.show_version_notification = config.show_version_note;
    lua_service_wrapper.setPriority(0);

    megamol::frontend::ProjectLoader_Service projectloader_service;
    megamol::frontend::ProjectLoader_Service::Config projectloaderConfig;
    projectloader_service.setPriority(1);

    megamol::frontend::ImagePresentation_Service imagepresentation_service;
    megamol::frontend::ImagePresentation_Service::Config imagepresentationConfig;
    imagepresentationConfig.local_framebuffer_resolution = config.local_framebuffer_resolution;

    // when there is no GL we should make sure the user defined some initial framebuffer size via CLI
    if (!with_gl) {
        if (!config.local_framebuffer_resolution.has_value()) {
            if (!config.window_size.has_value()) {
                log_error("Window and framebuffer size is not set. Abort.");
                return 1;
            }
            imagepresentationConfig.local_framebuffer_resolution = config.window_size;
        }
    }

    imagepresentationConfig.local_viewport_tile =
        config.local_viewport_tile.has_value()
            ? std::make_optional(megamol::frontend::ImagePresentation_Service::Config::Tile{
                  config.local_viewport_tile.value().global_framebuffer_resolution,
                  config.local_viewport_tile.value().tile_start_pixel,
                  config.local_viewport_tile.value().tile_resolution})
            : std::nullopt;
    imagepresentation_service.setPriority(3);

    megamol::frontend::VR_Service vr_service;
    vr_service.setPriority(imagepresentation_service.getPriority() - 1);
    megamol::frontend::VR_Service::Config vrConfig;
    vrConfig.mode = megamol::frontend::VR_Service::Config::Mode(static_cast<int>(config.vr_mode));
    const bool with_vr = vrConfig.mode != megamol::frontend::VR_Service::Config::Mode::Off;

    megamol::frontend::Command_Service command_service;
    // Should be applied after gui service to process only keyboard events not used by gui.
    command_service.setPriority(24);

    megamol::frontend::Profiling_Service profiling_service;
    megamol::frontend::Profiling_Service::Config profiling_config;
    profiling_config.log_file = config.profiling_output_file;
    profiling_config.flush_frequency = config.flush_frequency;
    profiling_config.autostart_profiling = config.autostart_profiling;
    profiling_config.include_graph_events = config.include_graph_events;

#ifdef MEGAMOL_USE_POWER
    std::list<std::string> power_str_container;
    megamol::frontend::Power_Service power_service;
    megamol::frontend::Power_Service::Config power_config;
    power_config.lpt = config.power_lpt;
    power_config.write_to_files = config.power_write_file;
    power_config.folder = config.power_folder;
    power_config.str_container = &power_str_container;
    power_service.setPriority(1);
#endif

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
#ifdef MEGAMOL_USE_POWER
    services.add(power_service, &power_config);
#endif
    if (with_gl) {
        services.add(gl_service, &openglConfig);
    }
    services.add(gui_service, &guiConfig);
    services.add(lua_service_wrapper, &luaConfig);
    services.add(screenshot_service, &screenshotConfig);
    services.add(framestatistics_service, &framestatisticsConfig);
    services.add(projectloader_service, &projectloaderConfig);
    services.add(imagepresentation_service, &imagepresentationConfig);
    services.add(command_service, nullptr);

    if (with_vr) {
        services.add(vr_service, &vrConfig);
    }

    services.add(profiling_service, &profiling_config);

#ifdef MM_CUDA_ENABLED
    services.add(cuda_service, nullptr);
#endif

    megamol::frontend::Remote_Service remote_service;
    megamol::frontend::Remote_Service::Config remoteConfig;
    if (auto remote_session_role = handle_remote_session_config(config, remoteConfig); !remote_session_role.empty()) {
        openglConfig.windowTitlePrefix += remote_session_role;
        remote_service.setPriority(
            lua_service_wrapper.getPriority() - 1); // remote does stuff before everything else, even before lua
        services.add(remote_service, &remoteConfig);
    }

    const bool init_ok = services.init(); // runs init(config_ptr) on all services with provided config sructs

    if (!init_ok) {
        log_error("Some frontend service could not be initialized successfully. Abort.");
        services.close();
        return 1;
    }

    megamol::frontend_resources::PluginsResource pluginsRes;
    loadPlugins(pluginsRes);
    services.getProvidedResources().push_back({"PluginsResource", pluginsRes});

    megamol::core::MegaMolGraph graph(pluginsRes.all_module_descriptions, pluginsRes.all_call_descriptions);

    // Graph and Config are also a resources that may be accessed by services
    services.getProvidedResources().push_back({megamol::frontend_resources::MegaMolGraph_Req_Name, graph});
    services.getProvidedResources().push_back(
        {megamol::frontend_resources::MegaMolGraph_SubscriptionRegistry_Req_Name, graph.GraphSubscribers()});
    services.getProvidedResources().push_back({"RuntimeConfig", config});
    services.getProvidedResources().push_back({"GlobalValueStore", global_value_store});

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

    const auto render_next_frame = [&]() -> bool {
#ifdef MEGAMOL_USE_TRACY
        ZoneScopedNC("RenderNextFrame", 0x0000FF);
#endif

        // services: receive inputs (GLFW poll events [keyboard, mouse, window], network, lua)
        services.updateProvidedResources();

        // aka simulation step
        // services: digest new inputs via FrontendResources (GUI digest user inputs, lua digest inputs, network ?)
        // e.g. graph updates, module and call creation via lua and GUI happen here
        services.digestChangedRequestedResources();

        // services tell us whether we should shut down megamol
        if (services.shouldShutdown())
            return false;

        // actual rendering
        {
            services.preGraphRender(); // e.g. start frame timer, clear render buffers

            imagepresentation_service
                .RenderNextFrame(); // executes graph views, those digest input events like keyboard/mouse, then render

            services.postGraphRender(); // render GUI, glfw swap buffers, stop frame timer
        }

        imagepresentation_service
            .PresentRenderedImages(); // draws rendering results to GLFW window, writes images to disk, sends images via network...

        services.resetProvidedResources(); // clear buffers holding glfw keyboard+mouse input

        return true;
    };

    // lua can issue rendering of frames, we provide a resource for this
    const std::function<bool()> render_next_frame_func = [&]() -> bool { return render_next_frame(); };
    services.getProvidedResources().push_back({"RenderNextFrame", render_next_frame_func});

    // image presentation service needs to assign frontend resources to entry points
    auto& frontend_resources = services.getProvidedResources();
    services.getProvidedResources().push_back({"FrontendResources", frontend_resources});

    int ret = 0;

    // distribute registered resources among registered services.
    const bool resources_ok = services.assignRequestedResources();
    // for each service we call their resource callbacks here:
    //    std::vector<FrontendResource>& getProvidedResources()
    //    std::vector<std::string> getRequestedResourceNames()
    //    void setRequestedResources(std::vector<FrontendResource>& resources)
    if (!resources_ok) {
        log_error("Frontend could not assign requested service resources. Abort.");
        run_megamol = false;
        ret += 1;
    }

    bool graph_resources_ok = graph.AddFrontendResources(frontend_resources);
    if (!graph_resources_ok) {
        log_error("Graph did not get resources he needs from frontend. Abort.");
        run_megamol = false;
        ret += 2;
    }

    // load project files via lua
    if (run_megamol && graph_resources_ok)
        for (auto& file : config.project_files) {
            if (!projectloader_service.load_file(file)) {
                log_error("Project file \"" + file + "\" did not execute correctly");
                run_megamol = false;
                ret += 4;

                // if interactive, continue to run MegaMol
                if (config.interactive) {
                    log_warning("Interactive mode: start MegaMol anyway");
                    run_megamol = true;
                }
            }
        }

    // execute Lua commands passed via CLI
    if (graph_resources_ok)
        if (!config.cli_execute_lua_commands.empty()) {
            std::string lua_result;
            bool cli_lua_ok = lua_api.RunString(config.cli_execute_lua_commands, lua_result);
            if (!cli_lua_ok) {
                run_megamol = false;
                ret += 8;
                log_error("Error in CLI Lua command: " + lua_result);
            }
        }

    while (run_megamol) {
#ifdef MEGAMOL_USE_TRACY
        ZoneScopedNC("MainLoop", 0x0000FF);
#endif
        run_megamol = render_next_frame();
    }

    graph.Clear();

    // close glfw context, network connections, other system resources
    services.close();

#ifdef MEGAMOL_USE_TRACY
    TracyCZoneEnd(main); 
    tracy::ShutdownProfiler();
#endif

    return ret;
}

void loadPlugins(megamol::frontend_resources::PluginsResource& pluginsRes) {
    for (auto const& pluginDesc : megamol::core::factories::PluginRegister::getAll()) {
        try {
            auto new_plugin = pluginDesc->create();
            pluginsRes.plugins.push_back(new_plugin);

            // report success
            Log::DefaultLog.WriteInfo("Plugin \"%s\" loaded: %u Modules, %u Calls",
                new_plugin->GetObjectFactoryName().c_str(), new_plugin->GetModuleDescriptionManager().Count(),
                new_plugin->GetCallDescriptionManager().Count());

            for (auto const& md : new_plugin->GetModuleDescriptionManager()) {
                try {
                    pluginsRes.all_module_descriptions.Register(md);
                } catch (std::invalid_argument const&) {
                    Log::DefaultLog.WriteError(
                        "Failed to load module description \"%s\": Naming conflict", md->ClassName());
                }
            }
            for (auto const& cd : new_plugin->GetCallDescriptionManager()) {
                try {
                    pluginsRes.all_call_descriptions.Register(cd);
                } catch (std::invalid_argument const&) {
                    Log::DefaultLog.WriteError(
                        "Failed to load call description \"%s\": Naming conflict", cd->ClassName());
                }
            }

        } catch (vislib::Exception const& vex) {
            Log::DefaultLog.WriteError(
                "Unable to load Plugin: %s (%s, &d)", vex.GetMsgA(), vex.GetFile(), vex.GetLine());
        } catch (std::exception const& ex) {
            Log::DefaultLog.WriteError("Unable to load Plugin: %s", ex.what());
        } catch (...) {
            Log::DefaultLog.WriteError("Unable to load Plugin: unknown exception");
        }
    }
}
