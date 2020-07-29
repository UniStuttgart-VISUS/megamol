#include "mmcore/CoreInstance.h"
#include "mmcore/MegaMolGraph.h"

#include "mmcore/utility/log/Log.h"

#include "AbstractFrontendService.hpp"
#include "FrontendServiceCollection.hpp"
#include "GUI_Service.hpp"
#include "OpenGL_GLFW_Service.hpp"

#include "mmcore/view/AbstractView_EventConsumption.h"

#include <cxxopts.hpp>
#include <filesystem>
#include "mmcore/LuaAPI.h"

bool set_up_graph(megamol::core::MegaMolGraph& graph, std::vector<megamol::frontend::ModuleResource>& module_resources);

int main(int argc, char* argv[]) {

    // setup log
    megamol::core::utility::log::Log::DefaultLog.SetLogFileName(static_cast<const char*>(NULL), false);
    megamol::core::utility::log::Log::DefaultLog.SetLevel(megamol::core::utility::log::Log::LEVEL_ALL);
    megamol::core::utility::log::Log::DefaultLog.SetEchoLevel(megamol::core::utility::log::Log::LEVEL_ALL);
    megamol::core::utility::log::Log::DefaultLog.SetEchoTarget(
        std::make_shared<megamol::core::utility::log::Log::StreamTarget>(
            std::cout, megamol::core::utility::log::Log::LEVEL_ALL));

    megamol::core::CoreInstance core;
    core.Initialise();

    const megamol::core::factories::ModuleDescriptionManager& moduleProvider = core.GetModuleDescriptionManager();
    const megamol::core::factories::CallDescriptionManager& callProvider = core.GetCallDescriptionManager();

    megamol::frontend::OpenGL_GLFW_Service gl_service;
    megamol::frontend::OpenGL_GLFW_Service::Config openglConfig;
    openglConfig.windowTitlePrefix = openglConfig.windowTitlePrefix + " ~ Main3000";
    openglConfig.versionMajor = 4;
    openglConfig.versionMinor = 6;
    gl_service.setPriority(1);

    megamol::frontend::GUI_Service gui_service;
    megamol::frontend::GUI_Service::Config guiConfig;
    guiConfig.imgui_api = megamol::frontend::GUI_Service::ImGuiAPI::OPEN_GL;
    guiConfig.core_instance = &core;
    // priority must be higher than priority of gl_service (=1)
    // service callbacks get called in order of priority of the service.
    // postGraphRender() and close() are called in reverse order of priorities.
    gui_service.setPriority(23);

    megamol::core::MegaMolGraph graph(core, moduleProvider, callProvider);

    // the main loop is organized around services that can 'do something' in different parts of the main loop
    // a service is something that implements the AbstractFrontendService interface from 'megamol\render_api\include'
    // a central mechanism that allows services to communicate with each other and with graph modules are _resources_
    // (see ModuleResource in 'megamol\render_api\include') services may provide resources to the system and they may
    // request resources they need themselves for functioning. think of a resource as a struct (or some type of your
    // choice) that gets wrapped by a helper structure and gets a name attached to it. the fronend makes sure (at least
    // attempts to) to hand each service the resources it requested, or else fail execution of megamol with an error
    // message. resource assignment is done by the name of the resource, so this is a very loose interface based on
    // trust. type safety of resources is ensured in the sense that extracting the wrong type from a ModuleResource will
    // lead to an unhandled bad type cast exception, leading to the shutdown of megamol.
    bool run_megamol = true;
    megamol::frontend::FrontendServiceCollection services;
    services.add(gl_service, &openglConfig);
    services.add(gui_service, &guiConfig);

    const bool init_ok = services.init(); // runs init(config_ptr) on all services with provided config sructs

    if (!init_ok) {
        std::cout << "ERROR: some service could not be initialized successfully. abort. " << std::endl;
        services.close();
        return 1;
    }

    // graph is also a resource that may be accessed by services
    // TODO: how to solve const and non-const resources?
    // TODO: graph manipulation during execution of graph modules is problematic, undefined?
    services.getProvidedResources().push_back({"MegaMolGraph", graph});

    // distribute registered resources among registered services.
    const bool resources_ok = services.assignRequestedResources();
    // for each service we call their resource callbacks here:
    //    std::vector<ModuleResource>& getProvidedResources()
    //    std::vector<std::string> getRequestedResourceNames()
    //    void setRequestedResources(std::vector<ModuleResource>& resources)
    if (!resources_ok) {
        std::cout << "ERROR: frontend could not assign requested service resources. abort. " << std::endl;
        run_megamol = false;
    }

    const bool graph_ok = set_up_graph(graph, services.getProvidedResources()); // fill graph with modules and calls
    if (!graph_ok) {
        std::cout << "ERROR: frontend could not build graph. abort. " << std::endl;
        run_megamol = false;
    }

    while (run_megamol) {
        // services: receive inputs (GLFW poll events [keyboard, mouse, window], network, lua)
        services.updateProvidedResources();

        // aka simulation step
        // services: digest new inputs via ModuleResources (GUI digest user inputs, lua digest inputs, network ?)
        // e.g. graph updates, module and call creation via lua and GUI happen here
        services.digestChangedRequestedResources();

        // services tell us wheter we should shut down megamol
        // TODO: service needs to mark intself as shutdown by calling this->setShutdown() during
        // digestChangedRequestedResources()
        if (services.shouldShutdown()) break;

        {                              // put this in render function so LUA can call it
            services.preGraphRender(); // e.g. start frame timer, clear render buffers

            graph.RenderNextFrame(); // executes graph views, those digest input events like keyboard/mouse, then render

            services.postGraphRender(); // render GUI, glfw swap buffers, stop frame timer
            // problem: guarantee correct order of pre- and post-render jobs, i.e. render gui before swapping buffers
        }

        services.resetProvidedResources(); // clear buffers holding glfw keyboard+mouse input
    }

    // close glfw context, network connections, other system resources
    services.close();

    // clean up modules, calls in graph
    // TODO: implement graph destructor

    return 0;
}

bool set_up_graph(megamol::core::MegaMolGraph& graph, std::vector<megamol::frontend::ModuleResource>& module_resources) {
#if 0
    megamol::core::LuaAPI lua_api(graph, true);

    cxxopts::Options options(argv[0], "MegaMol Frontend 3000");
    options.positional_help("<additional project files>");
    options.add_options()("project-files", "projects to load", cxxopts::value<std::vector<std::string>>());
    options.parse_positional({"project-files"});
    auto parsed_options = options.parse(argc, argv);
    std::string res;
    if (parsed_options.count("project-files")) {
        const auto& v = parsed_options["project-files"].as<std::vector<std::string>>();
        for (const auto& p : v) {
            if (std::filesystem::exists(p)) {
                if (!lua_api.RunFile(p, res)) {
                    std::cout << "Project file \"" << p << "\" did not execute correctly: " << res << std::endl;
                }
            } else {
                std::cout << "Project file \"" << p << "\" does not exist!" << std::endl;
            }
        }
    }

#else

#    define check(X)                                                                                                   \
        if (!X) return false;

    graph.AddModuleDependencies(module_resources);

    /// check(graph.CreateModule("GUIView", "::guiview"));
    check(graph.CreateModule("View3D_2", "::view"));
    check(graph.CreateModule("SphereRenderer", "::spheres"));
    check(graph.CreateModule("TestSpheresDataSource", "::datasource"));
    check(graph.CreateCall("CallRender3D_2", "::view::rendering", "::spheres::rendering"));
    check(graph.CreateCall("MultiParticleDataCall", "::spheres::getdata", "::datasource::getData"));
    /// check(graph.CreateCall("CallRenderView", "::guiview::renderview", "::view::render"));

    static std::vector<std::string> view_resource_requests = {
        "KeyboardEvents", "MouseEvents", "WindowEvents", "FramebufferEvents", "IOpenGL_Context"};

    // note: this is work in progress and more of a working prototype than a final design
    // callback executed by the graph for each frame
    // knows how to make a view module process input events and start the rendering
    auto view_rendering_execution = [&](megamol::core::Module::ptr_type module_ptr,
                                        std::vector<megamol::frontend::ModuleResource> const& resources) {
        megamol::core::view::AbstractView* view_ptr =
            dynamic_cast<megamol::core::view::AbstractView*>(module_ptr.get());

        assert(view_resource_requests.size() == resources.size());

        if (!view_ptr) {
            std::cout << "error. module is not a view module. could not set as graph rendering entry point."
                      << std::endl;
            return false;
        }

        megamol::core::view::AbstractView& view = *view_ptr;

        int i = 0;
        // resources are in order of initial requests
        megamol::core::view::view_consume_keyboard_events(view, resources[i++]);
        megamol::core::view::view_consume_mouse_events(view, resources[i++]);
        megamol::core::view::view_consume_window_events(view, resources[i++]);
        megamol::core::view::view_consume_framebuffer_events(view, resources[i++]);
        megamol::core::view::view_poke_rendering(view, resources[i++]);
    };

    /// check(graph.SetGraphEntryPoint("::guiview", view_resource_requests, view_rendering_execution));
    check(graph.SetGraphEntryPoint("::view", view_resource_requests, view_rendering_execution));

    std::string parameter_name("::datasource::numSpheres");
    auto parameterPtr = graph.FindParameter(parameter_name);
    if (parameterPtr) {
        parameterPtr->ParseValue("23");
    } else {
        std::cout << "ERROR: could not find parameter: " << parameter_name << std::endl;
        return false;
    }
#endif

    return true;
}
