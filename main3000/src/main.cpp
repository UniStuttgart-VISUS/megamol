#include "mmcore/CoreInstance.h"
#include "mmcore/MegaMolGraph.h"

#include "mmcore/utility/log/Log.h"
#include "mmcore/utility/log/StreamTarget.h"

#include "AbstractFrontendService.hpp"
#include "FrontendServiceCollection.hpp"
#include "GUI_Service.hpp"
#include "OpenGL_GLFW_Service.hpp"
#include "Lua_Service_Wrapper.hpp"

#include "mmcore/view/AbstractView_EventConsumption.h"

#include <cxxopts.hpp>
#include "mmcore/LuaAPI.h"

// Filesystem
#if defined(_HAS_CXX17) || ((defined(_MSC_VER) && (_MSC_VER > 1916))) // C++2017 or since VS2019
#    include <filesystem>
namespace stdfs = std::filesystem;
#else
// WINDOWS
#    ifdef _WIN32
#        include <filesystem>
namespace stdfs = std::experimental::filesystem;
#    else
// LINUX
#        include <experimental/filesystem>
namespace stdfs = std::experimental::filesystem;
#    endif
#endif

struct CLIConfig {
    std::vector<std::string> project_files;
};

CLIConfig handle_cli_inputs(int argc, char* argv[]);

bool set_up_graph(megamol::core::MegaMolGraph& graph);

int main(int argc, char* argv[]) {

    auto config = handle_cli_inputs(argc, argv);

    // setup log
    megamol::core::utility::log::Log::DefaultLog.SetLogFileName(static_cast<const char*>(NULL), false);
    megamol::core::utility::log::Log::DefaultLog.SetLevel(megamol::core::utility::log::Log::LEVEL_ALL);
    megamol::core::utility::log::Log::DefaultLog.SetEchoLevel(megamol::core::utility::log::Log::LEVEL_ALL);
    megamol::core::utility::log::Log::DefaultLog.SetEchoTarget(
        std::make_shared<megamol::core::utility::log::StreamTarget>(
            std::cout, megamol::core::utility::log::Log::LEVEL_ALL));

    megamol::core::CoreInstance core;
    core.Initialise(false); // false makes core not start his own lua service (else we collide on default port)

    const megamol::core::factories::ModuleDescriptionManager& moduleProvider = core.GetModuleDescriptionManager();
    const megamol::core::factories::CallDescriptionManager& callProvider = core.GetCallDescriptionManager();

    megamol::frontend::OpenGL_GLFW_Service gl_service;
    megamol::frontend::OpenGL_GLFW_Service::Config openglConfig;
    openglConfig.windowTitlePrefix = openglConfig.windowTitlePrefix + " ~ Main3000";
    openglConfig.versionMajor = 4;
    openglConfig.versionMinor = 5;
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

    bool lua_imperative_only = false; // allow mmFlush, mmList* and mmGetParam*
    megamol::core::LuaAPI lua_api(graph, lua_imperative_only);
    megamol::frontend::Lua_Service_Wrapper lua_service_wrapper;
    megamol::frontend::Lua_Service_Wrapper::Config luaConfig;
    luaConfig.lua_api_ptr = &lua_api;
    lua_service_wrapper.setPriority(0);

    // the main loop is organized around services that can 'do something' in different parts of the main loop
    // a service is something that implements the AbstractFrontendService interface from 'megamol\frontend_services\include'
    // a central mechanism that allows services to communicate with each other and with graph modules are _resources_
    // (see ModuleResource in 'megamol\module_resources\include') services may provide resources to the system and they may
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
        std::cout << "ERROR: some frontend service could not be initialized successfully. abort. " << std::endl;
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

    auto module_resources = services.getProvidedResources();
    graph.AddModuleDependencies(module_resources);

    const auto render_next_frame = [&]() -> bool {
        // services: receive inputs (GLFW poll events [keyboard, mouse, window], network, lua)
        services.updateProvidedResources();

        // aka simulation step
        // services: digest new inputs via ModuleResources (GUI digest user inputs, lua digest inputs, network ?)
        // e.g. graph updates, module and call creation via lua and GUI happen here
        services.digestChangedRequestedResources();

        // services tell us wheter we should shut down megamol
        // TODO: service needs to mark intself as shutdown by calling this->setShutdown() during
        // digestChangedRequestedResources()
        if (services.shouldShutdown())
            return false;

        {                              // put this in render function so LUA can call it
            services.preGraphRender(); // e.g. start frame timer, clear render buffers

            graph.RenderNextFrame(); // executes graph views, those digest input events like keyboard/mouse, then render

            services.postGraphRender(); // render GUI, glfw swap buffers, stop frame timer
            // problem: guarantee correct order of pre- and post-render jobs, i.e. render gui before swapping buffers
        }

        services.resetProvidedResources(); // clear buffers holding glfw keyboard+mouse input

        return true;
    };

    // lua can issue rendering of frames
    lua_api.setFlushCallback(render_next_frame);

	if (config.project_files.empty()) {
		const bool graph_ok = set_up_graph(graph); // fill graph with modules and calls
		if (!graph_ok) {
		    std::cout << "ERROR: frontend could not build graph. abort. " << std::endl;
		    run_megamol = false;
		}
    } else {
        // load project files via lua
        for (auto& file : config.project_files) {
		    std::string result;
            if (!lua_api.RunFile(file, result)) {
                std::cout << "Project file \"" << file << "\" did not execute correctly: " << result << std::endl;
		        run_megamol = false;
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

CLIConfig handle_cli_inputs(int argc, char* argv[]) {
    CLIConfig config;

    cxxopts::Options options(argv[0], "MegaMol Frontend 3000");

    // parse input project files
    options.positional_help("<additional project files>");
    options.add_options()("project-files", "projects to load", cxxopts::value<std::vector<std::string>>());
    options.parse_positional({"project-files"});

    auto parsed_options = options.parse(argc, argv);
    std::string res;

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

    return config;
}

bool set_up_graph(megamol::core::MegaMolGraph& graph) {
#define check(X) \
    if (!X) return false;

    check(graph.CreateModule("View3D_2", "::view"));
    check(graph.CreateModule("SphereRenderer", "::spheres"));
    check(graph.CreateModule("TestSpheresDataSource", "::datasource"));
    check(graph.CreateCall("CallRender3D_2", "::view::rendering", "::spheres::rendering"));
    check(graph.CreateCall("MultiParticleDataCall", "::spheres::getdata", "::datasource::getData"));

    check(graph.SetGraphEntryPoint("::view", megamol::core::view::get_gl_view_runtime_resources_requests(), megamol::core::view::view_rendering_execution));

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
