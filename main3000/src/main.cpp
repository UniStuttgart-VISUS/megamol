#include "mmcore/CoreInstance.h"
#include "mmcore/MegaMolGraph.h"

#include "AbstractRenderAPI.hpp"
#include "OpenGL_GLFW_RAPI.hpp"

#include "mmcore/view/AbstractView_EventConsumption.h"

#include "mmcore/LuaAPI.h"
#include <cxxopts.hpp>
#include <filesystem>

int main(int argc, char* argv[]) {
    megamol::core::CoreInstance core;
    core.Initialise();

    const megamol::core::factories::ModuleDescriptionManager& moduleProvider = core.GetModuleDescriptionManager();
    const megamol::core::factories::CallDescriptionManager& callProvider = core.GetCallDescriptionManager();

    megamol::frontend::OpenGL_GLFW_Service gl_service;

    megamol::frontend::OpenGL_GLFW_Service::Config openglConfig;
    openglConfig.windowTitlePrefix = openglConfig.windowTitlePrefix + " ~ Main3000";
    openglConfig.versionMajor = 4;
    openglConfig.versionMinor = 6;
    gl_service.init(&openglConfig);

    auto services;
    services.add(gl_service);

    megamol::core::MegaMolGraph graph(core, moduleProvider, callProvider);

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
	// for_all services: register render resources
	// graph.addResources(all_resources)

	// lua_service.set_zmq( central_resources.request(zmq_instance) )

    graph.AddModuleDependencies(gl_service.getModuleResources());

    graph.CreateModule("GUIView", "::guiview");
    graph.CreateModule("View3D_2", "::view");
    graph.CreateModule("SphereRenderer", "::spheres");
    graph.CreateModule("TestSpheresDataSource", "::datasource");
    graph.CreateCall("CallRender3D_2", "::view::rendering", "::spheres::rendering");
    graph.CreateCall("MultiParticleDataCall", "::spheres::getdata", "::datasource::getData");
    graph.CreateCall("CallRenderView", "::guiview::renderview", "::view::render");

    std::vector<std::string> view_resource_requests = {
        "KeyboardEvents", "MouseEvents", "WindowEvents", "FramebufferEvents", "IOpenGL_Context" };

	// callback executed by the graph for each frame
	// knows how to make a view module process input events and start the rendering
    auto view_rendering_execution = [&](megamol::core::Module::ptr_type module_ptr,
                                        std::vector<megamol::frontend::ModuleResource> resources) {
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
        //for (auto& dep : resources) {
        //    auto& depId = dep.getIdentifier();

        //    if (depId == "KeyboardEvents") {
        //        megamol::core::view::view_consume_keyboard_events(view, dep);
        //    }
        //    if (depId == "MouseEvents") {
        //        megamol::core::view::view_consume_mouse_events(view, dep);
        //    }
        //    if (depId == "WindowEvents") {
        //        megamol::core::view::view_consume_window_events(view, dep);
        //    }
        //    if (depId == "FramebufferEvents") {
        //        megamol::core::view::view_consume_framebuffer_events(view, dep);
        //    }
        //    if (depId == "IOpenGL_Context") {
        //        megamol::core::view::view_poke_rendering(view, dep);
        //    }
        //}
    };

    graph.SetGraphEntryPoint("::guiview", view_resource_requests, view_rendering_execution);

    std::string parameter_name("::datasource::numSpheres");
    auto parameterPtr = graph.FindParameter(parameter_name);
    if (parameterPtr) {
        parameterPtr->ParseValue("3");
    } else {
        std::cout << "ERROR: could not find parameter: " << parameter_name << std::endl;
    }
#endif

    while (true) {
		// services: receive inputs (GLFW poll events [keyboard, mouse, window], network, lua) 
		services.updateResources();

		// services: digest new inputs via ModuleResources (GUI digest user inputs, lua digest inputs, network ?)
		// e.g. graph updates via lua and GUI happen here
		services.digestChangedResources();

		// services tell us wheter we should shut down megamol
		if (services.shouldShutdown())
			break;

		{ // put this in render function so LUA can call it
			services.preGraphRender(); // glfw poll input events
			// the graph holds references to the input events structs filled by glfw
			// we should probably make this more explicit, i.e.: new_events = [gl_]service.preGraphRender(); graph.UpdateEvents(new_events); graph.Render();

			graph.RenderNextFrame();
			services.postGraphRender(); // swap buffers, clear input events
		}
    }

    // clean up modules, calls, graph

    services.close();

    // TODO: implement graph destructor

    return 0;
}
