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

    megamol::render_api::OpenGL_GLFW_RAPI gl_service;

    megamol::render_api::OpenGL_GLFW_RAPI::Config openglConfig;
    openglConfig.windowTitlePrefix = openglConfig.windowTitlePrefix + " ~ Main3000";
    openglConfig.versionMajor = 4;
    openglConfig.versionMinor = 6;
    gl_service.initAPI(&openglConfig);

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

    graph.AddModuleDependencies(gl_service.getRenderResources());

    graph.CreateModule("View3D_2", "::view");
    graph.CreateModule("SphereRenderer", "::spheres");
    graph.CreateModule("TestSpheresDataSource", "::datasource");
    graph.CreateCall("CallRender3D_2", "::view::rendering", "::spheres::rendering");
    graph.CreateCall("MultiParticleDataCall", "::spheres::getdata", "::datasource::getData");

    std::vector<std::string> view_dependency_requests = {
        "KeyboardEvents", "MouseEvents", "WindowEvents", "FramebufferEvents", "IOpenGL_Context"};

	// callback executed by the graph for each frame
	// knows how to make a view module process input events and start the rendering
    auto view_rendering_execution = [&](megamol::core::Module::ptr_type module_ptr,
                                        std::vector<megamol::render_api::RenderResource> dependencies) {
        megamol::core::view::AbstractView* view_ptr =
            dynamic_cast<megamol::core::view::AbstractView*>(module_ptr.get());

        assert(view_dependency_requests.size() == dependencies.size());

        if (!view_ptr) {
            std::cout << "error. module is not a view module. could not set as graph rendering entry point."
                      << std::endl;
            return false;
        }

        megamol::core::view::AbstractView& view = *view_ptr;

        for (auto& dep : dependencies) {
            auto& depId = dep.getIdentifier();

            if (depId == "KeyboardEvents") {
                megamol::core::view::view_consume_keyboard_events(view, dep);
            }
            if (depId == "MouseEvents") {
                megamol::core::view::view_consume_mouse_events(view, dep);
            }
            if (depId == "WindowEvents") {
                megamol::core::view::view_consume_window_events(view, dep);
            }
            if (depId == "FramebufferEvents") {
                megamol::core::view::view_consume_framebuffer_events(view, dep);
            }
            if (depId == "IOpenGL_Context") {
                megamol::core::view::view_poke_rendering(view, dep);
            }
        }
    };

    graph.SetGraphEntryPoint("::view", view_dependency_requests, view_rendering_execution);

    std::string parameter_name("::datasource::numSpheres");
    auto parameterPtr = graph.FindParameter(parameter_name);
    if (parameterPtr) {
        parameterPtr->ParseValue("3");
    } else {
        std::cout << "ERROR: could not find parameter: " << parameter_name << std::endl;
    }
#endif

    while (!gl_service.shouldShutdown()) {
        gl_service.preViewRender(); // glfw poll input events
		// the graph holds references to the input events structs filled by glfw
		// we should probably make this more explicit, i.e.: new_events = [gl_]service.preViewRender(); graph.UpdateEvents(new_events); graph.Render();

        graph.RenderNextFrame();

        gl_service.postViewRender(); // swap buffers, clear input events
    }

    // clean up modules, calls, graph

    gl_service.closeAPI();

    // TODO: implement graph destructor

    return 0;
}
