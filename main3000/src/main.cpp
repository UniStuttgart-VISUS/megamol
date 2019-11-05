#include "mmcore/CoreInstance.h"
#include "mmcore/MegaMolGraph.h"

#include "AbstractRenderAPI.hpp"
#include "OpenGL_GLFW_RAPI.hpp"

int main() {
    megamol::core::CoreInstance core;
    core.Initialise();

    const megamol::core::factories::ModuleDescriptionManager& moduleProvider = core.GetModuleDescriptionManager();
    const megamol::core::factories::CallDescriptionManager& callProvider = core.GetCallDescriptionManager();

    std::unique_ptr<megamol::render_api::AbstractRenderAPI> gl_api =
        std::make_unique<megamol::render_api::OpenGL_GLFW_RAPI>();
    std::string gl_api_name = "opengl";

	megamol::render_api::OpenGL_GLFW_RAPI::Config openglConfig;
    openglConfig.windowTitlePrefix = openglConfig.windowTitlePrefix + " ~ Main3000";
    gl_api->initAPI(&openglConfig);

    megamol::core::MegaMolGraph graph(core, moduleProvider, callProvider, std::move(gl_api), gl_api_name);

	// TODO: verify valid input IDs/names in graph instantiation methods - dont defer validation until executing the changes
	graph.QueueModuleInstantiation("View3D", gl_api_name + "::view");
	graph.QueueModuleInstantiation("SphereOutlineRenderer", gl_api_name + "::spheres");
	graph.QueueModuleInstantiation("TestSpheresDataSource", gl_api_name + "::datasource");
	graph.QueueCallInstantiation("CallRender3D", gl_api_name + "::view::rendering", gl_api_name + "::spheres::rendering");
	graph.QueueCallInstantiation("MultiParticleDataCall", gl_api_name + "::spheres::getdata", gl_api_name + "::datasource::getData");

	graph.ExecuteGraphUpdates();

	while (true) {
        graph.RenderNextFrame();
	}

	// clean up modules, calls, graph

	// TODO: implement graph destructor

    return 0;
}