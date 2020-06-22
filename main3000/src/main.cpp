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
    openglConfig.versionMajor = 4;
    openglConfig.versionMinor = 6;
    gl_api->initAPI(&openglConfig);

	auto* apiRawPtr = gl_api.get(); // TODO: this is dangerous and we need a graceful shutdown mechanism for the new graph

    megamol::core::MegaMolGraph graph(core, moduleProvider, callProvider, std::move(gl_api), gl_api_name);

	// TODO: verify valid input IDs/names in graph instantiation methods - dont defer validation until executing the changes
	graph.CreateModule("View3D_2", gl_api_name + "::view");
	graph.CreateModule("SphereRenderer", gl_api_name + "::spheres");
	graph.CreateModule("TestSpheresDataSource", gl_api_name + "::datasource");
	graph.CreateCall("CallRender3D_2", gl_api_name + "::view::rendering", gl_api_name + "::spheres::rendering");
	graph.CreateCall("MultiParticleDataCall", gl_api_name + "::spheres::getdata", gl_api_name + "::datasource::getData");


	while (!apiRawPtr->shouldShutdown()) {
        graph.RenderNextFrame();

		// must set paraeter after frame executed,
		// because module that contains parameter
		// only becomes created immediately before frame is rendered
		static bool first_frame_done = false;
        if (!first_frame_done) {
            std::string parameter_name(gl_api_name + "::datasource::numSpheres");
			auto parameterPtr = graph.FindParameter(parameter_name);
			if (parameterPtr) {
			    parameterPtr->ParseValue("3");
            } else {
                std::cout << "ERROR: could not find parameter: " << parameter_name << std::endl;
            }
            first_frame_done = true;
        }
	}

	// clean up modules, calls, graph

	// TODO: implement graph destructor

    return 0;
}