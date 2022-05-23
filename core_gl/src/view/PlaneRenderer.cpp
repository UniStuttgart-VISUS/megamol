#include "mmcore_gl/view/PlaneRenderer.h"

#include "mmcore/CallerSlot.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/Camera.h"

#include "mmcore_gl/view/CallRender3DGL.h"
#include "mmcore_gl/view/Renderer3DModuleGL.h"

#include "glowl/GLSLProgram.hpp"

#include "glm/mat4x4.hpp"

#include "vislib/math/Plane.h"

#include <exception>
#include <memory>
#include <string>
#include <utility>

namespace megamol {
namespace core_gl {
namespace view {

PlaneRenderer::PlaneRenderer() : input_plane_slot("input_plane", "Input (clip) plane to render"), initialized(false) {

    this->input_plane_slot.SetCompatibleCall<core::view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->input_plane_slot);
}

PlaneRenderer::~PlaneRenderer() {
    this->Release();
}

bool PlaneRenderer::create() {
    return true;
}

void PlaneRenderer::release() {}

bool PlaneRenderer::Render(CallRender3DGL& call) {
    // Get plane
    auto cp = this->input_plane_slot.CallAs<core::view::CallClipPlane>();

    if (cp == nullptr || !(*cp)(0)) {
        return false;
    }

    // Get clip plane
    this->plane = cp->GetPlane();
    this->plane.Normalise();

    this->color[0] = cp->GetColour()[0] / 255.0f;
    this->color[1] = cp->GetColour()[1] / 255.0f;
    this->color[2] = cp->GetColour()[2] / 255.0f;
    this->color[3] = cp->GetColour()[3] / 255.0f;

    // Get camera
    core::view::Camera cam;
    call.GetCamera(cam);

    const auto view = cam.getViewMatrix();
    const auto proj = cam.getProjectionMatrix();

    // On first execution, compile shader
    if (!this->initialized) {
        const std::string vertex_shader =
            "#version 330 \n"
            "uniform mat4 scale_mx; \n"
            "uniform mat4 model_mx; \n"
            "uniform mat4 view_mx; \n"
            "uniform mat4 proj_mx; \n"
            "uniform vec4 color; \n"
            "out vec4 vertex_color; \n"
            "void main() \n"
            "{ \n"
            "    const vec4 vertices[6] = vec4[6]( \n"
            "        vec4(-1.0, -1.0, 0.0, 1.0), \n"
            "        vec4( 1.0, -1.0, 0.0, 1.0), \n"
            "        vec4( 1.0,  1.0, 0.0, 1.0), \n"
            "        vec4(-1.0, -1.0, 0.0, 1.0), \n"
            "        vec4( 1.0,  1.0, 0.0, 1.0), \n"
            "        vec4(-1.0,  1.0, 0.0, 1.0) \n"
            "    ); \n"
            "    vertex_color = color;\n"
            "    gl_Position = proj_mx * view_mx * model_mx * scale_mx * vertices[gl_VertexID]; \n"
            "}";

        const std::string fragment_shader = "#version 330\n"
                                            "in vec4 vertex_color; \n"
                                            "out vec4 fragColor; \n"
                                            "void main() { \n"
                                            "    fragColor = vertex_color; \n"
                                            "}";

        try {
            glowl::GLSLProgram::ShaderSourceList shader{
                std::make_pair(glowl::GLSLProgram::ShaderType::Vertex, vertex_shader),
                std::make_pair(glowl::GLSLProgram::ShaderType::Fragment, fragment_shader)};

            this->render_data = std::make_unique<glowl::GLSLProgram>(shader);
        } catch (...) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Error compiling shaders. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);

            return false;
        }

        this->initialized = true;
    }

    // Create scaling matrix based on bounding box
    const auto& bb = call.GetBoundingBoxes().BoundingBox();
    const auto scale_factor = std::max(std::max(bb.Width(), bb.Height()), bb.Depth());

    glm::mat4 scale(1.0f);
    scale[0][0] = 0.75 * scale_factor;
    scale[1][1] = 0.75 * scale_factor;
    scale[2][2] = 0.75 * scale_factor;

    // Create model matrix from plane information...
    glm::mat4 model(1.0f);

    // ... construct plane with up point at the center of the bounding box and given normal
    const auto bb_center = call.GetBoundingBoxes().BoundingBox().CalcCenter();

    const auto normal = this->plane.Normal();

    const vislib::math::Plane<float> centered_plane(bb_center, this->plane.Normal());

    // ... extract translation
    const auto lambda = (this->plane.D() - centered_plane.D());

    const auto pos = bb_center - lambda * normal;

    // ... set translation to calculated position
    model[3][0] = pos[0];
    model[3][1] = pos[1];
    model[3][2] = pos[2];

    // ... set rotation for correct orientation
    vislib::math::Vector<float, 3> ortho_1, ortho_2;

    if (std::acos(normal.Dot(vislib::math::Vector<float, 3>(1.0f, 0.0f, 0.0f))) <
        std::acos(normal.Dot(vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f)))) {

        ortho_1 = normal.Cross(vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
    } else {
        ortho_1 = normal.Cross(vislib::math::Vector<float, 3>(1.0f, 0.0f, 0.0f));
    }

    ortho_1.Normalise();
    ortho_2 = normal.Cross(ortho_1);

    model[0][0] = ortho_1[0];
    model[0][1] = ortho_1[1];
    model[0][2] = ortho_1[2];

    model[1][0] = ortho_2[0];
    model[1][1] = ortho_2[1];
    model[1][2] = ortho_2[2];

    model[2][0] = normal[0];
    model[2][1] = normal[1];
    model[2][2] = normal[2];

    // Set state
    const auto depth_test = glIsEnabled(GL_DEPTH_TEST);
    if (!depth_test)
        glEnable(GL_DEPTH_TEST);

    const auto blending = glIsEnabled(GL_BLEND);
    if (!blending)
        glEnable(GL_BLEND);

    GLint src_color, dst_color, src_alpha, dst_alpha;
    glGetIntegerv(GL_BLEND_SRC_RGB, &src_color);
    glGetIntegerv(GL_BLEND_DST_RGB, &dst_color);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &src_alpha);
    glGetIntegerv(GL_BLEND_DST_ALPHA, &dst_alpha);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Render plane
    this->render_data->use();

    glUniformMatrix4fv(this->render_data->getUniformLocation("scale_mx"), 1, false, glm::value_ptr(scale));
    glUniformMatrix4fv(this->render_data->getUniformLocation("model_mx"), 1, false, glm::value_ptr(model));
    glUniformMatrix4fv(
        this->render_data->getUniformLocation("view_mx"), 1, false, glm::value_ptr(static_cast<glm::mat4>(view)));
    glUniformMatrix4fv(
        this->render_data->getUniformLocation("proj_mx"), 1, false, glm::value_ptr(static_cast<glm::mat4>(proj)));

    glUniform4fv(this->render_data->getUniformLocation("color"), 1, this->color.data());

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glUseProgram(0);

    // Restore state
    if (!depth_test)
        glDisable(GL_DEPTH_TEST);
    if (!blending)
        glDisable(GL_BLEND);

    glBlendFuncSeparate(src_color, dst_color, src_alpha, dst_alpha);

    return true;
}

bool PlaneRenderer::GetExtents(CallRender3DGL& call) {
    const auto& old_bb = call.GetBoundingBoxes().BoundingBox();

    const auto x_offset = old_bb.Width() / 2.0f;
    const auto y_offset = old_bb.Height() / 2.0f;
    const auto z_offset = old_bb.Depth() / 2.0f;

    call.AccessBoundingBoxes().SetClipBox(
        vislib::math::Cuboid<float>(old_bb.Left() - x_offset, old_bb.Bottom() - y_offset, old_bb.Back() - z_offset,
            old_bb.Left() + x_offset, old_bb.Bottom() + y_offset, old_bb.Back() + z_offset));

    return true;
}

} // namespace view
} // namespace core_gl
} // namespace megamol
