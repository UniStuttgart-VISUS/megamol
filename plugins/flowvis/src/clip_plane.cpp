#include "stdafx.h"
#include "clip_plane.h"

#include "matrix_call.h"

#include "flowvis/shader.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/Camera_2.h"

#include "glm/mat4x4.hpp"

#include "vislib/math/Cuboid.h"
#include "vislib/math/Plane.h"
#include "vislib/sys/Log.h"

#include <exception>
#include <string>

namespace megamol {
namespace flowvis {

clip_plane::clip_plane()
    : model_matrix_slot("model_matrix", "Model matrix for positioning the clip plane")
    , clip_plane_slot("clip_plane", "Clip plane")
    , initialized(false) {

    // Set output
    this->model_matrix_slot.SetCompatibleCall<matrix_call::matrix_call_description>();
    this->MakeSlotAvailable(&this->model_matrix_slot);

    // Connect input
    this->clip_plane_slot.SetCallback(
        core::view::CallClipPlane::ClassName(), "GetPlane", &clip_plane::get_clip_plane_callback);
    this->MakeSlotAvailable(&this->clip_plane_slot);
}

clip_plane::~clip_plane() { this->Release(); }

bool clip_plane::create() { return true; }

void clip_plane::release() {
    // Remove shaders
    if (this->initialized) {
        glDetachShader(this->render_data.prog, this->render_data.vs);
        glDetachShader(this->render_data.prog, this->render_data.fs);
        glDeleteProgram(this->render_data.prog);
    }
}

bool clip_plane::Render(core::view::CallRender3D_2& call) {
    // Get camera
    core::view::Camera_2 cam;
    call.GetCamera(cam);

    cam_type::matrix_type view, proj;
    cam.calc_matrices(view, proj);

    // On first execution, compile shader
    if (!this->initialized) {
        const std::string vertex_shader = "#version 330 \n"
                                          "uniform mat4 translate_mx; \n"
                                          "uniform mat4 scale_mx; \n"
                                          "uniform mat4 model_mx; \n"
                                          "uniform mat4 view_mx; \n"
                                          "uniform mat4 proj_mx; \n"
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
                                          "    vertex_color = vec4(0.7f, 0.7f, 0.7f, 0.3f);\n"
                                          "    gl_Position = proj_mx * view_mx * translate_mx * model_mx * scale_mx * vertices[gl_VertexID]; \n"
                                          "}";

        const std::string fragment_shader = "#version 330\n"
                                            "in vec4 vertex_color; \n"
                                            "out vec4 fragColor; \n"
                                            "void main() { \n"
                                            "    fragColor = vertex_color; \n"
                                            "}";

        try {
            this->render_data.vs = utility::make_shader(vertex_shader, GL_VERTEX_SHADER);
            this->render_data.fs = utility::make_shader(fragment_shader, GL_FRAGMENT_SHADER);

            this->render_data.prog = utility::make_program({this->render_data.vs, this->render_data.fs});
        } catch (const std::exception& e) {
            vislib::sys::Log::DefaultLog.WriteError(e.what());

            return false;
        }

        this->initialized = true;
    }

    // Get model matrix
    auto mc = this->model_matrix_slot.CallAs<matrix_call>();

    if (mc == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("Model matrix not connected to clip plane");
        return false;
    }

    if (!(*mc)(0)) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to get model matrix");
        return false;
    }

    // Create translation and scaling matrices based on bounding box
    const auto bb = call.GetBoundingBoxes().BoundingBox();
    const auto x_offset = 0.5f * (bb.Left() + bb.Right());
    const auto y_offset = 0.5f * (bb.Bottom() + bb.Top());
    const auto z_offset = 0.5f * (bb.Back() + bb.Front());
    const auto scale_factor = std::max(std::max(bb.Width(), bb.Height()), bb.Depth());

    glm::mat4 translate(1.0f);
    translate[3][0] = x_offset;
    translate[3][1] = y_offset;
    translate[3][2] = z_offset;

    glm::mat4 scale(1.0f);
    scale[0][0] = 0.75 * scale_factor;
    scale[1][1] = 0.75 * scale_factor;
    scale[2][2] = 0.75 * scale_factor;

    // Adjust translation of the model matrix to the difference of bounding box sizes
    auto model = mc->get_matrix();
    model[3][0] *= 0.5f * scale_factor;
    model[3][1] *= 0.5f * scale_factor;
    model[3][2] *= 0.5f * scale_factor;

    // Render plane
    glUseProgram(this->render_data.prog);
    
    glUniformMatrix4fv(glGetUniformLocation(this->render_data.prog, "translate_mx"), 1, false, glm::value_ptr(translate));
    glUniformMatrix4fv(glGetUniformLocation(this->render_data.prog, "scale_mx"), 1, false, glm::value_ptr(scale));
    glUniformMatrix4fv(glGetUniformLocation(this->render_data.prog, "model_mx"), 1, false, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(this->render_data.prog, "view_mx"), 1, false, glm::value_ptr(static_cast<glm::mat4>(view)));
    glUniformMatrix4fv(glGetUniformLocation(this->render_data.prog, "proj_mx"), 1, false, glm::value_ptr(static_cast<glm::mat4>(proj)));

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glUseProgram(0);

    return true;
}

bool clip_plane::GetExtents(core::view::CallRender3D_2& call) {
    const auto old_bb = call.GetBoundingBoxes().BoundingBox();

    const auto x_offset = old_bb.Width() / 2.0f;
    const auto y_offset = old_bb.Height() / 2.0f;
    const auto z_offset = old_bb.Depth() / 2.0f;

    call.AccessBoundingBoxes().SetClipBox(
        vislib::math::Cuboid<float>(old_bb.Left() - x_offset, old_bb.Bottom() - y_offset, old_bb.Back() - z_offset,
            old_bb.Left() + x_offset, old_bb.Bottom() + y_offset, old_bb.Back() + z_offset));

    return true;
}

bool clip_plane::get_clip_plane_callback(core::Call& call) {
    // TODO

    return true;
}

} // namespace flowvis
} // namespace megamol
