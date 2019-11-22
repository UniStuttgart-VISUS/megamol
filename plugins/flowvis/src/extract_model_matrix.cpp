#include "stdafx.h"
#include "extract_model_matrix.h"

#include "matrix_call.h"

#include "flowvis/shader.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/Camera_2.h"

#include "glm/mat4x4.hpp"

#include "vislib/math/Cuboid.h"
#include "vislib/sys/Log.h"

#include <exception>
#include <string>

namespace megamol {
namespace flowvis {

extract_model_matrix::extract_model_matrix()
    : model_matrix_slot("model_matrix", "Extracted model matrix"), initialized(false) {

    // Connect input
    this->model_matrix_slot.SetCallback(
        matrix_call::ClassName(), "get_data", &extract_model_matrix::get_matrix_callback);
    this->MakeSlotAvailable(&this->model_matrix_slot);
}

extract_model_matrix::~extract_model_matrix() { this->Release(); }

bool extract_model_matrix::create() { return true; }

void extract_model_matrix::release() {
    // Remove shaders
    if (this->initialized) {
        glDetachShader(this->render_data.prog, this->render_data.vs);
        glDetachShader(this->render_data.prog, this->render_data.fs);
        glDeleteProgram(this->render_data.prog);
    }
}

bool extract_model_matrix::Render(core::view::CallRender3D_2& call) {
    // Get camera
    core::view::Camera_2 cam;
    call.GetCamera(cam);

    cam_type::matrix_type view, proj;
    cam.calc_matrices(view, proj);

    // On first execution, set inverse initial matrix and compile shader
    if (!this->initialized) {
        this->inverse_initial_model_matrix = glm::inverse(static_cast<glm::mat4>(view));

        // Create shaders and link them
        const std::string vertex_shader =
            "#version 330 \n"
            "uniform mat4 view_mx; \n"
            "uniform mat4 proj_mx; \n"
            "out vec4 vertex_color; \n"
            "void main() \n"
            "{ \n"
            "    const vec4 vertices[36] = vec4[36]( \n"
            //       front
            "        vec4(-1.0, -1.0,  1.0, 1.0), \n"
            "        vec4( 1.0, -1.0,  1.0, 1.0), \n"
            "        vec4( 1.0,  1.0,  1.0, 1.0), \n"
            "        vec4(-1.0, -1.0,  1.0, 1.0), \n"
            "        vec4( 1.0,  1.0,  1.0, 1.0), \n"
            "        vec4(-1.0,  1.0,  1.0, 1.0), \n"
            //       back
            "        vec4(-1.0, -1.0, -1.0, 1.0), \n"
            "        vec4( 1.0,  1.0, -1.0, 1.0), \n"
            "        vec4( 1.0, -1.0, -1.0, 1.0), \n"
            "        vec4(-1.0, -1.0, -1.0, 1.0), \n"
            "        vec4(-1.0,  1.0, -1.0, 1.0), \n"
            "        vec4( 1.0,  1.0, -1.0, 1.0), \n"
            //       top
            "        vec4(-1.0,  1.0, -1.0, 1.0), \n"
            "        vec4( 1.0,  1.0, -1.0, 1.0), \n"
            "        vec4( 1.0,  1.0,  1.0, 1.0), \n"
            "        vec4(-1.0,  1.0, -1.0, 1.0), \n"
            "        vec4( 1.0,  1.0,  1.0, 1.0), \n"
            "        vec4(-1.0,  1.0,  1.0, 1.0), \n"
            //       bottom
            "        vec4(-1.0, -1.0, -1.0, 1.0), \n"
            "        vec4( 1.0, -1.0,  1.0, 1.0), \n"
            "        vec4( 1.0, -1.0, -1.0, 1.0), \n"
            "        vec4(-1.0, -1.0, -1.0, 1.0), \n"
            "        vec4(-1.0, -1.0,  1.0, 1.0), \n"
            "        vec4( 1.0, -1.0,  1.0, 1.0), \n"
            //       right
            "        vec4( 1.0, -1.0, -1.0, 1.0), \n"
            "        vec4( 1.0,  1.0, -1.0, 1.0), \n"
            "        vec4( 1.0,  1.0,  1.0, 1.0), \n"
            "        vec4( 1.0, -1.0, -1.0, 1.0), \n"
            "        vec4( 1.0,  1.0,  1.0, 1.0), \n"
            "        vec4( 1.0, -1.0,  1.0, 1.0), \n"
            //       left
            "        vec4(-1.0, -1.0, -1.0, 1.0), \n"
            "        vec4(-1.0,  1.0,  1.0, 1.0), \n"
            "        vec4(-1.0,  1.0, -1.0, 1.0), \n"
            "        vec4(-1.0, -1.0, -1.0, 1.0), \n"
            "        vec4(-1.0, -1.0,  1.0, 1.0), \n"
            "        vec4(-1.0,  1.0,  1.0, 1.0)  \n"
            "    ); \n"
            "    const vec4 colors[6] = vec4[6]( \n"
            "        vec4(0.0, 0.0, 1.0, 1.0), \n"
            "        vec4(0.0, 0.0, 0.5, 1.0), \n"
            "        vec4(0.0, 1.0, 0.0, 1.0), \n"
            "        vec4(0.0, 0.5, 0.0, 1.0), \n"
            "        vec4(1.0, 0.0, 0.0, 1.0), \n"
            "        vec4(0.5, 0.0, 0.0, 1.0)  \n"
            "    ); \n"
            "    vertex_color = colors[gl_VertexID / 6];\n"
            "    gl_Position = proj_mx * view_mx * vertices[gl_VertexID]; \n"
            "}";

        const std::string fragment_shader =
            "#version 330\n"
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

    // Calculate model matrix
    this->model_matrix = this->inverse_initial_model_matrix * static_cast<glm::mat4>(view);

    // Render view cube
    glUseProgram(this->render_data.prog);

    glUniformMatrix4fv(glGetUniformLocation(this->render_data.prog, "view_mx"), 1, false,
        glm::value_ptr(static_cast<glm::mat4>(view)));
    glUniformMatrix4fv(glGetUniformLocation(this->render_data.prog, "proj_mx"), 1, false,
        glm::value_ptr(static_cast<glm::mat4>(proj)));

    glDrawArrays(GL_TRIANGLES, 0, 36);

    glUseProgram(0);

    return true;
}

bool extract_model_matrix::GetExtents(core::view::CallRender3D_2& call) {
    call.AccessBoundingBoxes().SetBoundingBox(vislib::math::Cuboid<float>(-2.0f, -2.0f, -2.0f, 2.0f, 2.0f, 2.0f));
    call.AccessBoundingBoxes().SetClipBox(vislib::math::Cuboid<float>(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f));

    return true;
}

bool extract_model_matrix::get_matrix_callback(core::Call& call) {
    static_cast<matrix_call&>(call).set_matrix(this->model_matrix);

    return true;
}

} // namespace flowvis
} // namespace megamol
