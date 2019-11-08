#include "stdafx.h"
#include "draw_texture_3d.h"

#include "matrix_call.h"
#include "flowvis/shader.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/Camera_2.h"

#include "compositing/CompositingCalls.h"

#include "vislib/Exception.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/math/Cuboid.h"
#include "vislib/sys/Log.h"

#include "glowl/Texture2D.hpp"

#include "glm/vec4.hpp"

#include <exception>
#include <string>

namespace megamol {
namespace flowvis {

draw_texture_3d::draw_texture_3d()
    : texture_slot("texture", "Input texture")
    , model_matrix_slot("model_matrix", "Model matrix for positioning of the rendered texture quad") {

    // Connect input
    this->texture_slot.SetCompatibleCall<compositing::CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->texture_slot);

    this->model_matrix_slot.SetCompatibleCall<matrix_call::matrix_call_description>();
    this->MakeSlotAvailable(&this->model_matrix_slot);
}

draw_texture_3d::~draw_texture_3d() { this->Release(); }

bool draw_texture_3d::create() { return true; }

void draw_texture_3d::release() {
    // Remove shader
    if (this->render_data.initialized) {
        glDetachShader(this->render_data.prog, this->render_data.vs);
        glDetachShader(this->render_data.prog, this->render_data.fs);
        glDeleteProgram(this->render_data.prog);
    }
}

bool draw_texture_3d::Render(core::view::CallRender3D_2& call) {
    auto tc_ptr = this->texture_slot.CallAs<compositing::CallTexture2D>();

    // Get input texture
    if (tc_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("No input connected");
        return false;
    }

    auto& tc = *tc_ptr;

    if (!tc(0)) {
        vislib::sys::Log::DefaultLog.WriteError("Error getting texture");
        return false;
    }

    this->render_data.texture = tc.getData();

    // Get camera
    core::view::Camera_2 cam;
    call.GetCamera(cam);

    cam_type::matrix_type view, proj;
    cam.calc_matrices(view, proj);

    // Build shaders
    if (!this->render_data.initialized) {
        const std::string vertex_shader =
            "#version 330 \n" \
            "uniform mat4 model_mx; \n" \
            "uniform mat4 view_mx; \n" \
            "uniform mat4 proj_mx; \n" \
            "out vec2 tex_coords; \n" \
            "void main() { \n" \
            "    const vec2 vertices[6] =\n" \
            "        vec2[6](vec2(0.0f, 0.0f), vec2(1.0f, 0.0f), vec2(1.0f, 1.0f),\n" \
            "                vec2(0.0f, 0.0f), vec2(1.0f, 1.0f), vec2(0.0f, 1.0f));\n" \
            "    tex_coords = vertices[gl_VertexID]; \n" \
            "    gl_Position = proj_mx * view_mx * model_mx * vec4(tex_coords, 0.0f, 1.0f); \n" \
            "}";

        const std::string fragment_shader =
            "#version 330\n" \
            "uniform sampler2D tex2D; \n" \
            "in vec2 tex_coords; \n" \
            "out vec4 fragColor; \n" \
            "void main() { \n" \
            "    fragColor = texture(tex2D, tex_coords); \n" \
            "}";

        try
        {
            this->render_data.vs = utility::make_shader(vertex_shader, GL_VERTEX_SHADER);
            this->render_data.fs = utility::make_shader(fragment_shader, GL_FRAGMENT_SHADER);

            this->render_data.prog = utility::make_program({ this->render_data.vs, this->render_data.fs });
        }
        catch (const std::exception& e)
        {
            vislib::sys::Log::DefaultLog.WriteError(e.what());

            return false;
        }

        this->render_data.initialized = true;
    }

    // Draw quad with given texture and model matrix
    glUseProgram(this->render_data.prog);

    glActiveTexture(GL_TEXTURE0);
    this->render_data.texture->bindTexture();
    glUniform1i(glGetUniformLocation(this->render_data.prog, "tex2D"), 0);

    glUniformMatrix4fv(glGetUniformLocation(this->render_data.prog, "model_mx"), 1, GL_FALSE, glm::value_ptr(this->render_data.model_matrix));
    glUniformMatrix4fv(glGetUniformLocation(this->render_data.prog, "view_mx"), 1, GL_FALSE, glm::value_ptr(static_cast<glm::mat4>(view)));
    glUniformMatrix4fv(glGetUniformLocation(this->render_data.prog, "proj_mx"), 1, GL_FALSE, glm::value_ptr(static_cast<glm::mat4>(proj)));

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glBindTexture(GL_TEXTURE_2D, 0);

    glUseProgram(0);

    return true;
}

bool draw_texture_3d::GetExtents(core::view::CallRender3D_2& call) {
    auto tc_ptr = this->texture_slot.CallAs<compositing::CallTexture2D>();
    auto mc_ptr = this->model_matrix_slot.CallAs<matrix_call>();

    // Get input texture meta data
    if (tc_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("No input connected");
        return false;
    }

    auto& tc = *tc_ptr;

    if (!tc(1)) {
        vislib::sys::Log::DefaultLog.WriteError("Error getting texture extent");
        return false;
    }

    // Get model matrix
    if (mc_ptr == nullptr) {
        this->render_data.model_matrix = glm::mat4(1.0f);
    } else {
        auto& mc = *mc_ptr;

        if (!mc(0)) {
            vislib::sys::Log::DefaultLog.WriteWarn("Error getting model matrix. Using unit matrix instead");
            this->render_data.model_matrix = glm::mat4(1.0f);
        } else {
            this->render_data.model_matrix = mc.get_matrix();
        }
    }

    // Set bounding box
    const auto origin = this->render_data.model_matrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    const auto diagonal = this->render_data.model_matrix * glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);

    vislib::math::Cuboid<float> bounding_box;
    bounding_box.Set(origin.x, origin.y, origin.z, diagonal.x, diagonal.y, diagonal.z);

    auto& boxes = call.AccessBoundingBoxes();
    boxes.SetBoundingBox(bounding_box);
    boxes.SetClipBox(bounding_box);

    call.SetTimeFramesCount(1);

    return true;
}

} // namespace flowvis
} // namespace megamol
