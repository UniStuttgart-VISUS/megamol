/*,
 * WidgetPicking_gl.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "WidgetPicking_gl.h"


using namespace megamol;
using namespace megamol::gui;


#define GUI_INTERACTION_TUPLE_INIT                                                                                     \
    { false, -1, FLT_MAX }

#define GUI_GL_CHECK_ERROR                                                                                             \
    {                                                                                                                  \
        auto err = glGetError();                                                                                       \
        if (err != 0)                                                                                                  \
            megamol::core::utility::log::Log::DefaultLog.WriteError(                                                   \
                "OpenGL Error: %i. [%s, %s, line %d]\n", err, __FILE__, __FUNCTION__, __LINE__);                       \
    }


megamol::gui::PickingBuffer::PickingBuffer(void)
    : cursor_x(0.0)
    , cursor_y(0.0)
    , viewport_dim{0.0f, 0.0f}
    , cursor_on_interaction_obj(GUI_INTERACTION_TUPLE_INIT)
    , active_interaction_obj(GUI_INTERACTION_TUPLE_INIT)
    , available_interactions()
    , pending_manipulations()
    , fbo(nullptr)
    , enabled(false)
    , fbo_shader(nullptr) {}


megamol::gui::PickingBuffer::~PickingBuffer(void) { this->fbo.reset(); }


bool megamol::gui::PickingBuffer::ProcessMouseMove(double x, double y) {

    double dx = x - this->cursor_x;
    double dy = y - this->cursor_y;

    this->cursor_x = x;
    this->cursor_y = y;

    // double dx_fbo = x - this->cursor_x;
    // double dy_fbo = y - this->cursor_y;
    // if (this->fbo != nullptr) {
    //     dx_fbo = dx / this->fbo->getWidth();
    //     dy_fbo = -dy / this->fbo->getHeight();
    // }

    auto is_interaction_active = std::get<0>(this->active_interaction_obj);
    if (is_interaction_active) {

        auto active_id = std::get<1>(this->active_interaction_obj);
        auto interactions = this->get_available_interactions(active_id);
        for (auto& interaction : interactions) {

            if (interaction.type == InteractionType::MOVE_ALONG_AXIS_SCREEN) {

                glm::vec2 mouse_move = glm::vec2(static_cast<float>(dx), -static_cast<float>(dy));

                auto axis = glm::vec2(interaction.axis_x, interaction.axis_y);
                auto axis_norm = glm::normalize(axis);

                float scale = glm::dot(axis_norm, mouse_move);

                this->pending_manipulations.emplace_back(Manipulation{InteractionType::MOVE_ALONG_AXIS_SCREEN,
                    active_id, interaction.axis_x, interaction.axis_y, interaction.axis_z, scale});

            } else if (interaction.type == InteractionType::MOVE_ALONG_AXIS_3D) {
                /* FIXME
                glm::vec4 tgt_pos(interaction.origin_x, interaction.origin_y, interaction.origin_z, 1.0f);

                // Compute tgt pos and tgt + transform axisvector in screenspace
                glm::vec4 obj_ss = proj_mx_cpy * view_mx_cpy * tgt_pos;
                obj_ss /= obj_ss.w;

                glm::vec4 transfortgt = tgt_pos + glm::vec4(interaction.axis_x, interaction.axis_y, interaction.axis_z,
                0.0f); glm::vec4 transfortgt_ss = proj_mx_cpy * view_mx_cpy * transfortgt; transfortgt_ss /=
                transfortgt_ss.w;

                glm::vec2 transforaxis_ss =
                    glm::vec2(transfortgt_ss.x, transfortgt_ss.y) -
                    glm::vec2(obj_ss.x, obj_ss.y);

                glm::vec2 mouse_move =
                    glm::vec2(static_cast<float>(dx), static_cast<float>(dy)) * 2.0f;

                float scale = 0.0f;

                if (transforaxis_ss.length() > 0.0)
                {
                    auto mlenght = mouse_move.length();
                    auto ta_ss_length = transforaxis_ss.length();

                    auto mnorm = glm::normalize(mouse_move);
                    auto ta_ss_norm = glm::normalize(transforaxis_ss);

                    scale = glm::dot(mnorm, ta_ss_norm);
                    scale *= (mlenght / ta_ss_length);
                }

                std::cout << "Adding move manipulation: " << interaction.axis_x << " " << interaction.axis_y << " "
                    << interaction.axis_z << " " << scale << std::endl;

                this->interaction_collection->accessPendingManipulations().push(Manipulation{
                    InteractionType::MOVE_ALONG_AXIS, id,
                    interaction.axis_x, interaction.axis_y, interaction.axis_z, scale });
                // TODO add manipulation task with scale * axis
                */
            }
        }
    }

    // TODO compute manipulation based on mouse movement

    return false;
}


bool megamol::gui::PickingBuffer::ProcessMouseClick(megamol::core::view::MouseButton button,
    megamol::core::view::MouseButtonAction action, megamol::core::view::Modifiers mods) {

    // Enable/Disable cursor interaction
    if ((button == megamol::core::view::MouseButton::BUTTON_LEFT) &&
        (action == megamol::core::view::MouseButtonAction::PRESS)) {

        bool is_cursor_active = std::get<0>(this->cursor_on_interaction_obj);
        if (is_cursor_active) {

            this->active_interaction_obj = this->cursor_on_interaction_obj;
            auto active_id = std::get<1>(this->active_interaction_obj);
            this->pending_manipulations.emplace_back(
                Manipulation{InteractionType::SELECT, active_id, 0.0f, 0.0f, 0.0f, 0.0f});
            return true;
        }
    } else if ((button == megamol::core::view::MouseButton::BUTTON_LEFT) &&
               (action == megamol::core::view::MouseButtonAction::RELEASE)) {

        auto active_id = std::get<1>(this->active_interaction_obj);
        this->pending_manipulations.emplace_back(
            Manipulation{InteractionType::DESELECT, active_id, 0.0f, 0.0f, 0.0f, 0.0f});
        this->active_interaction_obj = GUI_INTERACTION_TUPLE_INIT;
    }

    return false;
}


bool megamol::gui::PickingBuffer::EnableInteraction(glm::vec2 vp_dim) {

    this->enabled = false;
    this->viewport_dim = vp_dim;

    // Clear available interactions
    this->available_interactions.clear();

    if (this->fbo == nullptr) {
        this->fbo = std::make_unique<glowl::FramebufferObject>(
            this->viewport_dim.x, this->viewport_dim.y, glowl::FramebufferObject::DepthStencilType::NONE);
        this->fbo->createColorAttachment(GL_RGBA32F, GL_RGBA, GL_FLOAT); // 0 Output Image
        this->fbo->createColorAttachment(GL_RG32F, GL_RG, GL_FLOAT);     // 1 Object ID(red) and Depth (green)
        GUI_GL_CHECK_ERROR
    } else if (this->fbo->getWidth() != this->viewport_dim.x || this->fbo->getHeight() != this->viewport_dim.y) {
        this->fbo->resize(this->viewport_dim.x, this->viewport_dim.y);
    }

    this->fbo->bind();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    GLint in[1] = {0};
    glClearBufferiv(GL_COLOR, 1, in);
    GUI_GL_CHECK_ERROR

    this->enabled = true;
    return this->enabled;
}


bool megamol::gui::PickingBuffer::DisableInteraction(void) {

    if (!this->enabled) {
        return false;
    }
    this->enabled = false;
    GUI_GL_CHECK_ERROR
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Clear pending manipulations
    this->pending_manipulations.clear();

    // Bind fbo to read buffer for retrieving pixel data
    GLfloat pixel_data[2] = {-1.0f, FLT_MAX};
    this->fbo->bindToRead(1);
    GUI_GL_CHECK_ERROR
    // Get object id and depth at cursor location from framebuffer's second color attachment
    /// TODO Check if cursor position is within framebuffer pixel range?
    glReadPixels(static_cast<GLint>(this->cursor_x), this->fbo->getHeight() - static_cast<GLint>(this->cursor_y), 1, 1,
        GL_RG, GL_FLOAT, pixel_data);
    GUI_GL_CHECK_ERROR
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    auto id = static_cast<int>(pixel_data[0]);
    auto depth = pixel_data[1];

    if (id > 0) {
        this->cursor_on_interaction_obj = {true, id, depth};
        this->pending_manipulations.emplace_back(Manipulation{InteractionType::HIGHLIGHT, id, 0.0f, 0.0f, 0.0f, 0.0f});
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("[[[DEBUG]]] ID = %i | Depth = %f", id, depth);
    } else {
        this->cursor_on_interaction_obj = GUI_INTERACTION_TUPLE_INIT;
    }

    // Draw fbo color buffer as texture because blending is required
    if (this->fbo_shader == nullptr) {
        std::string vertex_src = "#version 130 \n"
                                 "out vec2 uv_coord; \n"
                                 "void main(void) { \n"
                                 "    const vec4 vertices[6] = vec4[6](vec4(-1.0, -1.0, 0.0, 0.0), \n"
                                 "        vec4(1.0, 1.0, 1.0, 1.0), \n"
                                 "        vec4(-1.0, 1.0, 0.0, 1.0), \n"
                                 "        vec4(1.0, 1.0, 1.0, 1.0), \n"
                                 "        vec4(-1.0, -1.0, 0.0, 0.0), \n"
                                 "        vec4(1.0, -1.0, 1.0, 0.0)); \n"
                                 "    vec4 vertex = vertices[gl_VertexID]; \n"
                                 "    uv_coord = vertex.zw; \n"
                                 "    gl_Position = vec4(vertex.xy, -1.0, 1.0); \n"
                                 "} ";

        std::string fragment_src = "#version 130  \n"
                                   "#extension GL_ARB_explicit_attrib_location : require \n"
                                   "in vec2 uv_coord; \n"
                                   "uniform sampler2D col_tex; \n"
                                   "uniform sampler2D depth_tex; \n"
                                   "layout(location = 0) out vec4 outFragColor; \n"
                                   "void main(void) { \n"
                                   "    vec4 color = texture(col_tex, uv_coord).rgba; \n"
                                   "    if (color == vec4(0.0)) discard; \n"
                                   "    float depth = texture(depth_tex, uv_coord).g; \n"
                                   "    gl_FragDepth = depth; \n"
                                   "    outFragColor = color; \n"
                                   "} ";

        if (!PickingBuffer::CreatShader(this->fbo_shader, vertex_src, fragment_src)) return false;
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);

    this->fbo_shader->use();

    glActiveTexture(GL_TEXTURE0);
    this->fbo->bindColorbuffer(0);

    glActiveTexture(GL_TEXTURE1);
    this->fbo->bindColorbuffer(1);

    this->fbo_shader->setUniform("col_tex", 0);
    this->fbo_shader->setUniform("depth_tex", 1);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glUseProgram(0);
    glDisable(GL_BLEND);
    glBindTexture(GL_TEXTURE_2D, 0);

    return true;
}


bool megamol::gui::PickingBuffer::CreatShader(
    ShaderPtr& shader_ptr, const std::string& vertex_src, const std::string& fragment_src) {

    std::vector<std::pair<glowl::GLSLProgram::ShaderType, std::string>> shader_srcs;

    if (!vertex_src.empty()) shader_srcs.push_back({glowl::GLSLProgram::ShaderType::Vertex, vertex_src});
    if (!fragment_src.empty()) shader_srcs.push_back({glowl::GLSLProgram::ShaderType::Fragment, fragment_src});

    try {
        if (shader_ptr != nullptr) shader_ptr.reset();
        shader_ptr = std::make_shared<glowl::GLSLProgram>(shader_srcs);
    } catch (glowl::GLSLProgramException const& exc) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error during shader program creation of\"%s\": %s. [%s, %s, line %d]\n",
            shader_ptr->getDebugLabel().c_str(), exc.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


// Pickable Triangle ##########################################################

megamol::gui::PickableTriangle::PickableTriangle(void)
    : shader(nullptr), pixel_direction(100.0f, 200.0f), selected(false) {}


void megamol::gui::PickableTriangle::Draw(
    unsigned int id, glm::vec2 pixel_dir, glm::vec2 vp_dim, ManipVector& pending_manipulations) {

    // Shader data
    float depth = -0.9996f; /// <=== DEBUG !!!
    glm::mat4 ortho = glm::ortho(0.0f, vp_dim.x, 0.0f, vp_dim.y, -1.0f, 1.0f);
    glm::vec3 dir0 = glm::vec3(this->pixel_direction.x, this->pixel_direction.y, depth);
    glm::vec3 dir1 = glm::vec3(dir0.y / 2.0f, -dir0.x / 2.0f, depth);
    glm::vec3 dir2 = glm::vec3(-dir0.y / 2.0f, dir0.x / 2.0f, -0.9999f);

    glm::vec3 vp_vec = glm::vec3(vp_dim.x, vp_dim.y, 0.0f);
    dir0 = vp_vec / 2.0f + dir0;
    dir1 = vp_vec / 2.0f + dir1;
    dir2 = vp_vec / 2.0f + dir2;
    glm::vec4 color = glm::vec4(0.0f, 0.0f, 1.0, 1.0f);

    // Process pending manipulations
    for (auto& manip : pending_manipulations) {
        if (id == manip.obj_id) {
            if (manip.type == InteractionType::MOVE_ALONG_AXIS_SCREEN) {
                this->pixel_direction += glm::vec2(manip.axis_x, manip.axis_y) * manip.value;
            } else if (manip.type == InteractionType::SELECT) {
                this->selected = true;
            } else if (manip.type == InteractionType::DESELECT) {
                this->selected = false;
            } else if (manip.type == InteractionType::HIGHLIGHT) {
                color = glm::vec4(1.0f, 0.0f, 1.0, 1.0f);
            }
        }
    }
    if (selected) {
        color = glm::vec4(0.0f, 1.0f, 1.0, 1.0f);
    }

    // Create shader once
    if (this->shader == nullptr) {
        std::string vertex_src =
            "#version 130 \n"
            "uniform mat4 ortho; \n"
            "uniform vec3 dir0; \n"
            "uniform vec3 dir1; \n"
            "uniform vec3 dir2; \n"
            "uniform vec4 color; \n"
            "out vec4 frag_color; \n"
            "void main(void) {  \n"
            "    vec3 dir_pos = dir0; \n"
            "    frag_color = color; \n"
            "    if (gl_VertexID == 1) { dir_pos = dir1; frag_color = vec4(1.0, 0.0, 0.0, 1.0); } \n"
            "    else if (gl_VertexID == 2)  { dir_pos = dir2; frag_color = vec4(0.0, 1.0, 0.0, 1.0);  } \n"
            "    vec4 pos = ortho * vec4(dir_pos.xyz, 1.0); \n"
            "    pos /= pos.w; \n"
            "    gl_Position = pos; \n"
            "} ";

        std::string fragment_src = "#version 130  \n"
                                   "#extension GL_ARB_explicit_attrib_location : require \n"
                                   "in vec4 frag_color; \n"
                                   "uniform int id; \n"
                                   "layout(location = 0) out vec4 outFragColor; \n"
                                   "layout(location = 1) out vec2 outFragInfo; \n"
                                   "void main(void) { \n"
                                   "    float depth  = gl_FragCoord.z; \n"
                                   "    outFragColor = frag_color; \n"
                                   "    outFragInfo  = vec2(float(id), depth); \n"
                                   "} ";

        if (!PickingBuffer::CreatShader(this->shader, vertex_src, fragment_src)) return;
    }
    this->shader->use();

    // Create 2D orthographic mvp matrix
    // Vertex
    this->shader->setUniform("ortho", ortho);
    this->shader->setUniform("dir0", dir0);
    this->shader->setUniform("dir1", dir1);
    this->shader->setUniform("dir2", dir2);

    // Fragment
    this->shader->setUniform("color", color);
    this->shader->setUniform("id", static_cast<int>(id));

    // Vertex position is only given via uniforms.
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 3);

    glUseProgram(0);
}
