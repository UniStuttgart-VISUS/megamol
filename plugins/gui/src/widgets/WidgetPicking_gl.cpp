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


#define GUI_INTERACTION_TUPLE_INIT \
    { false, -1, FLT_MAX }

#define GUI_GL_CHECK_ERROR                                                                        \
    {                                                                                             \
        auto err = glGetError();                                                                  \
        if (err != 0)                                                                             \
            megamol::core::utility::log::Log::DefaultLog.WriteError(                              \
                "OpenGL Error: %i. [%s, %s, line %d]\n ", err, __FILE__, __FUNCTION__, __LINE__); \
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
        , fbo_shader(nullptr)
        , enabled(false) {}


megamol::gui::PickingBuffer::~PickingBuffer(void) {
    this->fbo.reset();
}


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
                /// TODO Add manipulation task with scale * axis
                */
            }
        }
    }

    /// TODO Compute manipulation based on mouse movement

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

            // Consume when interaction is started
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

    if (this->enabled) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Disable interaction before enabling again. [%s, %s, line %d]\n ", __FILE__, __FUNCTION__, __LINE__);
        return true;
    }

    // Enable interaction only if interactions have been added in previous frame
    if (this->available_interactions.empty()) {
        return false;
    }

    // Interactions are processed in ProcessMouseMove() and should be cleared each frame
    this->available_interactions.clear();
    this->enabled = false;
    this->viewport_dim = vp_dim;

    if (this->fbo == nullptr) {
        try {
            this->fbo = std::make_unique<glowl::FramebufferObject>(
                this->viewport_dim.x, this->viewport_dim.y, glowl::FramebufferObject::DepthStencilType::NONE);
        } catch (glowl::FramebufferObjectException& e) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Error during framebuffer object creation: '%s'. [%s, %s, line %d]\n ", e.what(), __FILE__,
                __FUNCTION__, __LINE__);
            return false;
        }
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
        // megamol::core::utility::log::Log::DefaultLog.WriteError(
        //    "[GUI] Enable interaction before disabling it. [%s, %s, line %d]\n ", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->enabled = false;

    // Clear pending manipulations
    this->pending_manipulations.clear();

    // Create FBO sahders if required -----------------------------------------
    if (this->fbo_shader == nullptr) {
        std::string vertex_src = "#version 130 \n "
                                 "out vec2 uv_coord; \n "
                                 "void main(void) { \n "
                                 "    const vec4 vertices[6] = vec4[6](vec4(-1.0, -1.0, 0.0, 0.0), \n "
                                 "        vec4(1.0, 1.0, 1.0, 1.0), \n "
                                 "        vec4(-1.0, 1.0, 0.0, 1.0), \n "
                                 "        vec4(1.0, 1.0, 1.0, 1.0), \n "
                                 "        vec4(-1.0, -1.0, 0.0, 0.0), \n "
                                 "        vec4(1.0, -1.0, 1.0, 0.0)); \n "
                                 "    vec4 vertex = vertices[gl_VertexID]; \n "
                                 "    uv_coord = vertex.zw; \n "
                                 "    gl_Position = vec4(vertex.xy, -1.0, 1.0); \n "
                                 "} ";

        std::string fragment_src = "#version 130  \n "
                                   "#extension GL_ARB_explicit_attrib_location : require \n "
                                   "in vec2 uv_coord; \n "
                                   "uniform sampler2D col_tex; \n "
                                   "uniform sampler2D depth_tex; \n "
                                   "layout(location = 0) out vec4 outFragColor; \n "
                                   "void main(void) { \n "
                                   "    vec4 color = texture(col_tex, uv_coord).rgba; \n "
                                   "    if (color == vec4(0.0)) discard; \n "
                                   "    float depth = texture(depth_tex, uv_coord).g; \n "
                                   "    gl_FragDepth = depth; \n "
                                   "    outFragColor = color; \n "
                                   "} ";

        if (!megamol::core::view::RenderUtils::CreateShader(this->fbo_shader, vertex_src, fragment_src))
            return false;
    }

    GUI_GL_CHECK_ERROR
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Bind fbo to read buffer for retrieving pixel data
    GLfloat pixel_data[2] = {-1.0f, FLT_MAX};
    this->fbo->bindToRead(1);
    GUI_GL_CHECK_ERROR
    // Get object id and depth at cursor location from framebuffer's second color attachment
    /// TODO Check if cursor position is within framebuffer pixel range -> ensured by GLFW?
    glReadPixels(static_cast<GLint>(this->cursor_x), this->fbo->getHeight() - static_cast<GLint>(this->cursor_y), 1, 1,
        GL_RG, GL_FLOAT, pixel_data);
    GUI_GL_CHECK_ERROR
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    auto id = static_cast<unsigned int>(pixel_data[0]);
    auto depth = pixel_data[1];

    if (id > 0) {
        this->cursor_on_interaction_obj = {true, id, depth};
        this->pending_manipulations.emplace_back(Manipulation{InteractionType::HIGHLIGHT, id, 0.0f, 0.0f, 0.0f, 0.0f});
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("[[[DEBUG]]] ID = %i | Depth = %f", id, depth);
    } else {
        this->cursor_on_interaction_obj = GUI_INTERACTION_TUPLE_INIT;
    }

    // Draw fbo color buffer as texture because blending is required
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
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    return true;
}


// Pickable Cube ##########################################################

megamol::gui::PickableCube::PickableCube(void) : shader(nullptr) {}


void megamol::gui::PickableCube::Draw(unsigned int id, int& inout_defaultview_index, int& out_hovered_view_index,
    const glm::vec4& view_orientation, const glm::vec2& vp_dim, ManipVector& pending_manipulations) {

    assert(ImGui::GetCurrentContext() != nullptr);

    // Info: IDs of the six cube faces are encoded via bit shift by face index of given parameter id.

    // Create shader once -----------------------------------------------------
    if (this->shader == nullptr) {
        std::string vertex_src =
            "#version 130 \n "
            "uniform mat4 rot_mx; \n "
            "uniform mat4 model_mx; \n "
            "uniform mat4 proj_mx; \n "
            "uniform int view_index; \n "
            "out vec4 vertex_color; \n "
            "flat out int face_index; \n "
            "void main() { \n "
            "    // Vertex indices must fit enum order in megamol::core::view::View3D_2::defaultview \n "
            "    const vec4 vertices[36] = vec4[36]( \n "
            "        // DEFAULTVIEW_FRONT = 0 \n "
            "        vec4(-1.0, -1.0, 1.0, 1.0), vec4(1.0, -1.0, 1.0, 1.0), vec4(1.0, 1.0, 1.0, 1.0), \n "
            "        vec4(-1.0, -1.0, 1.0, 1.0), vec4(1.0, 1.0, 1.0, 1.0), vec4(-1.0, 1.0, 1.0, 1.0), \n "
            "        // DEFAULTVIEW_BACK = 1 \n "
            "        vec4(-1.0, -1.0, -1.0, 1.0), vec4(1.0, 1.0, -1.0, 1.0), vec4(1.0, -1.0, -1.0, 1.0), \n "
            "        vec4(-1.0, -1.0, -1.0, 1.0), vec4(-1.0, 1.0, -1.0, 1.0), vec4(1.0, 1.0, -1.0, 1.0), \n "
            "        // DEFAULTVIEW_RIGHT = 2 \n "
            "        vec4(1.0, -1.0, -1.0, 1.0), vec4(1.0, 1.0, -1.0, 1.0), vec4(1.0, 1.0, 1.0, 1.0), \n "
            "        vec4(1.0, -1.0, -1.0, 1.0), vec4(1.0, 1.0, 1.0, 1.0), vec4(1.0, -1.0, 1.0, 1.0), \n "
            "        // DEFAULTVIEW_LEFT = 3 \n "
            "        vec4(-1.0, -1.0, -1.0, 1.0), vec4(-1.0, 1.0, 1.0, 1.0), vec4(-1.0, 1.0, -1.0, 1.0), \n "
            "        vec4(-1.0, -1.0, -1.0, 1.0), vec4(-1.0, -1.0, 1.0, 1.0), vec4(-1.0, 1.0, 1.0, 1.0), \n "
            "        // DEFAULTVIEW_TOP = 4 \n "
            "        vec4(-1.0, 1.0, -1.0, 1.0), vec4(1.0, 1.0, 1.0, 1.0), vec4(1.0, 1.0, -1.0, 1.0), \n "
            "        vec4(-1.0, 1.0, -1.0, 1.0), vec4(-1.0, 1.0, 1.0, 1.0), vec4(1.0, 1.0, 1.0, 1.0), \n "
            "        // DEFAULTVIEW_BOTTOM = 5 \n "
            "        vec4(-1.0, -1.0, -1.0, 1.0), vec4(1.0, -1.0, -1.0, 1.0), vec4(1.0, -1.0, 1.0, 1.0), \n "
            "        vec4(-1.0, -1.0, -1.0, 1.0), vec4(1.0, -1.0, 1.0, 1.0), vec4(-1.0, -1.0, 1.0, 1.0)); \n "
            "    const vec4 colors[6] = vec4[6](vec4(0.0, 0.0, 1.0, 1.0), vec4(0.0, 0.0, 0.5, 1.0), \n "
            "        vec4(0.0, 1.0, 0.0, 1.0),  vec4(0.0, 0.5, 0.0, 1.0), \n "
            "        vec4(1.0, 0.0, 0.0, 1.0), vec4(0.5, 0.0, 0.0, 1.0)); \n "
            "    face_index = int(gl_VertexID / 6); \n "
            "    vertex_color = colors[face_index]; \n "
            "    if (view_index != face_index) { \n "
            "        vertex_color *= 0.25; \n "
            "        vertex_color.w = 1.0; \n "
            "    } \n "
            "    gl_Position = proj_mx * model_mx * rot_mx * vertices[gl_VertexID]; \n "
            "}";

        std::string fragment_src = "#version 130  \n "
                                   "#extension GL_ARB_explicit_attrib_location : require \n "
                                   "in vec4 vertex_color; \n "
                                   "flat in int face_index; \n "
                                   "uniform int id; \n "
                                   "layout(location = 0) out vec4 outFragColor; \n "
                                   "layout(location = 1) out vec2 outFragInfo; \n "
                                   "void main(void) { \n "
                                   "    float depth  = gl_FragCoord.z; \n "
                                   "    outFragColor = vertex_color; \n "
                                   "    outFragInfo  = vec2(float(id << face_index), depth); \n "
                                   "} ";

        if (!megamol::core::view::RenderUtils::CreateShader(this->shader, vertex_src, fragment_src)) {
            return;
        }
    }

    // Process pending manipulations ------------------------------------------
    int shader_view_index = inout_defaultview_index;
    for (auto& manip : pending_manipulations) {

        /// Indices must fit enum order in megamol::core::view::View3D_2::defaultview
        int view_index;
        if (id == (manip.obj_id >> 0)) // DEFAULTVIEW_FRONT
            view_index = 0;
        if (id == (manip.obj_id >> 1)) // DEFAULTVIEW_BACK
            view_index = 1;
        if (id == (manip.obj_id >> 2)) // DEFAULTVIEW_RIGHT
            view_index = 2;
        if (id == (manip.obj_id >> 3)) // DEFAULTVIEW_LEFT
            view_index = 3;
        if (id == (manip.obj_id >> 4)) // DEFAULTVIEW_TOP
            view_index = 4;
        if (id == (manip.obj_id >> 5)) // DEFAULTVIEW_BOTTOM
            view_index = 5;

        if (view_index >= 0) {
            if (manip.type == InteractionType::SELECT) {
                inout_defaultview_index = view_index;
                shader_view_index = view_index;
            } else if (manip.type == InteractionType::HIGHLIGHT) {
                out_hovered_view_index = view_index;
                shader_view_index = view_index;
            }
        }
    }

    // Draw -------------------------------------------------------------------

    // Create view/model and projection matrices
    const auto rotation = glm::inverse(
        glm::mat4_cast(glm::quat(view_orientation.w, view_orientation.x, view_orientation.y, view_orientation.z)));
    const float dist = 2.0f / std::tan(megamol::core::thecam::math::angle_deg2rad(30.0f) / 2.0f);
    glm::mat4 model(1.0f);
    model[3][2] = -dist;
    const auto proj = glm::perspective(megamol::core::thecam::math::angle_deg2rad(30.0f), 1.0f, 0.1f, 100.0f);

    // Set state
    const auto culling = glIsEnabled(GL_CULL_FACE);
    if (!culling) {
        glEnable(GL_CULL_FACE);
    }
    std::array<GLint, 4> viewport;
    glGetIntegerv(GL_VIEWPORT, viewport.data());
    int size = (100 * static_cast<int>(megamol::gui::gui_scaling.Get()));
    int x = viewport[2] - size;
    int y = viewport[3] - size - ImGui::GetFrameHeightWithSpacing();
    glViewport(x, y, size, size);

    this->shader->use();

    this->shader->setUniform("rot_mx", rotation);
    this->shader->setUniform("model_mx", model);
    this->shader->setUniform("proj_mx", proj);

    this->shader->setUniform("view_index", shader_view_index);
    this->shader->setUniform("id", static_cast<int>(id));

    glDrawArrays(GL_TRIANGLES, 0, 36);

    glUseProgram(0);

    // Restore
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    if (!culling) {
        glDisable(GL_CULL_FACE);
    }
}


InteractVector megamol::gui::PickableCube::GetInteractions(unsigned int id) const {
    InteractVector interactions;
    interactions.emplace_back(Interaction({InteractionType::SELECT, id, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
    interactions.emplace_back(Interaction({InteractionType::HIGHLIGHT, id, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
    // interactions.emplace_back(
    //    Interaction({InteractionType::MOVE_ALONG_AXIS_SCREEN, id, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
    // interactions.emplace_back(
    //    Interaction({InteractionType::MOVE_ALONG_AXIS_SCREEN, id, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f}));
    // interactions.emplace_back(
    //    Interaction({InteractionType::DESELECT, id, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
    return interactions;
}
