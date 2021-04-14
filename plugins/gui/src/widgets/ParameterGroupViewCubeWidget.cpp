/*
 * ParameterGroupViewCubeWidget.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "widgets/ParameterGroupViewCubeWidget.h"
#include "graph/ParameterGroups.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::gui;


// *** Pickable Cube ******************************************************** //

megamol::gui::PickableCube::PickableCube(void) : shader(nullptr) {}


bool megamol::gui::PickableCube::Draw(unsigned int id, int& inout_view_index, int& inout_orientation_index,
    int& out_view_hover_index, int& out_orientation_hover_index, const glm::vec4& view_orientation,
    const glm::vec2& vp_dim, ManipVector& pending_manipulations) {

    assert(ImGui::GetCurrentContext() != nullptr);
    bool selected = false;

    // Info: IDs of the six cube faces are encoded via bit shift by face index of given parameter id.

    // Create shader once -----------------------------------------------------
    if (this->shader == nullptr) {
        std::string vertex_src =
            "#version 130 \n "
            "uniform int id; \n "
            "uniform mat4 rot_mx; \n "
            "uniform mat4 model_mx; \n "
            "uniform mat4 proj_mx; \n "
            "uniform int view_index; \n "
            "uniform int orientation_index; \n "
            "uniform int view_hover_index; \n "
            "uniform int orientation_hover_index; \n "
            "out vec4 vertex_color; \n "
            "flat out int face_id; \n "
            "void main() { \n "
            "    // Vertex indices must fit enum order in megamol::core::view::View3D_2::defaultview \n "
            "    const vec4 vertices[72] = vec4[72]( \n "
            "        // DEFAULTVIEW_FRONT = 0 \n "
            "        vec4(1.0, 1.0, 1.0, 1.0), vec4(-1.0, 1.0, 1.0, 1.0), vec4(0.0, 0.0, 1.0, 1.0), \n "
            "        vec4(1.0, -1.0, 1.0, 1.0), vec4(1.0, 1.0, 1.0, 1.0), vec4(0.0, 0.0, 1.0, 1.0), \n "
            "        vec4(-1.0, -1.0, 1.0, 1.0), vec4(1.0, -1.0, 1.0, 1.0), vec4(0.0, 0.0, 1.0, 1.0), \n "
            "        vec4(-1.0, 1.0, 1.0, 1.0), vec4(-1.0, -1.0, 1.0, 1.0), vec4(0.0, 0.0, 1.0, 1.0), \n "
            "        // DEFAULTVIEW_BACK = 1 \n "
            "        vec4(-1.0, 1.0, -1.0, 1.0), vec4(1.0, 1.0, -1.0, 1.0), vec4(0.0, 0.0, -1.0, 1.0), \n "
            "        vec4(-1.0, -1.0, -1.0, 1.0), vec4(-1.0, 1.0, -1.0, 1.0), vec4(0.0, 0.0, -1.0, 1.0), \n "
            "        vec4(1.0, -1.0, -1.0, 1.0), vec4(-1.0, -1.0, -1.0, 1.0), vec4(0.0, 0.0, -1.0, 1.0), \n "
            "        vec4(1.0, 1.0, -1.0, 1.0), vec4(1.0, -1.0, -1.0, 1.0), vec4(0.0, 0.0, -1.0, 1.0), \n "
            "        // DEFAULTVIEW_RIGHT = 2 \n "
            "        vec4(1.0, 1.0, -1.0, 1.0), vec4(1.0, 1.0, 1.0, 1.0), vec4(1.0, 0.0, 0.0, 1.0), \n "
            "        vec4(1.0, -1.0, -1.0, 1.0), vec4(1.0, 1.0, -1.0, 1.0), vec4(1.0, 0.0, 0.0, 1.0), \n "
            "        vec4(1.0, -1.0, 1.0, 1.0), vec4(1.0, -1.0, -1.0, 1.0), vec4(1.0, 0.0, 0.0, 1.0), \n "
            "        vec4(1.0, 1.0, 1.0, 1.0), vec4(1.0, -1.0, 1.0, 1.0), vec4(1.0, 0.0, 0.0, 1.0), \n "
            "        // DEFAULTVIEW_LEFT = 3 \n "
            "        vec4(-1.0, 1.0, 1.0, 1.0), vec4(-1.0, 1.0, -1.0, 1.0), vec4(-1.0, 0.0, 0.0, 1.0), \n "
            "        vec4(-1.0, -1.0, 1.0, 1.0), vec4(-1.0, 1.0, 1.0, 1.0), vec4(-1.0, 0.0, 0.0, 1.0), \n "
            "        vec4(-1.0, -1.0, -1.0, 1.0), vec4(-1.0, -1.0, 1.0, 1.0), vec4(-1.0, 0.0, 0.0, 1.0), \n "
            "        vec4(-1.0, 1.0, -1.0, 1.0), vec4(-1.0, -1.0, -1.0, 1.0), vec4(-1.0, 0.0, 0.0, 1.0), \n "
            "        // DEFAULTVIEW_TOP = 4 \n "
            "        vec4(1.0, 1.0, -1.0, 1.0), vec4(-1.0, 1.0, -1.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0), \n"
            "        vec4(1.0, 1.0, 1.0, 1.0), vec4(1.0, 1.0, -1.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0), \n "
            "        vec4(-1.0, 1.0, 1.0, 1.0), vec4(1.0, 1.0, 1.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0), \n "
            "        vec4(-1.0, 1.0, -1.0, 1.0), vec4(-1.0, 1.0, 1.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0), \n "
            "        // DEFAULTVIEW_BOTTOM = 5 \n "
            "        vec4(1.0, -1.0, 1.0, 1.0), vec4(-1.0, -1.0, 1.0, 1.0), vec4(0.0, -1.0, 0.0, 1.0), \n "
            "        vec4(1.0, -1.0, -1.0, 1.0), vec4(1.0, -1.0, 1.0, 1.0), vec4(0.0, -1.0, 0.0, 1.0), \n "
            "        vec4(-1.0, -1.0, -1.0, 1.0), vec4(1.0, -1.0, -1.0, 1.0), vec4(0.0, -1.0, 0.0, 1.0), \n "
            "        vec4(-1.0, -1.0, 1.0, 1.0), vec4(-1.0, -1.0, -1.0, 1.0), vec4(0.0, -1.0, 0.0, 1.0)); \n "
            "    \n"
            "    const vec4 colors[6] = vec4[6](vec4(0.0, 0.0, 1.0, 1.0), vec4(0.0, 1.0, 1.0, 1.0), \n "
            "                                   vec4(1.0, 0.0, 0.0, 1.0), vec4(1.0, 0.0, 1.0, 1.0), \n "
            "                                   vec4(0.0, 1.0, 0.0, 1.0), vec4(1.0, 1.0, 0.0, 1.0)); \n "
            "    // Calculate indices and IDs \n"
            "    float vertex_index = float(gl_VertexID); \n"
            "    float mod_index = vertex_index - (12.0 * floor(vertex_index/12.0)); \n"
            "    float mod_triangle_index = mod_index - (3.0 * floor(mod_index/3.0)); \n"
            "    int current_orientation_index = int(floor(mod_index / 3.0)); \n"
            "    int current_orientation_id = int(1 << current_orientation_index); // in range [0-3]\n "
            "    int current_view_index = int(gl_VertexID / 12);       // in range [0-5] \n "
            "    face_id = int((id << (current_view_index + 4)) | current_orientation_id);"
            "    \n"
            "    // Set colors depending on selected or hovered triangles \n"
            "    vertex_color = colors[current_view_index]; \n "
            "    if (view_index != current_view_index) { \n "
            "        vertex_color *= 0.25; \n "
            "    } \n "
            "    if (view_index == current_view_index) { \n "
            "        vertex_color *= (0.5 + (0.5 - 0.5*(current_orientation_index/2.0))); \n "
            "    } \n "
            "    if ((view_hover_index == current_view_index) && \n"
            "            (orientation_hover_index == current_orientation_index)) { \n "
            "        vertex_color = vec4(0.5); \n "
            "    } \n "
            "    vertex_color.w = 1.0; \n "
            "    \n"
            "    gl_Position = proj_mx * model_mx * rot_mx * vertices[gl_VertexID]; \n "
            "}";

        std::string fragment_src = "#version 130  \n "
                                   "#extension GL_ARB_explicit_attrib_location : require \n "
                                   "in vec4 vertex_color; \n "
                                   "flat in int face_id; \n "
                                   "layout(location = 0) out vec4 outFragColor; \n "
                                   "layout(location = 1) out vec2 outFragInfo; \n "
                                   "void main(void) { \n "
                                   "    outFragColor = vertex_color; \n "
                                   "    outFragInfo  = vec2(float(face_id), gl_FragCoord.z); \n "
                                   "} ";

        if (!megamol::core::view::RenderUtils::CreateShader(this->shader, vertex_src, fragment_src)) {
            return false;
        }
    }

    // Process pending manipulations ------------------------------------------
    out_orientation_hover_index = -1;
    out_view_hover_index = -1;
    for (auto& manip : pending_manipulations) {

        // ID is shifted by at least 4 bits and at most 9 bits.
        // Leaving at least 23 bit for actual id (meaning max id can be ....?)
        /// Indices must fit enum order in megamol::core::view::View3D_2::defaultview
        int view_index = -1;
        if (id == (id & (manip.obj_id >> 4))) // DEFAULTVIEW_FRONT
            view_index = 0;
        else if (id == (id & (manip.obj_id >> 5))) // DEFAULTVIEW_BACK
            view_index = 1;
        else if (id == (id & (manip.obj_id >> 6))) // DEFAULTVIEW_RIGHT
            view_index = 2;
        else if (id == (id & (manip.obj_id >> 7))) // DEFAULTVIEW_LEFT
            view_index = 3;
        else if (id == (id & (manip.obj_id >> 8))) // DEFAULTVIEW_TOP
            view_index = 4;
        else if (id == (id & (manip.obj_id >> 9))) // DEFAULTVIEW_BOTTOM
            view_index = 5;

        int orientation_index = -1;
        // First 4 bit indicate currently hovered orientation
        // Orientation is given by triangle order in shader of pickable cube
        if ((1 << 0) & manip.obj_id)
            orientation_index = 0; // TOP
        else if ((1 << 1) & manip.obj_id)
            orientation_index = 1; // RIGHT
        else if ((1 << 2) & manip.obj_id)
            orientation_index = 2; // BOTTOM
        else if ((1 << 3) & manip.obj_id)
            orientation_index = 3; // LEFT

        if (view_index >= 0) {
            if (manip.type == InteractionType::SELECT) {
                inout_view_index = view_index;
                inout_orientation_index = orientation_index;
                selected = true;
            } else if (manip.type == InteractionType::HIGHLIGHT) {
                out_view_hover_index = view_index;
                out_orientation_hover_index = orientation_index;
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
    this->shader->setUniform("view_index", inout_view_index);
    this->shader->setUniform("orientation_index", inout_orientation_index);
    this->shader->setUniform("view_hover_index", out_view_hover_index);
    this->shader->setUniform("orientation_hover_index", out_orientation_hover_index);
    this->shader->setUniform("id", static_cast<int>(id));

    glDrawArrays(GL_TRIANGLES, 0, 72);

    glUseProgram(0);

    // Restore
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    if (!culling) {
        glDisable(GL_CULL_FACE);
    }

    return selected;
}


InteractVector megamol::gui::PickableCube::GetInteractions(unsigned int id) const {

    InteractVector interactions;
    interactions.emplace_back(Interaction({InteractionType::SELECT, id, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
    interactions.emplace_back(Interaction({InteractionType::HIGHLIGHT, id, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
    return interactions;
}


// *** Parameter Group View Cube Widget ************************************ //

megamol::gui::ParameterGroupViewCubeWidget::ParameterGroupViewCubeWidget(void)
        : AbstractParameterGroupWidget(megamol::gui::GenerateUniqueID())
        , tooltip()
        , cube_widget()
        , last_presentation(param::AbstractParamPresentation::Presentation::Basic) {

    this->InitPresentation(ParamType_t::GROUP_3D_CUBE);
    this->name = "view";
}


bool megamol::gui::ParameterGroupViewCubeWidget::Check(bool only_check, ParamPtrVector_t& params) {

    bool param_cubeOrientation = false;
    bool param_defaultView = false;
    bool param_defaultOrientation = false;
    bool param_resetView = false;
    bool param_showCube = false;
    for (auto& param_ptr : params) {
        if ((param_ptr->Name() == "cubeOrientation") && (param_ptr->Type() == ParamType_t::VECTOR4F)) {
            param_cubeOrientation = true;
        } else if ((param_ptr->Name() == "defaultView") && (param_ptr->Type() == ParamType_t::ENUM)) {
            param_defaultView = true;
        } else if ((param_ptr->Name() == "defaultOrientation") && (param_ptr->Type() == ParamType_t::ENUM)) {
            param_defaultOrientation = true;
        } else if ((param_ptr->Name() == "resetView") && (param_ptr->Type() == ParamType_t::BUTTON)) {
            param_resetView = true;
        } else if ((param_ptr->Name() == "showViewCube") && (param_ptr->Type() == ParamType_t::BOOL)) {
            param_showCube = true;
        }
    }

    return (param_cubeOrientation && param_defaultView && param_showCube);
}


bool megamol::gui::ParameterGroupViewCubeWidget::Draw(ParamPtrVector_t params, const std::string& in_module_fullname,
    const std::string& in_search, megamol::gui::Parameter::WidgetScope in_scope, PickingBuffer* inout_picking_buffer) {

    if (ImGui::GetCurrentContext() == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Check required parameters ----------------------------------------------
    Parameter* param_cubeOrientation = nullptr;
    Parameter* param_defaultView = nullptr;
    Parameter* param_defaultOrientation = nullptr;
    Parameter* param_resetView = nullptr;
    Parameter* param_showCube = nullptr;
    /// Find specific parameters of group by name because parameter type can occure multiple times.
    for (auto& param_ptr : params) {
        if ((param_ptr->Name() == "cubeOrientation") && (param_ptr->Type() == ParamType_t::VECTOR4F)) {
            param_cubeOrientation = param_ptr;
        } else if ((param_ptr->Name() == "defaultView") && (param_ptr->Type() == ParamType_t::ENUM)) {
            param_defaultView = param_ptr;
        } else if ((param_ptr->Name() == "defaultOrientation") && (param_ptr->Type() == ParamType_t::ENUM)) {
            param_defaultOrientation = param_ptr;
        } else if ((param_ptr->Name() == "resetView") && (param_ptr->Type() == ParamType_t::BUTTON)) {
            param_resetView = param_ptr;
        } else if ((param_ptr->Name() == "showViewCube") && (param_ptr->Type() == ParamType_t::BOOL)) {
            param_showCube = param_ptr;
        }
    }
    if ((param_cubeOrientation == nullptr) || (param_defaultView == nullptr) || (param_defaultOrientation == nullptr) ||
        (param_showCube == nullptr)) {
        utility::log::Log::DefaultLog.WriteError("[GUI] Unable to find all required parameters by name "
                                                 "for '%s' group widget. [%s, %s, line %d]\n",
            this->name.c_str(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Parameter presentation -------------------------------------------------
    auto presentation = this->GetGUIPresentation();
    if (presentation != this->last_presentation) {
        param_showCube->SetValue((presentation == param::AbstractParamPresentation::Presentation::Group_3D_Cube));
        this->last_presentation = presentation;
    } else {
        if (std::get<bool>(param_showCube->GetValue())) {
            this->last_presentation = param::AbstractParamPresentation::Presentation::Group_3D_Cube;
            this->SetGUIPresentation(this->last_presentation);
        } else {
            this->last_presentation = param::AbstractParamPresentation::Presentation::Basic;
            this->SetGUIPresentation(this->last_presentation);
        }
    }

    if (presentation == param::AbstractParamPresentation::Presentation::Basic) {

        if (in_scope == Parameter::WidgetScope::LOCAL) {
            // LOCAL

            ParameterGroups::DrawGroupedParameters(
                this->name, params, in_module_fullname, in_search, in_scope, nullptr, nullptr, GUI_INVALID_ID);

            return true;

        } else if (in_scope == Parameter::WidgetScope::GLOBAL) {

            // no global implementation ...
            return true;
        }

    } else if (presentation == param::AbstractParamPresentation::Presentation::Group_3D_Cube) {

        if (in_scope == Parameter::WidgetScope::LOCAL) {
            // LOCAL

            ParameterGroups::DrawGroupedParameters(
                this->name, params, "", in_search, in_scope, nullptr, nullptr, GUI_INVALID_ID);

            return true;

        } else if (in_scope == Parameter::WidgetScope::GLOBAL) {
            // GLOBAL

            if (inout_picking_buffer == nullptr) {
                utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Pointer to required picking buffer is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                    __LINE__);
                return false;
            }

            ImGui::PushID(this->uid);

            auto id = param_defaultView->UID();
            inout_picking_buffer->AddInteractionObject(id, this->cube_widget.GetInteractions(id));

            ImGuiIO& io = ImGui::GetIO();
            auto default_view = std::get<int>(param_defaultView->GetValue());
            auto default_orientation = std::get<int>(param_defaultOrientation->GetValue());
            auto view_orientation = std::get<glm::vec4>(param_cubeOrientation->GetValue());
            auto viewport_dim = glm::vec2(io.DisplaySize.x, io.DisplaySize.y);
            int hovered_view = -1;
            int hovered_orientation = -1;

            bool selected = this->cube_widget.Draw(id, default_view, default_orientation, hovered_view,
                hovered_orientation, view_orientation, viewport_dim, inout_picking_buffer->GetPendingManipulations());

            std::string tooltip_text;
            /// Indices must fit enum order in view::View3D_2::defaultview
            if (hovered_view >= 0) {
                switch (hovered_view) {
                case (0): // DEFAULTVIEW_FRONT
                    tooltip_text += "[Front]";
                    break;
                case (1): // DEFAULTVIEW_BACK
                    tooltip_text += "[Back]";
                    break;
                case (2): // DEFAULTVIEW_RIGHT
                    tooltip_text += "[Right]";
                    break;
                case (3): // DEFAULTVIEW_LEFT
                    tooltip_text += "[Left]";
                    break;
                case (4): // DEFAULTVIEW_TOP
                    tooltip_text += "[Top]";
                    break;
                case (5): // DEFAULTVIEW_BOTTOM
                    tooltip_text += "[Bottom]";
                    break;
                default:
                    break;
                }
            }
            // Order is given by triangle order in shader of pickable cube
            if (hovered_orientation >= 0) {
                tooltip_text += " ";
                switch (hovered_orientation) {
                case (0): // TOP
                    tooltip_text += "0 degree";
                    break;
                case (1): // RIGHT
                    tooltip_text += "90 degree";
                    break;
                case (2): // BOTTOM
                    tooltip_text += "180 degree";
                    break;
                case (3): // LEFT
                    tooltip_text += "270 degree";
                    break;
                default:
                    break;
                }
            }
            if (!tooltip_text.empty()) {
                ImGui::BeginTooltip();
                ImGui::TextUnformatted(tooltip_text.c_str());
                ImGui::EndTooltip();
            }

            if (selected) {
                param_resetView->ForceSetValueDirty();
            }
            param_defaultOrientation->SetValue(default_orientation);
            param_defaultView->SetValue(default_view);

            ImGui::PopID();

            return true;
        }
    }
    return false;
}
