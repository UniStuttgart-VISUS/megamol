/*
 * ParameterGroupViewCubeWidget.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "widgets/ParameterGroupViewCubeWidget.h"
#include "graph/ParameterGroupsPresentation.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::gui;


megamol::gui::ParameterGroupViewCubeWidget::ParameterGroupViewCubeWidget(void)
        : AbstractParameterGroupWidget(megamol::gui::GenerateUniqueID()), tooltip(), cube_widget() {

    this->InitPresentation(Param_t::GROUP_3D_CUBE);
    this->name = "view";
}


bool megamol::gui::ParameterGroupViewCubeWidget::Check(bool only_check, ParamPtrVector_t& params) {

    bool param_cubeOrientation = false;
    bool param_defaultView = false;
    bool param_showCube = false;
    for (auto& param_ptr : params) {
        if ((param_ptr->GetName() == "cubeOrientation") && (param_ptr->type == Param_t::VECTOR4F)) {
            param_cubeOrientation = true;
        } else if ((param_ptr->GetName() == "defaultView") && (param_ptr->type == Param_t::ENUM)) {
            param_defaultView = true;
        } else if ((param_ptr->GetName() == "showViewCube") && (param_ptr->type == Param_t::BOOL)) {
            param_showCube = true;
        }
    }

    return (param_cubeOrientation && param_defaultView && param_showCube);
}


bool megamol::gui::ParameterGroupViewCubeWidget::Draw(ParamPtrVector_t params, const std::string& in_module_fullname,
    const std::string& in_search, megamol::gui::ParameterPresentation::WidgetScope in_scope,
    PickingBuffer* inout_picking_buffer) {

    if (ImGui::GetCurrentContext() == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Check required parameters ----------------------------------------------
    Parameter* param_cubeOrientation = nullptr;
    Parameter* param_defaultView = nullptr;
    Parameter* param_showCube = nullptr;
    /// Find specific parameters of group by name because parameter type can occure multiple times.
    for (auto& param_ptr : params) {
        if ((param_ptr->GetName() == "cubeOrientation") && (param_ptr->type == Param_t::VECTOR4F)) {
            param_cubeOrientation = param_ptr;
        } else if ((param_ptr->GetName() == "defaultView") && (param_ptr->type == Param_t::ENUM)) {
            param_defaultView = param_ptr;
        } else if ((param_ptr->GetName() == "showViewCube") && (param_ptr->type == Param_t::BOOL)) {
            param_showCube = param_ptr;
        }
    }
    if ((param_cubeOrientation == nullptr) || (param_defaultView == nullptr) || (param_showCube == nullptr)) {
        utility::log::Log::DefaultLog.WriteError("[GUI] Unable to find all required parameters by name "
                                                 "for '%s' group widget. [%s, %s, line %d]\n",
            this->name.c_str(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Parameter presentation -------------------------------------------------
    auto presentation = this->GetGUIPresentation();
    // Switch presentation via parameter
    if (param_showCube->IsValueDirty()) {
        if (std::get<bool>(param_showCube->GetValue())) {
            this->SetGUIPresentation(param::AbstractParamPresentation::Presentation::Group_3D_Cube);
        } else {
            this->SetGUIPresentation(param::AbstractParamPresentation::Presentation::Basic);
        }
    }
    param_showCube->SetValue((presentation == param::AbstractParamPresentation::Presentation::Group_3D_Cube));

    if (presentation == param::AbstractParamPresentation::Presentation::Basic) {

        if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
            // LOCAL

            ParameterGroupsPresentation::DrawGroupedParameters(
                this->name, params, in_module_fullname, in_search, in_scope, nullptr, nullptr, GUI_INVALID_ID);

            return true;

        } else if (in_scope == ParameterPresentation::WidgetScope::GLOBAL) {

            // no global implementation ...
            return true;
        }

    } else if (presentation == param::AbstractParamPresentation::Presentation::Group_3D_Cube) {

        if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
            // LOCAL

            ParameterGroupsPresentation::DrawGroupedParameters(
                this->name, params, "", in_search, in_scope, nullptr, nullptr, GUI_INVALID_ID);

            return true;

        } else if (in_scope == ParameterPresentation::WidgetScope::GLOBAL) {
            // GLOBAL

            if (inout_picking_buffer == nullptr) {
                utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Pointer to required picking buffer is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                    __LINE__);
                return false;
            }

            ImGui::PushID(this->uid);

            auto id = param_defaultView->uid;
            inout_picking_buffer->AddInteractionObject(id, this->cube_widget.GetInteractions(id));

            ImGuiIO& io = ImGui::GetIO();
            auto default_view = std::get<int>(param_defaultView->GetValue());
            auto view_orientation = std::get<glm::vec4>(param_cubeOrientation->GetValue());
            auto viewport_dim = glm::vec2(io.DisplaySize.x, io.DisplaySize.y);
            int hovered_view = -1;
            this->cube_widget.Draw(id, default_view, hovered_view, view_orientation, viewport_dim,
                inout_picking_buffer->GetPendingManipulations());

            std::string tooltip_text;
            /// Indices must fit enum order in view::View3D_2::defaultview
            switch (hovered_view) {
            case (0): // DEFAULTVIEW_FRONT
                tooltip_text = "Front";
                break;
            case (1): // DEFAULTVIEW_BACK
                tooltip_text = "Back";
                break;
            case (2): // DEFAULTVIEW_RIGHT
                tooltip_text = "Right";
                break;
            case (3): // DEFAULTVIEW_LEFT
                tooltip_text = "Left";
                break;
            case (4): // DEFAULTVIEW_TOP
                tooltip_text = "Top";
                break;
            case (5): // DEFAULTVIEW_BOTTOM
                tooltip_text = "Bottom";
                break;
            default:
                break;
            }
            if (!tooltip_text.empty()) {
                ImGui::BeginTooltip();
                ImGui::TextUnformatted(tooltip_text.c_str());
                ImGui::EndTooltip();
            }

            param_defaultView->SetValue(default_view);

            ImGui::PopID();

            return true;
        }
    }

    return false;
}
