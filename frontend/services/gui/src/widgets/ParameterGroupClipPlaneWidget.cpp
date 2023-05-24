/*
 * ParameterGroupClipPlaneWidget.cpp
 *
 * Copyright (C) 2023 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "widgets/ParameterGroupClipPlaneWidget.h"
#include "graph/ParameterGroups.h"
#include "mmcore/view/CameraSerializer.h"

#include <ImGuizmo.h>


using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::utility;
using namespace megamol::gui;


megamol::gui::ParameterGroupClipPlaneWidget::ParameterGroupClipPlaneWidget()
        : AbstractParameterGroupWidget(megamol::gui::GenerateUniqueID())
        , tooltip()
        , cameraSerializer()
        , guizmo_mat()
{
    this->InitPresentation(ParamType_t::GROUP_CLIPPLANE);
    this->name = "clip";
    this->guizmo_mat = glm::identity<glm::mat4>();
}


bool megamol::gui::ParameterGroupClipPlaneWidget::Check(ParamPtrVector_t& params) {

    int check = 1;
    for (auto& param_ptr : params) {
        if ((param_ptr->Name() == "enable") && (param_ptr->Type() == ParamType_t::BOOL)) {
            check = check << 1;
        } else if ((param_ptr->Name() == "colour") && (param_ptr->Type() == ParamType_t::COLOR)) {
            check = check << 1;
        } else if ((param_ptr->Name() == "normal") && (param_ptr->Type() == ParamType_t::VECTOR3F)) {
            check = check << 1;
        } else if ((param_ptr->Name() == "point") && (param_ptr->Type() == ParamType_t::VECTOR3F)) {
            check = check << 1;
        } else if ((param_ptr->Name() == "dist") && (param_ptr->Type() == ParamType_t::FLOAT)) {
            check = check << 1;
        } else if ((param_ptr->Name() == "camera") && (param_ptr->Type() == ParamType_t::STRING)) {
            check = check << 1;
        }
    }
    return (check == (1 << 6));
}


bool megamol::gui::ParameterGroupClipPlaneWidget::Draw(ParamPtrVector_t params, const std::string& in_search,
    megamol::gui::Parameter::WidgetScope in_scope, core::utility::PickingBuffer* inout_picking_buffer,
    ImGuiID in_override_header_state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        log::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Get required parameters ----------------------------------------------
    Parameter* param_enable = nullptr;
    Parameter* param_colour = nullptr;
    Parameter* param_normal = nullptr;
    Parameter* param_point = nullptr;
    Parameter* param_dist = nullptr;
    Parameter* param_camera = nullptr;
    /// Find specific parameters of group by name because parameter type can occure multiple times.
    for (auto& param_ptr : params) {
        if ((param_ptr->Name() == "enable") && (param_ptr->Type() == ParamType_t::BOOL)) {
            param_enable= param_ptr;
        } else if ((param_ptr->Name() == "colour") && (param_ptr->Type() == ParamType_t::COLOR)) {
            param_colour = param_ptr;
        } else if ((param_ptr->Name() == "normal") && (param_ptr->Type() == ParamType_t::VECTOR3F)) {
            param_normal = param_ptr;
        } else if ((param_ptr->Name() == "point") && (param_ptr->Type() == ParamType_t::VECTOR3F)) {
            param_point = param_ptr;
        } else if ((param_ptr->Name() == "dist") && (param_ptr->Type() == ParamType_t::FLOAT)) {
            param_dist = param_ptr;
        } else if ((param_ptr->Name() == "camera") && (param_ptr->Type() == ParamType_t::STRING)) {
            param_camera = param_ptr;
        }
    }
    if ((param_enable == nullptr) || (param_colour == nullptr) || (param_normal == nullptr) ||
        (param_point == nullptr) || (param_dist == nullptr) || (param_camera == nullptr)) {
        utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to find all required parameters by name for clip plane group widget. [%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    std::string camstring = std::get<std::string>(param_camera->GetValue());
    megamol::core::view::Camera cam;
    if (!this->cameraSerializer.deserialize(cam, camstring)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "The entered camera string was not valid. No change of the camera has been performed");
        return false;
    }

    ImGuiStyle& style = ImGui::GetStyle();
    ImGuiIO& io = ImGui::GetIO();

    // Parameter presentation -------------------------------------------------
    auto presentation = this->GetGUIPresentation();
    if (presentation == param::AbstractParamPresentation::Presentation::Basic) {

        if (in_scope == Parameter::WidgetScope::LOCAL) {

            ParameterGroups::DrawGroupedParameters(
                this->name, params, in_search, in_scope, nullptr, in_override_header_state);
            return true;

        } else if (in_scope == Parameter::WidgetScope::GLOBAL) {

            // no global implementation ...
            return true;
        }

    } else if (presentation == param::AbstractParamPresentation::Presentation::Group_ClipPlane) {

        if (in_scope == Parameter::WidgetScope::LOCAL) {

            ParameterGroups::DrawGroupedParameters(
                this->name, params, in_search, in_scope, nullptr, in_override_header_state);



            ImGui::Text("X: %f Y: %f", io.MousePos.x, io.MousePos.y);
            if (ImGuizmo::IsUsing()) {
                ImGui::Text("Using gizmo");
            } else {
                ImGui::Text(ImGuizmo::IsOver() ? "Over gizmo" : "");
                ImGui::SameLine();
                ImGui::Text(ImGuizmo::IsOver(ImGuizmo::TRANSLATE) ? "Over translate gizmo" : "");
                ImGui::SameLine();
                ImGui::Text(ImGuizmo::IsOver(ImGuizmo::ROTATE) ? "Over rotate gizmo" : "");
                ImGui::SameLine();
                ImGui::Text(ImGuizmo::IsOver(ImGuizmo::SCALE) ? "Over scale gizmo" : "");
            }



            return true;

        } else if (in_scope == Parameter::WidgetScope::GLOBAL) {

            ImGui::PushID(static_cast<int>(this->uid));



            // DRAW -------------------------------------------------------------------

            // Camera
            auto cam_view = cam.getViewMatrix();
            auto cam_proj = cam.getProjectionMatrix();
            auto cam_pose = cam.get<core::view::Camera::Pose>();



            // TODO ////////////////////////////
            ImGuizmo::BeginFrame();
            ImGuizmo::Enable(true);
            ImGuizmo::SetID(0);
            ImGuizmo::SetOrthographic(false);
            ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);

            ImGuizmo::OPERATION mCurrentGizmoOperation(ImGuizmo::ROTATE);
            ImGuizmo::MODE mCurrentGizmoMode(ImGuizmo::LOCAL);

            bool useSnap = false;
            bool boundSizing = false;
            bool boundSizingSnap = false;

            float snap[3] = {1.f, 1.f, 1.f};
            float bounds[] = {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f};
            float boundsSnap[] = {0.1f, 0.1f, 0.1f};

            /// Grid
            ImGuizmo::DrawGrid(glm::value_ptr(cam_view), glm::value_ptr(cam_proj), glm::value_ptr(this->guizmo_mat), 1.0f);

            /// Manipulator
            ImGuizmo::Manipulate(glm::value_ptr(cam_view), glm::value_ptr(cam_proj), mCurrentGizmoOperation,
                mCurrentGizmoMode,
                glm::value_ptr(this->guizmo_mat), nullptr, useSnap ? &snap[0] : nullptr, boundSizing ? bounds : nullptr,
                boundSizingSnap ? boundsSnap : nullptr);

            /// Cube inside manipulator
            //int gizmoCount = 1;
            //ImGuizmo::DrawCubes(glm::value_ptr(cam_view), glm::value_ptr(cam_proj), glm::value_ptr(this->guizmo_mat), gizmoCount);

            /// View Cube
            //float camDistance = 8.f;
            //float viewManipulateRight = io.DisplaySize.x;
            //float viewManipulateTop = 0;
            //ImGuizmo::ViewManipulate(glm::value_ptr(cam_view), camDistance, ImVec2(viewManipulateRight - 128, viewManipulateTop), ImVec2(128, 128), 0x10101010);


            std::string popup_label = "guizmo_popup";
            if ( ImGui::IsMouseClicked(ImGuiMouseButton_Right) &&
                !ImGui::IsPopupOpen(popup_label.c_str())) { // ImGuizmo::IsOver() &&
                ImGui::OpenPopup(popup_label.c_str());

            }
            if (ImGui::BeginPopup(popup_label.c_str(), ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove)) {

                ImGui::TextDisabled("Context Menu");

                ImGui::EndPopup();
            }

           


            //param_play->SetValue();

            
            ///////////////////////////////////

            ImGui::PopID();


            return true;
        }
    }

    return false;
}


void megamol::gui::ParameterGroupClipPlaneWidget::OrthoGraphic(
    const float l, float r, float b, const float t, float zn, const float zf, float* m16) {
    m16[0] = 2 / (r - l);
    m16[1] = 0.0f;
    m16[2] = 0.0f;
    m16[3] = 0.0f;
    m16[4] = 0.0f;
    m16[5] = 2 / (t - b);
    m16[6] = 0.0f;
    m16[7] = 0.0f;
    m16[8] = 0.0f;
    m16[9] = 0.0f;
    m16[10] = 1.0f / (zf - zn);
    m16[11] = 0.0f;
    m16[12] = (l + r) / (l - r);
    m16[13] = (t + b) / (b - t);
    m16[14] = zn / (zn - zf);
    m16[15] = 1.0f;
}
