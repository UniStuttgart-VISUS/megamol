/*
 * ParameterGroupClipPlaneWidget.cpp
 *
 * Copyright (C) 2023 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "widgets/ParameterGroupClipPlaneWidget.h"
#include "graph/ParameterGroups.h"
#include "mmcore/view/CameraSerializer.h"
#include "ButtonWidgets.h"
#include <glm/gtx/rotate_vector.hpp>


using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::utility;
using namespace megamol::gui;



megamol::gui::ParameterGroupClipPlaneWidget::ParameterGroupClipPlaneWidget()
        : AbstractParameterGroupWidget(megamol::gui::GenerateUniqueID())
        , tooltip()
        , camera_serializer()
        , guizmo_manipulation(glm::identity<glm::mat4>())
        , guizmo_operation(ImGuizmo::ROTATE) {

    this->InitPresentation(ParamType_t::GROUP_CLIPPLANE);
    this->name = "clip";
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
            param_enable = param_ptr;
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
    if (!this->camera_serializer.deserialize(cam, camstring)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "The entered camera string was not valid. No change of the camera has been performed");
        return false;
    }

    // Parameter presentation -------------------------------------------------
    ImGuiStyle& style = ImGui::GetStyle();
    ImGuiIO& io = ImGui::GetIO();

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

            ImGui::Text("Right-Click Guizmo for Context Menu.");

            ParameterGroups::DrawGroupedParameters(
                this->name, params, in_search, in_scope, nullptr, in_override_header_state);

            return true;

        } else if (in_scope == Parameter::WidgetScope::GLOBAL) {

            ImGui::PushID(static_cast<int>(this->uid));

            auto plane_enabled = std::get<bool>(param_enable->GetValue());
            auto plane_normal = std::get<glm::vec3>(param_normal->GetValue());
            auto plane_point = std::get<glm::vec3>(param_point->GetValue());
            auto plane_dist = std::get<float>(param_dist->GetValue());

            // Transform plane vectors to matrix
            auto plane_matrix = glm::orientation(plane_normal, glm::vec3(0.0f, 1.0f, 0.0f));
            plane_matrix = glm::translate(plane_matrix, plane_normal * plane_dist);

            // Camera
            auto cam_view = cam.getViewMatrix();
            auto cam_proj = cam.getProjectionMatrix();
            auto cam_pose = cam.get<core::view::Camera::Pose>();

            ImGuizmo::BeginFrame();
            ImGuizmo::Enable(plane_enabled);
            ImGuizmo::SetID(0);
            ImGuizmo::SetOrthographic(false);
            ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);

            bool useSnap = false;
            bool boundSizing = false;
            bool boundSizingSnap = false;
            float snap[3] = {1.f, 1.f, 1.f};
            float bounds[] = {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f};
            float boundsSnap[] = {0.1f, 0.1f, 0.1f};

            /// Grid
            ImGuizmo::DrawGrid(glm::value_ptr(cam_view), glm::value_ptr(cam_proj), glm::value_ptr(plane_matrix), 1.0f);

            /// Manipulator
            ImGuizmo::Manipulate(glm::value_ptr(cam_view), glm::value_ptr(cam_proj), this->guizmo_operation, ImGuizmo::LOCAL,
                glm::value_ptr(this->guizmo_manipulation), nullptr, useSnap ? &snap[0] : nullptr,
                boundSizing ? bounds : nullptr, boundSizingSnap ? boundsSnap : nullptr);

            /// Cube inside manipulator
            //ImGuizmo::DrawCubes(glm::value_ptr(cam_view), glm::value_ptr(cam_proj), glm::value_ptr(this->guizmo_matrix), 1);

            /// View Cube
            //float camDistance = 8.f;
            //float viewManipulateRight = io.DisplaySize.x;
            //float viewManipulateTop = 0;
            //ImGuizmo::ViewManipulate(glm::value_ptr(cam_view), camDistance, ImVec2(viewManipulateRight - 128, viewManipulateTop), ImVec2(128, 128), 0x10101010);

            // Transform plane matrix to vectors
            auto plane_vec = glm::vec3(this->guizmo_manipulation * glm::vec4(plane_point, 1.0f));

            // Write parameter values
            param_enable->SetValue(plane_enabled);
            param_point->SetValue(plane_vec);

            // Context Menu ---------------------------------------------------
            std::string popup_label = "guizmo_popup";
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) &&
                !ImGui::IsPopupOpen(popup_label.c_str()) &&  ImGuizmo::IsOver()) {
                ImGui::OpenPopup(popup_label.c_str());
            }
            if (ImGui::BeginPopup(popup_label.c_str(), ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove)) {

                ImGui::TextDisabled("Plane Widget");
                ImGui::Separator();
                ButtonWidgets::ToggleButton("Enable", plane_enabled);
                ImGui::Separator();
                bool mode_rot = (this->guizmo_operation == ImGuizmo::ROTATE);
                if (ImGui::Checkbox("Rotate", &mode_rot)) {
                    this->guizmo_operation = ImGuizmo::ROTATE;
                }
                ImGui::SameLine();
                bool mode_trans = (this->guizmo_operation == ImGuizmo::TRANSLATE);
                if (ImGui::Checkbox("Translate", &mode_trans)) {
                    this->guizmo_operation = ImGuizmo::TRANSLATE;
                }

                ImGui::EndPopup();
            }

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
