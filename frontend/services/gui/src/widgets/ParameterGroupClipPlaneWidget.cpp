/*
 * ParameterGroupClipPlaneWidget.cpp
 *
 * Copyright (C) 2023 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "widgets/ParameterGroupClipPlaneWidget.h"
#include "ButtonWidgets.h"
#include "graph/ParameterGroups.h"
#include "mmcore/view/CameraSerializer.h"
#include <glm/gtx/rotate_vector.hpp>


using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::utility;
using namespace megamol::gui;


megamol::gui::ParameterGroupClipPlaneWidget::ParameterGroupClipPlaneWidget()
        : AbstractParameterGroupWidget(megamol::gui::GenerateUniqueID())
        , tooltip()
        , camera_serializer()
        , guizmo_operation(ImGuizmo::TRANSLATE)
        , guizmo_draw_plane(false)
        , guizmo_draw_grid(true) {

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

            ParameterGroups::DrawGroupedParameters(
                this->name, params, in_search, in_scope, nullptr, in_override_header_state);

            this->widget_params(false);

            return true;

        } else if (in_scope == Parameter::WidgetScope::GLOBAL) {

            ImGui::PushID(static_cast<int>(this->uid));

            auto plane_enabled = std::get<bool>(param_enable->GetValue());
            auto plane_normal = std::get<glm::vec3>(param_normal->GetValue());
            plane_normal = glm::normalize(plane_normal);
            auto plane_point = std::get<glm::vec3>(param_point->GetValue());
            auto plane_dist = std::get<float>(param_dist->GetValue());
            auto plane_colour = std::get<glm::vec4>(param_colour->GetValue());

            auto cam_view = cam.getViewMatrix();
            auto cam_proj = cam.getProjectionMatrix();

            auto screen_pos = ImVec2(0.0f, 0.0f);
            auto screen_size = ImVec2(io.DisplaySize.x, io.DisplaySize.y);

            auto translate_mat = glm::translate(glm::identity<glm::mat4>(), plane_point);
            auto rotate_mat = glm::orientation(plane_normal, glm::vec3(0.0f, 1.0f, 0.0f));
            auto plane_mat = translate_mat * rotate_mat;
            auto mvp = cam_proj * cam_view * plane_mat;

            float plane_size = 2.0f;
            ImVec4 grid_color = ImVec4(plane_colour.x, plane_colour.y, plane_colour.z, plane_colour.w);
            ImVec4 plane_color = grid_color; // ImVec4(0.0f, 1.0f, 1.0f, 0.5f);

            ImGuizmo::BeginFrame();
            ImGuizmo::Enable(plane_enabled);
            ImGuizmo::SetID(0);
            ImGuizmo::SetOrthographic(false);
            ImGuizmo::SetRect(screen_pos.x, screen_pos.y, screen_size.x, screen_size.y);
            ImGuizmo::AllowAxisFlip(true);
            ImGuizmo::SetDrawlist(ImGui::GetBackgroundDrawList());

            /// Plane ---------------------------
            if (this->guizmo_draw_plane) {
                this->draw_plane(mvp, plane_size, plane_color, screen_pos, screen_size, plane_enabled);
            }
            /// Grid ----------------------------
            if (this->guizmo_draw_grid) {
                this->draw_grid(mvp, plane_size, plane_color, screen_pos, screen_size, plane_enabled);
            }
            ///ImGuizmo::DrawGrid(
            ///    glm::value_ptr(cam_view), glm::value_ptr(cam_proj), glm::value_ptr(plane_mat), plane_size);

            /// Cube inside manipulator ---------
            //ImGuizmo::DrawCubes(glm::value_ptr(cam_view), glm::value_ptr(cam_proj), glm::value_ptr(plane_mat), 1);

            /// View Cube -----------------------
            //float camDistance = 8.f;
            //float viewManipulateRight = io.DisplaySize.x;
            //float viewManipulateTop = 0;
            //ImGuizmo::ViewManipulate(glm::value_ptr(cam_view), camDistance, ImVec2(viewManipulateRight - 128, viewManipulateTop), ImVec2(128, 128), 0x10101010);

            /// Manipulator --------------------
            glm::mat4 delta_manipulation = glm::identity<glm::mat4>();
            if (ImGuizmo::Manipulate(glm::value_ptr(cam_view), glm::value_ptr(cam_proj), this->guizmo_operation,
                    ImGuizmo::LOCAL, glm::value_ptr(plane_mat), glm::value_ptr(delta_manipulation), nullptr, nullptr,
                    nullptr)) {

                switch (this->guizmo_operation) {
                case (ImGuizmo::ROTATE): {

                    param_normal->SetValue(
                        glm::normalize(glm::vec3(delta_manipulation * glm::vec4(plane_normal, 1.0f))));

                } break;
                case (ImGuizmo::TRANSLATE): {

                    param_point->SetValue(glm::vec3(delta_manipulation * glm::vec4(plane_point, 1.0f)));

                } break;
                default:
                    break;
                }
            }

            // Context Menu ---------------------------------------------------
            std::string popup_label = "guizmo_popup";
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !ImGui::IsPopupOpen(popup_label.c_str()) &&
                ImGuizmo::IsOver()) {
                ImGui::OpenPopup(popup_label.c_str());
            }
            if (ImGui::BeginPopup(popup_label.c_str(), ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove)) {
                this->widget_params(true);
                ImGui::EndPopup();
            }

            // Write parameter values
            param_enable->SetValue(plane_enabled);

            ImGui::PopID();

            return true;
        }
    }

    return false;
}


void megamol::gui::ParameterGroupClipPlaneWidget::draw_plane(
    const glm::mat4& mvp, float size, ImVec4 color, ImVec2 scree_pos, ImVec2 screen_size, bool plane_enabled) {

    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    assert(draw_list != nullptr);

    auto color_disabled = ImVec4(0.5f, 0.5f, 0.5f, 0.5f);

    auto p_1 = glm::vec4(size, 0.0f, size, 1.0f);
    auto p_2 = glm::vec4(size, 0.0f, -size, 1.0f);
    auto p_3 = glm::vec4(-size, 0.0f, size, 1.0f);
    auto p_4 = glm::vec4(-size, 0.0f, -size, 1.0f);

    auto p_1_screen = this->world_to_screen(p_1, mvp, scree_pos, screen_size);
    auto p_2_screen = this->world_to_screen(p_2, mvp, scree_pos, screen_size);
    auto p_3_screen = this->world_to_screen(p_3, mvp, scree_pos, screen_size);
    auto p_4_screen = this->world_to_screen(p_4, mvp, scree_pos, screen_size);

    draw_list->AddQuadFilled(
        p_1_screen, p_2_screen, p_4_screen, p_3_screen, ImGui::GetColorU32(plane_enabled ? color : color_disabled));
}


void megamol::gui::ParameterGroupClipPlaneWidget::draw_grid(
    const glm::mat4& mvp, float size, ImVec4 color, ImVec2 scree_pos, ImVec2 screen_size, bool plane_enabled) {

    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    assert(draw_list != nullptr);

    auto color_disabled = ImVec4(0.5f, 0.5f, 0.5f, 0.5f);

    for (float f = -size; f <= size; f += 0.5f) {
        for (int dir = 0; dir < 2; dir++) {
            auto p_a = glm::vec4((dir ? -size : f), 0.0f, (dir ? f : -size), 1.0f);
            auto p_b = glm::vec4((dir ? size : f), 0.0f, (dir ? f : size), 1.0f);

            float thickness = 1.5f;

            ImVec2 p_a_screen = this->world_to_screen(p_a, mvp, scree_pos, screen_size);
            ImVec2 p_b_screen = this->world_to_screen(p_b, mvp, scree_pos, screen_size);

            draw_list->AddLine(
                p_a_screen, p_b_screen, ImGui::GetColorU32(plane_enabled ? color : color_disabled), thickness);
        }
    }
}


ImVec2 megamol::gui::ParameterGroupClipPlaneWidget::world_to_screen(
    const glm::vec4& worldPos, const glm::mat4& mat, ImVec2 position, ImVec2 size) {

    glm::vec4 trans = mat * worldPos;
    trans *= 0.5f / trans.w;
    trans += glm::vec4(0.5f, 0.5f, 0.0f, 0.0f);
    trans.y = 1.0f - trans.y;
    trans.x *= size.x;
    trans.y *= size.y;
    trans.x += position.x;
    trans.y += position.y;

    return ImVec2(trans.x, trans.y);
}


void megamol::gui::ParameterGroupClipPlaneWidget::widget_params(bool pop_up) {

    /// ! The folowing options are not parameters of the calling module and are therefore NOT saved in the GUI stat.

    ImGui::Text("Plane Widget");
    if (!pop_up) {
        ImGui::TextDisabled("INFO: Right-Click Widget for Context Menu");
    }
    ImGui::Separator();

    auto column_flags = ImGuiTableColumnFlags_WidthStretch;
    auto table_flags = ImGuiTableColumnFlags_NoResize |
                       ImGuiTableFlags_BordersInnerV; // ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter
    if (ImGui::BeginTable("guizmo_widget_params", 2, table_flags)) {
        ImGui::TableSetupColumn("columns", column_flags);

        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        ImGui::TableNextColumn();
        ImGui::TextUnformatted("Manipulate:");
        ImGui::TableNextColumn();
        ImGui::TextUnformatted("Draw:");

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        if (ImGui::RadioButton("Rotate", (this->guizmo_operation == ImGuizmo::ROTATE))) {
            this->guizmo_operation = ImGuizmo::ROTATE;
            if (pop_up) {
                ImGui::CloseCurrentPopup();
            }
        }

        ImGui::TableNextColumn();
        if (ImGui::Checkbox("Plane", &this->guizmo_draw_plane)) {
            if (pop_up) {
                ImGui::CloseCurrentPopup();
            }
        }

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        if (ImGui::RadioButton("Translate", (this->guizmo_operation == ImGuizmo::TRANSLATE))) {
            this->guizmo_operation = ImGuizmo::TRANSLATE;
            if (pop_up) {
                ImGui::CloseCurrentPopup();
            }
        }

        ImGui::TableNextColumn();
        if (ImGui::Checkbox("Grid", &this->guizmo_draw_grid)) {
            if (pop_up) {
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndTable();
    }
}
