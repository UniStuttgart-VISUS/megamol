/*
 * ButtonWidgets.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ButtonWidgets.h"

#include "graph/Parameter.h"


using namespace megamol;
using namespace megamol::gui;


bool megamol::gui::ButtonWidgets::OptionButton(const std::string& id, const std::string& label, bool dirty) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    bool retval = false;
    std::string widget_name("option_button");
    std::string widget_id = widget_name + id;
    ImGui::PushID(widget_id.c_str());

    float button_size = ImGui::GetFrameHeight();
    float half_button_size = button_size / 2.0f;
    ImVec2 widget_start_pos = ImGui::GetCursorScreenPos();

    if (!label.empty()) {
        float text_x_offset_pos = button_size + style.ItemInnerSpacing.x;
        ImGui::SetCursorScreenPos(widget_start_pos + ImVec2(text_x_offset_pos, 0.0f));
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label.c_str());
        ImGui::SetCursorScreenPos(widget_start_pos);
    }

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_FrameBg]));
    auto child_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove;
    ImGui::BeginChild("special_button_background", ImVec2(button_size, button_size), false, child_flags);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    float thickness = button_size / 5.0f;
    ImVec2 center = widget_start_pos + ImVec2(half_button_size, half_button_size);
    ImU32 color_front = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonActive]);
    if (dirty) {
        color_front = ImGui::ColorConvertFloat4ToU32(GUI_COLOR_BUTTON_MODIFIED);
    }
    draw_list->AddCircleFilled(center, thickness, color_front, 12);
    draw_list->AddCircle(center, 2.0f * thickness, color_front, 12, (thickness / 2.0f));

    ImVec2 rect = ImVec2(button_size, button_size);
    retval = ImGui::InvisibleButton("special_button", rect);

    ImGui::EndChild();
    ImGui::PopStyleColor();

    ImGui::PopID();

    return retval;
}


bool megamol::gui::ButtonWidgets::KnobButton(
    const std::string& id, float size, float& inout_value, float minval, float maxval) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    bool retval = false;

    const float pi = 3.14159265358f;

    std::string widget_name("knob_widget_background");
    std::string widget_id = widget_name + id;
    ImGui::PushID(widget_id.c_str());

    ImVec2 widget_start_pos = ImGui::GetCursorScreenPos();

    const float thickness = size / 15.0f;
    const float knob_radius = thickness * 2.0f;

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_FrameBg]));
    auto child_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove;
    ImGui::BeginChild("knob_widget_background", ImVec2(size, size), false, child_flags);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    // Draw Outline
    float half_knob_size = size / 2.0f;

    ImVec2 widget_center = widget_start_pos + ImVec2(half_knob_size, half_knob_size);
    float half_thickness = (thickness / 2.0f);
    ImU32 outline_color_front = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonHovered]);
    ImU32 outline_color_shadow = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_FrameBgHovered]);

    // Shadow
    draw_list->AddCircle(widget_center, half_knob_size - thickness, outline_color_shadow, 24, thickness);
    // Outline
    draw_list->AddCircle(widget_center, half_knob_size - half_thickness, outline_color_front, 24, half_thickness);

    // Draw knob
    ImU32 knob_line_color = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Button]);
    ImU32 knob_color = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonHovered]);
    ImVec2 rect = ImVec2(knob_radius * 2.0f, knob_radius * 2.0f);
    ImVec2 half_rect = ImVec2(knob_radius, knob_radius);
    float knob_center_dist = (half_knob_size - knob_radius - half_thickness);

    // Adapt scaling of one round depending on min max delta
    float scaling = 1.0f;
    if ((minval > -FLT_MAX) && (maxval < FLT_MAX) && (maxval > minval)) {
        float delta = maxval - minval;
        scaling = delta / 100.0f; // 360 degree = 1%
    }

    // Calculate knob position
    ImVec2 knob_pos = ImVec2(0.0f, -(knob_center_dist));
    float tmp_value = inout_value / scaling;
    float angle = (tmp_value - floor(tmp_value)) * pi * 2.0f;
    float cos_angle = cosf(angle);
    float sin_angle = sinf(angle);
    knob_pos =
        ImVec2((cos_angle * knob_pos.x - sin_angle * knob_pos.y), (sin_angle * knob_pos.x + cos_angle * knob_pos.y));

    ImVec2 knob_button_pos = widget_center + knob_pos - half_rect;
    ImGui::SetCursorScreenPos(knob_button_pos);
    ImGui::InvisibleButton("special_button", rect);

    if (ImGui::IsItemActive()) {
        knob_color = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonActive]);
        ImVec2 p1 = knob_pos;
        float d1 = sqrtf((p1.x * p1.x) + (p1.y * p1.y));
        p1 /= d1;
        ImVec2 p2 = ImGui::GetMousePos() - widget_center;
        float d2 = sqrtf((p2.x * p2.x) + (p2.y * p2.y));
        p2 /= d2;
        float dot = (p1.x * p2.x) + (p1.y * p2.y); // dot product
        float det = (p1.x * p2.y) - (p1.y * p2.x); // determinant
        float angle = atan2(det, dot);
        float b = angle / (2.0f * pi); // b in [0,1] for [0,360] degree
        b *= scaling;
        knob_pos = (p2 * knob_center_dist);
        inout_value = std::min(maxval, (std::max(minval, inout_value + b)));
        retval = true;
    }
    draw_list->AddLine(widget_center, widget_center + knob_pos, knob_line_color, thickness);
    draw_list->AddCircleFilled(widget_center + knob_pos, knob_radius, knob_color, 12);

    ImGui::EndChild();
    ImGui::PopStyleColor();

    ImGui::PopID();

    return retval;
}


bool megamol::gui::ButtonWidgets::ExtendedModeButton(const std::string& id, bool& inout_extended_mode) {

    assert(ImGui::GetCurrentContext() != nullptr);

    bool retval = false;

    std::string widget_name("param_extend_button");
    std::string widget_id = widget_name + id;
    ImGui::PushID(widget_id.c_str());

    ImGui::BeginGroup();

    megamol::gui::ButtonWidgets::OptionButton("param_mode_button", "Mode");
    if (ImGui::BeginPopupContextItem("param_mode_button_context", 0)) { // 0 = left mouse button
        if (ImGui::MenuItem("Basic###param_mode_button_basic_mode", nullptr, !inout_extended_mode, true)) {
            inout_extended_mode = false;
            retval = true;
        }
        if (ImGui::MenuItem("Expert###param_mode_button_extended_mode", nullptr, inout_extended_mode, true)) {
            inout_extended_mode = true;
            retval = true;
        }
        ImGui::EndPopup();
    }
    ImGui::EndGroup();

    ImGui::PopID();

    return retval;
}


bool megamol::gui::ButtonWidgets::LuaButton(const std::string& id, const megamol::gui::Parameter& param,
    const std::string& param_fullname, const std::string& module_fullname) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    bool retval = false;

    std::string widget_name("lua_button");
    std::string widget_id = widget_name + id;
    ImGui::PushID(widget_id.c_str());

    float button_size = ImGui::GetFrameHeight();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_FrameBg]));
    auto child_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove;
    ImGui::BeginChild("lua_button_background", ImVec2(button_size, button_size), false, child_flags);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    const ImU32 COLOR_TEXT = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonHovered]);
    ImVec2 button_start_pos = ImGui::GetCursorScreenPos();
    ImVec2 button_middle = button_start_pos + ImVec2(button_size / 2.0f, button_size / 2.0f);
    const std::string button_label = "lua";
    ImVec2 text_size = ImGui::CalcTextSize(button_label.c_str());
    ImVec2 text_pos_left_upper = button_middle - ImVec2(text_size.x / 2.0f, text_size.y / 2.0f);
    draw_list->AddText(text_pos_left_upper, COLOR_TEXT, button_label.c_str());

    ImVec2 rect = ImVec2(button_size, button_size);
    retval = ImGui::InvisibleButton("lua_invisible_button", rect);

    ImGui::EndChild();
    ImGui::PopStyleColor();

    if (ImGui::BeginPopupContextItem("param_lua_button_context", 0)) {
        bool copy_to_clipboard = false;
        std::string lua_param_cmd;
        std::string mod_name(module_fullname.c_str()); /// local copy required
        if (ImGui::MenuItem("Copy mmSetParamValue")) {
            lua_param_cmd =
                "mmSetParamValue(\"" + mod_name + "::" + param_fullname + "\",[=[" + param.GetValueString() + "]=])";
            copy_to_clipboard = true;
        }
        if (ImGui::MenuItem("Copy mmGetParamValue")) {
            lua_param_cmd = "mmGetParamValue(\"" + mod_name + "::" + param_fullname + "\")";
            copy_to_clipboard = true;
        }

        if (copy_to_clipboard) {
            ImGui::SetClipboardText(lua_param_cmd.c_str());
        }
        ImGui::EndPopup();
    }

    ImGui::PopID();

    return retval;
}
