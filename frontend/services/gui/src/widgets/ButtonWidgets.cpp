/*
 * ButtonWidgets.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "ButtonWidgets.h"

#include <imgui.h>

#include "graph/Parameter.h"

using namespace megamol;
using namespace megamol::gui;


bool megamol::gui::ButtonWidgets::OptionButton(
    ButtonStyle button_style, const std::string& id, const std::string& label, bool dirty, bool read_only) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    std::string widget_name("option_button");
    std::string widget_id = widget_name + id;
    ImGui::PushID(widget_id.c_str());
    ImGui::BeginGroup();

    float button_size = ImGui::GetFrameHeight();
    float half_button_size = button_size / 2.0f;
    ImVec2 button_start_pos = ImGui::GetCursorScreenPos();
    ImVec2 rect = ImVec2(button_size, button_size);

    if (!label.empty()) {
        float text_x_offset_pos = button_size + style.ItemInnerSpacing.x;
        ImGui::SetCursorScreenPos(button_start_pos + ImVec2(text_x_offset_pos, 0.0f));
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label.c_str());
        ImGui::SetCursorScreenPos(button_start_pos);
    }

    auto child_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove;
    ImGui::BeginChild("option_button", rect, false, child_flags);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    bool retval = (!read_only && ImGui::InvisibleButton("special_button", rect));

    ImVec4 color_back = style.Colors[ImGuiCol_Button];
    ImVec4 color_front = style.Colors[ImGuiCol_ButtonActive];
    if (dirty) {
        color_front = GUI_COLOR_BUTTON_MODIFIED;
    }
    if (!read_only) {
        if (ImGui::IsItemHovered()) {
            color_back = style.Colors[ImGuiCol_ButtonHovered];
        }
        if (ImGui::IsItemActive()) {
            color_back = style.Colors[ImGuiCol_ButtonActive];
            color_front = color_front * ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
        }
    } else {
        color_back.w *= 0.5f;
        if (!dirty) {
            color_front.w *= 0.5f;
        }
    }

    const float thickness = button_size / 5.0f;
    const float half_thickness = thickness / 2.0f;

    ImVec2 center = button_start_pos + ImVec2(half_button_size, half_button_size);
    draw_list->AddRectFilled(button_start_pos, button_start_pos + rect, ImGui::ColorConvertFloat4ToU32(color_back));

    switch (button_style) {
    case (ButtonStyle::POINT_CIRCLE): {
        draw_list->AddCircleFilled(center, thickness, ImGui::ColorConvertFloat4ToU32(color_front));
        draw_list->AddCircle(center, 2.0f * thickness, ImGui::ColorConvertFloat4ToU32(color_front), 12, half_thickness);
    } break;
    case (ButtonStyle::GRID): {
        const float line_delta = button_size / 3.0f;
        draw_list->AddLine(button_start_pos + ImVec2(line_delta, half_thickness),
            button_start_pos + ImVec2(line_delta, button_size - half_thickness),
            ImGui::ColorConvertFloat4ToU32(color_front), half_thickness);
        draw_list->AddLine(button_start_pos + ImVec2(2.0f * line_delta, half_thickness),
            button_start_pos + ImVec2(2.0f * line_delta, button_size - half_thickness),
            ImGui::ColorConvertFloat4ToU32(color_front), half_thickness);
        draw_list->AddLine(button_start_pos + ImVec2(half_thickness, line_delta),
            button_start_pos + ImVec2(button_size - half_thickness, line_delta),
            ImGui::ColorConvertFloat4ToU32(color_front), half_thickness);
        draw_list->AddLine(button_start_pos + ImVec2(half_thickness, 2.0f * line_delta),
            button_start_pos + ImVec2(button_size - half_thickness, 2.0f * line_delta),
            ImGui::ColorConvertFloat4ToU32(color_front), half_thickness);
    } break;
    case (ButtonStyle::LINES): {
        const float line_delta = button_size / 4.0f;
        draw_list->AddLine(button_start_pos + ImVec2(thickness, 1.0f * line_delta),
            button_start_pos + ImVec2(button_size - thickness, 1.0f * line_delta),
            ImGui::ColorConvertFloat4ToU32(color_front), half_thickness);
        draw_list->AddLine(button_start_pos + ImVec2(thickness, 2.0f * line_delta),
            button_start_pos + ImVec2(button_size - thickness, 2.0f * line_delta),
            ImGui::ColorConvertFloat4ToU32(color_front), half_thickness);
        draw_list->AddLine(button_start_pos + ImVec2(thickness, 3.0f * line_delta),
            button_start_pos + ImVec2(button_size - thickness, 3.0f * line_delta),
            ImGui::ColorConvertFloat4ToU32(color_front), half_thickness);

    } break;
    case (ButtonStyle::POINTS): {
        draw_list->AddCircleFilled(center + ImVec2(-thickness, thickness), thickness,
            ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_SeparatorActive]));
        draw_list->AddCircleFilled(center + ImVec2(half_thickness, half_thickness), thickness,
            ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ScrollbarGrabHovered]));
        draw_list->AddCircleFilled(center + ImVec2(-half_thickness, -thickness), thickness,
            ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonActive]));
    } break;
    }

    ImGui::EndChild();

    ImGui::EndGroup();
    ImGui::PopID();

    return retval;
}


bool megamol::gui::ButtonWidgets::KnobButton(
    const std::string& id, float size, float& inout_value, float minval, float maxval, float step) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    bool retval = false;

    const float pi = 3.14159265358f;

    std::string widget_name("knob_widget_background");
    std::string widget_id = widget_name + id;
    ImGui::PushID(widget_id.c_str());

    ImVec2 button_start_pos = ImGui::GetCursorScreenPos();

    const float thickness = size / 15.0f;
    const float knob_radius = thickness * 2.0f;

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_FrameBg]));
    auto child_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove;
    ImGui::BeginChild("knob_widget_background", ImVec2(size, size), false, child_flags);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    // Draw Outline
    float half_knob_size = size / 2.0f;

    ImVec2 widget_center = button_start_pos + ImVec2(half_knob_size, half_knob_size);
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
    if (step == FLT_MAX) {
        if ((minval > -FLT_MAX) && (maxval < FLT_MAX) && (maxval > minval)) {
            float delta = maxval - minval;
            scaling = delta / 100.0f; // 360 degree = 1%
        }
    } else {
        scaling = step;
    }

    // Calculate knob position
    ImVec2 knob_pos = ImVec2(0.0f, -(knob_center_dist));
    float tmp_value = inout_value / scaling;
    float knob_angle = (tmp_value - floor(tmp_value)) * pi * 2.0f;
    float cos_angle = cosf(knob_angle);
    float sin_angle = sinf(knob_angle);
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
    draw_list->AddCircleFilled(widget_center + knob_pos, knob_radius, knob_color);

    ImGui::EndChild();
    ImGui::PopStyleColor();

    ImGui::PopID();

    return retval;
}


bool megamol::gui::ButtonWidgets::ExtendedModeButton(const std::string& id, bool& inout_extended_mode) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    std::string widget_name("param_extend_button");
    std::string widget_id = widget_name + id;
    ImGui::PushID(widget_id.c_str());

    bool retval = false;
    megamol::gui::ButtonWidgets::OptionButton(
        ButtonStyle::POINT_CIRCLE, "param_mode_button", "", inout_extended_mode, false);
    if (ImGui::BeginPopupContextItem("param_mode_button_context", ImGuiPopupFlags_MouseButtonLeft)) {
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
    ImGui::SameLine(ImGui::GetFrameHeight() + 1.5f * style.ItemSpacing.x);
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Mode");

    ImGui::PopID();

    return retval;
}


bool megamol::gui::ButtonWidgets::LuaButton(
    const std::string& id, const megamol::gui::Parameter& param, const std::string& param_fullname) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    std::string widget_name("lua_button");
    std::string widget_id = widget_name + id;
    ImGui::PushID(widget_id.c_str());

    ImVec2 button_start_pos = ImGui::GetCursorScreenPos();
    float button_size = ImGui::GetFrameHeight();
    ImVec2 rect = ImVec2(button_size, button_size);

    auto child_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove;
    ImGui::BeginChild("lua_button_background", rect, false, child_flags);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    bool retval = ImGui::InvisibleButton("lua_invisible_button", rect);

    ImU32 color_back = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Button]);
    ImU32 color_text = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonActive]);
    if (ImGui::IsItemHovered()) {
        color_back = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonHovered]);
        color_text = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ChildBg]);
    }
    if (ImGui::IsItemActive()) {
        color_back = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonActive]);
        color_text = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_TextDisabled]);
    }
    if (ImGui::BeginPopupContextItem("param_lua_button_context", ImGuiPopupFlags_MouseButtonLeft)) {
        bool copy_to_clipboard = false;
        std::string lua_param_cmd;
        if (ImGui::MenuItem("Copy mmSetParamValue")) {
            lua_param_cmd = "mmSetParamValue(\"" + param_fullname + "\",[=[" + param.GetValueString() + "]=])";
            copy_to_clipboard = true;
        }
        if (ImGui::MenuItem("Copy mmGetParamValue")) {
            lua_param_cmd = "mmGetParamValue(\"" + param_fullname + "\")";
            copy_to_clipboard = true;
        }

        if (copy_to_clipboard) {
            ImGui::SetClipboardText(lua_param_cmd.c_str());
        }
        ImGui::EndPopup();
    }

    ImVec2 button_middle = button_start_pos + ImVec2(button_size / 2.0f, button_size / 2.0f);
    ImVec2 text_size = ImGui::CalcTextSize("lua");
    ImVec2 text_pos_left_upper = button_middle - ImVec2(text_size.x / 2.0f, text_size.y / 2.0f);
    draw_list->AddRectFilled(button_start_pos, button_start_pos + rect, color_back);
    draw_list->AddText(text_pos_left_upper, color_text, "lua");

    ImGui::EndChild();

    ImGui::PopID();

    return retval;
}


bool megamol::gui::ButtonWidgets::ToggleButton(const std::string& id, bool& inout_bool) {

    bool retval = false;

    ImGuiStyle& style = ImGui::GetStyle();

    ImVec4* colors = ImGui::GetStyle().Colors;
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    float height = ImGui::GetFrameHeight();
    float width = height * 1.55f;
    float radius = height * 0.50f;

    ImGui::BeginGroup();

    ImGuiContext& g = *GImGui;

    std::string button_id = "toggle_button_" + id;
    ImGui::InvisibleButton(button_id.c_str(), ImVec2(width, height));
    if (ImGui::IsItemClicked()) {
        inout_bool = !inout_bool;
        retval = true;
    }
    if (ImGui::IsItemActivated()) {
        g.LastActiveId = ImGui::GetID(button_id.c_str());
        g.LastActiveIdTimer = 0.0f;
    }

    // Animate button changes
    float t = inout_bool ? 1.0f : 0.0f;
    float ANIM_SPEED = 0.25f;
    if ((g.LastActiveId == ImGui::GetID(button_id.c_str())) && (g.LastActiveIdTimer < ANIM_SPEED)) {
        float t_anim = ImSaturate(g.LastActiveIdTimer / ANIM_SPEED);
        t = inout_bool ? (t_anim) : (1.0f - t_anim);
    }

    ImVec4 col_knob = colors[ImGuiCol_Text];
    ImVec4 col_btn_hover_false = colors[ImGuiCol_Button];      // ImVec4(0.78f, 0.78f, 0.78f, 1.0f);
    ImVec4 col_btn_hover_true = colors[ImGuiCol_ButtonActive]; // ImVec4(0.64f, 0.83f, 0.34f, 1.0f);
    ImVec4 col_btn_false = colors[ImGuiCol_TextDisabled];      // ImVec4(0.85f, 0.85f, 0.85f, 1.0f);
    ImVec4 col_btn_true = colors[ImGuiCol_ButtonHovered];      // ImVec4(0.56f, 0.83f, 0.26f, 1.0f);
    ImU32 col_bg;
    if (ImGui::IsItemHovered()) {
        col_bg = ImGui::GetColorU32(ImLerp(col_btn_hover_false, col_btn_hover_true, t));
    } else {
        col_bg = ImGui::GetColorU32(ImLerp(col_btn_false, col_btn_true, t));
    }

    draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height), col_bg, height * 0.5f);
    draw_list->AddCircleFilled(
        ImVec2(p.x + radius + t * (width - radius * 2.0f), p.y + radius), radius - 1.5f, ImGui::GetColorU32(col_knob));

    if (!id.empty() && (id.find("###") != 0)) {
        ImGui::SameLine();
        ImGui::SetCursorScreenPos(ImGui::GetCursorScreenPos() + ImVec2(-style.ItemInnerSpacing.x, 0.0f));
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(id.c_str());
    }

    ImGui::EndGroup();

    return retval;
}
