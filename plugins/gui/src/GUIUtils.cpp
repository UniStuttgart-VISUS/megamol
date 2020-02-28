/*
 * PopUp.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "GUIUtils.h"

#include "vislib/UTF8Encoder.h"

#include <imgui_stdlib.h>
#include <vector>


using namespace megamol::gui;


GUIUtils::GUIUtils(void) : tooltipTime(0.0f), tooltipId(GUI_INVALID_ID), searchFocus(false), searchString() {}


void GUIUtils::HoverToolTip(const std::string& text, ImGuiID id, float time_start, float time_end) {
    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiIO& io = ImGui::GetIO();

    if (ImGui::IsItemHovered()) {
        bool show_tooltip = false;
        if (time_start > 0.0f) {
            if (this->tooltipId != id) {
                this->tooltipTime = 0.0f;
                this->tooltipId = id;
            } else {
                if ((this->tooltipTime > time_start) && (this->tooltipTime < (time_start + time_end))) {
                    show_tooltip = true;
                }
                this->tooltipTime += io.DeltaTime;
            }
        } else {
            show_tooltip = true;
        }

        if (show_tooltip) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(text.c_str());
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    } else {
        if ((time_start > 0.0f) && (this->tooltipId == id)) {
            this->tooltipTime = 0.0f;
        }
    }
}


void GUIUtils::HelpMarkerToolTip(const std::string& text, std::string label) {
    assert(ImGui::GetCurrentContext() != nullptr);

    if (!text.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled(label.c_str());
        this->HoverToolTip(text);
    }
}


float GUIUtils::TextWidgetWidth(const std::string& text) const {
    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    ImVec2 pos = ImGui::GetCursorPos();
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.0f);
    ImGui::Text(text.c_str());
    ImGui::PopStyleVar();
    ImGui::SetCursorPos(pos);

    return ImGui::GetItemRectSize().x;
}


bool GUIUtils::Utf8Decode(std::string& str) const {
    vislib::StringA dec_tmp;
    if (vislib::UTF8Encoder::Decode(dec_tmp, vislib::StringA(str.c_str()))) {
        str = std::string(dec_tmp.PeekBuffer());
        return true;
    }
    return false;
}


bool GUIUtils::Utf8Encode(std::string& str) const {
    vislib::StringA dec_tmp;
    if (vislib::UTF8Encoder::Encode(dec_tmp, vislib::StringA(str.c_str()))) {
        str = std::string(dec_tmp.PeekBuffer());
        return true;
    }
    return false;
}

void megamol::gui::GUIUtils::StringSearch(const std::string& id, const std::string& help) {
    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    ImGui::PushID(id.c_str());

    std::string help_label = "(?)";

    if (ImGui::Button("Clear")) {
        this->searchString = "";
    }
    ImGui::SameLine();

    // Set keyboard focus when hotkey is pressed
    if (this->searchFocus) {
        ImGui::SetKeyboardFocusHere();
        this->searchFocus = false;
    }

    std::string label = "Search";
    auto width = ImGui::GetContentRegionAvailWidth() - ImGui::GetCursorPosX() + 4.0f * style.ItemInnerSpacing.x -
                 this->TextWidgetWidth(label + help_label);
    const int min_width = 50.0f;
    width = (width < min_width) ? (min_width) : width;
    ImGui::PushItemWidth(width);

    /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
    this->Utf8Encode(this->searchString);
    ImGui::InputText("###Search Parameters", &this->searchString, ImGuiInputTextFlags_AutoSelectAll);
    this->Utf8Decode(this->searchString);

    ImGui::PopItemWidth();

    ImGui::SameLine();
    ImGui::Text(label.c_str());
    this->HelpMarkerToolTip(help, help_label);

    ImGui::PopID();
}


bool megamol::gui::GUIUtils::VerticalSplitter(float* size_left, float* size_right) {
    assert(ImGui::GetCurrentContext() != nullptr);

    const float thickness = 12.0f;

    bool split_vertically = true;
    float min_size = 1.0f; // >=1.0!
    float splitter_long_axis_size = ImGui::GetContentRegionAvail().y;

    float width_avail = ImGui::GetWindowSize().x - (3.0f * thickness);

    if (width_avail < thickness) return false;

    (*size_left) = std::min((*size_left), width_avail);
    (*size_right) = width_avail - (*size_left);

    ImGuiContext& g = *GImGui;
    ImGuiWindow* window = g.CurrentWindow;
    ImGuiID id = window->GetID("##Splitter");
    ImRect bb;
    bb.Min = window->DC.CursorPos +
             (split_vertically ? ImVec2((*size_left) + 1.0f, 0.0f) : ImVec2(0.0f, (*size_left) + 1.0f));
    bb.Max = bb.Min + ImGui::CalcItemSize(split_vertically ? ImVec2(thickness - 4.0f, splitter_long_axis_size)
                                                           : ImVec2(splitter_long_axis_size, thickness - 4.0f),
                          0.0f, 0.0f);
    return ImGui::SplitterBehavior(
        bb, id, split_vertically ? ImGuiAxis_X : ImGuiAxis_Y, size_left, size_right, min_size, min_size, 0.0f);
}
