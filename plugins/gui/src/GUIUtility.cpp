/*
 * GUIUtility.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "GUIUtility.h"


using namespace megamol::gui;


/**
 * Ctor
 */
megamol::gui::GUIUtility::GUIUtility(void) {

    // nothing to do here ...
}


/**
 * Dtor
 */
megamol::gui::GUIUtility::~GUIUtility(void) {

    // nothing to do here ...
}


/**
 * GUIUtility::ResetWindowSizePos
 */
void megamol::gui::GUIUtility::ResetWindowSizePos(std::string win_label, float min_height) {

    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    auto win_pos = ImGui::GetWindowPos();
    if (win_pos.x < 0) {
        win_pos.x = style.DisplayWindowPadding.x;
    }
    if (win_pos.y < 0) {
        win_pos.y = style.DisplayWindowPadding.y;
    }

    auto win_width = 0.0f; // width = 0 means auto resize
    auto win_height = io.DisplaySize.y - (win_pos.y + style.DisplayWindowPadding.y);
    if (win_height < min_height) {
        win_height = min_height;
        win_pos.y = io.DisplaySize.y - (min_height + style.DisplayWindowPadding.y);
    }

    ImGui::SetWindowSize(win_label.c_str(), ImVec2(win_width, win_height), ImGuiCond_Always);

    ImGui::SetWindowPos(win_label.c_str(), win_pos, ImGuiCond_Always);
}


/**
 * GUIUtility::HoverToolTip
 */
void megamol::gui::GUIUtility::HoverToolTip(std::string text, ImGuiID id, float time_start, float time_end) {

    ImGuiIO& io = ImGui::GetIO();

    if (ImGui::IsItemHovered()) {
        bool show_tooltip = false;
        if (time_start > 0.0f) {
            if (this->tooltip_id != id) {
                this->tooltip_time = 0.0f;
                this->tooltip_id = id;
            } else {
                if ((this->tooltip_time > time_start) && (this->tooltip_time < (time_start + time_end))) {
                    show_tooltip = true;
                }
                this->tooltip_time += io.DeltaTime;
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
        if ((time_start > 0.0f) && (this->tooltip_id == id)) {
            this->tooltip_time = 0.0f;
        }
    }
}


/**
 * GUIUtility::HelpMarkerToolTip
 */
void megamol::gui::GUIUtility::HelpMarkerToolTip(std::string text, std::string label) {

    if (!text.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled(label.c_str());
        this->HoverToolTip(text);
    }
}
