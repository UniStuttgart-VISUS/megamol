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
