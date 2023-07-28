/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "HoverToolTip.h"


using namespace megamol;
using namespace megamol::gui;


HoverToolTip::HoverToolTip() : tooltip_time(0.0f), tooltip_id(GUI_INVALID_ID) {}


bool HoverToolTip::ToolTip(const std::string& text, ImGuiID id, float time_start, float time_end) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiIO& io = ImGui::GetIO();

    bool retval = false;

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
            retval = true;
        }
    } else {
        if ((time_start > 0.0f) && (this->tooltip_id == id)) {
            this->tooltip_time = 0.0f;
        }
    }

    return retval;
}


bool HoverToolTip::Marker(const std::string& text, const std::string& label) {

    assert(ImGui::GetCurrentContext() != nullptr);

    if (!text.empty()) {
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled(label.c_str());
        return this->ToolTip(text);
    }
    return false;
}


void HoverToolTip::Reset() {
    this->tooltip_time = 0.0f;
    this->tooltip_id = GUI_INVALID_ID;
}
