/*
 * PopUp.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Popup.h"

#include <imgui_stdlib.h>
#include <vector>


using namespace megamol::gui;


Popup::Popup(void) : tooltipTime(0.0f), tooltipId(-1) {
    // nothing to do here ...
}

void Popup::HoverToolTip(std::string text, ImGuiID id, float time_start, float time_end) {
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

void Popup::HelpMarkerToolTip(std::string text, std::string label) {
    assert(ImGui::GetCurrentContext() != nullptr);

    if (!text.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled(label.c_str());
        this->HoverToolTip(text);
    }
}

std::string Popup::InputDialogPopUp(std::string title, std::string request, bool open) {
    assert(ImGui::GetCurrentContext() != nullptr);
    std::string response;

    if (open) {
        ImGui::OpenPopup(title.c_str());
    }
    if (ImGui::BeginPopupModal(title.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Enter %s:", request.c_str());
        this->HelpMarkerToolTip("Press [Enter] to confirm.");

        // ImGui::SetKeyboardFocusHere();
        if (ImGui::InputText("", &response, ImGuiInputTextFlags_EnterReturnsTrue)) {
            ImGui::CloseCurrentPopup();
        }

        if (ImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    return response;
}
