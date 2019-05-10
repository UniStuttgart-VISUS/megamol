/*
 * WidgetUtils.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Popup.h"

using namespace megamol::gui;

Popup::Popup(void) : tooltip_time(0.0f), tooltip_id(-1) {
    // nothing to do here ...
}

void Popup::HoverToolTip(std::string text, ImGuiID id, float time_start, float time_end) {
    assert(ImGui::GetCurrentContext() != nullptr);
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

void Popup::HelpMarkerToolTip(std::string text, std::string label) {
    assert(ImGui::GetCurrentContext() != nullptr);

    if (!text.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled(label.c_str());
        this->HoverToolTip(text);
    }
}

std::string Popup::InputDialogPopUp(std::string popup_name, std::string request, bool open) {
    assert(ImGui::GetCurrentContext() != nullptr);

    std::string outtext;

    char* buffer = new char[GUI_MAX_BUFFER_LEN];
    buffer[0] = '\0';

    if (open) {
        ImGui::OpenPopup(popup_name.c_str());
    }
    if (ImGui::BeginPopupModal(popup_name.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {

        ImGui::Text("Enter %s:", request.c_str());
        this->HelpMarkerToolTip("Press [Enter] to confirm input.");

        // ImGui::SetKeyboardFocusHere();
        if (ImGui::InputText("", buffer, GUI_MAX_BUFFER_LEN, ImGuiInputTextFlags_EnterReturnsTrue)) {
            outtext = buffer;
            ImGui::CloseCurrentPopup();
        }

        if (ImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    delete[] buffer;

    return outtext;
}
