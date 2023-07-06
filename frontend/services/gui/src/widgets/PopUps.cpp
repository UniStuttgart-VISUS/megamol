/*
 * PopUps.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "PopUps.h"
#include "imgui_stdlib.h"


using namespace megamol;
using namespace megamol::gui;


PopUps::PopUps() : rename_string(), rename_tooltip() {}


bool megamol::gui::PopUps::Rename(const std::string& label_id, bool open_popup, std::string& rename) {

    assert(ImGui::GetCurrentContext() != nullptr);

    bool retval = false;

    ImGui::PushID(label_id.c_str());

    if (open_popup) {
        this->rename_string = rename;
        ImGui::OpenPopup(label_id.c_str());
    }
    if (ImGui::BeginPopupModal(label_id.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {

        bool confirmed_edit = false;
        std::string text_label("New Name");
        auto flags = ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll;
        // Set focus on input text once
        if (open_popup) {
            ImGui::SetKeyboardFocusHere(0);
        }
        if (ImGui::InputText(text_label.c_str(), &this->rename_string, flags)) {
            confirmed_edit = true;
        }
        this->rename_tooltip.Marker("Leading '::' is forbidden.");

        if (ImGui::Button("OK") || confirmed_edit) {
            // Remove forbidden leading "::"
            if (this->rename_string.find_first_of("::") == 0) {
                this->rename_string = this->rename_string.substr(2);
            }
            rename = this->rename_string;
            retval = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    ImGui::PopID();

    return retval;
}


bool megamol::gui::PopUps::Minimal(const std::string& label_id, bool open_popup, const std::string& info_text,
    const std::string& confirm_btn_text, bool& confirmed, const std::string& abort_btn_text, bool& aborted) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    bool retval = false;

    ImGui::PushID(label_id.c_str());

    if (open_popup && !ImGui::IsPopupOpen(label_id.c_str())) {
        ImGui::OpenPopup(label_id.c_str());
        float max_width = std::max(ImGui::CalcTextSize(label_id.c_str()).x, ImGui::CalcTextSize(info_text.c_str()).x);
        max_width += (style.ItemSpacing.x * 4.0f);
        ImGui::SetNextWindowSize(ImVec2(max_width, 0.0f));
    }
    if (ImGui::BeginPopupModal(label_id.c_str(), nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove)) {
        retval = true;

        if (!info_text.empty()) {
            ImGui::TextUnformatted(info_text.c_str());
        }

        if (!confirm_btn_text.empty()) {
            if (ImGui::Button(confirm_btn_text.c_str())) {
                confirmed = true;
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
        }
        if (!abort_btn_text.empty()) {
            if (ImGui::Button(abort_btn_text.c_str())) {
                aborted = true;
                ImGui::CloseCurrentPopup();
            }
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    ImGui::PopID();

    return retval;
}
