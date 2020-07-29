/*
 * RenamePopUp.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "RenamePopUp.h"


using namespace megamol;
using namespace megamol::gui;


RenamePopUp::RenamePopUp(void) : rename_string() {}


bool megamol::gui::RenamePopUp::PopUp(const std::string& label_id, bool open_popup, std::string& rename) {

    assert(ImGui::GetCurrentContext() != nullptr);

    bool retval = false;

    ImGui::PushID(label_id.c_str());

    if (open_popup) {
        this->rename_string = rename;
        ImGui::OpenPopup(label_id.c_str());
    }
    if (ImGui::BeginPopupModal(label_id.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {

        std::string text_label("New Name");
        auto flags = ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll;
        if (ImGui::InputText(text_label.c_str(), &this->rename_string, flags)) {
            rename = this->rename_string;
            retval = true;
            ImGui::CloseCurrentPopup();
        }
        // Set focus on input text once (applied next frame)
        if (open_popup) {
            ImGuiID id = ImGui::GetID(text_label.c_str());
            ImGui::ActivateItem(id);
        }

        if (ImGui::Button("OK")) {
            rename = this->rename_string;
            retval = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    ImGui::PopID();

    return retval;
}
