/*
 * MinimalPopUp.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "MinimalPopUp.h"


using namespace megamol;
using namespace megamol::gui;


bool megamol::gui::MinimalPopUp::PopUp(const std::string& label_id, bool open_popup, const std::string& info_text,
    const std::string& confirm_btn_text, bool& confirmed, const std::string& abort_btn_text, bool& aborted) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    bool retval = false;

    ImGui::PushID(label_id.c_str());

    if (open_popup) {
        ImGui::OpenPopup(label_id.c_str());
        float max_width = std::max(ImGui::CalcTextSize(label_id.c_str()).x, ImGui::CalcTextSize(info_text.c_str()).x);
        max_width += (style.ItemSpacing.x * 2.0f);
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
        if (ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    ImGui::PopID();

    return retval;
}
