/*
 * ImageWidget_gl.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ImageWidget_gl.h"


using namespace megamol;
using namespace megamol::gui;


ImageWidget::ImageWidget(void) : tex_ptr(nullptr), tooltip() {}


void megamol::gui::ImageWidget::Widget(ImVec2 size, ImVec2 uv0, ImVec2 uv1) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    if (!this->IsLoaded()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No texture loaded. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    ImGui::Image(reinterpret_cast<ImTextureID>(this->tex_ptr->getName()), size, uv0, uv1,
        ImVec4(1.0f, 1.0f, 1.0f, 1.0f), style.Colors[ImGuiCol_Border]);
}


bool megamol::gui::ImageWidget::Button(const std::string& tooltip, ImVec2 size) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    if (!this->IsLoaded()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No texture loaded. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    bool retval = ImGui::ImageButton(reinterpret_cast<ImTextureID>(this->tex_ptr->getName()), size, ImVec2(0.0f, 0.0f),
        ImVec2(1.0f, 1.0f), 1, style.Colors[ImGuiCol_Button], style.Colors[ImGuiCol_ButtonActive]);
    this->tooltip.ToolTip(tooltip, ImGui::GetItemID(), 1.0f, 5.0f);

    return retval;
}
