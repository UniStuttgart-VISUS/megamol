/*
 * StringSearchWidget.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "StringSearchWidget.h"


using namespace megamol;
using namespace megamol::gui;


StringSearchWidget::StringSearchWidget(void) : search_focus(false), search_string(), tooltip() {}


bool megamol::gui::StringSearchWidget::Widget(const std::string& id, const std::string& help) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    bool retval = false;

    ImGui::BeginGroup();
    ImGui::PushID(id.c_str());

    if (ImGui::Button("Clear")) {
        this->search_string = "";
    }
    ImGui::SameLine();

    // Set keyboard focus when hotkey is pressed
    if (this->search_focus) {
        ImGui::SetKeyboardFocusHere();
        this->search_string = "";
        this->search_focus = false;
    }

    std::string complete_label("Search (?)");
    auto width = ImGui::GetContentRegionAvail().x - ImGui::GetCursorPosX() + 4.0f * style.ItemInnerSpacing.x -
                 ImGui::CalcTextSize(complete_label.c_str()).x;
    const float min_width = 50.0f;
    width = (width < min_width) ? (min_width) : width;
    ImGui::PushItemWidth(width);
    /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
    GUIUtils::Utf8Encode(this->search_string);
    ImGui::InputText("Search", &this->search_string, ImGuiInputTextFlags_AutoSelectAll);
    GUIUtils::Utf8Decode(this->search_string);
    if (ImGui::IsItemActive()) {
        retval = true;
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();

    this->tooltip.Marker(help);

    ImGui::PopID();
    ImGui::EndGroup();

    return retval;
}
