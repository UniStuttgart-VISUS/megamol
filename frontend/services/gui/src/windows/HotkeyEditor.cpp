/*
 * HotkeyEditor.cpp
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "HotkeyEditor.h"


using namespace megamol::gui;


megamol::gui::HotkeyEditor::HotkeyEditor(const std::string& window_name)
        : AbstractWindow(window_name, AbstractWindow::WINDOW_ID_HOTKEYEDITOR) {

    // Configure HOTKEY EDITOR Window
    this->win_config.size = ImVec2(100.0f * megamol::gui::gui_scaling.Get(), 200.0f * megamol::gui::gui_scaling.Get());
    this->win_config.reset_size = this->win_config.size;
    this->win_config.flags = ImGuiWindowFlags_None;
    this->win_config.hotkey =
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F6, core::view::Modifier::NONE);
}


HotkeyEditor::~HotkeyEditor() {

}


bool megamol::gui::HotkeyEditor::Update() {

    return true;
}


bool megamol::gui::HotkeyEditor::Draw() {

    return true;
}


void HotkeyEditor::SpecificStateFromJSON(const nlohmann::json& in_json) {

}


void HotkeyEditor::SpecificStateToJSON(nlohmann::json& inout_json) {

}
