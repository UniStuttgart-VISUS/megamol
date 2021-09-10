/*
 * HotkeyEditor.cpp
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "HotkeyEditor.h"


using namespace megamol::gui;


megamol::gui::HotkeyEditor::HotkeyEditor(const std::string& window_name)
    : AbstractWindow(window_name, AbstractWindow::WINDOW_ID_HOTKEYEDITOR)
    , command_registry_ptr(nullptr) {

    // Configure HOTKEY EDITOR Window
    this->win_config.size = ImVec2(0.0f * megamol::gui::gui_scaling.Get(), 0.0f * megamol::gui::gui_scaling.Get());
    this->win_config.reset_size = this->win_config.size;
    this->win_config.flags = ImGuiWindowFlags_None;
    this->win_config.hotkey =
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F6, core::view::Modifier::NONE);
}


HotkeyEditor::~HotkeyEditor() {

}


void megamol::gui::HotkeyEditor::SetData(megamol::core::view::CommandRegistry* cmdregistry) {

    this->command_registry_ptr = cmdregistry;
}


bool megamol::gui::HotkeyEditor::Update() {

    return true;
}


bool megamol::gui::HotkeyEditor::Draw() {

    auto table_flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableColumnFlags_NoResize;
    auto column_flags = ImGuiTableColumnFlags_WidthFixed; // ImGuiTableColumnFlags_WidthStretch;
    if (ImGui::BeginTable("megamol_hotkeys", 3, table_flags)) {
        ImGui::TableSetupColumn("", column_flags);

        if (this->command_registry_ptr != nullptr) {
            for (auto& cmd : this->command_registry_ptr->list_commands()) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(cmd.name.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(cmd.key.ToString().c_str());
                ImGui::TableNextColumn();
                if (ImGui::Button(std::string("execute##" + cmd.name).c_str())) {
                    cmd.execute();
                }
            }
        }
        ImGui::EndTable();
    }

    return true;
}


void HotkeyEditor::SpecificStateFromJSON(const nlohmann::json& in_json) {

    if (this->command_registry_ptr != nullptr) {
        for (auto& header_item : in_json.items()) {
            if (header_item.key() == GUI_JSON_TAG_WINDOW_CONFIGS) {
                for (auto& config_item : header_item.value().items()) {
                    if (config_item.key() == this->Name()) {
                        auto config_values = config_item.value();
                        megamol::frontend_resources::Command cmd;
                        for (auto& cmd_json : config_values) {
                            megamol::frontend_resources::from_json(cmd_json, cmd);
                            this->command_registry_ptr->add_command(cmd);
                        }
                    }
                }
            }
        }
    }
}


void HotkeyEditor::SpecificStateToJSON(nlohmann::json& inout_json) {

    if (this->command_registry_ptr != nullptr) {
        for (auto& cmd : this->command_registry_ptr->list_commands()) {
            nlohmann::basic_json cmd_json;
            megamol::frontend_resources::to_json(cmd_json, cmd);
            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()].push_back(cmd_json);
        }
    }
}
