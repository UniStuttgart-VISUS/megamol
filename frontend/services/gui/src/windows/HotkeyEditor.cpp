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
        , command_registry_ptr(nullptr)
        , search_widget()
        , tooltip_widget() {

    // Configure HOTKEY EDITOR Window
    this->win_config.size = ImVec2(0.0f * megamol::gui::gui_scaling.Get(), 0.0f * megamol::gui::gui_scaling.Get());
    this->win_config.reset_size = this->win_config.size;
    this->win_config.flags = ImGuiWindowFlags_None;
    this->win_config.hotkey =
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F6, core::view::Modifier::NONE);
}


HotkeyEditor::~HotkeyEditor() {}


void megamol::gui::HotkeyEditor::SetData(megamol::core::view::CommandRegistry* cmdregistry) {

    this->command_registry_ptr = cmdregistry;
}


bool megamol::gui::HotkeyEditor::Update() {

    return true;
}


bool megamol::gui::HotkeyEditor::Draw() {

    this->search_widget.Widget("hotkeyeditor_search", "Case insensitive substring search in names and hotkeys.");
    auto search_string = this->search_widget.GetSearchString();

    auto table_flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableColumnFlags_NoResize;
    auto column_flags =
        ImGuiTableColumnFlags_None; // ImGuiTableColumnFlags_WidthFixed; // ImGuiTableColumnFlags_WidthStretch;
    if (ImGui::BeginTable("megamol_hotkeys", 2, table_flags)) {

        ImGui::TableSetupColumn("Name\n(Click to execute)", column_flags);
        ImGui::TableSetupColumn("Hotkey\n(Click to edit)", column_flags);
        ImGui::TableHeadersRow();

        if (this->command_registry_ptr != nullptr) {
            for (auto& cmd : this->command_registry_ptr->list_commands()) {

                if (search_string.empty() ||
                    (gui_utils::FindCaseInsensitiveSubstring(cmd.name, search_string) ||
                        gui_utils::FindCaseInsensitiveSubstring(cmd.key.ToString(), search_string))) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    if (ImGui::Button(std::string(cmd.name + "##" + cmd.name).c_str())) {
                        cmd.execute();
                    }
                    ImGui::TableNextColumn();
                    if (ImGui::Button(std::string(cmd.key.ToString() + "##" + cmd.name).c_str())) {
                        /// TODO Catch next key code and assign -> pop-up?
                    }
                }
            }
        }
        ImGui::EndTable();
    }

    return true;
}


void HotkeyEditor::SpecificStateFromJSON(const nlohmann::json& in_json) {

    /// TODO command_registry_ptr is nullptr when called on project load ...

    if (this->command_registry_ptr != nullptr) {
        for (auto& header_item : in_json.items()) {
            if (header_item.key() == GUI_JSON_TAG_WINDOW_CONFIGS) {
                for (auto& window_item : header_item.value().items()) {
                    if (window_item.key() == this->Name()) {
                        for (auto& config_item : window_item.value().items()) {
                            if (config_item.key() == "hotkey_list") {
                                megamol::frontend_resources::Command cmd;
                                /* TODO
                                for (auto& cmd_json : config_item.value().items()) {
                                    megamol::frontend_resources::from_json(cmd_json.value(), cmd);
                                    this->command_registry_ptr->add_command(cmd);
                                }
                                */
                            }
                        }
                    }
                }
            }
        }
    }
}


void HotkeyEditor::SpecificStateToJSON(nlohmann::json& inout_json) {

    nlohmann::json cmdlist_json;
    if (this->command_registry_ptr != nullptr) {
        for (auto& cmd : this->command_registry_ptr->list_commands()) {
            nlohmann::json cmd_json;
            megamol::frontend_resources::to_json(cmd_json, cmd);
            cmdlist_json += cmd_json;
        }
    }
    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["hotkey_list"] = cmdlist_json;
}
