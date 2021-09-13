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
        , tooltip_widget()
        , pending_hotkey_assignment(0)
        , pending_hotkey() {

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
    if (ImGui::BeginTable("megamol_hotkeys", 2, table_flags)) {

        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn(
            "Hotkey\n[Left-click to execute]\n[Right-click to edit]", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableHeadersRow();

        if (this->command_registry_ptr != nullptr) {
            for (auto& cmd : this->command_registry_ptr->list_commands()) {

                ImGui::PushID(std::string("hotkey_editor_" + cmd.name).c_str());

                if (search_string.empty() ||
                    (gui_utils::FindCaseInsensitiveSubstring(cmd.name, search_string) ||
                        gui_utils::FindCaseInsensitiveSubstring(cmd.key.ToString(), search_string))) {

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(cmd.name.c_str());
                    ImGui::TableNextColumn();
                    if (ImGui::Button(cmd.key.ToString().c_str())) {
                        cmd.execute();
                    }
                    if (ImGui::BeginPopupContextItem("Edit Hotkey", ImGuiPopupFlags_MouseButtonRight)) {

                        const auto close_popup = [&]() {
                            this->pending_hotkey_assignment = 0;
                            ImGui::CloseCurrentPopup();
                        };

                        ImGui::TextUnformatted("Press new hotkey to overwrite existing one... \n[Press ESC to abort]");
                        if (this->pending_hotkey_assignment == 0) {
                            // First: Wait for first user key press
                            if (is_any_key_pressed()) {
                                this->pending_hotkey_assignment = 1;
                            }
                        } else if (this->pending_hotkey_assignment == 1) {
                            // Second: Collect all pressed keys
                            ImGuiIO& io = ImGui::GetIO();
                            for (int i = 0; i < 337; i++) { // Exclude modifiers (total range of array is 512)
                                if (io.KeysDown[i]) {
                                    this->pending_hotkey.key = static_cast<megamol::frontend_resources::Key>(i);
                                    this->pending_hotkey.mods = core::view::Modifier::NONE;
                                    if (io.KeyAlt)
                                        this->pending_hotkey.mods |= core::view::Modifier::ALT;
                                    if (io.KeyCtrl)
                                        this->pending_hotkey.mods |= core::view::Modifier::CTRL;
                                    if (io.KeyShift)
                                        this->pending_hotkey.mods |= core::view::Modifier::SHIFT;
                                }
                            }
                            if (!is_any_key_pressed()) {
                                this->pending_hotkey_assignment = 2;
                            }
                        } else if (this->pending_hotkey_assignment == 2) {
                            // Third: Ask for new hotkey assignment
                            const auto oldc = this->command_registry_ptr->get_command(this->pending_hotkey);
                            if (oldc.key.key != frontend_resources::Key::KEY_UNKNOWN) {
                                ImGui::Text("Re-assign existing hotkey: %s - currently assigned to %s",
                                    this->pending_hotkey.ToString().c_str(), oldc.name.c_str());
                            } else {
                                ImGui::Text("New hotkey: %s", this->pending_hotkey.ToString().c_str());
                            }
                            if (ImGui::Button("Confirm")) {
                                if (oldc.key.key == frontend_resources::Key::KEY_UNKNOWN) {
                                    this->command_registry_ptr->update_hotkey(cmd.name, this->pending_hotkey);
                                } else {
                                    this->command_registry_ptr->remove_hotkey(this->pending_hotkey);
                                    this->command_registry_ptr->update_hotkey(cmd.name, this->pending_hotkey);
                                }
                                close_popup();
                            }
                            ImGui::SameLine();
                            if (ImGui::Button("Cancel")) {
                                close_popup();
                            }
                        }

                        if (ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape))) {
                            close_popup();
                        }
                        ImGui::EndPopup();
                    }
                }

                ImGui::PopID();
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
                for (auto& window_item : header_item.value().items()) {
                    if (window_item.key() == this->Name()) {
                        for (auto& config_item : window_item.value().items()) {
                            if (config_item.key() == "hotkey_list") {
                                /// TODO Add verbose log on error?
                                if (config_item.value().is_array()) {
                                    for (auto& cmds_array : config_item.value().items()) {
                                        if (cmds_array.value().is_array()) {
                                            for (auto& cmd_json : cmds_array.value().items()) {
                                                megamol::frontend_resources::Command cmd;
                                                megamol::frontend_resources::from_json(cmd_json.value(), cmd);
                                                if (!this->command_registry_ptr->update_hotkey(cmd.name, cmd.key)) {
                                                    this->command_registry_ptr->add_command(cmd);
                                                }
                                            }
                                        }
                                    }
                                }
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
            cmdlist_json.push_back(cmd_json);
        }
    }
    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["hotkey_list"] = cmdlist_json;
}


bool HotkeyEditor::is_any_key_pressed() {

    ImGuiIO& io = ImGui::GetIO();
    bool regular_key_pressed = false;
    for (auto& k : io.KeysDown) {
        if (k) {
            regular_key_pressed = true;
        }
    }
    return (regular_key_pressed || io.KeyAlt || io.KeyCtrl || io.KeyShift);
}
