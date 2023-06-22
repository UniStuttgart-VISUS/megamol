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
        , search_widget()
        , tooltip_widget()
        , pending_hotkey_assignment(0)
        , pending_hotkey()
        , command_registry_ptr(nullptr)
        , window_collection_ptr(nullptr)
        , gui_hotkey_ptr(nullptr)
        , megamolgraph_ptr(nullptr)
        , parent_gui_hotkey_lambda()
        , parent_gui_window_lambda()
        , parent_gui_window_hotkey_lambda() {

    // Configure HOTKEY EDITOR Window
    this->win_config.size = ImVec2(0.0f * megamol::gui::gui_scaling.Get(), 0.0f * megamol::gui::gui_scaling.Get());
    this->win_config.reset_size = this->win_config.size;
    this->win_config.flags = ImGuiWindowFlags_NoNavInputs;
    this->win_config.hotkey =
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F6, core::view::Modifier::NONE);
}


HotkeyEditor::~HotkeyEditor() {

    this->command_registry_ptr = nullptr;
    this->window_collection_ptr = nullptr;
    this->gui_hotkey_ptr = nullptr;
    this->megamolgraph_ptr = nullptr;
}


void megamol::gui::HotkeyEditor::RegisterHotkeys(megamol::core::view::CommandRegistry* cmdregistry,
    megamol::core::MegaMolGraph* megamolgraph, megamol::gui::WindowCollection* wincollection,
    megamol::gui::HotkeyMap_t* guihotkeys) {

    assert(cmdregistry != nullptr);
    assert(wincollection != nullptr);
    assert(guihotkeys != nullptr);
    assert(megamolgraph != nullptr);

    this->command_registry_ptr = cmdregistry;
    this->window_collection_ptr = wincollection;
    this->gui_hotkey_ptr = guihotkeys;
    this->megamolgraph_ptr = megamolgraph;

    this->graph_parameter_lambda = [&](const frontend_resources::Command* self) {
        auto my_p = this->megamolgraph_ptr->FindParameter(self->parent);
        if (my_p != nullptr) {
            my_p->setDirty();
        }
    };

    this->parent_gui_hotkey_lambda = [&](const frontend_resources::Command* self) {
        for (auto& hotkey : *this->gui_hotkey_ptr) {
            if (hotkey.second.name == self->name) {
                hotkey.second.is_pressed = !hotkey.second.is_pressed;
            }
        }
    };

    this->parent_gui_window_lambda = [&](const frontend_resources::Command* self) {
        std::stringstream sstream(self->parent);
        size_t parent_hash = 0;
        sstream >> parent_hash;
        const auto wf = [&](megamol::gui::AbstractWindow& wc) {
            if (wc.Hash() == parent_hash) {
                wc.Config().show = !wc.Config().show;
            }
        };
        this->window_collection_ptr->EnumWindows(wf);
    };

    this->parent_gui_window_hotkey_lambda = [&](const frontend_resources::Command* self) {
        const auto wf = [&](megamol::gui::AbstractWindow& wc) {
            for (auto& hotkey : wc.GetHotkeys()) {
                if (hotkey.second.name == self->name) {
                    hotkey.second.is_pressed = !hotkey.second.is_pressed;
                }
            }
        };
        this->window_collection_ptr->EnumWindows(wf);
    };


    // Add new commands -------------------------------------------------------

    // GUI
    frontend_resources::Command hkcmd;
    for (auto& hotkey : *this->gui_hotkey_ptr) {
        hkcmd.parent_type = megamol::frontend_resources::Command::parent_type_c::PARENT_GUI_HOTKEY;
        hkcmd.key = hotkey.second.keycode;
        hkcmd.name = hotkey.second.name;
        hkcmd.parent = std::string();
        hkcmd.effect = this->parent_gui_hotkey_lambda;
        cmdregistry->add_command(hkcmd);
    }

    // Hotkeys of window(s)
    const auto windows_func = [&](AbstractWindow& wc) {
        // Check "Show/Hide Window"-Hotkey
        hkcmd.parent_type = megamol::frontend_resources::Command::parent_type_c::PARENT_GUI_WINDOW;
        hkcmd.key = wc.Config().hotkey;
        hkcmd.name = std::string("_hotkey_gui_window_" + wc.Name());
        hkcmd.parent = std::to_string(wc.Hash());
        hkcmd.effect = this->parent_gui_window_lambda;
        cmdregistry->add_command(hkcmd);

        // Check for additional hotkeys of window
        for (auto& hotkey : wc.GetHotkeys()) {
            hkcmd.parent_type = megamol::frontend_resources::Command::parent_type_c::PARENT_GUI_WINDOW_HOTKEY;
            hkcmd.key = hotkey.second.keycode;
            hkcmd.name = hotkey.second.name;
            hkcmd.parent = std::string();
            hkcmd.effect = this->parent_gui_window_hotkey_lambda;
            cmdregistry->add_command(hkcmd);
        }
    };
    this->window_collection_ptr->EnumWindows(windows_func);
}


bool megamol::gui::HotkeyEditor::Update() {

    return true;
}


bool megamol::gui::HotkeyEditor::Draw() {

    this->search_widget.Widget("hotkeyeditor_search", "Case insensitive substring search in names and hotkeys.");
    auto search_string = this->search_widget.GetSearchString();

    ImGui::TextUnformatted("Left-click on button: Execute hotkey\nRight-click on button: Edit hotkey");

    auto table_flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableColumnFlags_NoResize;
    if (ImGui::BeginTable("megamol_hotkeys", 2, table_flags)) {

        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Hotkey", ImGuiTableColumnFlags_WidthFixed);
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

                        ImGui::TextUnformatted("Press new hotkey to overwrite existing one\n[Press ESC to abort]");
                        if (this->pending_hotkey_assignment == 0) {
                            // First: Wait for first user key press
                            if (this->is_any_key_down()) {
                                this->pending_hotkey_assignment = 1;
                            }
                        } else if (this->pending_hotkey_assignment == 1) {
                            // Second: Collect all pressed keys
                            ImGuiIO& io = ImGui::GetIO();
                            for (int i = static_cast<int>(ImGuiKey_NamedKey_BEGIN);
                                 i < static_cast<int>(ImGuiKey_NamedKey_END); i++) {
                                auto imgui_key = static_cast<ImGuiKey>(i);
                                if (ImGui::IsKeyDown(imgui_key) && !this->is_key_modifier(imgui_key)) {
                                    this->pending_hotkey.key = gui_utils::ImGuiKeyToGlfwKey(imgui_key);
                                    this->pending_hotkey.mods = core::view::Modifier::NONE;
                                    if (ImGui::IsKeyDown(ImGuiMod_Alt))
                                        this->pending_hotkey.mods |= core::view::Modifier::ALT;
                                    if (ImGui::IsKeyDown(ImGuiMod_Ctrl))
                                        this->pending_hotkey.mods |= core::view::Modifier::CTRL;
                                    if (ImGui::IsKeyDown(ImGuiMod_Shift))
                                        this->pending_hotkey.mods |= core::view::Modifier::SHIFT;
                                    break;
                                }
                            }
                            if (!this->is_any_key_down()) {
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

                        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
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


void megamol::gui::HotkeyEditor::SpecificStateFromJSON(const nlohmann::json& in_json) {

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
                                                    switch (cmd.parent_type) {
                                                    case (megamol::frontend_resources::Command::parent_type_c::
                                                            PARENT_PARAM):
                                                        cmd.effect = this->graph_parameter_lambda;
                                                        break;
                                                    case (megamol::frontend_resources::Command::parent_type_c::
                                                            PARENT_GUI_HOTKEY):
                                                        cmd.effect = this->parent_gui_hotkey_lambda;
                                                        break;
                                                    case (megamol::frontend_resources::Command::parent_type_c::
                                                            PARENT_GUI_WINDOW):
                                                        cmd.effect = this->parent_gui_window_lambda;
                                                        break;
                                                    case (megamol::frontend_resources::Command::parent_type_c::
                                                            PARENT_GUI_WINDOW_HOTKEY):
                                                        cmd.effect = this->parent_gui_window_hotkey_lambda;
                                                        break;
                                                    }
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


void megamol::gui::HotkeyEditor::SpecificStateToJSON(nlohmann::json& inout_json) {

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


bool megamol::gui::HotkeyEditor::is_any_key_down() {

    ImGuiIO& io = ImGui::GetIO();
    for (int i = static_cast<int>(ImGuiKey_NamedKey_BEGIN); i < static_cast<int>(ImGuiKey_NamedKey_END); i++) {
        if (ImGui::IsKeyDown(static_cast<ImGuiKey>(i))) {
            return true;
        }
    }
    return false;
}


bool megamol::gui::HotkeyEditor::is_key_modifier(ImGuiKey k) {

    if ((k == ImGuiKey_LeftCtrl) || (k == ImGuiKey_LeftShift) || (k == ImGuiKey_LeftAlt) || (k == ImGuiKey_LeftSuper) ||
        (k == ImGuiKey_RightCtrl) || (k == ImGuiKey_RightShift) || (k == ImGuiKey_RightAlt) ||
        (k == ImGuiKey_RightSuper) || (k == ImGuiMod_Ctrl) || (k == ImGuiMod_Shift) || (k == ImGuiMod_Alt) ||
        (k == ImGuiMod_Super)) {
        return true;
    }
    return false;
}
