/*
 * ParameterList.cpp
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "ParameterList.h"
#include "widgets/ButtonWidgets.h"


using namespace megamol::gui;



ParameterList::ParameterList() : WindowConfiguration("Parameters", WindowConfiguration::WINDOW_ID_MAIN_PARAMETERS) {

    // Configure PARAMETER LIST Window
    this->config.show = true;
    this->config.size = ImVec2(400.0f, 500.0f);
    this->config.reset_size = this->config.size;
    this->config.hotkey = core::view::KeyCode(core::view::Key::KEY_F10);
    this->config.flags = ImGuiWindowFlags_NoScrollbar;
}

void ParameterList::Draw() {

    // Mode
    megamol::gui::ButtonWidgets::ExtendedModeButton(
            "draw_param_window_callback", wc.config.specific.param_extended_mode);
    this->tooltip.Marker("Expert mode enables options for additional parameter presentation options.");
    ImGui::SameLine();

    // Options
    ImGuiID override_header_state = GUI_INVALID_ID;
    if (ImGui::Button("Expand All")) {
        override_header_state = 1; // open
    }
    ImGui::SameLine();
    if (ImGui::Button("Collapse All")) {
        override_header_state = 0; // close
    }
    ImGui::SameLine();

    // Info
    std::string help_marker = "[INFO]";
    std::string param_help = "[Hover] Show Parameter Description Tooltip\n"
                             "[Right Click] Context Menu\n"
                             "[Drag & Drop] Move Module to other Parameter Window\n"
                             "[Enter], [Tab], [Left Click outside Widget] Confirm input changes";
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled(help_marker.c_str());
    this->tooltip.ToolTip(param_help);

    // Paramter substring name filtering (only for main parameter view)
    if (wc.CallbackID() == WindowConfiguration::WINDOW_ID_MAIN_PARAMETERS) {
        if (this->hotkeys[GUIManager::GuiHotkeyIndex::PARAMETER_SEARCH].is_pressed) {
            this->search_widget.SetSearchFocus(true);
            this->hotkeys[GUIManager::GuiHotkeyIndex::PARAMETER_SEARCH].is_pressed = false;
        }
        std::string help_test = "[" + this->hotkeys[GUIManager::GuiHotkeyIndex::PARAMETER_SEARCH].keycode.ToString() +
                                "] Set keyboard focus to search input field.\n"
                                "Case insensitive substring search in module and parameter names.\nSearches globally "
                                "in all parameter windows.\n";
        this->search_widget.Widget("guiwindow_parameter_earch", help_test);
    }

    ImGui::Separator();

    // Create child window for sepearte scroll bar and keeping header always visible on top of parameter list
    ImGui::BeginChild("###ParameterList", ImVec2(0.0f, 0.0f), false, ImGuiWindowFlags_HorizontalScrollbar);

    // Listing modules and their parameters
    const size_t dnd_size = 2048; // Set same max size of all module labels for drag and drop.
    if (auto graph_ptr = this->configurator.GetGraphCollection().GetRunningGraph()) {

        // Get module groups
        std::map<std::string, std::vector<ModulePtr_t>> group_map;
        for (auto& module_ptr : graph_ptr->Modules()) {
            auto group_name = module_ptr->GroupName();
            if (!group_name.empty()) {
                group_map["::" + group_name].emplace_back(module_ptr);
            } else {
                group_map[""].emplace_back(module_ptr);
            }
        }
        for (auto& group : group_map) {
            std::string search_string = this->search_widget.GetSearchString();
            bool indent = false;
            bool group_header_open = group.first.empty();
            if (!group_header_open) {
                group_header_open = gui_utils::GroupHeader(
                        megamol::gui::HeaderType::MODULE_GROUP, group.first, search_string, override_header_state);
                indent = true;
                ImGui::Indent();
            }
            if (group_header_open) {
                for (auto& module_ptr : group.second) {
                    std::string module_label = module_ptr->FullName();
                    ImGui::PushID(module_ptr->UID());

                    // Check if module should be considered.
                    if (!this->considerModule(module_label, wc.config.specific.param_modules_list)) {
                        continue;
                    }

                    // Draw module header
                    bool module_header_open = gui_utils::GroupHeader(
                            megamol::gui::HeaderType::MODULE, module_label, search_string, override_header_state);
                    // Module description as hover tooltip
                    this->tooltip.ToolTip(module_ptr->Description(), ImGui::GetID(module_label.c_str()), 0.5f, 5.0f);

                    // Context menu
                    if (ImGui::BeginPopupContextItem()) {
                        if (ImGui::MenuItem("Copy to new Window")) {
                            std::srand(std::time(nullptr));
                            std::string window_name = "Parameters###parameters_" + std::to_string(std::rand());
                            WindowConfiguration wc_param(window_name, WindowConfiguration::WINDOW_ID_PARAMETERS);
                            wc_param.config.basic.show = true;
                            wc_param.config.basic.flags = ImGuiWindowFlags_NoScrollbar;
                            wc_param.config.specific.param_show_hotkeys = false;
                            wc_param.config.basic.position =
                                    ImVec2(ImGui::GetTextLineHeightWithSpacing(), ImGui::GetTextLineHeightWithSpacing());
                            wc_param.config.basic.size = ImVec2(
                                    (400.0f * megamol::gui::gui_scaling.Get()), (600.0f * megamol::gui::gui_scaling.Get()));
                            wc_param.config.specific.param_modules_list.emplace_back(module_label);
                            this->windows.AddWindowConfiguration(wc_param);
                        }

                        // Deleting module's parameters is not available in main parameter window.
                        if (wc.CallbackID() != WindowConfiguration::WINDOW_ID_MAIN_PARAMETERS) {
                            if (ImGui::MenuItem("Delete from List")) {
                                auto find_iter = std::find(wc.config.specific.param_modules_list.begin(),
                                                           wc.config.specific.param_modules_list.end(), module_label);
                                // Break if module name is not contained in list
                                if (find_iter != wc.config.specific.param_modules_list.end()) {
                                    wc.config.specific.param_modules_list.erase(find_iter);
                                }
                                if (wc.config.specific.param_modules_list.empty()) {
                                    this->state.win_delete_hash_id = wc.Hash();
                                }
                            }
                        }
                        ImGui::EndPopup();
                    }

                    // Drag source
                    module_label.resize(dnd_size);
                    if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
                        ImGui::SetDragDropPayload(
                                "DND_COPY_MODULE_PARAMETERS", module_label.c_str(), (module_label.size() * sizeof(char)));
                        ImGui::TextUnformatted(module_label.c_str());
                        ImGui::EndDragDropSource();
                    }

                    // Draw parameters
                    if (module_header_open) {
                        bool out_open_external_tf_editor;
                        module_ptr->GUIParameterGroups().Draw(module_ptr->Parameters(), search_string,
                                                              vislib::math::Ternary(wc.config.specific.param_extended_mode), true,
                                                              Parameter::WidgetScope::LOCAL, this->tf_editor_ptr, &out_open_external_tf_editor,
                                                              override_header_state, nullptr);
                        if (out_open_external_tf_editor) {
                            const auto func = [](WindowConfiguration& wc) {
                                if (wc.CallbackID() == WindowConfiguration::WINDOW_ID_TRANSFER_FUNCTION) {
                                    wc.config.basic.show = true;
                                }
                            };
                            this->windows.EnumWindows(func);
                        }
                    }

                    ImGui::PopID();
                }
            }
            if (indent) {
                ImGui::Unindent();
            }
        }
    }

    // Drop target
    ImGui::Dummy(ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetFontSize()));
    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("DND_COPY_MODULE_PARAMETERS")) {

            IM_ASSERT(payload->DataSize == (dnd_size * sizeof(char)));
            std::string payload_id = (const char*) payload->Data;

            // Insert dragged module name only if not contained in list
            if (!this->considerModule(payload_id, wc.config.specific.param_modules_list)) {
                wc.config.specific.param_modules_list.emplace_back(payload_id);
            }
        }
        ImGui::EndDragDropTarget();
    }

    ImGui::EndChild();
}


bool GUIManager::considerModule(const std::string& modname, std::vector<std::string>& modules_list) {
    bool retval = false;
    // Empty module list means that all modules should be considered.
    if (modules_list.empty()) {
        retval = true;
    } else {
        retval = (std::find(modules_list.begin(), modules_list.end(), modname) != modules_list.end());
    }
    return retval;
}