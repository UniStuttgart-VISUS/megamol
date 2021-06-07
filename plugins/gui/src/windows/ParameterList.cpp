/*
 * ParameterList.cpp
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "ParameterList.h"
#include "graph/Parameter.h"
#include "widgets/ButtonWidgets.h"


using namespace megamol::gui;


ParameterList::ParameterList(const std::string& window_name, AbstractWindow::WindowConfigID win_id, const std::string& initial_module, std::shared_ptr<Configurator> win_configurator,
                             std::shared_ptr<TransferFunctionEditor> win_tfeditor, const RequestParamWindowCallback_t& add_parameter_window)
        : AbstractWindow(window_name, win_id)
        , win_configurator_ptr(win_configurator)
        , win_tfeditor_ptr(win_tfeditor)
        , request_new_parameter_window_func(add_parameter_window)
        , win_modules_list()
        , win_extended_mode(false)
        , search_widget()
        , tooltip() {

    assert((this->WindowID() == AbstractWindow::WINDOW_ID_MAIN_PARAMETERS) || (this->WindowID() == AbstractWindow::WINDOW_ID_PARAMETERS));

    // Configure PARAMETER LIST Window
    this->win_config.show = true;
    this->win_config.size = ImVec2(400.0f * megamol::gui::gui_scaling.Get(), 500.0f * megamol::gui::gui_scaling.Get());
    this->win_config.reset_size = this->win_config.size;
    this->win_config.flags = ImGuiWindowFlags_NoScrollbar;

    if (this->WindowID() == AbstractWindow::WINDOW_ID_PARAMETERS) {
        if (!initial_module.empty()) {
            this->win_modules_list.emplace_back(ModuleIDPair_t(initial_module, GUI_INVALID_ID));
        }
    } else if (this->WindowID() == AbstractWindow::WINDOW_ID_MAIN_PARAMETERS) {
        this->hotkeys[HOTKEY_GUI_PARAMETER_SEARCH] = {
                megamol::core::view::KeyCode(megamol::core::view::Key::KEY_P, core::view::Modifier::CTRL), false};
        this->win_config.hotkey =
                megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F10, core::view::Modifier::NONE);
    }
}


bool ParameterList::Update() {

    // UNUSED

    return true;
}


bool ParameterList::Draw() {

    // Mode
    megamol::gui::ButtonWidgets::ExtendedModeButton("draw_param_window_callback", this->win_extended_mode);
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

    // Parameter substring name filtering (only for main parameter view)
    if (this->WindowID() == AbstractWindow::WINDOW_ID_MAIN_PARAMETERS) {
        if (this->hotkeys[HOTKEY_GUI_PARAMETER_SEARCH].is_pressed) {
            this->search_widget.SetSearchFocus(true);
            this->hotkeys[HOTKEY_GUI_PARAMETER_SEARCH].is_pressed = false;
        }
        std::string help_test = "[" + this->hotkeys[HOTKEY_GUI_PARAMETER_SEARCH].keycode.ToString() +
                                "] Set keyboard focus to search input field.\n"
                                "Case insensitive substring search in module and parameter names.\nSearches globally "
                                "in all parameter windows.\n";
        this->search_widget.Widget("guiwindow_parameter_earch", help_test);
    }

    ImGui::Separator();

    // Create child window for separate scroll bar and keeping header always visible on top of parameter list
    ImGui::BeginChild("###parameter_child", ImVec2(0.0f, 0.0f), false, ImGuiWindowFlags_HorizontalScrollbar);

    // Listing modules and their parameters
    const size_t dnd_size = 2048; // Set same max size of all module labels for drag and drop.
    if (auto graph_ptr = this->win_configurator_ptr->GetGraphCollection().GetRunningGraph()) {

        // Get module groups
        std::map<std::string, std::vector<ModulePtr_t>> group_map;
        for (auto& module_ptr : graph_ptr->Modules()) {
            std::string module_label = module_ptr->FullName();
            // Consider always all modules for main parameter window
            if (this->WindowID() == AbstractWindow::WINDOW_ID_PARAMETERS) {
                // Check if module should be considered.
                if (std::find(this->win_modules_list.begin(), this->win_modules_list.end(), module_label) == this->win_modules_list.end()) {
                    continue;
                }
            }
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
                            this->request_new_parameter_window_func(window_name, AbstractWindow::WINDOW_ID_PARAMETERS, module_label);
                        }

                        // Deleting module's parameters is not available in main parameter window.
                        if (this->WindowID() == AbstractWindow::WINDOW_ID_PARAMETERS) {
                            if (ImGui::MenuItem("Delete from List")) {
                                auto find_iter = std::find(this->win_modules_list.begin(), this->win_modules_list.end(), module_label);
                                // Break if module name is not contained in list
                                if (find_iter != this->win_modules_list.end()) {
                                    this->win_modules_list.erase(find_iter);
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
                        module_ptr->GUIParameterGroups().Draw(module_ptr->Parameters(), search_string,
                            vislib::math::Ternary(this->win_extended_mode), true, Parameter::WidgetScope::LOCAL,
                            this->win_tfeditor_ptr, override_header_state, nullptr);
                    }
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
            if (std::find(this->win_modules_list.begin(), this->win_modules_list.end(), payload_id) == this->win_modules_list.end()) {
                this->win_modules_list.emplace_back(ModuleIDPair_t(payload_id));
            }
        }
        ImGui::EndDragDropTarget();
    }

    ImGui::EndChild();

    return true;
}


void ParameterList::PopUps() {

    // UNUSED
}


void ParameterList::SpecificStateFromJSON(const nlohmann::json& in_json) {

    for (auto& header_item : in_json.items()) {
        if (header_item.key() == GUI_JSON_TAG_WINDOW_CONFIGS) {
            for (auto& config_item : header_item.value().items()) {
                if (config_item.key() == this->Name()) {
                    auto config_values = config_item.value();

                    this->win_modules_list.clear();
                    if (config_values.at("param_modules_list").is_array()) {
                        size_t tmp_size = config_values.at("param_modules_list").size();
                        for (size_t i = 0; i < tmp_size; ++i) {
                            std::string value;
                            megamol::core::utility::get_json_value<std::string>(
                                config_values.at("param_modules_list")[i], {}, &value);
                            this->win_modules_list.emplace_back(value);
                        }
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'param_modules_list' as array. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                    }

                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"param_extended_mode"}, &this->win_extended_mode);
                }
            }
        }
    }
}


void ParameterList::SpecificStateToJSON(nlohmann::json& inout_json) {

    for (auto& pm : this->win_modules_list) {
        gui_utils::Utf8Encode(pm);
    }
    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["param_modules_list"] = this->win_modules_list;
    for (auto& pm : this->win_modules_list) {
        gui_utils::Utf8Decode(pm);
    }
    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["param_extended_mode"] = this->win_extended_mode;
}
