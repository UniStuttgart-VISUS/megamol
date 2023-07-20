/*
 * Configurator.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "Configurator.h"


using namespace megamol::gui;


megamol::gui::Configurator::Configurator(
    const std::string& window_name, std::shared_ptr<TransferFunctionEditor> win_tfe_ptr)
        : AbstractWindow(window_name, AbstractWindow::WINDOW_ID_CONFIGURATOR)
        , graph_state()
        , graph_collection()
        , win_tfeditor_ptr(win_tfe_ptr)
        , module_list_sidebar_width(250.0f)
        , selected_list_module_id(GUI_INVALID_ID)
        , add_project_graph_uid(GUI_INVALID_ID)
        , module_list_popup_hovered_group_uid(GUI_INVALID_ID)
        , show_module_list_sidebar(false)
        , show_module_list_popup(false)
        , last_selected_callslot_uid(GUI_INVALID_ID)
        , open_popup_load(false)
        , file_browser()
        , search_widget()
        , splitter_widget()
        , tooltip() {

    assert(this->win_tfeditor_ptr != nullptr);

    // init hotkeys
    this->win_hotkeys[HOTKEY_CONFIGURATOR_MODULE_SEARCH] = {"_hotkey_gui_configurator_module_search",
        core::view::KeyCode(core::view::Key::KEY_M, (core::view::Modifier::CTRL | core::view::Modifier::SHIFT)), false};
    this->win_hotkeys[HOTKEY_CONFIGURATOR_PARAMETER_SEARCH] = {"_hotkey_gui_configurator_param_search",
        core::view::KeyCode(core::view::Key::KEY_P, (core::view::Modifier::CTRL | core::view::Modifier::SHIFT)), false};
    this->win_hotkeys[HOTKEY_CONFIGURATOR_DELETE_GRAPH_ITEM] = {
        "_hotkey_gui_configurator_delete_graph_entry", core::view::KeyCode(core::view::Key::KEY_DELETE), false};
    this->win_hotkeys[HOTKEY_CONFIGURATOR_SAVE_PROJECT] = {"_hotkey_gui_configurator_save_project",
        megamol::core::view::KeyCode(core::view::Key::KEY_S, core::view::Modifier::CTRL | core::view::Modifier::SHIFT),
        false};
    this->win_hotkeys[HOTKEY_CONFIGURATOR_LAYOUT_GRAPH] = {"_hotkey_gui_configurator_layout_graph",
        core::view::KeyCode(core::view::Key::KEY_R, core::view::Modifier::SHIFT | core::view::Modifier::CTRL), false};

    this->graph_state.graph_zoom_font_scalings = {0.85f, 0.95f, 1.0f, 1.5f, 2.5f};
    this->graph_state.graph_width = 0.0f;
    this->graph_state.show_parameter_sidebar = false;
    this->graph_state.show_profiling_bar = false;
    this->graph_state.graph_selected_uid = GUI_INVALID_ID;
    this->graph_state.graph_delete = false;
    this->graph_state.configurator_graph_save = false;
    this->graph_state.global_graph_save = false;
    this->graph_state.new_running_graph_uid = GUI_INVALID_ID;

    // Configure CONFIGURATOR Window
    this->win_config.size = ImVec2(750.0f * megamol::gui::gui_scaling.Get(), 500.0f * megamol::gui::gui_scaling.Get());
    this->win_config.reset_size = this->win_config.size;
    this->win_config.flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoNavInputs;
    this->win_config.hotkey =
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F11, core::view::Modifier::NONE);
}


bool Configurator::Update() {

    auto tf_param_connect_request = this->win_tfeditor_ptr->ProcessParameterConnectionRequest();
    if (!tf_param_connect_request.empty()) {
        if (auto graph_ptr = this->GetGraphCollection().GetRunningGraph()) {
            for (auto& module_ptr : graph_ptr->Modules()) {
                std::string module_full_name = module_ptr->FullName();
                for (auto& param : module_ptr->Parameters()) {
                    std::string param_full_name = param.FullName();
                    if (gui_utils::CaseInsensitiveStringEqual(tf_param_connect_request, param_full_name) &&
                        (param.Type() == ParamType_t::TRANSFERFUNCTION)) {
                        win_tfeditor_ptr->SetConnectedParameter(&param, param_full_name);
                        param.TransferFunctionEditor_ConnectExternal(this->win_tfeditor_ptr, true);
                    }
                }
            }
        }
    }

    return true;
}


bool megamol::gui::Configurator::Draw() {

    if (ImGui::GetCurrentContext() == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (!this->graph_collection.IsCallStockLoaded()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Calls stock is not loaded. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (!this->graph_collection.IsModuleStockLoaded()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Modules stock is not loaded. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Update state -------------------------------------------------------
    this->graph_state.hotkeys = this->win_hotkeys;

    // Process hotkeys
    /// HOTKEY_CONFIGURATOR_SAVE_PROJECT
    if (this->graph_state.hotkeys[HOTKEY_CONFIGURATOR_SAVE_PROJECT].is_pressed &&
        (this->graph_state.graph_selected_uid != GUI_INVALID_ID)) {

        bool is_running_graph = false;
        if (auto graph_ptr = this->graph_collection.GetGraph(this->graph_state.graph_selected_uid)) {
            is_running_graph = graph_ptr->IsRunning();
        }
        if (is_running_graph) {
            this->graph_state.global_graph_save = true;
        } else {
            this->graph_state.configurator_graph_save = true;
        }
        this->graph_state.hotkeys[HOTKEY_CONFIGURATOR_SAVE_PROJECT].is_pressed = false;
    }
    /// HOTKEY_CONFIGURATOR_MODULE_SEARCH
    if (this->graph_state.hotkeys[HOTKEY_CONFIGURATOR_MODULE_SEARCH].is_pressed) {
        this->search_widget.SetSearchFocus();
        this->graph_state.hotkeys[HOTKEY_CONFIGURATOR_MODULE_SEARCH].is_pressed = false;
    }

    if (this->graph_state.hotkeys[HOTKEY_CONFIGURATOR_LAYOUT_GRAPH].is_pressed) {
        this->graph_collection.GetGraph(this->graph_state.graph_selected_uid)->SetLayoutGraph(true);
        this->graph_state.hotkeys[HOTKEY_CONFIGURATOR_LAYOUT_GRAPH].is_pressed = false;
    }

    // Draw Windows -------------------------------------------------------

    // Menu
    this->draw_window_menu();

    // Splitter
    this->graph_state.graph_width = 0.0f;
    if (this->show_module_list_sidebar) {
        this->module_list_sidebar_width *= megamol::gui::gui_scaling.Get();

        this->splitter_widget.Widget("module_splitter", true, 0.0f, SplitterWidget::FixedSplitterSide::LEFT_TOP,
            this->module_list_sidebar_width, this->graph_state.graph_width, ImGui::GetCursorScreenPos());
        this->draw_window_module_list(this->module_list_sidebar_width, 0.0f, this->show_module_list_popup);

        this->module_list_sidebar_width /= megamol::gui::gui_scaling.Get();
        ImGui::SameLine();
    }
    // Graphs
    this->graph_collection.Draw(this->graph_state);

    // Reset state --------------------------------------------------------

    // Only reset 'externally' processed hotkeys
    this->graph_state.hotkeys[HOTKEY_CONFIGURATOR_PARAMETER_SEARCH].is_pressed = false;
    this->graph_state.hotkeys[HOTKEY_CONFIGURATOR_DELETE_GRAPH_ITEM].is_pressed = false;
    this->win_hotkeys = this->graph_state.hotkeys;

    return true;
}


void megamol::gui::Configurator::PopUps() {

    // Load Project -----------------------------------------------------------
    bool popup_failed = false;
    std::string project_filename;
    if (auto graph_ptr = this->graph_collection.GetGraph(this->graph_state.graph_selected_uid)) {
        project_filename = graph_ptr->GetFilename();
    }
    if (this->file_browser.PopUp_Load("Load Project", project_filename, this->open_popup_load, {"lua"},
            megamol::core::param::FilePathParam::Flag_File_RestrictExtension)) {

        popup_failed = !this->graph_collection.LoadOrAddProjectFromFile(this->add_project_graph_uid, project_filename);
        this->add_project_graph_uid = GUI_INVALID_ID;
    }
    PopUps::Minimal("Failed to Load Project", popup_failed, "See console log output for more information.", "Cancel");

    // Module Stock List Child Window ------------------------------------------
    if (auto selected_graph_ptr = this->graph_collection.GetGraph(this->graph_state.graph_selected_uid)) {

        ImGuiID selected_callslot_uid = selected_graph_ptr->GetSelectedCallSlot();
        ImGuiID selected_group_uid = selected_graph_ptr->GetSelectedGroup();
        bool is_any_module_hovered = selected_graph_ptr->IsModuleHovered();

        bool valid_double_click =
            (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && selected_graph_ptr->IsCanvasHovered() &&
                (selected_group_uid == GUI_INVALID_ID) && (!this->show_module_list_popup) && (!is_any_module_hovered));
        bool valid_double_click_callslot =
            (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && selected_graph_ptr->IsCanvasHovered() &&
                (selected_callslot_uid != GUI_INVALID_ID) &&
                ((!this->show_module_list_popup) || (this->last_selected_callslot_uid != selected_callslot_uid)));

        if (valid_double_click || valid_double_click_callslot) {
            this->graph_state.hotkeys[HOTKEY_CONFIGURATOR_MODULE_SEARCH].is_pressed = true;
            this->last_selected_callslot_uid = selected_callslot_uid;
        }
        if (this->graph_state.hotkeys[HOTKEY_CONFIGURATOR_MODULE_SEARCH].is_pressed) {
            this->module_list_popup_hovered_group_uid = selected_graph_ptr->GetHoveredGroup();
        }
    }

    ImVec2 module_list_popup_pos;
    if (this->graph_state.hotkeys[HOTKEY_CONFIGURATOR_MODULE_SEARCH].is_pressed) {
        this->show_module_list_popup = true;
        module_list_popup_pos = ImGui::GetMousePos();
    }
    if (this->show_module_list_popup) {

        ImGuiStyle& style = ImGui::GetStyle();
        float offset_x = 2.0f * style.WindowPadding.x;
        float offset_y = 2.0f * style.WindowPadding.y;
        float popup_width = (250.0f * megamol::gui::gui_scaling.Get()) + offset_x;
        float popup_height = (350.0f * megamol::gui::gui_scaling.Get()) + offset_y;
        std::string pop_up_id = "module_list_child";
        if (!ImGui::IsPopupOpen(pop_up_id.c_str())) {
            ImGui::OpenPopup(pop_up_id.c_str(), ImGuiPopupFlags_None);
            this->search_widget.SetSearchFocus();

            float diff_width = (this->win_config.position.x + this->win_config.size.x - module_list_popup_pos.x);
            float diff_height = (this->win_config.position.y + this->win_config.size.y - module_list_popup_pos.y);
            if (diff_width < popup_width) {
                module_list_popup_pos.x -= ((popup_width - diff_width) + offset_x);
            }
            module_list_popup_pos.x = std::max(module_list_popup_pos.x, this->win_config.position.x);
            if (diff_height < popup_height) {
                module_list_popup_pos.y -= ((popup_height - diff_height) + offset_y);
            }
            module_list_popup_pos.y = std::max(module_list_popup_pos.y, this->win_config.position.y);
            ImGui::SetNextWindowPos(module_list_popup_pos);
            ImGui::SetNextWindowSize(ImVec2(10.0f, 10.0f));
        }
        auto popup_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar |
                           ImGuiWindowFlags_NoCollapse;
        if (ImGui::BeginPopup(pop_up_id.c_str(), popup_flags)) {

            ImVec2 popup_position = ImGui::GetWindowPos();

            this->draw_window_module_list(
                std::max(0.0f, (popup_width - offset_x)), std::max(0.0f, (popup_height - offset_y)), false);

            bool module_list_popup_hovered = false;
            if ((ImGui::GetMousePos().x >= popup_position.x) &&
                (ImGui::GetMousePos().x <= (popup_position.x + popup_width)) &&
                (ImGui::GetMousePos().y >= popup_position.y) &&
                (ImGui::GetMousePos().y <= (popup_position.y + popup_height))) {
                module_list_popup_hovered = true;
            }
            if (((ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !module_list_popup_hovered)) ||
                ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                this->show_module_list_popup = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }
}


void megamol::gui::Configurator::draw_window_menu() {

    bool is_running_graph_active = false;
    if (auto graph_ptr = this->graph_collection.GetGraph(this->graph_state.graph_selected_uid)) {
        is_running_graph_active = graph_ptr->IsRunning();
    }

    ImGui::PushID("Configurator::Menu");
    // Menu
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {

            if (ImGui::MenuItem("New Empty Project")) {
                this->graph_collection.AddEmptyProject();
            }

            if (ImGui::MenuItem("Open Project")) {
                this->add_project_graph_uid = GUI_INVALID_ID;
                this->open_popup_load = true;
            }

            if (ImGui::MenuItem(
                    "Add Project", nullptr, false, (this->graph_state.graph_selected_uid != GUI_INVALID_ID))) {
                this->add_project_graph_uid = this->graph_state.graph_selected_uid;
                this->open_popup_load = true;
            }

            // Save currently active project to LUA file
            if (ImGui::MenuItem("Save Project",
                    this->graph_state.hotkeys[HOTKEY_CONFIGURATOR_SAVE_PROJECT].keycode.ToString().c_str(), false,
                    ((this->graph_state.graph_selected_uid != GUI_INVALID_ID)))) {
                if (is_running_graph_active) {
                    this->graph_state.global_graph_save = true;
                } else {
                    this->graph_state.configurator_graph_save = true;
                }
            }
            ImGui::EndMenu();
        }
        ImGui::Separator();

        if (ImGui::BeginMenu("View")) {
            if (ImGui::MenuItem("Modules Sidebar", nullptr, this->show_module_list_sidebar)) {
                this->show_module_list_sidebar = !this->show_module_list_sidebar;
            }
            if (ImGui::MenuItem("Parameter Sidebar", nullptr, this->graph_state.show_parameter_sidebar,
                    (this->graph_state.graph_selected_uid != GUI_INVALID_ID))) {
                this->graph_state.show_parameter_sidebar = !this->graph_state.show_parameter_sidebar;
            }
#ifdef MEGAMOL_USE_PROFILING
            if (ImGui::MenuItem(
                    "Profiling Bar", nullptr, this->graph_state.show_profiling_bar, is_running_graph_active)) {
                this->graph_state.show_profiling_bar = !this->graph_state.show_profiling_bar;
            }
#endif // MEGAMOL_USE_PROFILING
            ImGui::EndMenu();
        }
        ImGui::Separator();

        if (ImGui::BeginMenu("Help")) {

            ImGui::TextUnformatted("Graph Interactions");
            auto table_flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableColumnFlags_NoResize;
            if (ImGui::BeginTable("configurator_help_table", 2, table_flags)) {
                //ImGui::TableSetupColumn("Description", ImGuiTableColumnFlags_WidthStretch);
                //ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed);
                //ImGui::TableHeadersRow();

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(
                    "Spawn module selection pop-up. Double Left-Click on call slots shows only compatible modules.");
                ImGui::TableNextColumn();
                ImGui::TextUnformatted("Mouse Double Left-Click");

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted("Show context menu of module/call/group");
                ImGui::TableNextColumn();
                ImGui::TextUnformatted("Mouse Right-Click");

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(
                    "Drag call slot of module to other compatible call slot to create call between modules.");
                ImGui::TableNextColumn();
                ImGui::TextUnformatted("Mouse Left Drag & Drop");

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted("Zoom graph");
                ImGui::TableNextColumn();
                ImGui::TextUnformatted("Mouse Wheel");

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted("Scroll graph");
                ImGui::TableNextColumn();
                ImGui::TextUnformatted("Mouse Middle Drag");

                ImGui::EndTable();
            }

            ImGui::EndMenu();
        }
        ImGui::Separator();

        ImGui::EndMenuBar();
    }
    ImGui::PopID();
}


void megamol::gui::Configurator::draw_window_module_list(float width, float height, bool omit_focus) {

    ImGui::BeginGroup();

    const float search_child_height = ImGui::GetFrameHeightWithSpacing() * 2.5f;
    auto child_flags = ImGuiWindowFlags_AlwaysUseWindowPadding | ImGuiWindowFlags_NoScrollbar;
    ImGui::BeginChild("module_search_child_window", ImVec2(width, search_child_height), false, child_flags);

    ImGui::TextUnformatted("Available Modules");
    ImGui::Separator();

    std::string help_text = "[" + this->graph_state.hotkeys[HOTKEY_CONFIGURATOR_MODULE_SEARCH].keycode.ToString() +
                            "] Set keyboard focus to search input field.\n"
                            "Case insensitive substring search in module names.";
    this->search_widget.Widget("configurator_module_search", help_text, omit_focus);
    auto search_string = this->search_widget.GetSearchString();

    ImGui::EndChild();

    // ------------------------------------------------------------------------

    child_flags = ImGuiWindowFlags_None;
    ImGui::BeginChild(
        "module_list_child_window", ImVec2(width, std::max(0.0f, height - search_child_height)), true, child_flags);

    bool interfaceslot_selected = false;
    std::string compat_callslot_name;
    CallSlotPtr_t selected_callslot_ptr;
    if (auto selected_graph_ptr = this->graph_collection.GetGraph(this->graph_state.graph_selected_uid)) {
        auto callslot_id = selected_graph_ptr->GetSelectedCallSlot();
        if (callslot_id != GUI_INVALID_ID) {
            for (auto& module_ptr : selected_graph_ptr->Modules()) {
                if (auto callslot_ptr = module_ptr->CallSlotPtr(callslot_id)) {
                    selected_callslot_ptr = callslot_ptr;
                }
            }
        }
        auto interfaceslot_id = selected_graph_ptr->GetSelectedInterfaceSlot();
        if (interfaceslot_id != GUI_INVALID_ID) {
            for (auto& group_ptr : selected_graph_ptr->GetGroups()) {
                if (auto interfaceslot_ptr = group_ptr->InterfaceSlotPtr(interfaceslot_id)) {
                    if (auto callslot_ptr = interfaceslot_ptr->GetCompatibleCallSlot()) {
                        selected_callslot_ptr = callslot_ptr;
                        interfaceslot_selected = true;
                    }
                }
            }
        }
    }

    ImGuiID id = 1;
    for (auto& mod : this->graph_collection.GetModulesStock()) {
        // Filter module by given search string
        bool search_filter = true;
        if (!search_string.empty()) {
            search_filter = gui_utils::FindCaseInsensitiveSubstring(mod.class_name, search_string);
        }

        // Filter module by compatible call slots
        bool compat_filter = true;
        if (selected_callslot_ptr != nullptr) {
            compat_filter = false;
            for (auto& stock_callslot_map : mod.callslots) {
                for (auto& stock_callslot : stock_callslot_map.second) {
                    if (CallSlot::GetCompatibleCallIndex(selected_callslot_ptr, stock_callslot) != GUI_INVALID_ID) {
                        compat_callslot_name = stock_callslot.name;
                        compat_filter = true;
                    }
                }
            }
        }

        if (search_filter && compat_filter) {
            ImGui::PushID(static_cast<int>(id));

            std::string label = mod.class_name + " (" + mod.plugin_name + ")";
            if (mod.is_view) {
                label += " [View]";
            }

            bool add_module = false;
            if (ImGui::Selectable(label.c_str(), (id == this->selected_list_module_id))) {
                this->selected_list_module_id = id;
            }
            // Left mouse button double click action
            if ((ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && ImGui::IsItemHovered()) || // Mouse Double Click
                (!ImGui::IsMouseClicked(ImGuiMouseButton_Left) && ImGui::IsItemFocused() &&
                    ImGui::IsItemActivated())) { // Selection via key ('Space')
                add_module = true;
            }
            // Context menu
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Add", "'Double-Click'")) {
                    add_module = true;
                }
                ImGui::EndPopup();
            }

            if (add_module) {
                if (auto selected_graph_ptr = this->graph_collection.GetGraph(this->graph_state.graph_selected_uid)) {

                    // If there is a call slot selected, create call to compatible call slot of new module
                    bool add_call = compat_filter && (selected_callslot_ptr != nullptr);

                    // If there is a group selected or hovered or the new call is connected to module which is part
                    // of group, add module to this group
                    std::string group_name = "";
                    if (!interfaceslot_selected) {
                        ImGuiID connected_group = GUI_INVALID_ID;
                        if (add_call && selected_callslot_ptr->IsParentModuleConnected()) {
                            connected_group = selected_callslot_ptr->GetParentModule()->GroupUID();
                        }
                        ImGuiID selected_group_uid = selected_graph_ptr->GetSelectedGroup();
                        ImGuiID group_uid = (connected_group != GUI_INVALID_ID)
                                                ? (connected_group)
                                                : ((selected_group_uid != GUI_INVALID_ID)
                                                          ? (selected_group_uid)
                                                          : (this->module_list_popup_hovered_group_uid));
                        if (auto group_ptr = selected_graph_ptr->GetGroup(group_uid)) {
                            group_name = group_ptr->Name();
                        }
                    }

                    // Add new module
                    if (auto module_ptr = selected_graph_ptr->AddModule(
                            this->graph_collection.GetModulesStock(), mod.class_name, "", group_name)) {

                        // Add new call after module is created and after possible renaming due to group joining of module!
                        if (add_call) {
                            // Get call slots of last added module
                            for (auto& callslot_map : module_ptr->CallSlots()) {
                                for (auto& callslot_ptr : callslot_map.second) {
                                    if (callslot_ptr->Name() == compat_callslot_name) {
                                        if (selected_graph_ptr->AddCall(this->graph_collection.GetCallsStock(),
                                                selected_callslot_ptr, callslot_ptr)) {
                                            module_ptr->SetSelectedSlotPosition();
                                        }
                                    }
                                }
                            }
                        }
                        // Place new module at mouse pos if added via separate module list child window.
                        else if (this->show_module_list_popup) {
                            module_ptr->SetScreenPosition(ImGui::GetMousePos());
                        }
                    }
                    if (this->show_module_list_popup) {
                        this->show_module_list_popup = false;
                        // ImGui::CloseCurrentPopup();
                    }
                } else {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] No project loaded. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                }
            }
            // Hover tool tip
            this->tooltip.ToolTip(mod.description, id, 0.5f, 5.0f);

            ImGui::PopID();
        }
        id++;
    }

    ImGui::EndChild();

    ImGui::EndGroup();
}


void megamol::gui::Configurator::SpecificStateToJSON(nlohmann::json& inout_json) {

    try {
        // Write configurator state
        inout_json[GUI_JSON_TAG_CONFIGURATOR]["show_module_list_sidebar"] = this->show_module_list_sidebar;
        inout_json[GUI_JSON_TAG_CONFIGURATOR]["module_list_sidebar_width"] = this->module_list_sidebar_width;

        // Write graph states
        for (auto& graph_ptr : this->GetGraphCollection().GetGraphs()) {
            // For graphs with no interface to core save only file name of loaded project
            if (graph_ptr->IsRunning()) {
                graph_ptr->StateToJSON(inout_json);
            } else {
                auto graph_filename = graph_ptr->GetFilename();
                if (!graph_filename.empty()) {
                    inout_json[GUI_JSON_TAG_GRAPHS][graph_filename] = nlohmann::json::object();
                }
            }
        }

        // Write GUI state of parameters (groups) of running graph
        if (auto graph_ptr = this->GetGraphCollection().GetRunningGraph()) {
            for (auto& module_ptr : graph_ptr->Modules()) {
                module_ptr->StateToJSON(inout_json);
            }
        }

#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Wrote configurator state to JSON.");
#endif // GUI_VERBOSE

    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON Error - Unable to write state to JSON. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


void megamol::gui::Configurator::SpecificStateFromJSON(const nlohmann::json& in_json) {

    try {
        if (!in_json.is_object()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Invalid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return;
        }

        // Read configurator state
        for (auto& header_item : in_json.items()) {
            if (header_item.key() == GUI_JSON_TAG_CONFIGURATOR) {
                auto config_state = header_item.value();

                megamol::core::utility::get_json_value<bool>(
                    config_state, {"show_module_list_sidebar"}, &this->show_module_list_sidebar);

                megamol::core::utility::get_json_value<float>(
                    config_state, {"module_list_sidebar_width"}, &this->module_list_sidebar_width);
            }
        }

        // Read graph states
        for (auto& graph_ptr : this->GetGraphCollection().GetGraphs()) {
            if (graph_ptr->IsRunning()) {
                if (graph_ptr->StateFromJSON(in_json)) {
                    // Disable layouting if graph state was found
                    graph_ptr->SetLayoutGraph(false);
                }
            }
        }
        for (auto& graph_header_item : in_json.items()) {
            if (graph_header_item.key() == GUI_JSON_TAG_GRAPHS) {
                for (auto& graph_item : graph_header_item.value().items()) {
                    std::string json_graph_id = graph_item.key();
                    if (json_graph_id != GUI_JSON_TAG_PROJECT) {
                        // Otherwise load additonal graph from given file name
                        this->GetGraphCollection().LoadOrAddProjectFromFile(GUI_INVALID_ID, json_graph_id);
                    }
                }
            }
        }

        // Read GUI state of parameters (groups) of running graph
        if (auto graph_ptr = this->GetGraphCollection().GetRunningGraph()) {
            for (auto& module_ptr : graph_ptr->Modules()) {
                module_ptr->StateFromJSON(in_json);
            }
        }

#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Read configurator state from JSON.");
#endif // GUI_VERBOSE

    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON Error - Unable to read state from JSON. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}
