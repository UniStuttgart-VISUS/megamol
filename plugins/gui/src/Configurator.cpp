/*
 * Configurator.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Configurator.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::Configurator::Configurator()
        : graph_collection()
        , init_state(0)
        , module_list_sidebar_width(250.0f)
        , selected_list_module_uid(GUI_INVALID_ID)
        , add_project_graph_uid(GUI_INVALID_ID)
        , module_list_popup_hovered_group_uid(GUI_INVALID_ID)
        , show_module_list_sidebar(true)
        , show_module_list_popup(false)
        , module_list_popup_pos()
        , last_selected_callslot_uid(GUI_INVALID_ID)
        , graph_state()
        , open_popup_load(false)
        , project_file_drop_valid(false)
        , file_browser()
        , search_widget()
        , splitter_widget()
        , tooltip() {

    this->graph_state.hotkeys[megamol::gui::HotkeyIndex::MODULE_SEARCH] = {
        core::view::KeyCode(core::view::Key::KEY_M, (core::view::Modifier::CTRL | core::view::Modifier::SHIFT)), false};
    this->graph_state.hotkeys[megamol::gui::HotkeyIndex::PARAMETER_SEARCH] = {
        core::view::KeyCode(core::view::Key::KEY_P, (core::view::Modifier::CTRL | core::view::Modifier::SHIFT)), false};
    this->graph_state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM] = {
        core::view::KeyCode(core::view::Key::KEY_DELETE), false};
    this->graph_state.hotkeys[megamol::gui::HotkeyIndex::SAVE_PROJECT] = {
        megamol::core::view::KeyCode(core::view::Key::KEY_S, core::view::Modifier::CTRL | core::view::Modifier::SHIFT),
        false};
    this->graph_state.font_scalings = {0.85f, 0.95f, 1.0f, 1.5f, 2.5f};
    this->graph_state.graph_width = 0.0f;
    this->graph_state.show_parameter_sidebar = false;
    this->graph_state.graph_selected_uid = GUI_INVALID_ID;
    this->graph_state.graph_delete = false;
    this->graph_state.configurator_graph_save = false;
    this->graph_state.global_graph_save = false;
}


Configurator::~Configurator() {}


bool megamol::gui::Configurator::Draw(
    WindowCollection::WindowConfiguration& wc, megamol::core::CoreInstance* core_instance) {

    if (ImGui::GetCurrentContext() == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    if (core_instance == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Pointer to Core Instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Draw
    if (this->init_state < 2) {
        /// Step 1] (two frames!)

        // Show pop-up before calling this->graph_collection.LoadModulesCallsStock().
        /// Rendering of pop-up requires two complete draw calls!
        bool open = true;
        std::string popup_label("Loading");
        if (this->init_state == 0) {
            ImGui::OpenPopup(popup_label.c_str());
        }
        ImGuiWindowFlags popup_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove;
        if (ImGui::BeginPopupModal(popup_label.c_str(), &open, popup_flags)) {
            ImGui::TextUnformatted("Please wait...\nLoading available modules and calls for configurator.");
            ImGui::EndPopup();
        }

        this->init_state++;

    } else if (this->init_state == 2) {
        /// Step 2] (one frame)

        // Load available modules and calls (if they are not loaded already)
        this->graph_collection.LoadCallStock(core_instance);
        this->graph_collection.LoadModuleStock(core_instance);

        // Loading separate gui graph for running graph of core instance,
        // because initial gui graph is hidden. It should not be manipulated
        // since there is no synchronization for the core instance graph (yet)
        auto graph_count = this->graph_collection.GetGraphs().size();
        if (graph_count != 1) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Invalid expected number of graphs: %i (should be 1, for loaded running graph). [%s, %s, line "
                "%d]\n",
                graph_count, __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        // Load configurator state once after available modules and call are loaded
        auto graph_ptr = this->graph_collection.GetGraphs().front();
        this->load_graph_state_from_file(graph_ptr->GetFilename());

        this->init_state++;
    } else {
        /// Step 3]
        // Render configurator gui content

        // Update state -------------------------------------------------------

        // Process hotkeys
        /// SAVE_PROJECT
        if (this->graph_state.hotkeys[megamol::gui::HotkeyIndex::SAVE_PROJECT].is_pressed &&
            (this->graph_state.graph_selected_uid != GUI_INVALID_ID)) {

            bool graph_has_core_interface = false;
            GraphPtr_t graph_ptr;
            if (this->graph_collection.GetGraph(this->graph_state.graph_selected_uid, graph_ptr)) {
                graph_has_core_interface = graph_ptr->HasCoreInterface();
            }
            if (graph_has_core_interface) {
                this->graph_state.global_graph_save = true;
            } else {
                this->graph_state.configurator_graph_save = true;
            }
            this->graph_state.hotkeys[megamol::gui::HotkeyIndex::SAVE_PROJECT].is_pressed = false;
        }
        /// MODULE_SEARCH
        if (this->graph_state.hotkeys[megamol::gui::HotkeyIndex::MODULE_SEARCH].is_pressed) {

            this->search_widget.SetSearchFocus(true);
            this->graph_state.hotkeys[megamol::gui::HotkeyIndex::MODULE_SEARCH].is_pressed = false;
        }

        this->project_file_drop_valid = (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows));

        // Draw Windows -------------------------------------------------------

        // Menu
        this->draw_window_menu(core_instance);

        // Splitter
        if (megamol::gui::gui_scaling.PendingChange()) {
            this->module_list_sidebar_width *= megamol::gui::gui_scaling.TransitonFactor();
        }
        this->graph_state.graph_width = 0.0f;
        if (this->show_module_list_sidebar) {
            this->splitter_widget.Widget(SplitterWidget::FixedSplitterSide::LEFT, this->module_list_sidebar_width,
                this->graph_state.graph_width);

            // Module List
            this->draw_window_module_list(this->module_list_sidebar_width, 0.0f, !this->show_module_list_popup);
            ImGui::SameLine();
        }
        // Graphs
        this->graph_collection.PresentGUI(this->graph_state);

        // Process Pop-ups
        this->drawPopUps(core_instance);

        // Reset state --------------------------------------------------------

        // Only reset 'externally' processed hotkeys
        this->graph_state.hotkeys[megamol::gui::HotkeyIndex::PARAMETER_SEARCH].is_pressed = false;
        this->graph_state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM].is_pressed = false;
    }

    return true;
}


void megamol::gui::Configurator::draw_window_menu(megamol::core::CoreInstance* core_instance) {

    if (core_instance == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Pointer to Core Instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
    ImGui::PushID("Configurator::Menu");
    // Menu
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("New Project")) {
                this->graph_collection.AddEmptyProject();
            }

            if (ImGui::BeginMenu("Load Project")) {
                // Load project from LUA file
                if (ImGui::MenuItem("File")) {
                    this->add_project_graph_uid = GUI_INVALID_ID;
                    this->open_popup_load = true;
                }
                if (ImGui::MenuItem("Running")) {
                    this->graph_collection.LoadProjectFromCore(core_instance, nullptr);
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Add Project")) {
                // Add project from LUA file to current project
                if (ImGui::MenuItem("File", nullptr, false, (this->graph_state.graph_selected_uid != GUI_INVALID_ID))) {
                    this->add_project_graph_uid = this->graph_state.graph_selected_uid;
                    this->open_popup_load = true;
                }
                if (ImGui::MenuItem(
                        "Running", nullptr, false, (this->graph_state.graph_selected_uid != GUI_INVALID_ID))) {
                    this->graph_collection.AddUpdateProjectFromCore(
                        this->graph_state.graph_selected_uid, core_instance, nullptr, true);
                }
                ImGui::EndMenu();
            }

            // Save currently active project to LUA file
            if (ImGui::MenuItem("Save Project",
                    this->graph_state.hotkeys[megamol::gui::HotkeyIndex::SAVE_PROJECT].keycode.ToString().c_str(),
                    false, ((this->graph_state.graph_selected_uid != GUI_INVALID_ID)))) {
                bool graph_has_core_interface = false;
                GraphPtr_t graph_ptr;
                if (this->graph_collection.GetGraph(this->graph_state.graph_selected_uid, graph_ptr)) {
                    graph_has_core_interface = graph_ptr->HasCoreInterface();
                }
                if (graph_has_core_interface) {
                    this->graph_state.global_graph_save = true;
                } else {
                    this->graph_state.configurator_graph_save = true;
                }
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {
            if (ImGui::MenuItem("Modules Sidebar", nullptr, this->show_module_list_sidebar)) {
                this->show_module_list_sidebar = !this->show_module_list_sidebar;
            }
            if (ImGui::MenuItem("Parameter Sidebar", nullptr, this->graph_state.show_parameter_sidebar,
                    (this->graph_state.graph_selected_uid != GUI_INVALID_ID))) {
                this->graph_state.show_parameter_sidebar = !this->graph_state.show_parameter_sidebar;
            }
            ImGui::EndMenu();
        }

        ImGui::SameLine();

        ImGui::EndMenuBar();
    }
    ImGui::PopID();
}


void megamol::gui::Configurator::draw_window_module_list(float width, float height, bool apply_focus) {

    ImGui::BeginGroup();

    const float search_child_height = ImGui::GetFrameHeightWithSpacing() * 2.5f;
    auto child_flags = ImGuiWindowFlags_AlwaysUseWindowPadding | ImGuiWindowFlags_NoScrollbar;
    ImGui::BeginChild("module_search_child_window", ImVec2(width, search_child_height), false, child_flags);

    ImGui::TextUnformatted("Available Modules");
    ImGui::Separator();

    std::string help_text = "[" +
                            this->graph_state.hotkeys[megamol::gui::HotkeyIndex::MODULE_SEARCH].keycode.ToString() +
                            "] Set keyboard focus to search input field.\n"
                            "Case insensitive substring search in module names.";
    this->search_widget.Widget("configurator_module_search", help_text, apply_focus);
    auto search_string = this->search_widget.GetSearchString();

    ImGui::EndChild();

    // ------------------------------------------------------------------------

    child_flags = ImGuiWindowFlags_None;
    ImGui::BeginChild(
        "module_list_child_window", ImVec2(width, std::max(0.0f, height - search_child_height)), true, child_flags);

    bool search_filter = true;
    bool compat_filter = true;

    bool interfaceslot_selected = false;
    std::string compat_callslot_name;
    CallSlotPtr_t selected_callslot_ptr;
    GraphPtr_t selected_graph_ptr;
    if (this->graph_collection.GetGraph(this->graph_state.graph_selected_uid, selected_graph_ptr)) {
        auto callslot_id = selected_graph_ptr->present.GetSelectedCallSlot();
        if (callslot_id != GUI_INVALID_ID) {
            for (auto& module_ptr : selected_graph_ptr->GetModules()) {
                CallSlotPtr_t callslot_ptr;
                if (module_ptr->GetCallSlot(callslot_id, callslot_ptr)) {
                    selected_callslot_ptr = callslot_ptr;
                }
            }
        }
        auto interfaceslot_id = selected_graph_ptr->present.GetSelectedInterfaceSlot();
        if (interfaceslot_id != GUI_INVALID_ID) {
            for (auto& group_ptr : selected_graph_ptr->GetGroups()) {
                InterfaceSlotPtr_t interfaceslot_ptr;
                if (group_ptr->GetInterfaceSlot(interfaceslot_id, interfaceslot_ptr)) {
                    CallSlotPtr_t callslot_ptr;
                    if (interfaceslot_ptr->GetCompatibleCallSlot(callslot_ptr)) {
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
        search_filter = true;
        if (!search_string.empty()) {
            search_filter = megamol::gui::GUIUtils::FindCaseInsensitiveSubstring(mod.class_name, search_string);
        }

        // Filter module by compatible call slots
        compat_filter = true;
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
            ImGui::PushID(id);

            std::string label = mod.class_name + " (" + mod.plugin_name + ")";
            if (mod.is_view) {
                label += " [View]";
            }
            if (ImGui::Selectable(label.c_str(), (id == this->selected_list_module_uid))) {
                this->selected_list_module_uid = id;
            }
            bool add_module = false;
            // Left mouse button double click action
            if ((ImGui::IsMouseDoubleClicked(0) && ImGui::IsItemHovered()) ||
                (ImGui::IsItemFocused() && ImGui::IsItemActivated())) {
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
                if (selected_graph_ptr != nullptr) {
                    ImGuiID module_uid =
                        selected_graph_ptr->AddModule(this->graph_collection.GetModulesStock(), mod.class_name);
                    ModulePtr_t module_ptr;
                    if (selected_graph_ptr->GetModule(module_uid, module_ptr)) {

                        // If there is a call slot selected, create call to compatible call slot of new module
                        bool added_call = false;
                        if (compat_filter && (selected_callslot_ptr != nullptr)) {
                            // Get call slots of last added module
                            for (auto& callslot_map : module_ptr->GetCallSlots()) {
                                for (auto& callslot_ptr : callslot_map.second) {
                                    if (callslot_ptr->name == compat_callslot_name) {
                                        added_call = selected_graph_ptr->AddCall(this->graph_collection.GetCallsStock(),
                                            selected_callslot_ptr, callslot_ptr);
                                        if (added_call) {
                                            module_ptr->present.SetSelectedSlotPosition();
                                        }
                                    }
                                }
                            }
                        }
                        // Place new module at mouse pos if added via separate module list child window.
                        else if (this->show_module_list_popup) {
                            module_ptr->present.SetScreenPosition(ImGui::GetMousePos());
                        }

                        // If there is a group selected or hoverd or the new call is connceted to module which is part
                        // of group, add module to this group
                        if (!interfaceslot_selected) {
                            ImGuiID connceted_group = GUI_INVALID_ID;
                            if (added_call && selected_callslot_ptr->IsParentModuleConnected()) {
                                connceted_group = selected_callslot_ptr->GetParentModule()->present.group.uid;
                            }
                            ImGuiID selected_group_uid = selected_graph_ptr->present.GetSelectedGroup();
                            ImGuiID group_uid = (connceted_group != GUI_INVALID_ID)
                                                    ? (connceted_group)
                                                    : ((selected_group_uid != GUI_INVALID_ID)
                                                              ? (selected_group_uid)
                                                              : (this->module_list_popup_hovered_group_uid));

                            if (group_uid != GUI_INVALID_ID) {
                                for (auto& group_ptr : selected_graph_ptr->GetGroups()) {
                                    if (group_ptr->uid == group_uid) {
                                        Graph::QueueData queue_data;
                                        queue_data.name_id = module_ptr->FullName();
                                        selected_graph_ptr->present.ResetStatePointers();
                                        group_ptr->AddModule(module_ptr);
                                        queue_data.rename_id = module_ptr->FullName();
                                        selected_graph_ptr->PushSyncQueue(
                                            Graph::QueueAction::RENAME_MODULE, queue_data);
                                    }
                                }
                            }
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


bool megamol::gui::Configurator::StateToJSON(nlohmann::json& inout_json) {

    try {
        // Write configurator state
        inout_json[GUI_JSON_TAG_CONFIGURATOR]["show_module_list_sidebar"] = this->show_module_list_sidebar;
        inout_json[GUI_JSON_TAG_CONFIGURATOR]["module_list_sidebar_width"] = this->module_list_sidebar_width;

        /// Make sure configurator processed loading of modules and calls
        if (this->init_state > 1) {

            // Write graph states
            for (auto& graph_ptr : this->GetGraphCollection().GetGraphs()) {
                // For graphs with no interface to core save only file name of loaded project
                if (graph_ptr->HasCoreInterface()) {
                    graph_ptr->StateToJSON(inout_json);
                } else {
                    std::string filename = graph_ptr->GetFilename();
                    GUIUtils::Utf8Encode(filename);
                    if (!filename.empty()) {
                        inout_json[GUI_JSON_TAG_GRAPHS][filename] = nlohmann::json::object();
                    }
                }
            }
        }

#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Wrote configurator state to JSON.");
#endif // GUI_VERBOSE

    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON Error - Unable to write state to JSON. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


bool megamol::gui::Configurator::StateFromJSON(const nlohmann::json& in_json) {

    try {
        if (!in_json.is_object()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Invalid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
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

        /// Make sure configurator processed loading of modules and calls
        if (this->init_state > 1) {

            // Read graph states
            for (auto& graph_ptr : this->GetGraphCollection().GetGraphs()) {
                if (graph_ptr->HasCoreInterface()) {
                    if (graph_ptr->StateFromJSON(in_json)) {
                        // Disable layouting if graph state was found
                        graph_ptr->present.SetLayoutGraph(false);
                    }
                }
            }
            for (auto& graph_header_item : in_json.items()) {
                if (graph_header_item.key() == GUI_JSON_TAG_GRAPHS) {
                    for (auto& graph_item : graph_header_item.value().items()) {
                        std::string json_graph_id = graph_item.key();
                        GUIUtils::Utf8Decode(json_graph_id);
                        if (json_graph_id != GUI_JSON_TAG_PROJECT) {
                            // Otherwise load additonal graph from given file name
                            this->GetGraphCollection().LoadAddProjectFromFile(GUI_INVALID_ID, json_graph_id);
                        }
                    }
                }
            }
        }
#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Read configurator state from JSON.");
#endif // GUI_VERBOSE

    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON Error - Unable to read state from JSON. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


void megamol::gui::Configurator::drawPopUps(megamol::core::CoreInstance* core_instance) {

    bool confirmed, aborted;

    // Load Project -----------------------------------------------------------
    bool popup_failed = false;
    std::string project_filename;
    GraphPtr_t graph_ptr;
    if (this->graph_collection.GetGraph(this->graph_state.graph_selected_uid, graph_ptr)) {
        project_filename = graph_ptr->GetFilename();
    }
    // Try to get current project path from lua in core instance
    if (project_filename.empty() && (core_instance != nullptr)) {
        project_filename = core_instance->GetLuaState()->GetScriptPath();
    }
    if (this->file_browser.PopUp(
            project_filename, FileBrowserWidget::FileBrowserFlag::LOAD, "Load Project", this->open_popup_load, "lua")) {
        popup_failed = (GUI_INVALID_ID ==
                        this->graph_collection.LoadAddProjectFromFile(this->add_project_graph_uid, project_filename));
        this->add_project_graph_uid = GUI_INVALID_ID;
    }
    MinimalPopUp::PopUp("Failed to Load Project", popup_failed, "See console log output for more information.", "",
        confirmed, "Cancel", aborted);
    this->open_popup_load = false;

    // Module Stock List Child Window ------------------------------------------
    GraphPtr_t selected_graph_ptr;
    if (this->graph_collection.GetGraph(this->graph_state.graph_selected_uid, selected_graph_ptr)) {

        ImGuiID selected_callslot_uid = selected_graph_ptr->present.GetSelectedCallSlot();
        ImGuiID selected_group_uid = selected_graph_ptr->present.GetSelectedGroup();

        bool valid_double_click = (ImGui::IsMouseDoubleClicked(0) && selected_graph_ptr->present.IsCanvasHoverd() &&
                                   (selected_group_uid == GUI_INVALID_ID) && (!this->show_module_list_popup));
        bool valid_double_click_callslot =
            (ImGui::IsMouseDoubleClicked(0) && selected_graph_ptr->present.IsCanvasHoverd() &&
                (selected_callslot_uid != GUI_INVALID_ID) &&
                ((!this->show_module_list_popup) || (this->last_selected_callslot_uid != selected_callslot_uid)));

        if (valid_double_click || valid_double_click_callslot) {
            this->graph_state.hotkeys[megamol::gui::HotkeyIndex::MODULE_SEARCH].is_pressed = true;
            this->last_selected_callslot_uid = selected_callslot_uid;
            // Force consume double click!
            ImGui::GetIO().MouseDoubleClicked[0] = false;
        }
    }
    if ( // !this->show_module_list_sidebar &&
        this->graph_state.hotkeys[megamol::gui::HotkeyIndex::MODULE_SEARCH].is_pressed) {
        this->show_module_list_popup = true;
        this->module_list_popup_pos = ImGui::GetMousePos();
        this->module_list_popup_hovered_group_uid = selected_graph_ptr->present.GetHoveredGroup();
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

            float diff_width = (ImGui::GetWindowPos().x + ImGui::GetWindowSize().x - this->module_list_popup_pos.x);
            float diff_height = (ImGui::GetWindowPos().y + ImGui::GetWindowSize().y - this->module_list_popup_pos.y);
            if (diff_width < popup_width) {
                this->module_list_popup_pos.x -= ((popup_width - diff_width) + offset_x);
            }
            this->module_list_popup_pos.x = std::max(this->module_list_popup_pos.x, ImGui::GetWindowPos().x);
            if (diff_height < popup_height) {
                this->module_list_popup_pos.y -= ((popup_height - diff_height) + offset_y);
            }
            this->module_list_popup_pos.y = std::max(this->module_list_popup_pos.y, ImGui::GetWindowPos().y);
            ImGui::SetNextWindowPos(this->module_list_popup_pos);
            ImGui::SetNextWindowSize(ImVec2(10.0f, 10.0f));
        }
        auto popup_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar |
                           ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove;
        if (ImGui::BeginPopup(pop_up_id.c_str(), popup_flags)) {

            this->draw_window_module_list(
                std::max(0.0f, (popup_width - offset_x)), std::max(0.0f, (popup_height - offset_y)), true);

            bool module_list_popup_hovered = false;
            if ((ImGui::GetMousePos().x >= this->module_list_popup_pos.x) &&
                (ImGui::GetMousePos().x <= (this->module_list_popup_pos.x + popup_width)) &&
                (ImGui::GetMousePos().y >= this->module_list_popup_pos.y) &&
                (ImGui::GetMousePos().y <= (this->module_list_popup_pos.y + popup_height))) {
                module_list_popup_hovered = true;
            }
            if ((ImGui::IsMouseClicked(0) && !module_list_popup_hovered) ||
                ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape))) {
                this->show_module_list_popup = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }
}


bool megamol::gui::Configurator::load_graph_state_from_file(const std::string& filename) {

    std::string state_str;
    if (megamol::core::utility::FileUtils::ReadFile(filename, state_str, true)) {
        state_str = GUIUtils::ExtractGUIState(state_str);
        if (state_str.empty())
            return false;
        nlohmann::json in_json = nlohmann::json::parse(state_str);
        return this->StateFromJSON(in_json);
    }

    return false;
}
