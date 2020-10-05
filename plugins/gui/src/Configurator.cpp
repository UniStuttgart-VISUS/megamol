/*
 * Configurator.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

/**
 * USED HOTKEYS:
 *
 * - Search Module:        Ctrl + Shift + m
 * - Search Parameter:     Ctrl + Shift + p
 * - Save Edited Project:  Ctrl + Shift + s
 * - Delete Graph Item:    Delete
 */

#include "stdafx.h"
#include "Configurator.h"


using namespace megamol;
using namespace megamol::gui;


std::vector<std::string> megamol::gui::Configurator::dropped_files;


megamol::gui::Configurator::Configurator()
    : graph_collection()
    , param_slots()
    , state_param(GUI_CONFIGURATOR_STATE_PARAM_NAME, "State of the configurator.")
    , init_state(0)
    , module_list_sidebar_width(250.0f)
    , selected_list_module_uid(GUI_INVALID_ID)
    , add_project_graph_uid(GUI_INVALID_ID)
    , module_list_popup_hovered_group_uid(GUI_INVALID_ID)
    , show_module_list_sidebar(false)
    , show_module_list_child(false)
    , module_list_popup_pos()
    , module_list_popup_hovered(false)
    , last_selected_callslot_uid(GUI_INVALID_ID)
    , graph_state()
    , open_popup_load(false)
    , file_browser()
    , search_widget()
    , splitter_widget()
    , tooltip() {

    this->state_param << new core::param::StringParam("");
    this->state_param.Parameter()->SetGUIVisible(false);
    this->state_param.Parameter()->SetGUIReadOnly(true);

    this->param_slots.clear();
    this->param_slots.push_back(&this->state_param);

    this->graph_state.hotkeys[megamol::gui::HotkeyIndex::MODULE_SEARCH] = megamol::gui::HotkeyData_t(
        core::view::KeyCode(core::view::Key::KEY_M, (core::view::Modifier::CTRL | core::view::Modifier::SHIFT)), false);
    this->graph_state.hotkeys[megamol::gui::HotkeyIndex::PARAMETER_SEARCH] = megamol::gui::HotkeyData_t(
        core::view::KeyCode(core::view::Key::KEY_P, (core::view::Modifier::CTRL | core::view::Modifier::SHIFT)), false);
    this->graph_state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM] =
        megamol::gui::HotkeyData_t(core::view::KeyCode(core::view::Key::KEY_DELETE), false);
    this->graph_state.hotkeys[megamol::gui::HotkeyIndex::SAVE_PROJECT] = megamol::gui::HotkeyData_t(
        megamol::core::view::KeyCode(core::view::Key::KEY_S, core::view::Modifier::CTRL | core::view::Modifier::SHIFT),
        false);
    this->graph_state.font_scalings = {0.85f, 0.95f, 1.0f, 1.5f, 2.5f};
    this->graph_state.graph_width = 0.0f;
    this->graph_state.show_parameter_sidebar = false;
    this->graph_state.graph_selected_uid = GUI_INVALID_ID;
    this->graph_state.graph_delete = false;
    this->graph_state.graph_save = false;
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

        // Load available modules and calls and currently loaded project from core once(!)
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
        }

        // Enable drag and drop of files for configurator (if glfw is available here)
#ifdef GUI_USE_GLFW
        auto glfw_win = ::glfwGetCurrentContext();
        ::glfwSetDropCallback(glfw_win, this->file_drop_callback);
#endif

        this->init_state++;
    } else {
        /// Step 3]
        // Render configurator gui content

        // Update state -------------------------------------------------------
        // Check for configurator parameter changes
        if (this->state_param.IsDirty()) {
            std::string state = std::string(this->state_param.Param<core::param::StringParam>()->Value().PeekBuffer());
            this->configurator_state_from_json_string(state);
            this->state_param.ResetDirty();
        }
        // Hotkeys
        if (std::get<1>(this->graph_state.hotkeys[megamol::gui::HotkeyIndex::SAVE_PROJECT]) &&
            (this->graph_state.graph_selected_uid != GUI_INVALID_ID)) {

            bool graph_has_core_interface = false;
            GraphPtr_t graph_ptr;
            if (this->graph_collection.GetGraph(this->graph_state.graph_selected_uid, graph_ptr)) {
                graph_has_core_interface = graph_ptr->HasCoreInterface();
            }
            this->graph_state.graph_save = !graph_has_core_interface;
        }

        // Clear dropped file list (when configurator window is opened, after it was closed)
        if (ImGui::IsWindowAppearing()) {
            megamol::gui::Configurator::dropped_files.clear();
        }
        // Process dropped files ...
        if (!megamol::gui::Configurator::dropped_files.empty()) {
            // ... only if configurator is focused.
            if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
                for (auto& dropped_file : megamol::gui::Configurator::dropped_files) {
                    this->graph_collection.LoadAddProjectFromFile(this->graph_state.graph_selected_uid, dropped_file);
                }
            }
            megamol::gui::Configurator::dropped_files.clear();
        }

        // Draw Windows -------------------------------------------------------

        // Menu
        this->draw_window_menu(core_instance);

        // Splitter
        this->graph_state.graph_width = 0.0f;
        if (this->show_module_list_sidebar) {
            this->splitter_widget.Widget(SplitterWidget::FixedSplitterSide::LEFT, this->module_list_sidebar_width,
                this->graph_state.graph_width);

            // Module List
            this->draw_window_module_list(this->module_list_sidebar_width);
            ImGui::SameLine();
        }
        // Graphs
        this->graph_collection.PresentGUI(this->graph_state);

        // Process Pop-ups
        this->drawPopUps();

        // Reset state -------------------------------------------------------
        for (auto& h : this->graph_state.hotkeys) {
            std::get<1>(h) = false;
        }
    }

    return true;
}


void megamol::gui::Configurator::UpdateStateParameter(void) {

    // Save current state of configurator to state parameter
    nlohmann::json configurator_json;
    if (this->configurator_state_to_json(configurator_json)) {
        std::string state;

        try {
            state = configurator_json.dump(2);
        } catch (std::exception e) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
            return;
        } catch (...) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Unknown Error - Unable to dump JSON string. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
            return;
        }

        this->state_param.Param<core::param::StringParam>()->SetValue(state.c_str(), false);
    }
}


void megamol::gui::Configurator::draw_window_menu(megamol::core::CoreInstance* core_instance) {

    if (core_instance == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Pointer to Core Instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

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
            bool graph_has_core_interface = false;
            GraphPtr_t graph_ptr;
            if (this->graph_collection.GetGraph(this->graph_state.graph_selected_uid, graph_ptr)) {
                graph_has_core_interface = graph_ptr->HasCoreInterface();
            }
            bool enable_save_graph = !graph_has_core_interface;
            if (ImGui::MenuItem("Save Editor Project",
                    std::get<0>(this->graph_state.hotkeys[megamol::gui::HotkeyIndex::SAVE_PROJECT]).ToString().c_str(),
                    false, ((this->graph_state.graph_selected_uid != GUI_INVALID_ID) && enable_save_graph))) {
                this->graph_state.graph_save = true;
            }
            if (!enable_save_graph) {
                this->tooltip.ToolTip("Save running project using global menu.");
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

        if (ImGui::BeginMenu("Help")) {
            const std::string docu_link =
                "https://github.com/UniStuttgart-VISUS/megamol/tree/master/plugins/gui#configurator";
            if (ImGui::Button("Readme on GitHub (Copy Link)")) {
#ifdef GUI_USE_GLFW
                auto glfw_win = ::glfwGetCurrentContext();
                ::glfwSetClipboardString(glfw_win, docu_link.c_str());
#elif _WIN32
                ImGui::SetClipboardText(docu_link.c_str());
#else // LINUX
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "[GUI] No clipboard use provided. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Readme Link:\n%s", docu_link.c_str());
#endif
            }
            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }
}


void megamol::gui::Configurator::draw_window_module_list(float width) {

    ImGui::BeginGroup();

    const float search_child_height = ImGui::GetFrameHeightWithSpacing() * 2.5f;
    auto child_flags =
        ImGuiWindowFlags_AlwaysUseWindowPadding | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NavFlattened;
    ImGui::BeginChild("module_search_child_window", ImVec2(width, search_child_height), false, child_flags);

    ImGui::TextUnformatted("Available Modules");
    ImGui::Separator();

    if (std::get<1>(this->graph_state.hotkeys[megamol::gui::HotkeyIndex::MODULE_SEARCH])) {
        this->search_widget.SetSearchFocus(true);
    }
    std::string help_text =
        "[" + std::get<0>(this->graph_state.hotkeys[megamol::gui::HotkeyIndex::MODULE_SEARCH]).ToString() +
        "] Set keyboard focus to search input field.\n"
        "Case insensitive substring search in module names.";
    this->search_widget.Widget("configurator_module_search", help_text);
    auto search_string = this->search_widget.GetSearchString();

    ImGui::EndChild();

    // ------------------------------------------------------------------------

    child_flags = ImGuiWindowFlags_NavFlattened;
    ImGui::BeginChild("module_list_child_window", ImVec2(width, 0.0f), true, child_flags);

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
            search_filter = StringSearchWidget::FindCaseInsensitiveSubstring(mod.class_name, search_string);
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
                        else if (this->show_module_list_child) {
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
                                        selected_graph_ptr->present.ResetStatePointers();
                                        group_ptr->AddModule(module_ptr);
                                    }
                                }
                            }
                        }
                    }
                    this->show_module_list_child = false;
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


bool megamol::gui::Configurator::configurator_state_from_json_string(const std::string& in_json_string) {

    try {
        if (in_json_string.empty()) {
            return false;
        }
        nlohmann::json json;
        json = nlohmann::json::parse(in_json_string);
        if (!json.is_object()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] State is no valid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        bool found = false;
        for (auto& header_item : json.items()) {
            if (header_item.key() == GUI_JSON_TAG_CONFIGURATOR) {
                auto config_state = header_item.value();
                found = true;

                megamol::core::utility::get_json_value<bool>(
                    config_state, {"show_module_list_sidebar"}, &this->show_module_list_sidebar);

                megamol::core::utility::get_json_value<float>(
                    config_state, {"module_list_sidebar_width"}, &this->module_list_sidebar_width);

            } else if (header_item.key() == GUI_JSON_TAG_GRAPHS) {
                // Check for configurator settings of previously loaded graphs
                for (auto& config_item : header_item.value().items()) {
                    std::string json_graph_id = config_item.key();
                    GUIUtils::Utf8Decode(json_graph_id);
                    if (json_graph_id == GUI_JSON_TAG_PROJECT_GRAPH) {
                        // Read configurator state for graph connected to core
                        for (auto& graph_ptr : this->graph_collection.GetGraphs()) {
                            if (graph_ptr->HasCoreInterface()) {
                                if (graph_ptr->StateFromJsonString(in_json_string)) {
                                    // Disable layouting if graph state was found
                                    graph_ptr->present.SetLayoutGraph(false);
                                }
                                break;
                            }
                        }
                    } else {
                        // Otherwise load additonal graph from given file name
                        auto graph_uid = this->graph_collection.LoadAddProjectFromFile(GUI_INVALID_ID, json_graph_id);
                    }
                }
            }
        }

        if (found) {
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Read configurator state from JSON string.");
#endif // GUI_VERBOSE
        } else {
            return false;
        }
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error - Unable to parse JSON string. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


bool megamol::gui::Configurator::configurator_state_to_json(nlohmann::json& out_json) {

    try {
        out_json[GUI_JSON_TAG_CONFIGURATOR]["show_module_list_sidebar"] = this->show_module_list_sidebar;
        out_json[GUI_JSON_TAG_CONFIGURATOR]["module_list_sidebar_width"] = this->module_list_sidebar_width;

        for (auto& graph_ptr : this->graph_collection.GetGraphs()) {
            graph_ptr->StateToJSON(out_json);
        }
#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Wrote configurator state to JSON.");
#endif // GUI_VERBOSE

    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error - Unable to write JSON of state. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
        return false;
    }

    return true;
}


void megamol::gui::Configurator::drawPopUps(void) {

    bool confirmed, aborted;

    // Pop-ups-----------------------------------
    // LOAD
    bool popup_failed = false;
    std::string project_filename;
    GraphPtr_t graph_ptr;
    if (this->graph_collection.GetGraph(this->add_project_graph_uid, graph_ptr)) {
        project_filename = graph_ptr->GetFilename();
    }
    if (this->file_browser.PopUp(
            FileBrowserWidget::FileBrowserFlag::LOAD, "Load Project", this->open_popup_load, project_filename)) {
        popup_failed = !this->graph_collection.LoadAddProjectFromFile(this->add_project_graph_uid, project_filename);
        this->add_project_graph_uid = GUI_INVALID_ID;
    }
    MinimalPopUp::PopUp("Failed to Load Project", popup_failed, "See console log output for more information.", "",
        confirmed, "Cancel", aborted);
    this->open_popup_load = false;

    // Module Stock List Child Window ------------------------------------------
    GraphPtr_t selected_graph_ptr;
    if (this->graph_collection.GetGraph(this->graph_state.graph_selected_uid, selected_graph_ptr)) {

        if (this->show_module_list_child && ((ImGui::IsMouseClicked(0) && !this->module_list_popup_hovered) ||
                                                ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape)))) {
            this->show_module_list_child = false;
        }

        ImGuiID selected_callslot_uid = selected_graph_ptr->present.GetSelectedCallSlot();
        ImGuiID selected_group_uid = selected_graph_ptr->present.GetSelectedGroup();

        bool valid_double_click = (ImGui::IsMouseDoubleClicked(0) && selected_graph_ptr->present.IsCanvasHoverd() &&
                                   (selected_group_uid == GUI_INVALID_ID) && (!this->show_module_list_child));
        bool valid_double_click_callslot =
            (ImGui::IsMouseDoubleClicked(0) && selected_graph_ptr->present.IsCanvasHoverd() &&
                (selected_callslot_uid != GUI_INVALID_ID) &&
                ((!this->show_module_list_child) || (this->last_selected_callslot_uid != selected_callslot_uid)));

        if (valid_double_click || valid_double_click_callslot) {
            std::get<1>(this->graph_state.hotkeys[megamol::gui::HotkeyIndex::MODULE_SEARCH]) = true;
            this->last_selected_callslot_uid = selected_callslot_uid;
            // Force consume double click!
            ImGui::GetIO().MouseDoubleClicked[0] = false;
        }
    }
    if (std::get<1>(this->graph_state.hotkeys[megamol::gui::HotkeyIndex::MODULE_SEARCH])) {
        this->show_module_list_child = true;
        this->module_list_popup_pos = ImGui::GetMousePos();
        this->module_list_popup_hovered_group_uid = selected_graph_ptr->present.GetHoveredGroup();
    }
    if (this->show_module_list_child) {
        ImGuiStyle& style = ImGui::GetStyle();
        ImVec4 tmpcol = style.Colors[ImGuiCol_ChildBg];
        tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, tmpcol);
        ImGui::SetCursorScreenPos(this->module_list_popup_pos);
        const float child_width = 250.0f;
        const float child_height = 350.0f;
        float diff_width = (ImGui::GetWindowPos().x + ImGui::GetWindowSize().x - this->module_list_popup_pos.x);
        float diff_height = (ImGui::GetWindowPos().y + ImGui::GetWindowSize().y - this->module_list_popup_pos.y);
        if (diff_width < child_width) {
            this->module_list_popup_pos.x -= (child_width - diff_width);
        }
        this->module_list_popup_pos.x = std::max(this->module_list_popup_pos.x, ImGui::GetWindowPos().x);
        if (diff_height < child_height) {
            this->module_list_popup_pos.y -= (child_height - diff_height);
        }
        this->module_list_popup_pos.y = std::max(this->module_list_popup_pos.y, ImGui::GetWindowPos().y);
        ImGui::SetCursorScreenPos(this->module_list_popup_pos);
        auto child_flags = ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NavFlattened;
        ImGui::BeginChild("module_list_child", ImVec2(child_width, child_height), true, child_flags);
        /// if (ImGui::Button("Close") || ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape))) {
        ///    this->show_module_list_child = false;
        ///}
        /// ImGui::Separator();
        this->draw_window_module_list(0.0f);
        ImGui::EndChild();
        ImGui::PopStyleColor();
        this->module_list_popup_hovered = false;
        if ((ImGui::GetMousePos().x >= this->module_list_popup_pos.x) &&
            (ImGui::GetMousePos().x <= (this->module_list_popup_pos.x + child_width)) &&
            (ImGui::GetMousePos().y >= this->module_list_popup_pos.y) &&
            (ImGui::GetMousePos().y <= (this->module_list_popup_pos.y + child_height))) {
            this->module_list_popup_hovered = true;
        }
    }
}


#ifdef GUI_USE_GLFW
void megamol::gui::Configurator::file_drop_callback(::GLFWwindow* window, int count, const char* paths[]) {

    for (int i = 0; i < count; i++) {
        megamol::gui::Configurator::dropped_files.emplace_back(std::string(paths[i]));
    }
}
#endif
