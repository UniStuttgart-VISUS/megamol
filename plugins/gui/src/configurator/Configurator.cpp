/*
 * Configurator.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

/**
 * USED HOTKEYS:
 *
 * - Search module:        Shift + Ctrl + m
 * - Search parameter:     Shift + Ctrl + p
 * - Delete module/call:   Delete
 */

#include "stdafx.h"
#include "Configurator.h"


using namespace megamol;
using namespace megamol::gui;
using namespace megamol::gui::configurator;


megamol::gui::configurator::Configurator::Configurator()
    : hotkeys()
    , graph_manager()
    , utils()
    , window_state(0)
    , project_filename("")
    , graph_uid(GUI_INVALID_ID)
    , selected_list_module_uid(GUI_INVALID_ID)
    , graph_font(nullptr)
    , child_split_width(250.0f)
    , add_project_graph_uid(GUI_INVALID_ID)
    , project_uid(0) {

    // Define HotKeys
    this->hotkeys[HotkeyIndex::MODULE_SEARCH] =
        HotkeyDataType(megamol::core::view::KeyCode(
                           megamol::core::view::Key::KEY_M, core::view::Modifier::CTRL | core::view::Modifier::SHIFT),
            false);
    this->hotkeys[HotkeyIndex::PARAMETER_SEARCH] =
        HotkeyDataType(megamol::core::view::KeyCode(
                           megamol::core::view::Key::KEY_P, core::view::Modifier::CTRL | core::view::Modifier::SHIFT),
            false);
    this->hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM] =
        HotkeyDataType(megamol::core::view::KeyCode(megamol::core::view::Key::KEY_DELETE), false);
}


Configurator::~Configurator() {}


bool megamol::gui::configurator::Configurator::CheckHotkeys(void) {

    if (ImGui::GetCurrentContext() == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    ImGuiIO& io = ImGui::GetIO();

    bool hotkey_pressed = false;
    for (auto& h : this->hotkeys) {
        auto key = std::get<0>(h).key;
        auto mods = std::get<0>(h).mods;
        if (ImGui::IsKeyDown(static_cast<int>(key)) && (mods.test(core::view::Modifier::CTRL) == io.KeyCtrl) &&
            (mods.test(core::view::Modifier::ALT) == io.KeyAlt) &&
            (mods.test(core::view::Modifier::SHIFT) == io.KeyShift)) {
            std::get<1>(h) = true;
            hotkey_pressed = true;
        }
    }

    return hotkey_pressed;
}


bool megamol::gui::configurator::Configurator::Draw(
    WindowManager::WindowConfiguration& wc, megamol::core::CoreInstance* core_instance) {

    if (core_instance == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to Core Instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (ImGui::GetCurrentContext() == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    if (this->window_state < 2) {
        // 1] Show pop-up before calling UpdateAvailableModulesCallsOnce of graph.

        /// Rendering of pop-up requires two complete Draw calls!
        bool open = true;
        std::string popup_label = "Loading";
        if (this->window_state == 0) {
            ImGui::OpenPopup(popup_label.c_str());
        }
        ImGuiWindowFlags popup_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove;
        if (ImGui::BeginPopupModal(popup_label.c_str(), &open, popup_flags)) {
            ImGui::Text("Please wait...\nLoading available modules and calls for configurator.");
            ImGui::EndPopup();
        }
        this->window_state++;

    } else if (this->window_state == 2) {
        // 2] Load available modules and calls and currently loaded project from core once(!)

        this->graph_manager.UpdateModulesCallsStock(core_instance);
        this->add_empty_project();
        this->window_state++;
    } else {
        // 3] Render configurator gui content

        this->draw_window_menu(core_instance);

        float child_width_auto = 0.0f;
        this->utils.VerticalSplitter(GUIUtils::FixedSplitterSide::LEFT, this->child_split_width, child_width_auto);

        this->draw_window_module_list(this->child_split_width);

        ImGui::SameLine();

        this->graph_uid = this->graph_manager.GUI_Present(child_width_auto, this->graph_font, this->hotkeys);
    }

    return true;
}


void megamol::gui::configurator::Configurator::draw_window_menu(megamol::core::CoreInstance* core_instance) {

    if (core_instance == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to Core Instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    bool popup_save_file = false;
    bool popup_load_file = false;

    if (ImGui::BeginMenuBar()) {

        if (ImGui::BeginMenu("File")) {

            if (ImGui::BeginMenu("New Project")) {

                if (ImGui::MenuItem("Empty", nullptr)) {
                    this->add_empty_project();
                }

#ifdef GUI_USE_FILESYSTEM
                // Load project from LUA file
                if (ImGui::MenuItem("LUA File", nullptr)) {
                    this->add_project_graph_uid = GUI_INVALID_ID;
                    popup_load_file = true;
                }
#endif // GUI_USE_FILESYSTEM

                if (ImGui::MenuItem("Running Project")) {
                    this->graph_manager.LoadProjectCore(this->get_unique_project_name(), core_instance);
                    // this->GetCoreInstance()->LoadProject(vislib::StringA(projectFilename.c_str()));
                }

                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Add Project")) {

#ifdef GUI_USE_FILESYSTEM
                // Add project from LUA file to current project
                if (ImGui::MenuItem("LUA File", nullptr, false, (this->graph_uid != GUI_INVALID_ID))) {
                    this->add_project_graph_uid = this->graph_uid;
                    popup_load_file = true;
                }
#endif // GUI_USE_FILESYSTEM

                if (ImGui::MenuItem("Running Project", nullptr, false, (this->graph_uid != GUI_INVALID_ID))) {
                    this->graph_manager.AddProjectCore(this->graph_uid, core_instance);
                    // this->GetCoreInstance()->LoadProject(vislib::StringA(projectFilename.c_str()));
                }

                ImGui::EndMenu();
            }

#ifdef GUI_USE_FILESYSTEM
            // Save currently active project to LUA file
            if (ImGui::MenuItem("Save Project", nullptr, false, (this->graph_uid != GUI_INVALID_ID))) {
                popup_save_file = true;
            }
#endif // GUI_USE_FILESYSTEM

            ImGui::EndMenu();
        }

        ImGui::SameLine();
        std::string info_text = "----- Additonal Options -----\n"
                                "- Add Module from Stock List to Graph\n"
                                "    - [Double Left Click]\n"
                                "    - [Richt Click] on Selected Module -> Context Menu: Add\n"
                                "- Delete Selected Module/Call from Graph\n"
                                "    - Select item an press [Delete]\n"
                                "    - [Richt Click] on Selected Item -> Context Menu: Delete\n"
                                "- Rename Graph or Module\n"
                                "    - [Richt Click] on Graph Tab or Module -> Context Menu: Rename\n"
                                "- Collapse/Expand Splitter\n"
                                "    - [Double Richt Click] on Splitter\n"
                                "- Create Call between Module Slots\n"
                                "    - Select Slot and Drag&Drop Call to other Highlighted Compatible Slot.";
        this->utils.HelpMarkerToolTip(info_text.c_str(), "[?]");

        // Info text ----------------------------------------------------------
        ImGui::SameLine(260.0f);
        std::string label = "This is a PROTOTYPE. Changes will NOT effect the currently loaded MegaMol project.";
        ImGui::TextColored(ImVec4(1.0f, 0.25f, 0.25f, 1.0f), label.c_str());

        ImGui::EndMenuBar();
    }

    // SAVE/LOAD PROJECT pop-up
#ifdef GUI_USE_FILESYSTEM
    bool popup_save_failed = false;
    bool popup_load_failed = false;
    if (this->utils.FileBrowserPopUp(
            GUIUtils::FileBrowserFlag::LOAD, "Load Project", popup_load_file, this->project_filename)) {
        popup_load_failed = !this->graph_manager.LoadAddProjectFile(add_project_graph_uid, this->project_filename);
        this->add_project_graph_uid = GUI_INVALID_ID;
    }
    if (this->utils.FileBrowserPopUp(
            GUIUtils::FileBrowserFlag::SAVE, "Save Project", popup_save_file, this->project_filename)) {
        popup_save_failed = !this->graph_manager.SaveProjectFile(this->graph_uid, this->project_filename);
    }
    bool confirmed, aborted;
    this->utils.MinimalPopUp("Failed to Save Project", popup_save_failed,
        "See console log output for more information.", "", confirmed, "Cancel", aborted);
    this->utils.MinimalPopUp("Failed to Load Project", popup_load_failed,
        "See console log output for more information.", "", confirmed, "Cancel", aborted);

#endif // GUI_USE_FILESYSTEM
}


void megamol::gui::configurator::Configurator::draw_window_module_list(float width) {

    ImGui::BeginGroup();

    const float search_child_height = ImGui::GetItemsLineHeightWithSpacing() * 2.25f;
    ImGui::BeginChild("module_search_child_window", ImVec2(width, search_child_height), false,
        ImGuiWindowFlags_AlwaysUseWindowPadding | ImGuiWindowFlags_NoScrollbar);

    ImGui::Text("Available Modules");
    ImGui::Separator();

    if (std::get<1>(this->hotkeys[HotkeyIndex::MODULE_SEARCH])) {
        std::get<1>(this->hotkeys[HotkeyIndex::MODULE_SEARCH]) = false;
        this->utils.SetSearchFocus(true);
    }
    std::string help_text = "[" + std::get<0>(this->hotkeys[HotkeyIndex::MODULE_SEARCH]).ToString() +
                            "] Set keyboard focus to search input field.\n"
                            "Case insensitive substring search in module names.";
    this->utils.StringSearch("configurator_module_search", help_text);
    auto search_string = this->utils.GetSearchString();

    ImGui::EndChild();

    ImGui::BeginChild("module_list_child_window", ImVec2(width, 0.0f), true, ImGuiWindowFlags_None);

    ImGuiID id = 1;

    bool search_filter = true;
    bool compat_filter = true;

    std::string compat_call_slot_name;
    CallSlotPtrType selected_call_slot_ptr;
    auto graph_ptr = this->graph_manager.GetGraph(this->graph_uid);
    if (graph_ptr != nullptr) {
        auto call_slot_id = graph_ptr->GUI_GetSelectedCallSlot();
        for (auto& mods : graph_ptr->GetGraphModules()) {
            CallSlotPtrType call_slot_ptr = mods->GetCallSlot(call_slot_id);
            if (call_slot_ptr != nullptr) {
                selected_call_slot_ptr = call_slot_ptr;
            }
        }
    }

    for (auto& mod : this->graph_manager.GetModulesStock()) {

        // Filter module by given search string
        search_filter = true;
        if (!search_string.empty()) {
            search_filter = this->utils.FindCaseInsensitiveSubstring(mod.class_name, search_string);
        }

        // Filter module by compatible call slots
        compat_filter = true;
        if (selected_call_slot_ptr != nullptr) {
            compat_filter = false;
            for (auto& stock_call_slot_map : mod.call_slots) {
                for (auto& stock_call_slot : stock_call_slot_map.second) {
                    ImGuiID cpcidx = CallSlot::GetCompatibleCallIndex(selected_call_slot_ptr, stock_call_slot);
                    if (cpcidx != GUI_INVALID_ID) {
                        compat_call_slot_name = stock_call_slot.name;
                        compat_filter = true;
                    }
                }
            }
        }

        if (search_filter && compat_filter) {
            ImGui::PushID(id);

            std::string label = mod.class_name + " (" + mod.plugin_name + ")"; /// std::to_string(id) + " " +
            if (mod.is_view) {
                label += " [View]";
            }
            if (ImGui::Selectable(label.c_str(), (id == this->selected_list_module_uid))) {
                this->selected_list_module_uid = id;
            }
            bool add_module = false;
            // Left mouse button double click action
            if (ImGui::IsMouseDoubleClicked(0) && ImGui::IsItemHovered()) {
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
                if (graph_ptr != nullptr) {
                    graph_ptr->AddModule(this->graph_manager.GetModulesStock(), mod.class_name);
                    // If there is a call slot selected, create call to compatible call slot of new module
                    if (compat_filter && (selected_call_slot_ptr != nullptr)) {
                        // Get call slots of last added module
                        for (auto& call_slot_map : graph_ptr->GetGraphModules().back()->GetCallSlots()) {
                            for (auto& call_slot : call_slot_map.second) {
                                if (call_slot->name == compat_call_slot_name) {
                                    if (graph_ptr->AddCall(
                                            this->graph_manager.GetCallsStock(), selected_call_slot_ptr, call_slot)) {
                                    }
                                }
                            }
                        }
                    }
                } else {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "No project loaded. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                }
            }

            // Hover tool tip
            this->utils.HoverToolTip(mod.description, id, 0.5f, 5.0f);

            ImGui::PopID();
            id++;
        }
    };

    ImGui::EndChild();

    ImGui::EndGroup();
}


void megamol::gui::configurator::Configurator::add_empty_project(void) {

    if (this->graph_manager.AddGraph(this->get_unique_project_name())) {

        // Add initial GUIView and set as view instance
        auto graph_ptr = this->graph_manager.GetGraphs().back();
        if (graph_ptr != nullptr) {
            std::string guiview_class_name = "GUIView";
            if (graph_ptr->AddModule(this->graph_manager.GetModulesStock(), guiview_class_name)) {
                auto graph_module = graph_ptr->GetGraphModules().back();
                graph_module->is_view_instance = true;
            } else {
                vislib::sys::Log::DefaultLog.WriteError(
                    "Unable to add initial gui view module: '%s'. [%s, %s, line %d]\n", guiview_class_name.c_str(),
                    __FILE__, __FUNCTION__, __LINE__);
            }
        } else {
            vislib::sys::Log::DefaultLog.WriteError(
                "Unable to get last added graph. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        }
    } else {
        vislib::sys::Log::DefaultLog.WriteError(
            "Unable to create new graph. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
}
