/*
 * Configurator.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

/**
 * USED HOTKEYS:
 *
 * - Search module:        Shift + Ctrl  + m
 * - Search parameter:     Shift + Ctrl  + p
 * - Delete module/call:   Delete
 */

#include "stdafx.h"
#include "Configurator.h"


using namespace megamol::gui::configurator;


megamol::gui::configurator::Configurator::Configurator()
    : hotkeys()
    , graph_manager()
    , utils()
    , window_state(0)
    , project_filename("")
    , graph_ptr(nullptr)
    , selected_list_module_id(GUI_INVALID_ID)
    , graph_font(nullptr)
    , split_width(250.0f)
    , unique_project_id(0) {

    // Define HotKeys
    this->hotkeys[HotkeyIndex::MODULE_SEARCH] =
        HotkeyData(megamol::core::view::KeyCode(
                       megamol::core::view::Key::KEY_M, core::view::Modifier::CTRL | core::view::Modifier::SHIFT),
            false);
    this->hotkeys[HotkeyIndex::PARAMETER_SEARCH] =
        HotkeyData(megamol::core::view::KeyCode(
                       megamol::core::view::Key::KEY_P, core::view::Modifier::CTRL | core::view::Modifier::SHIFT),
            false);
    this->hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM] =
        HotkeyData(megamol::core::view::KeyCode(megamol::core::view::Key::KEY_DELETE), false);
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
        auto key = std::get<0>(h).GetKey();
        auto mods = std::get<0>(h).GetModifiers();
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
        this->graph_manager.AddGraph(this->get_unique_project_name());
        this->window_state++;
    } else {
        // 3] Render configurator gui content

        this->draw_window_menu(core_instance);

        float child_width_auto = 0.0f;
        this->utils.VerticalSplitter(&this->split_width, &child_width_auto);

        this->draw_window_module_list(this->split_width);

        ImGui::SameLine();

        this->graph_ptr = nullptr;

        this->graph_manager.GUI_Present(child_width_auto, this->graph_font,
            this->hotkeys[HotkeyIndex::PARAMETER_SEARCH], this->hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM]);

        this->graph_ptr = this->graph_manager.GUI_GetPresentedGraph();
    }

    return true;
}


void megamol::gui::configurator::Configurator::draw_window_menu(megamol::core::CoreInstance* core_instance) {

    if (core_instance == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to Core Instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    bool open_save_popup = false;
    bool open_load_popup = false;
    if (ImGui::BeginMenuBar()) {

        if (ImGui::BeginMenu("File")) {

            if (ImGui::MenuItem("New Project", nullptr)) {
                this->graph_manager.AddGraph(this->get_unique_project_name());
            }

            if (ImGui::MenuItem("Load Running Project")) {
                size_t graph_count = this->graph_manager.GetGraphs().size();
                std::string graph_name = "Project_" + std::to_string(graph_count + 1);
                this->graph_manager.LoadCurrentCoreProject(this->get_unique_project_name(), core_instance);
                // this->GetCoreInstance()->LoadProject(vislib::StringA(projectFilename.c_str()));
            }

#ifdef GUI_USE_FILESYSTEM
            // Load project from LUA file
            if (ImGui::MenuItem("Load Project", nullptr)) {
                open_load_popup = true;
            }

            // Save currently active project to LUA file
            if (ImGui::MenuItem("Save Project", nullptr, false, (this->graph_ptr != nullptr))) {
                open_save_popup = true;
            }
#endif // GUI_USE_FILESYSTEM

            ImGui::EndMenu();
        }

        // Info text ----------------------------------------------------------
        ImGui::SameLine(260.0f);
        std::string label = "This is a PROTOTYPE. Changes will NOT effect the currently loaded MegaMol project.";
        ImGui::TextColored(ImVec4(0.75f, 0.2f, 0.2f, 1.0f), label.c_str());

        ImGui::EndMenuBar();
    }

    // SAVE/LOAD PROJECT pop-up
#ifdef GUI_USE_FILESYSTEM
    if (this->utils.FileBrowserPopUp(
            GUIUtils::FileBrowserFlag::LOAD, open_load_popup, "Load Project", this->project_filename)) {
        this->graph_manager.LoadProjectFile(this->graph_ptr->GetUID(), this->project_filename, core_instance);
    }
    if (this->utils.FileBrowserPopUp(
            GUIUtils::FileBrowserFlag::SAVE, open_save_popup, "Save Project", this->project_filename)) {
        this->graph_manager.SaveProjectFile(this->graph_ptr->GetUID(), this->project_filename, core_instance);
    }
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

    int id = 1;
    for (auto& mod : this->graph_manager.GetModulesStock()) {

        // Filter module by given search string
        bool search_filter = true;
        if (!search_string.empty()) {

            search_filter = this->utils.FindCaseInsensitiveSubstring(mod.class_name, search_string);
        }

        // Filter module by compatible call slots
        bool compat_filter = true;
        int compat_call_index = GUI_INVALID_ID;
        std::string compat_call_slot_name;
        if (this->graph_ptr != nullptr) {
            if (this->graph_ptr->GUI_GetSelectedSlot() != nullptr) {
                compat_filter = false;
                for (auto& cst : mod.call_slots) {
                    for (auto& cs : cst.second) {
                        int cpidx = CallSlot::GetCompatibleCallIndex(this->graph_ptr->GUI_GetSelectedSlot(), cs);
                        if (cpidx > 0) {
                            compat_call_index = cpidx;
                            compat_call_slot_name = cs.name;
                            compat_filter = true;
                        }
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
            if (ImGui::Selectable(label.c_str(), (id == this->selected_list_module_id))) {
                this->selected_list_module_id = id;
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
                if (this->graph_ptr != nullptr) {
                    this->graph_ptr->AddModule(this->graph_manager.GetModulesStock(), mod.class_name);

                    auto selected_slot = this->graph_ptr->GUI_GetSelectedSlot();
                    // If there is a call slot selected, create call to compatible call slot of new module
                    if (selected_slot != nullptr) {
                        // Get call slots of last added module
                        for (auto& call_slot_map : this->graph_ptr->GetGraphModules().back()->GetCallSlots()) {
                            for (auto& call_slot : call_slot_map.second) {
                                if (call_slot->name == compat_call_slot_name) {
                                    if (this->graph_ptr->AddCall(this->graph_manager.GetCallsStock(), compat_call_index,
                                            selected_slot, call_slot)) {
                                        /// XXXthis->selected_slot_ptr = nullptr;
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
