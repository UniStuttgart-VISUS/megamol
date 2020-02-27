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
 * - Delete module/call:   Delete
 */

#include "stdafx.h"
#include "Configurator.h"


using namespace megamol::gui::configurator;


megamol::gui::configurator::Configurator::Configurator() : hotkeys(), graph_manager(), utils(), gui() {

    // Init HotKeys
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

    // Init state
    this->gui.window_state = 0;
    this->gui.project_filename = "";
    this->gui.graph_ptr = nullptr;
    this->gui.selected_list_module_id = -1;
    this->gui.graph_font = nullptr;
    this->gui.split_width = 250.0f;
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

    if (this->gui.window_state < 2) {
        // 1] Show pop-up before calling UpdateAvailableModulesCallsOnce of graph.

        /// Rendering of pop-up requires two complete Draw calls!
        bool open = true;
        std::string popup_label = "Loading";
        if (this->gui.window_state == 0) {
            ImGui::OpenPopup(popup_label.c_str());
        }
        ImGuiWindowFlags popup_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove;
        if (ImGui::BeginPopupModal(popup_label.c_str(), &open, popup_flags)) {
            ImGui::Text("Please wait...\nLoading available modules and calls for configurator.");
            ImGui::EndPopup();
        }
        this->gui.window_state++;

    } else if (this->gui.window_state == 2) {
        // 2] Load available modules and calls and currently loaded project from core once(!)

        this->graph_manager.UpdateModulesCallsStock(core_instance);
        this->gui.window_state++;
    } else {
        // 3] Render configurator gui content

        this->draw_window_menu(core_instance);

        const float split_thickness = 10.0f;
        float child_width_auto = 0.0f;
        this->utils.VerticalSplitter(split_thickness, &this->gui.split_width, &child_width_auto);

        this->draw_window_module_list(this->gui.split_width);
        ImGui::SameLine();
        this->graph_manager.Present(child_width_auto, this->gui.graph_font);

        this->gui.graph_ptr = this->graph_manager.GetActiveGraph();
    }

    return true;
}


void megamol::gui::configurator::Configurator::draw_window_menu(megamol::core::CoreInstance* core_instance) {

    if (core_instance == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to Core Instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    bool open_popup_project = false;
    if (ImGui::BeginMenuBar()) {

        if (ImGui::BeginMenu("File")) {

            if (ImGui::MenuItem("New Project", nullptr)) {
                this->graph_manager.AddGraph(this->get_unique_project_name());
            }

            if (ImGui::MenuItem("Load Running Project")) {
                int graph_count = this->graph_manager.GetGraphs().size();
                std::string graph_name = "Project_" + std::to_string(graph_count + 1);
                this->graph_manager.LoadCurrentCoreProject(this->get_unique_project_name(), core_instance);
                // this->GetCoreInstance()->LoadProject(vislib::StringA(projectFilename.c_str()));
            }

#ifdef GUI_USE_FILESYSTEM
            // Load/save parameter values to LUA file
            if (ImGui::MenuItem("Save Project", nullptr)) {
                open_popup_project = true;
            }
#endif // GUI_USE_FILESYSTEM

            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Info")) {
            std::string info_text = "Additonal Options:\n\n"
                                    "- Add module from stock list to graph\n"
                                    "     [Double Click] with left mouse button"
                                    " | [Richt Click] on selected module -> Context Menu: Add\n\n"
                                    "- Delete selected module/call from graph\n"
                                    "     Select item an press [Delete]"
                                    " | [Richt Click] on selected item -> Context Menu: Delete\n\n"
                                    "- Rename graph or module\n"
                                    "     [Richt Click] on graph tab or module -> Context Menu: Rename\n\n";
            ImGui::Text(info_text.c_str());
            ImGui::EndMenu();
        }

        ImGui::SameLine(260.0f);

        // Info text for PROTOTYPE --------------------------------------------
        std::string label = "This is a PROTOTYPE. Changes will NOT effect the currently loaded MegaMol project.";
        ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.2f, 1.0f), label.c_str());

        ImGui::EndMenuBar();
    }

    // Pop-Up(s)
    this->popup_save_project(open_popup_project, core_instance);
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
    this->utils.StringSearch("Search", help_text);
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
        int compat_call_index = -1;
        std::string compat_call_slot_name;
        if (this->gui.graph_ptr != nullptr) {
            if (this->gui.graph_ptr->GetSelectedSlot() != nullptr) {
                compat_filter = false;
                for (auto& cst : mod.call_slots) {
                    for (auto& cs : cst.second) {
                        int cpidx =
                            this->graph_manager.GetCompatibleCallIndex(this->gui.graph_ptr->GetSelectedSlot(), cs);
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
            if (ImGui::Selectable(label.c_str(), (id == this->gui.selected_list_module_id))) {
                this->gui.selected_list_module_id = id;
            }
            // Left mouse button double click action
            if (ImGui::IsMouseDoubleClicked(0) && ImGui::IsItemHovered()) {
                /// XXX this->add_new_module_to_graph(mod, compat_call_index, compat_call_slot_name);
            }
            // Context menu
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Add", "'Double-Click'")) {
                    /// XXX this->add_new_module_to_graph(mod, compat_call_index, compat_call_slot_name);
                }
                ImGui::EndPopup();
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


bool megamol::gui::configurator::Configurator::popup_save_project(
    bool open, megamol::core::CoreInstance* core_instance) {

#ifdef GUI_USE_FILESYSTEM
    std::string save_project_label = "Save Project";

    if (open) {
        ImGui::OpenPopup(save_project_label.c_str());
    }
    if (ImGui::BeginPopupModal(save_project_label.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {

        bool save_project = false;

        std::string label = "File Name";
        auto flags = ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll;
        /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
        this->utils.Utf8Encode(this->gui.project_filename);
        if (ImGui::InputText(label.c_str(), &this->gui.project_filename, flags)) {
            save_project = true;
        }
        this->utils.Utf8Decode(this->gui.project_filename);
        // Set focus on input text once (applied next frame)
        if (open) {
            ImGuiID id = ImGui::GetID(label.c_str());
            ImGui::ActivateItem(id);
        }

        bool valid_ending = true;
        if (!HasFileExtension(this->gui.project_filename, std::string(".lua"))) {
            ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.0f, 1.0f), "Appending required file ending '.lua'");
            valid_ending = false;
        }
        // Warn when file already exists
        if (PathExists(this->gui.project_filename) || PathExists(this->gui.project_filename + ".lua")) {
            ImGui::TextColored(ImVec4(0.9f, 0.0f, 0.0f, 1.0f), "Overwriting existing file.");
        }
        if (ImGui::Button("Save (Enter)")) {
            save_project = true;
        }

        if (save_project) {
            if (this->gui.graph_ptr != nullptr) {
                if (!valid_ending) {
                    this->gui.project_filename.append(".lua");
                }
                if (this->graph_manager.PROTOTYPE_SaveGraph(
                        this->gui.graph_ptr->GetUID(), this->gui.project_filename, core_instance)) {
                    ImGui::CloseCurrentPopup();
                }
            } else {
                vislib::sys::Log::DefaultLog.WriteWarn("No project available for saving.");
            }
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
#endif // GUI_USE_FILESYSTEM

    return true;
}

/*
bool megamol::gui::configurator::Configurator::add_new_module_to_graph(
    const megamol::gui::configurator::StockModule& mod, int compat_call_idx, const std::string& compat_call_slot_name) {

    bool retval = false;
    if (this->gui.graph_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "No project available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        // Alternatively create new graph ... but not here :-/
        return false;
    }

    // Process module adding
    for (auto& graph : this->graph_manager.GetGraphs()) {
        // Look up currently active graph
        if (graph->GetUID() == this->gui.graph_ptr->GetUID()) {
            // Add new module
            retval = graph->AddModule(this->graph_manager.GetModulesStock(), mod.class_name);
            // If there is a call slot selected, create call to compatible call slot of new module
            if (graph->gui.selected_slot_ptr != nullptr) {
                // Get call slots of last added module
                for (auto& call_slot_map : graph->GetGraphModules().back()->GetCallSlots()) {
                    for (auto& call_slot : call_slot_map.second) {
                        if (call_slot->name == compat_call_slot_name) {
                            if (graph->AddCall(this->graph_manager.GetCallsStock(), compat_call_idx,
                                    graph->gui.selected_slot_ptr, call_slot)) {
                                graph->gui.selected_slot_ptr = nullptr;
                                retval = true;
                            }
                        }
                    }
                }
            }
        }
    }
    return retval;
}
*/