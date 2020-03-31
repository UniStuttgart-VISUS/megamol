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
    : graph_manager()
    , file_utils()
    , utils()
    , init_state(0)
    , left_child_width(250.0f)
    , selected_list_module_uid(GUI_INVALID_ID)
    , add_project_graph_uid(GUI_INVALID_ID)
    , show_module_list_sidebar(true)    
    , project_filename("")    
    , state() {           

    this->state.hotkeys[megamol::gui::HotkeyIndex::MODULE_SEARCH] = megamol::gui::HotkeyDataType(
        core::view::KeyCode(core::view::Key::KEY_M, (core::view::Modifier::CTRL | core::view::Modifier::SHIFT)), false);
    this->state.hotkeys[megamol::gui::HotkeyIndex::PARAMETER_SEARCH] = megamol::gui::HotkeyDataType(
        core::view::KeyCode(core::view::Key::KEY_P, (core::view::Modifier::CTRL | core::view::Modifier::SHIFT)), false);
    this->state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM] =
        megamol::gui::HotkeyDataType(core::view::KeyCode(core::view::Key::KEY_DELETE), false);
    this->state.hotkeys[megamol::gui::HotkeyIndex::SAVE_PROJECT] = megamol::gui::HotkeyDataType(
        megamol::core::view::KeyCode(core::view::Key::KEY_S, core::view::Modifier::CTRL), false);
    this->state.font = nullptr;
    this->state.child_width = 0.0f;
    this->state.show_parameter_sidebar = true;
    this->state.graph_selected_uid = GUI_INVALID_ID;
    this->state.graph_delete = false;
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
    for (auto& h : this->state.hotkeys) {
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

    // Draw
    if (this->init_state < 2) {
        // 1] Show pop-up before calling UpdateAvailableModulesCallsOnce of graph.
        /// Rendering of pop-up requires two complete Draw calls!
        bool open = true;
        std::string popup_label = "Loading";
        if (this->init_state == 0) {
            ImGui::OpenPopup(popup_label.c_str());
        }
        ImGuiWindowFlags popup_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove;
        if (ImGui::BeginPopupModal(popup_label.c_str(), &open, popup_flags)) {
            ImGui::Text("Please wait...\nLoading available modules and calls for configurator.");
            ImGui::EndPopup();
        }
        this->init_state++;

    } else if (this->init_state == 2) {
        // 2] Load available modules and calls and currently loaded project from core once(!)
        this->graph_manager.UpdateModulesCallsStock(core_instance);
        // Load once inital project
        this->graph_manager.LoadProjectCore(core_instance);
        /// this->add_empty_project();
        this->init_state++;
    } else {
        // 3] Render configurator gui content
        this->draw_window_menu(core_instance);
        this->state.child_width = 0.0f;
        if (this->show_module_list_sidebar) {
            this->utils.VerticalSplitter(
                GUIUtils::FixedSplitterSide::LEFT, this->left_child_width, this->state.child_width);
            this->draw_window_module_list(this->left_child_width);
            ImGui::SameLine();
        }
        this->graph_manager.GUI_Present(this->state);
    }

    // Reset hotkeys
    for (auto& h : this->state.hotkeys) {
        std::get<1>(h) = false;
    }

    return true;
}


void megamol::gui::configurator::Configurator::draw_window_menu(megamol::core::CoreInstance* core_instance) {

    if (core_instance == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to Core Instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    auto selected_graph_ptr = this->graph_manager.GetGraph(this->state.graph_selected_uid);
    bool group_save = false;
    ImGuiID group_selected_uid = GUI_INVALID_ID;
    if (selected_graph_ptr != nullptr) {
        group_save = selected_graph_ptr->GUI_GetGroupSave();
        group_selected_uid = selected_graph_ptr->GUI_GetSelectedGroup();
    }

    bool confirmed, aborted;
    bool popup_save_project_file = false;
    bool popup_save_group_file = group_save;
    bool popup_load_file = false;

    // Check hotkeys
    if (std::get<1>(this->state.hotkeys[megamol::gui::HotkeyIndex::MODULE_SEARCH])) {
        this->show_module_list_sidebar = true; // !this->show_module_list_sidebar
    }
    if (std::get<1>(this->state.hotkeys[megamol::gui::HotkeyIndex::PARAMETER_SEARCH])) {
        this->state.show_parameter_sidebar = true; // !this->state.show_parameter_sidebar
    }
    if (std::get<1>(this->state.hotkeys[megamol::gui::HotkeyIndex::SAVE_PROJECT])) {
        popup_save_project_file = true;
    }

    // Draw menu
    if (ImGui::BeginMenuBar()) {

        if (ImGui::BeginMenu("File")) {

            if (ImGui::BeginMenu("Load Project")) {

                if (ImGui::MenuItem("New", nullptr)) {
                    this->add_empty_project();
                }

                // Load project from LUA file
                if (ImGui::MenuItem("File", nullptr)) {
                    this->add_project_graph_uid = GUI_INVALID_ID;
                    popup_load_file = true;
                }

                if (ImGui::MenuItem("Running")) {
                    this->graph_manager.LoadProjectCore(core_instance);
                    // this->GetCoreInstance()->LoadProject(vislib::StringA(projectFilename.c_str()));
                }

                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Add Project")) {

                // Add project from LUA file to current project
                if (ImGui::MenuItem("File", nullptr, false, (this->state.graph_selected_uid != GUI_INVALID_ID))) {
                    this->add_project_graph_uid = this->state.graph_selected_uid;
                    popup_load_file = true;
                }

                if (ImGui::MenuItem("Running", nullptr, false, (this->state.graph_selected_uid != GUI_INVALID_ID))) {
                    this->graph_manager.AddProjectCore(this->state.graph_selected_uid, core_instance);
                    // this->GetCoreInstance()->LoadProject(vislib::StringA(projectFilename.c_str()));
                }

                ImGui::EndMenu();
            }

            // Save currently active project to LUA file
            if (ImGui::MenuItem("Save Project",
                    std::get<0>(this->state.hotkeys[megamol::gui::HotkeyIndex::SAVE_PROJECT]).ToString().c_str(), false,
                    (this->state.graph_selected_uid != GUI_INVALID_ID))) {
                popup_save_project_file = true;
            }
            // Save currently active group to LUA file
            /*
            if (ImGui::MenuItem("Save Group", nullptr, false, (group_selected_uid != GUI_INVALID_ID))) {
                popup_save_group_file = true;
            }
            */

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {
            if (ImGui::MenuItem("Modules Sidebar",
                    std::get<0>(this->state.hotkeys[megamol::gui::HotkeyIndex::MODULE_SEARCH]).ToString().c_str(),
                    this->show_module_list_sidebar)) {
                this->show_module_list_sidebar = !this->show_module_list_sidebar;
            }
            if (ImGui::MenuItem("Parameter Sidebar",
                    std::get<0>(this->state.hotkeys[megamol::gui::HotkeyIndex::PARAMETER_SEARCH]).ToString().c_str(),
                    this->state.show_parameter_sidebar)) {
                this->state.show_parameter_sidebar = !this->state.show_parameter_sidebar;
            }
            ImGui::EndMenu();
        }

        ImGui::SameLine();

        if (ImGui::BeginMenu("Help")) {
            const std::string docu_link =
                "https://github.com/UniStuttgart-VISUS/megamol/tree/master/plugins/gui#configurator";
            if (ImGui::Button("Readme (Copy Link)")) {
#ifdef GUI_USE_GLFW
                auto glfw_win = ::glfwGetCurrentContext();
                ::glfwSetClipboardString(glfw_win, docu_link.c_str());
#elif _WIN32
                ImGui::SetClipboardText(docu_link.c_str());
#else // LINUX
                vislib::sys::Log::DefaultLog.WriteWarn(
                    "No clipboard use provided. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                vislib::sys::Log::DefaultLog.WriteInfo("Readme Link:\n%s", docu_link.c_str());
#endif
            }

            ImGui::EndMenu();
        }

        // Info text ----------------------------------------------------------
        ImGui::SameLine(260.0f);
        std::string label = "Changes will not affect the currently loaded MegaMol project.";
        ImGui::TextColored(ImVec4(1.0f, 0.25f, 0.25f, 1.0f), label.c_str());

        ImGui::EndMenuBar();
    }

    // Pop-ups-----------------------------------
    bool popup_failed = false;
    if (this->file_utils.FileBrowserPopUp(
            FileUtils::FileBrowserFlag::LOAD, "Load Project", popup_load_file, this->project_filename)) {
        popup_failed = !this->graph_manager.LoadAddProjectFile(add_project_graph_uid, this->project_filename);
        this->add_project_graph_uid = GUI_INVALID_ID;
    }
    this->utils.MinimalPopUp("Failed to Load Project", popup_failed, "See console log output for more information.", "",
        confirmed, "Cancel", aborted);

    popup_failed = false;
    if (this->file_utils.FileBrowserPopUp(
            FileUtils::FileBrowserFlag::SAVE, "Save Project", popup_save_project_file, this->project_filename)) {
        popup_failed = !this->graph_manager.SaveProjectFile(this->state.graph_selected_uid, this->project_filename);
    }
    this->utils.MinimalPopUp("Failed to Save Project", popup_failed, "See console log output for more information.", "",
        confirmed, "Cancel", aborted);

    popup_failed = false;
    if (this->file_utils.FileBrowserPopUp(
            FileUtils::FileBrowserFlag::SAVE, "Save Group", popup_save_group_file, this->project_filename)) {
        popup_failed = !this->graph_manager.SaveGroupFile(group_selected_uid, this->project_filename);
    }
    this->utils.MinimalPopUp("Failed to Save Group", popup_failed, "See console log output for more information.", "",
        confirmed, "Cancel", aborted);
}


void megamol::gui::configurator::Configurator::draw_window_module_list(float width) {

    ImGui::BeginGroup();

    const float search_child_height = ImGui::GetFrameHeightWithSpacing() * 2.25f;
    auto child_flags =
        ImGuiWindowFlags_AlwaysUseWindowPadding | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NavFlattened;
    ImGui::BeginChild("module_search_child_window", ImVec2(width, search_child_height), false, child_flags);

    ImGui::Text("Available Modules");
    ImGui::Separator();

    if (std::get<1>(this->state.hotkeys[megamol::gui::HotkeyIndex::MODULE_SEARCH])) {
        this->show_module_list_sidebar = true;
        this->utils.SetSearchFocus(true);
    }
    std::string help_text = "[" +
                            std::get<0>(this->state.hotkeys[megamol::gui::HotkeyIndex::MODULE_SEARCH]).ToString() +
                            "] Set keyboard focus to search input field.\n"
                            "Case insensitive substring search in module names.";
    this->utils.StringSearch("configurator_module_search", help_text);
    auto search_string = this->utils.GetSearchString();

    ImGui::EndChild();

    child_flags = ImGuiWindowFlags_NavFlattened;
    ImGui::BeginChild("module_list_child_window", ImVec2(width, 0.0f), true, child_flags);

    bool search_filter = true;
    bool compat_filter = true;

    std::string compat_call_slot_name;
    CallSlotPtrType selected_call_slot_ptr;
    auto graph_ptr = this->graph_manager.GetGraph(this->state.graph_selected_uid);
    if (graph_ptr != nullptr) {
        auto call_slot_id = graph_ptr->GUI_GetSelectedCallSlot();
        if (call_slot_id != GUI_INVALID_ID) {
            for (auto& mods : graph_ptr->GetGraphModules()) {
                CallSlotPtrType call_slot_ptr = mods->GetCallSlot(call_slot_id);
                if (call_slot_ptr != nullptr) {
                    selected_call_slot_ptr = call_slot_ptr;
                }
            }
        }
    }

    ImGuiID id = 1;
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
                if (graph_ptr != nullptr) {
                    ImGuiID module_uid = graph_ptr->AddModule(this->graph_manager.GetModulesStock(), mod.class_name);
                    auto module_ptr = graph_ptr->GetModule(module_uid);
                    if (module_ptr != nullptr) {
                        // If there is a call slot selected, create call to compatible call slot of new module
                        if (compat_filter && (selected_call_slot_ptr != nullptr)) {
                            // Get call slots of last added module
                            for (auto& call_slot_map : module_ptr->GetCallSlots()) {
                                for (auto& call_slot : call_slot_map.second) {
                                    if (call_slot->name == compat_call_slot_name) {
                                        if (graph_ptr->AddCall(this->graph_manager.GetCallsStock(), selected_call_slot_ptr, call_slot)) {
                                            // Caluculate nice position of newly added module
                                            if (selected_call_slot_ptr->ParentModuleConnected()) {
                                                auto size = selected_call_slot_ptr->GetParentModule()->GUI_GetSize();
                                                const float offset = (GUI_CALL_SLOT_RADIUS * 4.0f);
                                                float x_offset, y_offset;
                                                // Callers have only one connected call
                                                if (selected_call_slot_ptr->type == CallSlotType::CALLER) {
                                                    std::string call_name = selected_call_slot_ptr->GetConnectedCalls()[0]->class_name;
                                                    x_offset = (size.x + (1.5f * GUIUtils::TextWidgetWidth(call_name) + offset));
                                                    y_offset = size.y + offset;
                                                }
                                                else if (call_slot->type == CallSlotType::CALLER) {
                                                    auto size = selected_call_slot_ptr->GetParentModule()->GUI_GetSize();
                                                    std::string call_name = call_slot->GetConnectedCalls()[0]->class_name;
                                                    x_offset = -1.0f * (1.5f * GUIUtils::TextWidgetWidth(call_name) + offset);
                                                    y_offset = -1.0f * (size.y + offset);
                                                }
                                                ImVec2 new_module_pos = selected_call_slot_ptr->GetParentModule()->GUI_GetPosition();
                                                new_module_pos.x += x_offset;
                                                new_module_pos.y += y_offset;
                                                module_ptr->GUI_SetPosition(new_module_pos);
                                            }
                                        }
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
        }
        id++;
    }

    ImGui::EndChild();
    ImGui::EndGroup();
}


void megamol::gui::configurator::Configurator::add_empty_project(void) {

    ImGuiID graph_uid = this->graph_manager.AddGraph();
    if (graph_uid != GUI_INVALID_ID) {

        // Add initial GUIView and set as view instance
        auto graph_ptr = this->graph_manager.GetGraph(graph_uid);
        if (graph_ptr != nullptr) {
            std::string guiview_class_name = "GUIView";
            ImGuiID module_uid = graph_ptr->AddModule(this->graph_manager.GetModulesStock(), guiview_class_name);
            auto module_ptr = graph_ptr->GetModule(module_uid);
            if (module_ptr != nullptr) {
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
