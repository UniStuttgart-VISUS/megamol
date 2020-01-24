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

using namespace megamol;
using namespace megamol::gui;
using vislib::sys::Log;


Configurator::Configurator() : hotkeys(), graph_manager(), utils(), state() {

    // Init HotKeys
    this->hotkeys[HotkeyIndex::MODULE_SEARCH] =
        HotkeyData(megamol::core::view::KeyCode(
                       megamol::core::view::Key::KEY_M, core::view::Modifier::CTRL | core::view::Modifier::SHIFT),
            false);
    this->hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM] =
        HotkeyData(megamol::core::view::KeyCode(megamol::core::view::Key::KEY_DELETE), false);

    // Init state
    this->state.window_rendering_state = 0;
    this->state.project_filename == "";
    this->state.slot_radius = 8.0f;
    this->state.active_graph_uid = -1;
    this->state.selected_list_module = -1;
    this->state.selected_module_uid = -1;
    this->state.selected_call_uid = -1;
    this->state.hovered_call_slot_uid = -1;
    this->state.selected_call_slot_ptr = nullptr;
    this->state.process_selected_slot = 0;
    this->state.canvas_position = ImVec2(0.0f, 0.0f);
    this->state.canvas_size = ImVec2(0.0f, 0.0f);
    this->state.open_rename_popup = false;
    this->state.rename = nullptr;
    this->state.scrolling = ImVec2(0.0f, 0.0f);
    this->state.show_grid = true;
    this->state.show_call_names = true;
    this->state.small_modules = false;
}


Configurator::~Configurator() {}


bool megamol::gui::Configurator::CheckHotkeys(void) {

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


bool megamol::gui::Configurator::Draw(
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

    ImGuiStyle& style = ImGui::GetStyle();

    if (this->state.window_rendering_state < 2) {
        // 1] Show pop-up before before calling UpdateAvailableModulesCallsOnce of graph.

        /// Rendering of pop-up requires two complete Draw calls!
        bool open = true;
        std::string popup_label = "Loading";
        if (this->state.window_rendering_state == 0) {
            ImGui::OpenPopup(popup_label.c_str());
        }
        ImGuiWindowFlags popup_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove;
        if (ImGui::BeginPopupModal(popup_label.c_str(), &open, popup_flags)) {
            ImGui::Text("Please wait...\nLoading available modules and calls for configurator.");
            ImGui::EndPopup();
        }
        this->state.window_rendering_state++;
    } else if (this->state.window_rendering_state == 2) {
        // 2] Load available modules and calls once(!)

        this->graph_manager.UpdateModulesCallsStock(core_instance);
        // Load graph of currently loaded project from core
        this->graph_manager.LoadCurrentCoreProjectToGraph(core_instance);
        this->state.window_rendering_state++;
    } else {
        // 3] Render configurator gui content

        this->draw_window_menu(core_instance);
        this->draw_window_module_list();

        // Draws module list and graph canvas tabs next to each other
        ImGui::SameLine();
        ImGui::BeginGroup();

        // Info text for PROTOTYPE --------------------------------------------
        ImGui::BeginChild("info", ImVec2(0.0f, 1.5f * ImGui::GetItemsLineHeightWithSpacing()), true,
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);
        std::string label = "This is a PROTOTYPE. Changes will NOT effect the currently loaded MegaMol project.";
        ImGui::TextColored(ImVec4(0.9f, 0.2f, 0.2f, 1.0f), label.c_str());
        ImGui::EndChild();
        // --------------------------------------------------------------------

        // Project (graph) tabs
        int delete_graph_uid = -1; // (Assuming only one closed tab per frame)

        ImGuiTabBarFlags tabbar_flags = ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_Reorderable;
        ImGui::BeginTabBar("Graphs", tabbar_flags);
        for (auto& graph : this->graph_manager.GetGraphs()) {
            bool open = true;

            // Tab showing one graph
            ImGuiTabItemFlags tab_flags = ImGuiTabItemFlags_None;
            if (graph->IsDirty()) {
                tab_flags |= ImGuiTabItemFlags_UnsavedDocument;
            }
            std::string graph_label = "    " + graph->GetName() + "  ";
            if (ImGui::BeginTabItem(graph_label.c_str(), &open, tab_flags)) {
                if (ImGui::BeginPopupContextItem()) {
                    if (ImGui::MenuItem("Rename")) {
                        this->state.open_rename_popup = true;
                        this->state.rename = &graph->GetName();
                    }
                    ImGui::EndPopup();
                }

                this->state.active_graph_uid = graph->GetUID();
                this->draw_canvas_graph(graph);
                ImGui::EndTabItem();
            }
            // (Do not delete graph while looping through graphs list!)
            if (!open) {
                delete_graph_uid = graph->GetUID();
            }
        }
        ImGui::EndTabBar();

        // Delete marked graph when tab closed
        this->graph_manager.DeleteGraph(delete_graph_uid);

        // Rename pop-up (grpah or module name)
        if (this->state.open_rename_popup) {
            ImGui::OpenPopup("Rename");
        }
        if (ImGui::BeginPopupModal("Rename", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {

            std::string label = "Enter new  project name";
            auto flags = ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll;
            if (ImGui::InputText("Enter new  project name", this->state.rename, flags)) {
                this->state.rename = nullptr;
                ImGui::CloseCurrentPopup();
            }
            // Set focus on input text once (applied next frame)
            if (this->state.open_rename_popup) {
                ImGuiID id = ImGui::GetID(label.c_str());
                ImGui::ActivateItem(id);
            }

            ImGui::EndPopup();
        }
        this->state.open_rename_popup = false;

        ImGui::EndGroup();
    }

    return true;
}


bool megamol::gui::Configurator::draw_window_menu(megamol::core::CoreInstance* core_instance) {

    if (core_instance == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to Core Instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    bool open_popup_project = false;
    if (ImGui::BeginMenuBar()) {
        /// , "(no hotkey)")) {

        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("New Project (Graph)", nullptr)) {
                this->add_new_graph();
            }
#ifdef GUI_USE_FILEUTILS
            // Load/save parameter values to LUA file
            if (ImGui::MenuItem("Save Project (Graph)", nullptr)) {
                open_popup_project = true;
            }
            /// TODO: Load parameter file
            // if (ImGui::MenuItem("Load Project")) {
            // std::string projectFilename;
            // this->GetCoreInstance()->LoadProject(vislib::StringA(projectFilename.c_str()));
            // Load to new graph ...
            //}
            ImGui::EndMenu();
        }
#endif // GUI_USE_FILEUTILS

        if (ImGui::BeginMenu("Graph")) {
            if (ImGui::MenuItem("Reset Scrolling", nullptr)) {
                this->state.scrolling = ImVec2(0.0f, 0.0f);
            }
            if (ImGui::MenuItem("Show Grid", nullptr, this->state.show_grid)) {
                this->state.show_grid = !this->state.show_grid;
            }
            if (ImGui::MenuItem("Show Call Names", nullptr, this->state.show_call_names)) {
                this->state.show_call_names = !this->state.show_call_names;
            }
            if (ImGui::MenuItem("Show Small Modules", nullptr, this->state.small_modules)) {
                this->state.small_modules = !this->state.small_modules;
                // Update all module gui parameters when rendered next time
                for (auto& graph : this->graph_manager.GetGraphs()) {
                    for (auto& mod : graph->GetGraphModules()) {
                        mod->gui.update = true;
                    }
                }
                // Change slot radius depending on module size
                this->state.slot_radius = (this->state.small_modules) ? (5.0f) : (8.0f);
            }
            if (ImGui::MenuItem("Layout Graph", nullptr)) {
                for (auto& graph : this->graph_manager.GetGraphs()) {
                    if (graph->GetUID() == this->state.active_graph_uid) {
                        this->layout_graph(graph);
                    }
                }
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Info")) {
            std::string info_text = "Additonal Options:\n"
                                    "- Add selected module from stock list\n"
                                    "     [Double Click] with left mouse button"
                                    " | [Richt Click] on selected module -> Context Menu: Add  \n"
                                    "- Delete selected module/call from graph\n"
                                    "     Select item an press [Delete]"
                                    " | [Richt Click] on selected item -> Context Menu: Delete  \n"
                                    "- Rename graph or module\n"
                                    "     [Richt Click] on graph tab or module -> Context Menu: Rename  \n";
            ImGui::Text(info_text.c_str());
            ImGui::EndMenu();
        }

        ImGui::SameLine(260.0f);

        ImGui::Separator();
        ImGui::Text("Scrolling: %.2f,%.2f (Middle Mouse Button)", this->state.scrolling.x, this->state.scrolling.y);
        ImGui::Separator();

        ImGui::EndMenuBar();
    }

    // Pop-Up(s)
    this->popup_save_project(open_popup_project, core_instance);

    return true;
}


bool megamol::gui::Configurator::draw_window_module_list(void) {

    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    const float child_width = 250.0f;
    const float child_height = ImGui::GetItemsLineHeightWithSpacing() * 2.5f;

    ImGui::BeginGroup();
    ImGui::BeginChild("module_search", ImVec2(child_width, child_height), true, ImGuiWindowFlags_None);

    ImGui::Text("Available Modules");
    ImGui::Separator();

    if (std::get<1>(this->hotkeys[HotkeyIndex::MODULE_SEARCH])) {
        std::get<1>(this->hotkeys[HotkeyIndex::MODULE_SEARCH]) = false;
        this->utils.SetSearchFocus(true);
    }
    std::string help_text = "[" + std::get<0>(this->hotkeys[HotkeyIndex::MODULE_SEARCH]).ToString() +
                            "] Set keyboard focus to search input field.\n"
                            "Case insensitive substring search in module names.";
    this->utils.StringSearch("Search Modules", help_text);
    auto search_string = this->utils.GetSearchString();

    ImGui::EndChild();

    ImGui::BeginChild("module_list", ImVec2(child_width, 0.0f), true, ImGuiWindowFlags_HorizontalScrollbar);

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
        if (this->state.selected_call_slot_ptr != nullptr) {
            compat_filter = false;
            for (auto& cst : mod.call_slots) {
                for (auto& cs : cst.second) {
                    int cpidx = this->graph_manager.GetCompatibleCallIndex(this->state.selected_call_slot_ptr, cs);
                    if (cpidx > 0) {
                        compat_call_index = cpidx;
                        compat_call_slot_name = cs.name;
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
            if (ImGui::Selectable(label.c_str(), (id == this->state.selected_list_module))) {
                this->state.selected_list_module = id;
            }
            // Left mouse button double click action
            if (ImGui::IsMouseDoubleClicked(0) && ImGui::IsItemHovered()) {
                this->add_new_module_to_graph(mod, compat_call_index, compat_call_slot_name);
            }
            // Context menu
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Add", "'Double-Click'")) {
                    this->add_new_module_to_graph(mod, compat_call_index, compat_call_slot_name);
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

    return true;
}


bool megamol::gui::Configurator::draw_canvas_graph(GraphManager::GraphPtrType graph) {

    ImGuiIO& io = ImGui::GetIO();

    const ImU32 COLOR_CANVAS_BACKGROUND = IM_COL32(75, 75, 75, 255);

    // Process module deletion
    if (std::get<1>(this->hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM])) {
        std::get<1>(this->hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM]) = false;
        this->state.selected_call_slot_ptr = nullptr;
        if (this->state.selected_module_uid > 0) {
            graph->DeleteModule(this->state.selected_module_uid);
        }
        if (this->state.selected_call_uid > 0) {
            graph->DeleteCall(this->state.selected_call_uid);
        }
    }

    // Register trigger for connecting call
    if ((this->state.selected_call_slot_ptr != nullptr) && (io.MouseReleased[0])) {
        this->state.process_selected_slot = 2;
    }

    // Draw child canvas ------------------------------------------------------
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(1, 1));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleColor(ImGuiCol_ChildWindowBg, COLOR_CANVAS_BACKGROUND);
    ImGui::BeginChild("region", ImVec2(0.0f, 0.0f), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);
    // ImGui::PushItemWidth(120.0f);
    this->state.canvas_position = ImGui::GetCursorScreenPos();
    this->state.canvas_size = ImGui::GetWindowSize();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);
    draw_list->ChannelsSplit(2);

    if (ImGui::IsMouseClicked(0) && ImGui::IsWindowHovered()) {
        this->state.selected_module_uid = -1;
        this->state.selected_call_uid = -1;
        this->state.selected_call_slot_ptr = nullptr;
    }

    // Display grid -------------------------------------------------------
    if (this->state.show_grid) {
        this->draw_canvas_grid();
    }
    ImGui::PopStyleVar(2);

    // Draw modules -----------------------------------------------------
    this->draw_canvas_modules(graph);

    // Draw call --------------------------------------------------------
    this->draw_canvas_calls(graph);

    // Draw dragged call --------------------------------------------------------
    this->draw_canvas_dragged_call();

    // Zoomin and Scaling  ----------------------------------------------------
    if (ImGui::IsWindowHovered() && !ImGui::IsAnyItemActive()) {
        // Scrolling (2 = Middle Mouse Button)
        if (ImGui::IsMouseDragging(2, 0.0f)) {
            this->state.scrolling = this->state.scrolling + ImGui::GetIO().MouseDelta;
        }
    }

    draw_list->ChannelsMerge();
    // ImGui::PopItemWidth();
    ImGui::EndChild();
    ImGui::PopStyleColor();


    if (this->state.process_selected_slot > 0) {
        this->state.process_selected_slot--;
    }

    return true;
}


bool megamol::gui::Configurator::draw_canvas_grid(void) {

    try {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        assert(draw_list != nullptr);
        draw_list->ChannelsSetCurrent(0); // Background

        const ImU32 COLOR_GRID = IM_COL32(192, 192, 192, 40);
        const float GRID_SIZE = 64.0f;

        for (float x = std::fmodf(this->state.scrolling.x, GRID_SIZE); x < this->state.canvas_size.x; x += GRID_SIZE) {
            draw_list->AddLine(ImVec2(x, 0.0f) + this->state.canvas_position,
                ImVec2(x, this->state.canvas_size.y) + this->state.canvas_position, COLOR_GRID);
        }

        for (float y = std::fmodf(this->state.scrolling.y, GRID_SIZE); y < this->state.canvas_size.y; y += GRID_SIZE) {
            draw_list->AddLine(ImVec2(0.0f, y) + this->state.canvas_position,
                ImVec2(this->state.canvas_size.x, y) + this->state.canvas_position, COLOR_GRID);
        }
    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


bool megamol::gui::Configurator::draw_canvas_calls(GraphManager::GraphPtrType graph) {

    try {
        ImGuiStyle& style = ImGui::GetStyle();
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        assert(draw_list != nullptr);

        const ImU32 COLOR_CALL_CURVE = IM_COL32(225, 225, 0, 255);
        const ImU32 COLOR_CALL_BACKGROUND = IM_COL32(64, 61, 64, 255);
        const ImU32 COLOR_CALL_HIGHTLIGHT = IM_COL32(92, 92, 92, 255);
        const ImU32 COLOR_CALL_BORDER = IM_COL32(128, 128, 128, 255);

        int hovered_call = -1;

        for (auto& call : graph->GetGraphCalls()) {
            const int id = call->uid;
            ImGui::PushID(id);

            if (call->IsConnected()) {

                ImVec2 position_offset = this->state.canvas_position + this->state.scrolling;
                ImVec2 p1 = position_offset + call->GetCallSlot(Graph::CallSlotType::CALLER)->gui.position;
                ImVec2 p2 = position_offset + call->GetCallSlot(Graph::CallSlotType::CALLEE)->gui.position;

                draw_list->ChannelsSetCurrent(0); // Background
                draw_list->AddBezierCurve(
                    p1, p1 + ImVec2(50.0f, 0.0f), p2 + ImVec2(-50.0f, 0.0f), p2, COLOR_CALL_CURVE, 3.0f);

                if (this->state.show_call_names) {
                    draw_list->ChannelsSetCurrent(1); // Foreground

                    ImVec2 call_center = ImVec2(p1.x + (p2.x - p1.x) / 2.0f, p1.y + (p2.y - p1.y) / 2.0f);

                    // Draw text
                    auto call_name_width = this->utils.TextWidgetWidth(call->class_name);
                    ImGui::SetCursorScreenPos(
                        call_center + ImVec2(-(call_name_width / 2.0f), -0.5f * ImGui::GetFontSize()));
                    ImGui::Text(call->class_name.c_str());

                    // Draw box
                    draw_list->ChannelsSetCurrent(0); // Background

                    ImVec2 rect_size = ImVec2(call_name_width + (2.0f * style.ItemSpacing.x),
                        ImGui::GetFontSize() + (2.0f * style.ItemSpacing.y));
                    ImVec2 call_rect_min =
                        ImVec2(call_center.x - (rect_size.x / 2.0f), call_center.y - (rect_size.y / 2.0f));
                    ImVec2 call_rect_max = ImVec2((call_rect_min.x + rect_size.x), (call_rect_min.y + rect_size.y));
                    ImGui::SetCursorScreenPos(call_rect_min);
                    std::string label = "call_" + call->class_name + std::to_string(id);
                    ImGui::InvisibleButton(label.c_str(), rect_size);
                    bool call_active = ImGui::IsItemActive();
                    if (call_active) {
                        this->state.selected_call_uid = id;
                        this->state.selected_module_uid = -1;
                    }
                    if (ImGui::IsItemHovered() && (hovered_call < 0)) {
                        hovered_call = id;
                    }
                    ImU32 call_bg_color = (hovered_call == id || this->state.selected_call_uid == id)
                                              ? COLOR_CALL_HIGHTLIGHT
                                              : COLOR_CALL_BACKGROUND;
                    draw_list->AddRectFilled(call_rect_min, call_rect_max, call_bg_color, 4.0f);
                    draw_list->AddRect(call_rect_min, call_rect_max, COLOR_CALL_BORDER, 4.0f);
                }
            }

            ImGui::PopID();
        }
    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


bool megamol::gui::Configurator::draw_canvas_modules(GraphManager::GraphPtrType graph) {

    try {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        assert(draw_list != nullptr);

        const ImU32 COLOR_MODULE_BACKGROUND = IM_COL32(64, 61, 64, 255);
        const ImU32 COLOR_MODULE_HIGHTLIGHT = IM_COL32(92, 92, 92, 255);
        const ImU32 COLOR_MODULE_BORDER = IM_COL32(128, 128, 128, 255);

        int hovered_module = -1;
        this->state.hovered_call_slot_uid = -1;
        ImVec2 position_offset = this->state.canvas_position + this->state.scrolling;

        for (auto& mod : graph->GetGraphModules()) {

            // Draw call slots ------------------------------------------------
            // Draw call slots prior to modules to catch mouse clicks on slot area lying over module box!
            this->draw_canvas_module_call_slots(graph, mod);

            // Draw module ----------------------------------------------------
            const int id = mod->uid;
            ImGui::PushID(id);

            if (mod->gui.update) {
                this->update_module_size(mod);
            }

            // Draw text
            draw_list->ChannelsSetCurrent(1); // Foreground
            ImGui::BeginGroup();

            ImVec2 module_rect_min = position_offset + mod->gui.position;
            ImVec2 module_rect_max = module_rect_min + mod->gui.size;
            ImVec2 module_center = module_rect_min + ImVec2(mod->gui.size.x / 2.0f, mod->gui.size.y / 2.0f);

            float line_offset = 0.0f;
            if (mod->is_view) {
                line_offset = -0.5f * ImGui::GetItemsLineHeightWithSpacing();
            }

            std::string label = mod->gui.class_label + mod->class_name;
            auto class_name_width = this->utils.TextWidgetWidth(label);
            ImGui::SetCursorScreenPos(module_center + ImVec2(-(class_name_width / 2.0f),
                                                          line_offset - ImGui::GetItemsLineHeightWithSpacing()));
            ImGui::Text(label.c_str());

            label = mod->gui.name_label + mod->name;
            auto name_width = this->utils.TextWidgetWidth(label);
            ImGui::SetCursorScreenPos(module_center + ImVec2(-(name_width / 2.0f), line_offset));
            ImGui::Text(label.c_str());

            if (mod->is_view) {
                if (this->state.small_modules) {
                    std::string view_label = "[View]";
                    if (mod->is_view_instance) {
                        view_label = "[Main View]";
                    }
                    name_width = this->utils.TextWidgetWidth(view_label);
                    ImGui::SetCursorScreenPos(module_center + ImVec2(-(name_width / 2.0f), -line_offset));
                    ImGui::Text(view_label.c_str());
                } else {
                    std::string view_label = "Main View";
                    name_width = this->utils.TextWidgetWidth(view_label);
                    ImGui::SetCursorScreenPos(module_center + ImVec2(-(name_width / 2.0f) - 20.0f, -line_offset));
                    ImGui::Checkbox(view_label.c_str(), &mod->is_view_instance);
                    ImGui::SameLine();
                    this->utils.HelpMarkerToolTip(
                        "There should be only one main view.\nOtherwise first one found is used.");
                    /// TODO ensure that there is always just one main view ...
                }
            }

            ImGui::EndGroup();

            // Draw box
            draw_list->ChannelsSetCurrent(0); // Background

            ImGui::SetCursorScreenPos(module_rect_min);
            label = "module_" + mod->full_name + std::to_string(mod->uid);
            ImGui::InvisibleButton(label.c_str(), mod->gui.size);
            // Gives slots which overlap modules priority for ToolTip and Context Menu.
            if (this->state.hovered_call_slot_uid < 0) {
                this->utils.HoverToolTip(mod->description, ImGui::GetID(label.c_str()), 0.5f, 5.0f);
                // Context menu
                if (ImGui::BeginPopupContextItem()) {
                    if (ImGui::MenuItem(
                            "Delete", std::get<0>(this->hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM]).ToString().c_str())) {
                        std::get<1>(this->hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM]) = true;
                    }
                    if (ImGui::MenuItem("Rename")) {
                        this->state.open_rename_popup = true;
                        this->state.rename = &mod->name;
                    }

                    ImGui::EndPopup();
                }
            }
            bool module_active = ImGui::IsItemActive();
            if (module_active) {
                this->state.selected_module_uid = id;
                this->state.selected_call_uid = -1;
            }
            if (module_active && ImGui::IsMouseDragging(0)) {
                mod->gui.position = mod->gui.position + ImGui::GetIO().MouseDelta;
            }
            if (ImGui::IsItemHovered() && (hovered_module < 0)) {
                hovered_module = id;
            }
            ImU32 module_bg_color = (hovered_module == id || this->state.selected_module_uid == id)
                                        ? COLOR_MODULE_HIGHTLIGHT
                                        : COLOR_MODULE_BACKGROUND;
            draw_list->AddRectFilled(module_rect_min, module_rect_max, module_bg_color, 4.0f);
            draw_list->AddRect(module_rect_min, module_rect_max, COLOR_MODULE_BORDER, 4.0f);


            ImGui::PopID();
            // --------------------------------------------------------------------
        }
    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


bool megamol::gui::Configurator::draw_canvas_module_call_slots(
    GraphManager::GraphPtrType graph, Graph::ModulePtrType mod) {

    try {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        assert(draw_list != nullptr);
        draw_list->ChannelsSetCurrent(1); // Foreground

        const ImU32 COLOR_SLOT = IM_COL32(175, 175, 175, 255);
        const ImU32 COLOR_SLOT_BORDER = IM_COL32(225, 225, 225, 255);
        const ImU32 COLOR_SLOT_CALLER_LABEL = IM_COL32(0, 255, 255, 255);
        const ImU32 COLOR_SLOT_CALLER_HIGHTLIGHT = IM_COL32(0, 255, 255, 255);
        const ImU32 COLOR_SLOT_CALLEE_LABEL = IM_COL32(255, 92, 255, 255);
        const ImU32 COLOR_SLOT_CALLEE_HIGHTLIGHT = IM_COL32(255, 92, 255, 255);
        const ImU32 COLOR_SLOT_COMPATIBLE = IM_COL32(0, 255, 0, 255);

        ImU32 slot_color = COLOR_SLOT;
        ImU32 slot_highl_color;
        ImU32 slot_label_color;

        ImVec2 position_offset = this->state.canvas_position + this->state.scrolling;

        // Draw call slots for given module
        for (auto& slot_pair : mod->GetCallSlots()) {

            if (slot_pair.first == Graph::CallSlotType::CALLER) {
                slot_highl_color = COLOR_SLOT_CALLER_HIGHTLIGHT;
                slot_label_color = COLOR_SLOT_CALLER_LABEL;
            } else if (slot_pair.first == Graph::CallSlotType::CALLEE) {
                slot_highl_color = COLOR_SLOT_CALLEE_HIGHTLIGHT;
                slot_label_color = COLOR_SLOT_CALLEE_LABEL;
            }

            for (auto& slot : slot_pair.second) {
                ImGui::PushID(slot->uid);

                slot->UpdateGuiPos();

                ImVec2 slot_position = position_offset + slot->gui.position;
                std::string slot_name = slot->name;
                slot_color = COLOR_SLOT;

                ImGui::SetCursorScreenPos(slot_position - ImVec2(this->state.slot_radius, this->state.slot_radius));
                std::string label = "slot_" + mod->full_name + slot_name + std::to_string(slot->uid);
                ImGui::InvisibleButton(
                    label.c_str(), ImVec2(this->state.slot_radius * 2.0f, this->state.slot_radius * 2.0f));
                std::string tooltip = slot->description;
                if (this->state.small_modules) {
                    tooltip = slot->name + " | " + tooltip;
                }
                this->utils.HoverToolTip(tooltip, ImGui::GetID(label.c_str()), 0.5f, 5.0f);
                auto hovered = ImGui::IsItemHovered();
                auto clicked = ImGui::IsItemClicked();
                int compat_call_idx =
                    this->graph_manager.GetCompatibleCallIndex(this->state.selected_call_slot_ptr, slot);
                if (hovered) {
                    this->state.hovered_call_slot_uid = slot->uid;
                    // Check if selected call slot should be connected with current slot
                    if (this->state.process_selected_slot > 0) {
                        if (graph->AddCall(this->graph_manager.GetCallsStock(), compat_call_idx,
                                this->state.selected_call_slot_ptr, slot)) {
                            this->state.selected_call_slot_ptr = nullptr;
                        }
                    }
                }
                if (clicked) {
                    // Select / Unselect call slot
                    if (this->state.selected_call_slot_ptr != slot) {
                        this->state.selected_call_slot_ptr = slot;
                    } else {
                        this->state.selected_call_slot_ptr = nullptr;
                    }
                }
                if (hovered || (this->state.selected_call_slot_ptr == slot)) {
                    slot_color = slot_highl_color;
                }
                // Highlight if compatible to selected slot
                if (compat_call_idx > 0) {
                    slot_color = COLOR_SLOT_COMPATIBLE;
                }

                ImGui::SetCursorScreenPos(slot_position);
                draw_list->AddCircleFilled(slot_position, this->state.slot_radius, slot_color);
                draw_list->AddCircle(slot_position, this->state.slot_radius, COLOR_SLOT_BORDER);

                if (!this->state.small_modules) {
                    ImVec2 text_pos;
                    text_pos.y = slot_position.y - ImGui::GetFontSize() / 2.0f;
                    if (slot_pair.first == Graph::CallSlotType::CALLER) {
                        text_pos.x =
                            slot_position.x - this->utils.TextWidgetWidth(slot_name) - (2.0f * this->state.slot_radius);
                    } else if (slot_pair.first == Graph::CallSlotType::CALLEE) {
                        text_pos.x = slot_position.x + (2.0f * this->state.slot_radius);
                    }
                    draw_list->AddText(text_pos, slot_label_color, slot_name.c_str());
                }

                ImGui::PopID();
            }
        }
    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


bool megamol::gui::Configurator::draw_canvas_dragged_call(void) {

    try {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        assert(draw_list != nullptr);
        draw_list->ChannelsSetCurrent(0); // Background

        const auto COLOR_CALL_CURVE = IM_COL32(128, 128, 0, 255);

        ImVec2 position_offset = this->state.canvas_position + this->state.scrolling;

        if ((this->state.selected_call_slot_ptr != nullptr) && (this->state.hovered_call_slot_uid < 0)) {
            ImVec2 current_pos = ImGui::GetMousePos();
            bool mouse_inside_canvas = false;
            if ((current_pos.x >= this->state.canvas_position.x) &&
                (current_pos.x <= (this->state.canvas_position.x + this->state.canvas_size.x)) &&
                (current_pos.y >= this->state.canvas_position.y) &&
                (current_pos.y <= (this->state.canvas_position.y + this->state.canvas_size.y))) {
                mouse_inside_canvas = true;
            }
            if (ImGui::IsMouseDown(0) && mouse_inside_canvas) {
                ImVec2 p1 = position_offset + this->state.selected_call_slot_ptr->gui.position;
                ImVec2 p2 = ImGui::GetMousePos();
                if (this->state.selected_call_slot_ptr->type == Graph::CallSlotType::CALLEE) {
                    ImVec2 tmp = p1;
                    p1 = p2;
                    p2 = tmp;
                }
                draw_list->AddBezierCurve(p1, p1 + ImVec2(+50, 0), p2 + ImVec2(-50, 0), p2, COLOR_CALL_CURVE, 3.0f);
            }
        }
    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


bool megamol::gui::Configurator::update_module_size(Graph::ModulePtrType mod) {

    mod->gui.class_label = "Class: ";
    if (this->state.small_modules) mod->gui.class_label.clear();
    float class_name_length = this->utils.TextWidgetWidth(mod->gui.class_label + mod->class_name);

    mod->gui.name_label = "Name: ";
    if (this->state.small_modules) mod->gui.name_label.clear();
    float name_length = this->utils.TextWidgetWidth(mod->gui.name_label + mod->name);

    float max_label_length = std::max(class_name_length, name_length);

    float max_slot_name_length = 0.0f;
    if (!this->state.small_modules) {
        for (auto& call_slot_type_list : mod->GetCallSlots()) {
            for (auto& call_slot : call_slot_type_list.second) {
                float slot_name_length = this->utils.TextWidgetWidth(call_slot->name);
                max_slot_name_length = std::max(slot_name_length, max_slot_name_length);
            }
        }
        max_slot_name_length = (2.0f * max_slot_name_length) + (2.0f * this->state.slot_radius);
    }

    float module_width = (max_label_length + max_slot_name_length) + (4.0f * this->state.slot_radius);
    auto max_slot_count = std::max(
        mod->GetCallSlots(Graph::CallSlotType::CALLEE).size(), mod->GetCallSlots(Graph::CallSlotType::CALLER).size());
    float module_slot_height = (static_cast<float>(max_slot_count) * (this->state.slot_radius * 2.0f) * 1.5f) +
                               ((this->state.slot_radius * 2.0f) * 0.5f);
    float module_height =
        std::max(module_slot_height, ImGui::GetItemsLineHeightWithSpacing() * ((mod->is_view) ? (4.0f) : (3.0f)));
    mod->gui.size = ImVec2(module_width, module_height);

    mod->gui.update = false;
    return true;
}


bool megamol::gui::Configurator::layout_graph(GraphManager::GraphPtrType graph) {

    // Really simple layouting sorting modules into differnet layers

    std::vector<std::vector<Graph::ModulePtrType>> layers;
    layers.clear();

    // Fill first layer with modules having no connected callee
    layers.emplace_back();
    for (auto& mod : graph->GetGraphModules()) {
        bool any_connected_callee = false;
        for (auto& callee_slot : mod->GetCallSlots(Graph::CallSlotType::CALLEE)) {
            if (callee_slot->CallsConnected()) {
                any_connected_callee = true;
            }
        }
        if (!any_connected_callee) {
            layers.back().emplace_back(mod);
        }
    }

    size_t layer_idx = 0;
    bool added_module = true;
    while (added_module) {
        added_module = false;
        layers.emplace_back();
        for (auto& mod : layers[layer_idx]) {
            for (auto& caller_slot : mod->GetCallSlots(Graph::CallSlotType::CALLER)) {
                if (caller_slot->CallsConnected()) {
                    for (auto& call : caller_slot->GetConnectedCalls()) {
                        auto add_mod = call->GetCallSlot(Graph::CallSlotType::CALLEE)->GetParentModule();
                        // Check if module was already added
                        bool found_module = false;
                        for (auto& layer : layers) {
                            for (auto& m : layer) {
                                if (m == add_mod) {
                                    found_module = true;
                                }
                            }
                        }
                        if (!found_module) {
                            layers.back().emplace_back(add_mod);
                            added_module = true;
                        }
                    }
                }
            }
        }
        layer_idx++;
    }

    // Calculate new positions of modules
    const float border_offset = this->state.slot_radius * 2.0f;
    const float call_offset = 50.0f;
    ImVec2 position_offset = this->state.scrolling;
    ImVec2 pos = position_offset;
    for (auto& layer : layers) {
        float max_module_width = 0.0f;
        size_t layer_mod_cnt = layer.size();
        pos.x += border_offset;
        pos.y = border_offset;
        for (int i = 0; i < layer_mod_cnt; i++) {
            auto mod = layer[i];
            mod->gui.position = pos;
            pos.y += mod->gui.size.y + border_offset;
            max_module_width = (max_module_width < mod->gui.size.x) ? (mod->gui.size.x) : (max_module_width);
        }
        pos.x += (max_module_width + call_offset);
    }


    return true;
}


bool megamol::gui::Configurator::popup_save_project(bool open, megamol::core::CoreInstance* core_instance) {

#ifdef GUI_USE_FILEUTILS
    std::string save_project_label = "Save Project";

    if (open) {
        ImGui::OpenPopup(save_project_label.c_str());
    }
    if (ImGui::BeginPopupModal(save_project_label.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {

        bool save_project = false;

        std::string label = "File Name";
        auto flags = ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll;
        /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
        this->utils.Utf8Encode(this->state.project_filename);
        if (ImGui::InputText(label.c_str(), &this->state.project_filename, flags)) {
            save_project = true;
        }
        this->utils.Utf8Decode(this->state.project_filename);
        // Set focus on input text once (applied next frame)
        if (open) {
            ImGuiID id = ImGui::GetID(label.c_str());
            ImGui::ActivateItem(id);
        }

        bool valid = true;
        if (!HasFileExtension(this->state.project_filename, std::string(".lua"))) {
            ImGui::TextColored(ImVec4(0.9f, 0.0f, 0.0f, 1.0f), "File name needs to have the ending '.lua'");
            valid = false;
        }
        // Warn when file already exists
        if (PathExists(this->state.project_filename)) {
            ImGui::TextColored(ImVec4(0.9f, 0.0f, 0.0f, 1.0f), "File name already exists and will be overwritten.");
        }
        if (ImGui::Button("Save (Enter)")) {
            save_project = true;
        }

        if (save_project && valid) {
            if (!this->graph_manager.GetGraphs().empty()) {
                if (this->graph_manager.PROTOTYPE_SaveGraph(
                        this->state.active_graph_uid, this->state.project_filename, core_instance)) {
                    ImGui::CloseCurrentPopup();
                }
            }
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
#endif // GUI_USE_FILEUTILS

    return true;
}


bool megamol::gui::Configurator::add_new_graph(void) {

    int graph_count = this->graph_manager.GetGraphs().size();
    std::string graph_name = "Project_" + std::to_string(graph_count + 1);
    return this->graph_manager.AddGraph(graph_name);
}


bool megamol::gui::Configurator::add_new_module_to_graph(
    Graph::StockModule& mod, int compat_call_idx, const std::string& compat_call_slot_name) {

    bool retval = false;
    // Process module adding
    for (auto& graph : this->graph_manager.GetGraphs()) {
        // Look up currently active graph
        if (graph->GetUID() == this->state.active_graph_uid) {
            // Add new module
            retval = graph->AddModule(this->graph_manager.GetModulesStock(), mod.class_name);
            // If there is a call slot selected, create call to compatible call slot of new module
            if (this->state.selected_call_slot_ptr != nullptr) {
                // Get call slots of last added module
                for (auto& call_slot_map : graph->GetGraphModules().back()->GetCallSlots()) {
                    for (auto& call_slot : call_slot_map.second) {
                        if (call_slot->name == compat_call_slot_name) {
                            if (graph->AddCall(this->graph_manager.GetCallsStock(), compat_call_idx,
                                    this->state.selected_call_slot_ptr, call_slot)) {
                                this->state.selected_call_slot_ptr = nullptr;
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
