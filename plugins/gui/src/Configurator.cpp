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


#define CONFIGURATOR_SLOT_RADIUS (8.0f)


using namespace megamol;
using namespace megamol::gui;
using vislib::sys::Log;


Configurator::Configurator() : hotkeys(), graph_manager(), utils(), state() {

    // Init HotKeys
    this->hotkeys[HotkeyIndex::MODULE_SEARCH] =
        HotkeyData(megamol::core::view::KeyCode(
                       megamol::core::view::Key::KEY_M, core::view::Modifier::CTRL | core::view::Modifier::SHIFT),
            false);
    this->hotkeys[HotkeyIndex::DELETE_MODULE] =
        HotkeyData(megamol::core::view::KeyCode(megamol::core::view::Key::KEY_DELETE), false);

    // Init state
    this->state.window_rendering_state = 0;
    this->state.project_filename == "";
    this->state.active_graph_uid = -1;
    this->state.selected_module_list_uid = -1;
    this->state.selected_module_graph_uid = -1;
    this->state.hovered_call_slot_uid = -1;
    this->state.selected_call_slot = nullptr;
    this->state.process_selected_slot = 0;
    this->state.canvas_position = ImVec2(0.0f, 0.0f);
    this->state.popup_project_name = nullptr;
    // Menu
    this->state.scrolling = ImVec2(0.0f, 0.0f);
    this->state.zooming = 1.0f;
    this->state.show_grid = true;
    this->state.show_call_names = true;
    this->state.minimize_modules = false;
    this->state.relayout_graph = false;
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
        ImGui::BeginChild("info", ImVec2(0.0f, (2.0f * ImGui::GetItemsLineHeightWithSpacing())), true,
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);
        std::string label =
            "This is a PROTOTYPE. Any changes will NOT EFFECT the currently loaded project.\n"
            "You can SAVE the modified graph to a separate PROJECT FILE (parameters are not considered yet).";
        ImGui::TextColored(ImVec4(0.9f, 0.2f, 0.2f, 1.0f), label.c_str());
        ImGui::EndChild();
        // --------------------------------------------------------------------

        // Project (graph) tabs
        int delete_graph_uid = -1; // (Assuming only one closed tab per frame)
        bool open_rename_popup = false;

        ImGuiTabBarFlags flags = ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_Reorderable;
        ImGui::BeginTabBar("Graphs", flags);
        for (auto& graph : this->graph_manager.GetGraphs()) {
            bool open = true;

            // Tab showing one graph
            if (ImGui::BeginTabItem(graph->GetName().c_str(), &open, ImGuiTabItemFlags_None)) {
                // Tab context menu
                /* DISBALED since changing instance name is not serialized properly yet.
                if (ImGui::BeginPopupContextItem()) {
                    if (ImGui::MenuItem("Rename Project")) {
                        open_rename_popup = true;
                        this->state.popup_project_name = &graph->GetName();
                    }
                    ImGui::EndPopup();
                }
                */

                this->state.active_graph_uid = graph->GetUID();
                this->draw_window_graph_canvas(graph);
                ImGui::EndTabItem();
            }
            // (Do not delete graph while looping through graphs list)
            if (!open) {
                delete_graph_uid = graph->GetUID();
            }
        }
        ImGui::EndTabBar();

        // Delete closed tab
        if (delete_graph_uid > 0) {
            this->graph_manager.DeleteGraph(delete_graph_uid);
        }

        // Rename project tab pop-up
        if (open_rename_popup) {
            ImGui::OpenPopup("Rename Project");
        }
        if (ImGui::BeginPopupModal("Rename Project", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            auto flags = ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll;

            if (ImGui::InputText("Enter new  project name", this->state.popup_project_name, flags)) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

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

        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("New Project (Graph)", "(no hotkey)")) {
                this->add_new_graph();
            }
#ifdef GUI_USE_FILEUTILS
            // Load/save parameter values to LUA file
            if (ImGui::MenuItem("Save Project (Graph)", "(no hotkey)")) {
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
            if (ImGui::MenuItem("Reset Scrolling", "(no hotkey)")) {
                this->state.scrolling = ImVec2(0.0f, 0.0f);
            }
            if (ImGui::MenuItem("Reset Zooming", "(no hotkey)")) {
                this->state.zooming = 1.0f;
            }
            if (ImGui::MenuItem("Show Grid", "(no hotkey)", this->state.show_grid)) {
                this->state.show_grid = !this->state.show_grid;
            }
            if (ImGui::MenuItem("Show Call Names", "(no hotkey)", this->state.show_call_names)) {
                this->state.show_call_names = !this->state.show_call_names;
            }
            if (ImGui::MenuItem("Minimize Modules", "(no hotkey)", this->state.minimize_modules)) {
                this->state.minimize_modules = !this->state.minimize_modules;
            }
            if (ImGui::MenuItem("Layout Graph", "(no hotkey)")) {
                this->state.relayout_graph = true;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Info")) {
            std::string info_text = "Additonal supported actions:\n"
                                    "- Add selected module from stock list\n"
                                    "     [Double Click] with left mouse button"
                                    " | [Richt Click] on selected module / Context Menu: Add \n"
                                    "- Delete selected module/call from graph\n"
                                    "     Select item an press [Delete]"
                                    " | [Richt Click] on selected item / Context Menu: Delete \n";
            ImGui::Text(info_text.c_str());
            ImGui::EndMenu();
        }

        ImGui::SameLine(260.0f);

        ImGui::Separator();
        ImGui::Text("Scrolling: %.2f,%.2f (Middle Mouse Button)", this->state.scrolling.x, this->state.scrolling.y);
        ImGui::Separator();
        /* DISABLED since not implemented yet
        ImGui::Text("Zooming: %.2f (Mouse Wheel)", this->state.zooming);
        ImGui::Separator();
        */

        ImGui::EndMenuBar();
    }

    // Pop-Up(s)
    this->save_project_popup(open_popup_project, core_instance);

    return true;
}


bool megamol::gui::Configurator::draw_window_module_list(void) {

    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    const ImU32 COLOR_MODULE_VIEW = IM_COL32(0, 225, 225, 255);

    const float child_width = 250.0f;
    const float child_height = 2.5f * ImGui::GetItemsLineHeightWithSpacing();

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
        if (this->state.selected_call_slot != nullptr) {
            compat_filter = false;
            for (auto& cst : mod.call_slots) {
                for (auto& cs : cst.second) {
                    int cpidx = this->graph_manager.GetCompatibleCallIndex(this->state.selected_call_slot, cs);
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
            // Changing color for views
            if (mod.is_view) {
                ImGui::PushStyleColor(ImGuiCol_Text, COLOR_MODULE_VIEW);
            }

            std::string label = std::to_string(id) + " " + mod.class_name + " (" + mod.plugin_name + ")";
            if (ImGui::Selectable(label.c_str(), (id == this->state.selected_module_list_uid))) {
                this->state.selected_module_list_uid = id;
            }
            // Left mouse button double click action
            if (ImGui::IsMouseDoubleClicked(0) && ImGui::IsItemHovered()) {
                this->add_new_module_to_graph(mod, compat_call_index, compat_call_slot_name);
            }
            // Context menu
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Add", "Double-Click")) {
                    this->add_new_module_to_graph(mod, compat_call_index, compat_call_slot_name);
                }
                ImGui::EndPopup();
            }
            // Hover tool tip
            this->utils.HoverToolTip(mod.description, id, 0.5f, 5.0f);

            if (mod.is_view) {
                ImGui::PopStyleColor();
            }
            ImGui::PopID();
            id++;
        }
    };

    ImGui::EndChild();
    ImGui::EndGroup();

    return true;
}


bool megamol::gui::Configurator::draw_window_graph_canvas(GraphManager::GraphPtrType graph) {

    ImGuiIO& io = ImGui::GetIO();
    /// Font scaling with zooming factor is not possible locally within window (only prior to ImGui::Begin()).
    /// Add separate font for graph?!

    const ImU32 COLOR_MODULE_BACKGROUND = IM_COL32(60, 60, 70, 255);

    // Process module deletion
    if (std::get<1>(this->hotkeys[HotkeyIndex::DELETE_MODULE])) {
        std::get<1>(this->hotkeys[HotkeyIndex::DELETE_MODULE]) = false;
        this->state.selected_call_slot = nullptr;
        graph->DeleteModule(this->state.selected_module_graph_uid);
    }

    // Register trigger for connecting call
    if ((this->state.selected_call_slot != nullptr) && (io.MouseReleased[0])) {
        this->state.process_selected_slot = 2;
    }

    // Draw child canvas ------------------------------------------------------
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(1, 1));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleColor(ImGuiCol_ChildWindowBg, COLOR_MODULE_BACKGROUND);
    ImGui::BeginChild("region", ImVec2(0.0f, 0.0f), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);
    ImGui::PushItemWidth(120.0f);
    this->state.canvas_position = ImGui::GetCursorScreenPos();
    ImVec2 position_offset = ImGui::GetCursorScreenPos() + this->state.scrolling;
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);
    draw_list->ChannelsSplit(2);

    if (ImGui::IsMouseClicked(0) && ImGui::IsWindowHovered()) {
        this->state.selected_module_graph_uid = -1;
        this->state.selected_call_slot = nullptr;
    }

    // Display grid -------------------------------------------------------
    if (this->state.show_grid) {
        this->draw_canvas_grid(this->state.scrolling, this->state.zooming);
    }
    ImGui::PopStyleVar(2);

    // Draw modules -----------------------------------------------------
    this->draw_canvas_modules(graph, position_offset);

    // Draw call --------------------------------------------------------
    this->draw_canvas_calls(graph, position_offset);

    // Draw dragged call --------------------------------------------------------
    this->draw_canvas_dragged_call(position_offset);

    // Zoomin and Scaling  ----------------------------------------------------
    if (ImGui::IsWindowHovered() && !ImGui::IsAnyItemActive()) {
        // Scrolling (2 = Middle Mouse Button)
        if (ImGui::IsMouseDragging(2, 0.0f)) {
            this->state.scrolling = this->state.scrolling + ImGui::GetIO().MouseDelta;
        }
        /* DISABLED since not implemented yet
        // Zooming (Mouse Wheel)
        float last_zooming = this->state.zooming;
        this->state.zooming = this->state.zooming + io.MouseWheel / 10.0f;
        this->state.zooming = (this->state.zooming < 0.1f) ? (0.1f) : (this->state.zooming);
        if (last_zooming != this->state.zooming) {
           /// TODO process changed zooming ...
        }
        */
    }

    draw_list->ChannelsMerge();
    ImGui::PopItemWidth();
    ImGui::EndChild();
    ImGui::PopStyleColor();

    if (this->state.process_selected_slot > 0) {
        this->state.process_selected_slot--;
    }

    return true;
}


bool megamol::gui::Configurator::draw_canvas_grid(ImVec2 scrolling, float zooming) {

    try {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        assert(draw_list != nullptr);
        draw_list->ChannelsSetCurrent(0); // Background

        const ImU32 COLOR_GRID = IM_COL32(200, 200, 200, 40);
        const float GRID_SIZE = 64.0f;

        ImVec2 canvas_size = ImGui::GetWindowSize();
        ImVec2 win_pos = ImGui::GetCursorScreenPos();
        float grid_size = GRID_SIZE * zooming;

        for (float x = std::fmodf(scrolling.x, grid_size); x < canvas_size.x; x += grid_size) {
            draw_list->AddLine(ImVec2(x, 0.0f) + win_pos, ImVec2(x, canvas_size.y) + win_pos, COLOR_GRID);
        }

        for (float y = std::fmodf(scrolling.y, grid_size); y < canvas_size.y; y += grid_size) {
            draw_list->AddLine(ImVec2(0.0f, y) + win_pos, ImVec2(canvas_size.x, y) + win_pos, COLOR_GRID);
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


bool megamol::gui::Configurator::draw_canvas_calls(GraphManager::GraphPtrType graph, ImVec2 position_offset) {

    try {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        assert(draw_list != nullptr);
        draw_list->ChannelsSetCurrent(1); // Foreground

        const ImU32 COLOR_CALL_CURVE = IM_COL32(200, 200, 100, 255);

        for (auto& call : graph->GetGraphCalls()) {
            ImGui::PushID(call->uid);

            if (call->IsConnected()) {

                ImVec2 p1 = position_offset + call->GetCallSlot(Graph::CallSlotType::CALLER)->GetGuiPos();
                ImVec2 p2 = position_offset + call->GetCallSlot(Graph::CallSlotType::CALLEE)->GetGuiPos();


                if (this->state.show_call_names) {
                    /// TODO
                    /*
                    ImVec2 button_cursour_pos;
                    ImVec2 button_size;

                    ImVec2 module_rect_min = position_offset + mod->gui.position;
                    ImVec2 module_rect_max = module_rect_min + mod->gui.size;
                    ImVec2 module_center = module_rect_min + ImVec2(mod->gui.size.x / 2.0f, mod->gui.size.y / 2.0f);

                    ImGui::BeginGroup();

                    float line_offset = 0.0f;
                    if (mod->is_view) {
                        line_offset = -(ImGui::GetItemsLineHeightWithSpacing() / 2.0f);
                    }

                    auto class_name_width = this->utils.TextWidgetWidth(mod->class_name);
                    ImGui::SetCursorScreenPos(module_center + ImVec2(-(class_name_width / 2.0f),
                        line_offset - ImGui::GetItemsLineHeightWithSpacing()));
                    ImGui::Text(mod->class_name.c_str());


                    ImGui::SetCursorScreenPos(button_cursour_pos);
                    std::string label = "call_" + call->class_name + std::to_string(call->uid);
                    ImGui::InvisibleButton(label.c_str(), button_size);

                    draw_list->AddRectFilled(module_rect_min, module_rect_max, module_bg_color, 4.0f);
                    draw_list->AddRect(module_rect_min, module_rect_max, COLOR_MODULE_BORDER, 4.0f);
                    */
                }


                draw_list->AddBezierCurve(
                    p1, p1 + ImVec2(50.0f, 0.0f), p2 + ImVec2(-50.0f, 0.0f), p2, COLOR_CALL_CURVE, 3.0f);
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


bool megamol::gui::Configurator::draw_canvas_modules(GraphManager::GraphPtrType graph, ImVec2 position_offset) {

    try {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        assert(draw_list != nullptr);

        const ImU32 COLOR_MODULE = IM_COL32(60, 60, 60, 255);
        const ImU32 COLOR_MODULE_HIGHTL = IM_COL32(75, 75, 75, 255);
        const ImU32 COLOR_MODULE_BORDER = IM_COL32(100, 100, 100, 255);
        const ImVec4 COLOR_VIEW_LABEL = ImVec4(0.5f, 0.5f, 0.0f, 1.0f);

        int hovered_module = -1;
        this->state.hovered_call_slot_uid = -1;
        for (auto& mod : graph->GetGraphModules()) {

            // Draw call slots ------------------------------------------------
            // Draw call slots prior to modules to catch mouse clicks on slot area lying over module box!
            this->draw_canvas_module_call_slots(graph, mod, position_offset);

            // Draw module ----------------------------------------------------
            const int id = mod->uid;
            ImGui::PushID(id);

            if (!mod->gui.initialized) {
                this->init_module_gui_params(mod);
            }

            // Draw text
            draw_list->ChannelsSetCurrent(1); // Foreground

            ImVec2 module_rect_min = position_offset + mod->gui.position;
            ImVec2 module_rect_max = module_rect_min + mod->gui.size;
            ImVec2 module_center = module_rect_min + ImVec2(mod->gui.size.x / 2.0f, mod->gui.size.y / 2.0f);

            ImGui::BeginGroup();

            float line_offset = 0.0f;
            if (mod->is_view) {
                line_offset = -(ImGui::GetItemsLineHeightWithSpacing() / 2.0f);
            }

            auto class_name_width = this->utils.TextWidgetWidth(mod->class_name);
            ImGui::SetCursorScreenPos(module_center + ImVec2(-(class_name_width / 2.0f),
                                                          line_offset - ImGui::GetItemsLineHeightWithSpacing()));
            ImGui::Text(mod->class_name.c_str());

            auto name_width = this->utils.TextWidgetWidth(mod->name);
            ImGui::SetCursorScreenPos(module_center + ImVec2(-(name_width / 2.0f), line_offset));
            ImGui::Text(mod->name.c_str());

            if (mod->is_view) {
                // std::string view_label = "[view]";
                // name_width = this->utils.TextWidgetWidth(view_label);
                // ImGui::SetCursorScreenPos(module_center + ImVec2(-(name_width / 2.0f), -line_offset));
                // ImGui::TextColored(COLOR_VIEW_LABEL, view_label.c_str());

                std::string view_label = "Main View";
                name_width = this->utils.TextWidgetWidth(view_label);
                ImGui::SetCursorScreenPos(module_center + ImVec2(-(name_width / 2.0f), -line_offset));
                ImGui::Checkbox(view_label.c_str(), &mod->is_view_instance);
            }

            ImGui::EndGroup();

            // Draw box
            draw_list->ChannelsSetCurrent(0); // Background

            ImGui::SetCursorScreenPos(module_rect_min);
            std::string label = "module_" + mod->full_name + std::to_string(mod->uid);
            ImGui::InvisibleButton(label.c_str(), mod->gui.size);
            // Gives slots which overlap modules priority for ToolTip and Context Menu.
            if (this->state.hovered_call_slot_uid < 0) {
                this->utils.HoverToolTip(mod->description, ImGui::GetID(label.c_str()), 0.5f, 5.0f);
                // Context menu
                if (ImGui::BeginPopupContextItem()) {
                    if (ImGui::MenuItem(
                            "Delete", std::get<0>(this->hotkeys[HotkeyIndex::DELETE_MODULE]).ToString().c_str())) {
                        std::get<1>(this->hotkeys[HotkeyIndex::DELETE_MODULE]) = true;
                    }
                    ImGui::EndPopup();
                }
            }
            bool module_active = ImGui::IsItemActive();
            if (module_active) {
                this->state.selected_module_graph_uid = id;
            }
            if (module_active && ImGui::IsMouseDragging(0)) {
                mod->gui.position = mod->gui.position + ImGui::GetIO().MouseDelta;
            }
            if (ImGui::IsItemHovered() && (hovered_module < 0)) {
                hovered_module = id;
            }
            ImU32 module_bg_color = (hovered_module == id || this->state.selected_module_graph_uid == id)
                                        ? COLOR_MODULE_HIGHTL
                                        : COLOR_MODULE;
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
    GraphManager::GraphPtrType graph, Graph::ModulePtrType mod, ImVec2 position_offset) {

    try {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        assert(draw_list != nullptr);
        draw_list->ChannelsSetCurrent(1); // Foreground

        const ImU32 COLOR_SLOT = IM_COL32(175, 175, 175, 255);
        const ImU32 COLOR_SLOT_BORDER = IM_COL32(225, 225, 225, 255);
        const ImU32 COLOR_SLOT_CALLER_LABEL = IM_COL32(0, 192, 192, 255);
        const ImU32 COLOR_SLOT_CALLER_HIGHTL = IM_COL32(0, 192, 192, 255);
        const ImU32 COLOR_SLOT_CALLEE_LABEL = IM_COL32(192, 192, 0, 255);
        const ImU32 COLOR_SLOT_CALLEE_HIGHTL = IM_COL32(192, 192, 0, 255);
        const ImU32 COLOR_SLOT_COMPATIBLE = IM_COL32(25, 225, 25, 255);

        ImU32 slot_color = COLOR_SLOT;
        ImU32 slot_highl_color;
        ImU32 slot_label_color;

        // Draw call slots for given module
        for (auto& slot_pair : mod->GetCallSlots()) {

            if (slot_pair.first == Graph::CallSlotType::CALLER) {
                slot_highl_color = COLOR_SLOT_CALLER_HIGHTL;
                slot_label_color = COLOR_SLOT_CALLER_LABEL;
            } else if (slot_pair.first == Graph::CallSlotType::CALLEE) {
                slot_highl_color = COLOR_SLOT_CALLEE_HIGHTL;
                slot_label_color = COLOR_SLOT_CALLEE_LABEL;
            }

            for (auto& slot : slot_pair.second) {
                ImGui::PushID(slot->uid);

                ImVec2 slot_position = position_offset + slot->GetGuiPos();
                std::string slot_name = slot->name;
                slot_color = COLOR_SLOT;

                ImGui::SetCursorScreenPos(slot_position - ImVec2(CONFIGURATOR_SLOT_RADIUS, CONFIGURATOR_SLOT_RADIUS));
                std::string label = "slot_" + mod->full_name + slot_name + std::to_string(slot->uid);
                ImGui::InvisibleButton(
                    label.c_str(), ImVec2(CONFIGURATOR_SLOT_RADIUS * 2.0f, CONFIGURATOR_SLOT_RADIUS * 2.0f));
                this->utils.HoverToolTip(slot->description, ImGui::GetID(label.c_str()), 0.5f, 5.0f);
                auto hovered = ImGui::IsItemHovered();
                auto clicked = ImGui::IsItemClicked();
                int compat_call_idx = this->graph_manager.GetCompatibleCallIndex(this->state.selected_call_slot, slot);
                if (hovered) {
                    this->state.hovered_call_slot_uid = slot->uid;
                    // Check if selected call slot should be connected with current slot
                    if (this->state.process_selected_slot > 0) {
                        if (graph->AddCall(this->graph_manager.GetCallsStock(), compat_call_idx,
                                this->state.selected_call_slot, slot)) {
                            this->state.selected_call_slot = nullptr;
                        }
                    }
                }
                if (clicked) {
                    // Select / Unselect call slot
                    if (this->state.selected_call_slot != slot) {
                        this->state.selected_call_slot = slot;
                    } else {
                        this->state.selected_call_slot = nullptr;
                    }
                }
                if (hovered || (this->state.selected_call_slot == slot)) {
                    slot_color = slot_highl_color;
                }
                // Highlight if compatible to selected slot
                if (compat_call_idx > 0) {
                    slot_color = COLOR_SLOT_COMPATIBLE;
                }

                ImGui::SetCursorScreenPos(slot_position);
                draw_list->AddCircleFilled(slot_position, CONFIGURATOR_SLOT_RADIUS, slot_color);
                draw_list->AddCircle(slot_position, CONFIGURATOR_SLOT_RADIUS, COLOR_SLOT_BORDER);

                ImVec2 text_pos;
                text_pos.y = slot_position.y - ImGui::GetFontSize() / 2.0f;
                if (slot_pair.first == Graph::CallSlotType::CALLER) {
                    text_pos.x =
                        slot_position.x - this->utils.TextWidgetWidth(slot_name) - (2.0f * CONFIGURATOR_SLOT_RADIUS);
                } else if (slot_pair.first == Graph::CallSlotType::CALLEE) {
                    text_pos.x = slot_position.x + (2.0f * CONFIGURATOR_SLOT_RADIUS);
                }
                draw_list->AddText(text_pos, slot_label_color, slot_name.c_str());

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


bool megamol::gui::Configurator::draw_canvas_dragged_call(ImVec2 position_offset) {

    try {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        assert(draw_list != nullptr);
        draw_list->ChannelsSetCurrent(0); // Background

        const auto COLOR_CALL_CURVE = IM_COL32(200, 200, 100, 255);

        if ((this->state.selected_call_slot != nullptr) && (this->state.hovered_call_slot_uid < 0)) {
            ImVec2 canvas_size = ImGui::GetWindowSize();
            ImVec2 canvas_pos = this->state.canvas_position;
            ImVec2 current_pos = ImGui::GetMousePos();
            bool mouse_inside_canvas = false;
            if ((current_pos.x >= canvas_pos.x) && (current_pos.x <= (canvas_pos.x + canvas_size.x)) &&
                (current_pos.y >= canvas_pos.y) && (current_pos.y <= (canvas_pos.y + canvas_size.y))) {
                mouse_inside_canvas = true;
            }
            if (ImGui::IsMouseDown(0) && mouse_inside_canvas) {
                ImVec2 p1 = position_offset + this->state.selected_call_slot->GetGuiPos();
                ImVec2 p2 = ImGui::GetMousePos();
                if (this->state.selected_call_slot->type == Graph::CallSlotType::CALLEE) {
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


bool megamol::gui::Configurator::init_module_gui_params(Graph::ModulePtrType mod) {

    // Init size of module (prior to position) --------------------------------

    float max_class_name_length = this->utils.TextWidgetWidth(mod->class_name);
    float max_full_name_length = 0.0f; // this->utils.TextWidgetWidth(mod->full_name); // Not displayed
    float max_name_length = this->utils.TextWidgetWidth(mod->name);
    float max_label_length = std::max(std::max(max_class_name_length, max_full_name_length), max_name_length);

    float max_slot_name_length = 0.0f;
    for (auto& call_slot_type_list : mod->GetCallSlots()) {
        for (auto& call_slot : call_slot_type_list.second) {
            float slot_name_length = this->utils.TextWidgetWidth(call_slot->name);
            max_slot_name_length = std::max(slot_name_length, max_slot_name_length);
        }
    }
    float module_width = (max_label_length + 2.0f * max_slot_name_length) + (6.0f * CONFIGURATOR_SLOT_RADIUS);
    auto max_slot_count = std::max(
        mod->GetCallSlots(Graph::CallSlotType::CALLEE).size(), mod->GetCallSlots(Graph::CallSlotType::CALLER).size());
    float module_slot_height = (static_cast<float>(max_slot_count) * (CONFIGURATOR_SLOT_RADIUS * 2.0f) * 1.5f) +
                               ((CONFIGURATOR_SLOT_RADIUS * 2.0f) * 0.5f);
    float module_height =
        std::max(module_slot_height, ImGui::GetItemsLineHeightWithSpacing() * ((mod->is_view) ? (4.0f) : (3.0f)));
    mod->gui.size = ImVec2(module_width, module_height);

    // Init position ----------------------------------------------------------

    ImVec2 canvas_size = ImGui::GetWindowSize();
    mod->gui.position = ImVec2((canvas_size.x - mod->gui.size.x) / 2.0f, (canvas_size.y - mod->gui.size.y) / 2.0f);

    mod->gui.initialized = true;

    return true;
}


bool megamol::gui::Configurator::save_project_popup(bool open, megamol::core::CoreInstance* core_instance) {

#ifdef GUI_USE_FILEUTILS
    std::string save_project_label = "Save Project";

    if (open) {
        ImGui::OpenPopup(save_project_label.c_str());
    }
    if (ImGui::BeginPopupModal(save_project_label.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {

        std::string label = "File Name###Save Project";
        if (open) {
            ImGuiID id = ImGui::GetID(label.c_str());
            ImGui::ActivateItem(id);
        }
        /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
        this->utils.Utf8Encode(this->state.project_filename);
        ImGui::InputText(label.c_str(), &this->state.project_filename, ImGuiInputTextFlags_None);
        this->utils.Utf8Decode(this->state.project_filename);

        bool valid = true;
        if (!HasFileExtension(this->state.project_filename, std::string(".lua"))) {
            ImGui::TextColored(ImVec4(0.9f, 0.0f, 0.0f, 1.0f), "File name needs to have the ending '.lua'");
            valid = false;
        }
        // Warn when file already exists
        if (PathExists(this->state.project_filename)) {
            ImGui::TextColored(ImVec4(0.9f, 0.0f, 0.0f, 1.0f), "File name already exists and will be overwritten.");
        }
        if (ImGui::Button("Save")) {
            if (valid) {
                if (!this->graph_manager.GetGraphs().empty()) {
                    if (this->graph_manager.PROTOTYPE_SaveGraph(
                            this->state.active_graph_uid, this->state.project_filename, core_instance)) {
                        ImGui::CloseCurrentPopup();
                    }
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
    this->graph_manager.AddGraph(graph_name);
    return true;
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
            // If there is a call slot selected, create a call connection to compatible call slot of new
            // module
            if (this->state.selected_call_slot != nullptr) {
                // Get call slots of last added module
                for (auto& call_slot_map : graph->GetGraphModules().back()->GetCallSlots()) {
                    for (auto& call_slot : call_slot_map.second) {
                        if (call_slot->name == compat_call_slot_name) {
                            if (graph->AddCall(this->graph_manager.GetCallsStock(), compat_call_idx,
                                    this->state.selected_call_slot, call_slot)) {
                                this->state.selected_call_slot = nullptr;
                            }
                        }
                    }
                }
            }
        }
    }
    return retval;
}
