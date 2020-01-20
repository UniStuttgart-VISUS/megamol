/*
 * Configurator.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

/**
 * USED HOTKEYS:
 *
 * - Search module:   Shift + Ctrl  + m
 */

#include "stdafx.h"
#include "Configurator.h"


using namespace megamol;
using namespace megamol::gui;
using vislib::sys::Log;


Configurator::Configurator() : hotkeys(), graph(), utils(), window_rendering_state(0), project_filename(), state() {

    // Init HotKeys
    this->hotkeys[HotkeyIndex::MODULE_SEARCH] =
        HotkeyData(megamol::core::view::KeyCode(
                       megamol::core::view::Key::KEY_M, core::view::Modifier::CTRL | core::view::Modifier::SHIFT),
            false);
    this->hotkeys[HotkeyIndex::DELETE_MODULE] =
        HotkeyData(megamol::core::view::KeyCode(megamol::core::view::Key::KEY_DELETE), false);

    // Init state
    this->state.selected_module_list = -1;
    this->state.scrolling = ImVec2(0.0f, 0.0f);
    this->state.zooming = 1.0f;
    this->state.show_grid = true;
    this->state.selected_module_graph = -1;
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

    if (this->window_rendering_state < 2) {
        // 1] Show pop-up before before calling UpdateAvailableModulesCallsOnce of graph.
        /// Rendering of pop-up requires two complete Draw calls!
        bool open = true;
        std::string popup_label = "Loading";
        if (this->window_rendering_state == 0) {
            ImGui::OpenPopup(popup_label.c_str());
        }
        ImGuiWindowFlags popup_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove;
        if (ImGui::BeginPopupModal(popup_label.c_str(), &open, popup_flags)) {
            ImGui::Text("Please wait...\nLoading available modules and calls for configurator.");
            ImGui::EndPopup();
        }
        this->window_rendering_state++;
    } else if (this->window_rendering_state == 2) {
        // 2] Load available modules and calls once(!)
        this->graph.UpdateAvailableModulesCallsOnce(core_instance);
        this->window_rendering_state++;
    } else {
        // 3] Render configurator gui content
        this->draw_window_menu(core_instance);
        this->draw_window_module_list();
        ImGui::SameLine(); // Draws module list and graph canvas next to each other
        this->draw_window_graph_canvas();
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
    std::string save_project_label = "Save Project";
    if (ImGui::BeginMenuBar()) {

        if (ImGui::BeginMenu("File")) {
#ifdef GUI_USE_FILEUTILS
            // Load/save parameter values to LUA file
            if (ImGui::MenuItem(save_project_label.c_str(), "no hotkey set")) {
                open_popup_project = true;
            }
            /// TODO: Load parameter file
            // if (ImGui::MenuItem("Load Project", "no hotkey set")) {
            //     std::string projectFilename;
            //     this->GetCoreInstance()->LoadProject(vislib::StringA(projectFilename.c_str()));
            // }
#endif // GUI_USE_FILEUTILS
            ImGui::EndMenu();
        }

        ImGui::Separator();
        if (ImGui::BeginMenu("Graph")) {
            ImGui::Checkbox("Show Grid", &this->state.show_grid);
            if (ImGui::MenuItem("Reset Scrolling")) {
                this->state.scrolling = ImVec2(0.0f, 0.0f);
            }
            if (ImGui::MenuItem("Reset Zooming")) {
                this->state.zooming = 1.0f;
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();
        ImGui::Text("Scrolling: %.2f,%.2f (Middle Mouse Button)", this->state.scrolling.x, this->state.scrolling.y);
        ImGui::Separator();
        ImGui::Text("Zooming: %.2f (Mouse Wheel)", this->state.zooming);
        ImGui::Separator();

        ImGui::EndMenuBar();
    }

    // Pop-Up(s)
#ifdef GUI_USE_FILEUTILS
    if (open_popup_project) {
        ImGui::OpenPopup(save_project_label.c_str());
    }
    if (ImGui::BeginPopupModal(save_project_label.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {

        std::string label = "File Name###Save Project";
        if (open_popup_project) {
            ImGuiID id = ImGui::GetID(label.c_str());
            ImGui::ActivateItem(id);
        }
        /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
        this->utils.Utf8Encode(project_filename);
        ImGui::InputText(label.c_str(), &project_filename, ImGuiInputTextFlags_None);
        this->utils.Utf8Decode(project_filename);

        bool valid = true;
        if (!HasFileExtension(project_filename, std::string(".lua"))) {
            ImGui::TextColored(ImVec4(0.9f, 0.0f, 0.0f, 1.0f), "File name needs to have the ending '.lua'");
            valid = false;
        }
        // Warn when file already exists
        if (PathExists(project_filename)) {
            ImGui::TextColored(ImVec4(0.9f, 0.0f, 0.0f, 1.0f), "File name already exists and will be overwritten.");
        }
        if (ImGui::Button("Save")) {
            if (valid) {
                if (this->graph.PROTOTYPE_SaveGraph(project_filename, core_instance)) {
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


bool megamol::gui::Configurator::draw_window_module_list(void) {

    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

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

    int id = 1; // Start with 1 because it is used as enumeration
    for (auto& mod : this->graph.GetAvailableModulesList()) {
        if (search_string.empty() || this->utils.FindCaseInsensitiveSubstring(mod.class_name, search_string)) {
            ImGui::PushID(id);
            std::string label = std::to_string(id) + " " + mod.class_name + " (" + mod.plugin_name + ")";
            if (ImGui::Selectable(label.c_str(), (id == this->state.selected_module_list))) {
                this->state.selected_module_list = id;
            }
            // Left mouse button double click action
            if (ImGui::IsMouseDoubleClicked(0) && ImGui::IsItemHovered()) {
                this->graph.AddModule(mod.class_name);
            }
            // Context menu
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Add Module", "Double-Click")) {
                    this->graph.AddModule(mod.class_name);
                }
                ImGui::EndPopup();
            }
            // Hover tool tip
            this->utils.HoverToolTip(mod.description, id, 0.5f, 5.0f);
            ImGui::PopID();
        }
        id++;
    };

    ImGui::EndChild();
    ImGui::EndGroup();

    return true;
}


bool megamol::gui::Configurator::draw_window_graph_canvas(void) {

    ImGuiIO& io = ImGui::GetIO();
    /// Font scaling with zooming factor is not possible locally within window (only prior to ImGui::Begin()).
    /// (io.FontDefault->Scale = this->state.zooming;)

    // Process module deletion
    if (std::get<1>(this->hotkeys[HotkeyIndex::DELETE_MODULE])) {
        std::get<1>(this->hotkeys[HotkeyIndex::DELETE_MODULE]) = false;
        this->graph.DeleteModule(this->state.selected_module_graph);
    }

    ImGui::BeginGroup();

    // Info text --------------------------------------------------------------
    ImGui::BeginChild("info", ImVec2(0.0f, (2.0f * ImGui::GetItemsLineHeightWithSpacing())), true,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);
    std::string label =
        "This is a PROTOTYPE. Any changes will NOT EFFECT the currently loaded project.\n"
        "You can SAVE the modified graph to a separate PROJECT FILE (parameters are not considered yet).";
    ImGui::TextColored(ImVec4(0.9f, 0.2f, 0.2f, 1.0f), label.c_str());
    ImGui::EndChild();

    // Draw child canvas ------------------------------------------------------
    {
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(1.0f, 1.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleColor(ImGuiCol_ChildWindowBg, IM_COL32(60, 60, 70, 200));
        ImGui::BeginChild("region", ImVec2(0.0f, 0.0f), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);
        ImGui::PushItemWidth(120.0f);
        ImVec2 position_offset = ImGui::GetCursorScreenPos() + this->state.scrolling;
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        draw_list->ChannelsSplit(2);
        draw_list->ChannelsSetCurrent(0); // Background

        // Display grid -----------------------------------------------------------
        if (this->state.show_grid) {
            this->draw_canvas_grid(this->state.scrolling, this->state.zooming);
        }

        // Draw selected call slot link -------------------------------------------
        /// Call this before rendering module call slots.
        this->draw_canvas_selected_call(position_offset);

        // Display call links -----------------------------------------------------
        this->draw_canvas_calls(position_offset);

        // Display modules -----------------------------------------------------
        this->draw_canvas_modules(position_offset);

        draw_list->ChannelsMerge();
    }

    // Zoomin and Scaling  ----------------------------------------------------

    if (ImGui::IsWindowHovered() && !ImGui::IsAnyItemActive()) {
        // Scrolling (2 = Middle Mouse Button)
        if (ImGui::IsMouseDragging(2, 0.0f)) {
            this->state.scrolling = this->state.scrolling + ImGui::GetIO().MouseDelta;
        }
        // Zooming (Mouse Wheel)
        float last_zooming = this->state.zooming;
        this->state.zooming = this->state.zooming + io.MouseWheel / 10.0f;
        this->state.zooming = (this->state.zooming < 0.1f) ? (0.1f) : (this->state.zooming);
        if (last_zooming != this->state.zooming) {
            /// TODO ...
        }
    }

    ImGui::PopItemWidth();
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);

    // Hovered text -----------------------------------------------------------
    /*
    ImGui::BeginChild("desc", ImVec2(0.0f, 0.0f), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);



    ImGui::EndChild();
    */

    ImGui::EndGroup();

    return true;
}


bool megamol::gui::Configurator::draw_canvas_grid(ImVec2 scrolling, float zooming) {

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_size = ImGui::GetWindowSize();
    ImVec2 win_pos = ImGui::GetCursorScreenPos();
    ImU32 grid_color = IM_COL32(200, 200, 200, 40);
    float grid_size = 64.0f * zooming;

    try {
        for (float x = std::fmodf(scrolling.x, grid_size); x < canvas_size.x; x += grid_size) {
            draw_list->AddLine(ImVec2(x, 0.0f) + win_pos, ImVec2(x, canvas_size.y) + win_pos, grid_color);
        }

        for (float y = std::fmodf(scrolling.y, grid_size); y < canvas_size.y; y += grid_size) {
            draw_list->AddLine(ImVec2(0.0f, y) + win_pos, ImVec2(canvas_size.x, y) + win_pos, grid_color);
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


bool megamol::gui::Configurator::draw_canvas_calls(ImVec2 position_offset) {

    const auto COLOR_CALL_CURVE = IM_COL32(200, 200, 100, 255);
    try {
        for (auto& call : this->graph.GetGraphCalls()) {
            /// Assuming connected calls are not nullptr
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            if (call->IsConnected()) {
                ImVec2 p1 = position_offset + call->GetCallSlot(Graph::CallSlotType::CALLER)->GetGuiPos();
                ImVec2 p2 = position_offset + call->GetCallSlot(Graph::CallSlotType::CALLEE)->GetGuiPos();
                draw_list->AddBezierCurve(
                    p1, p1 + ImVec2(50.0f, 0.0f), p2 + ImVec2(-50.0f, 0.0f), p2, COLOR_CALL_CURVE, 3.0f);
            } else {
                vislib::sys::Log::DefaultLog.WriteError(
                    "BUG: Call is has no connected call slots. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
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


bool megamol::gui::Configurator::draw_canvas_modules(ImVec2 position_offset) {

    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    const ImU32 COLOR_MODULE = IM_COL32(60, 60, 60, 255);
    const ImU32 COLOR_MODULE_HIGHTL = IM_COL32(75, 75, 75, 255);
    const ImU32 COLOR_MODULE_BORDER = IM_COL32(100, 100, 100, 255);
    const ImVec4 COLOR_VIEW_LABEL = ImVec4(0.5f, 0.5f, 0.0f, 1.0f);
    const float SLOT_LABEL_OFFSET = 5.0f;
    const float MODULE_SLOT_RADIUS = 8.0f;
    int module_hovered_in_scene = -1;

    try {
        int id = 0;
        for (auto& mod : this->graph.GetGraphModules()) {

            // Draw CALL SLOTS ----------------------------------------------------
            /// Draw call slots before module to catch mouse clicks for slot area lying over module box.
            this->draw_canvas_module_call_slots(mod, position_offset, MODULE_SLOT_RADIUS, SLOT_LABEL_OFFSET);

            // Draw MODULE --------------------------------------------------------
            ImGui::PushID(id);
            // Update of mod->id is crucial -> see graph delete module function.
            mod->gui.id = id;

            // Init size of module once (prior to position)
            if ((mod->gui.size.x < 0.0f) && (mod->gui.size.y < 0.0f)) {
                /// Assuming mod->name is longest used string for module label
                float max_label_length = this->utils.TextWidgetWidth(mod->class_name); /// TODO change to full_name
                float max_slot_name_length = 0.0f;
                for (auto& call_slot_type_list : mod->GetCallSlots()) {
                    for (auto& call_slot : call_slot_type_list.second) {
                        float slot_name_length = this->utils.TextWidgetWidth(call_slot->name);
                        max_slot_name_length = std::max(slot_name_length, max_slot_name_length);
                    }
                }
                float module_width = (max_label_length + 2.0f * max_slot_name_length) + (2.0f * MODULE_SLOT_RADIUS) +
                                     (4.0f * SLOT_LABEL_OFFSET);

                auto max_slot_count = std::max(mod->GetCallSlots(Graph::CallSlotType::CALLEE).size(),
                    mod->GetCallSlots(Graph::CallSlotType::CALLER).size());
                float module_slot_height = (static_cast<float>(max_slot_count) * (MODULE_SLOT_RADIUS * 2.0f) * 1.5f) +
                                           ((MODULE_SLOT_RADIUS * 2.0f) * 0.5f);
                float module_height = std::max(
                    module_slot_height, ImGui::GetItemsLineHeightWithSpacing() * ((mod->is_view) ? (4.0f) : (3.0f)));

                mod->gui.size = ImVec2(module_width, module_height);
            }
            // Init position of module once
            if ((mod->gui.position.x < 0.0f) && (mod->gui.position.y < 0.0f)) {
                ImVec2 canvas_size = ImGui::GetWindowSize();
                mod->gui.position =
                    ImVec2((canvas_size.x - mod->gui.size.x) / 2.0f, (canvas_size.y - mod->gui.size.y) / 2.0f);
            }

            // Draw text ------------------------------------------------------
            draw_list->ChannelsSetCurrent(1); // Foreground

            ImVec2 module_rect_min = position_offset + mod->gui.position;
            ImVec2 module_rect_max = module_rect_min + mod->gui.size;
            ImVec2 module_center = module_rect_min + ImVec2(mod->gui.size.x / 2.0f, mod->gui.size.y / 2.0f);

            bool old_any_active = ImGui::IsAnyItemActive();
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
                std::string view_label = "[view]";
                name_width = this->utils.TextWidgetWidth(view_label);
                ImGui::SetCursorScreenPos(module_center + ImVec2(-(name_width / 2.0f), -line_offset));
                ImGui::TextColored(COLOR_VIEW_LABEL, view_label.c_str());
            }

            ImGui::EndGroup();

            // Draw box -------------------------------------------------------
            draw_list->ChannelsSetCurrent(0); // Background

            ImGui::SetCursorScreenPos(module_rect_min);

            // Save whether any of the widgets are being used
            bool module_widgets_active = (!old_any_active && ImGui::IsAnyItemActive());

            ImGui::InvisibleButton("module", mod->gui.size);
            // Context menu
            if (ImGui::BeginPopupContextItem()) {
                auto hotkey = megamol::core::view::KeyCode(megamol::core::view::Key::KEY_DELETE);
                if (ImGui::MenuItem(
                        "Delete Module", std::get<0>(this->hotkeys[HotkeyIndex::DELETE_MODULE]).ToString().c_str())) {
                    this->graph.DeleteModule(mod->gui.id);
                    break;
                }
                ImGui::EndPopup();
            }

            bool module_moving_active = ImGui::IsItemActive();
            if (module_widgets_active || module_moving_active) {
                this->state.selected_module_graph = id;
            }
            if (module_moving_active && ImGui::IsMouseDragging(0)) {
                mod->gui.position = mod->gui.position + ImGui::GetIO().MouseDelta;
            }

            // Hovered
            if (ImGui::IsItemHovered()) {
                module_hovered_in_scene = id;
            }

            ImU32 module_bg_color = (module_hovered_in_scene == id || this->state.selected_module_graph == id)
                                        ? COLOR_MODULE_HIGHTL
                                        : COLOR_MODULE;

            draw_list->AddRectFilled(module_rect_min, module_rect_max, module_bg_color, 4.0f);
            draw_list->AddRect(module_rect_min, module_rect_max, COLOR_MODULE_BORDER, 4.0f);

            // --------------------------------------------------------------------

            ImGui::PopID();
            id++;
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
    Graph::ModulePtr mod, ImVec2 position_offset, float slot_radius, float slot_label_offset) {

    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    const ImU32 COLOR_SLOT = IM_COL32(175, 175, 175, 255);
    const ImU32 COLOR_SLOT_BORDER = IM_COL32(225, 225, 225, 255);
    const ImU32 COLOR_SLOT_CALLER_LABEL = IM_COL32(0, 192, 192, 255);
    const ImU32 COLOR_SLOT_CALLER_HIGHTL = IM_COL32(0, 192, 192, 255);
    const ImU32 COLOR_SLOT_CALLEE_LABEL = IM_COL32(192, 192, 0, 255);
    const ImU32 COLOR_SLOT_CALLEE_HIGHTL = IM_COL32(192, 192, 0, 255);
    ImU32 slot_color = COLOR_SLOT;
    ImU32 slot_highl_color;
    ImU32 slot_label_color;

    std::string selected_slot_name;
    if (this->graph.IsCallSlotSelected()) {
        selected_slot_name = this->graph.GetSelectedCallSlot()->name;
    }

    try {
        for (auto& slot_pair : mod->GetCallSlots()) {

            if (slot_pair.first == Graph::CallSlotType::CALLER) {
                slot_highl_color = COLOR_SLOT_CALLER_HIGHTL;
                slot_label_color = COLOR_SLOT_CALLER_LABEL;
            } else if (slot_pair.first == Graph::CallSlotType::CALLEE) {
                slot_highl_color = COLOR_SLOT_CALLEE_HIGHTL;
                slot_label_color = COLOR_SLOT_CALLEE_LABEL;
            }

            for (auto& slot : slot_pair.second) {
                draw_list->ChannelsSetCurrent(1); // Foreground

                ImVec2 slot_position = position_offset + slot->GetGuiPos();
                std::string slot_name = slot->name;

                ImGui::SetCursorScreenPos(slot_position - ImVec2(slot_radius, slot_radius));
                // (Label must be unique)
                std::string label = slot_name + "###" + mod->full_name;
                ImGui::InvisibleButton(label.c_str(), ImVec2(slot_radius * 2.0f, slot_radius * 2.0f));
                slot_color = COLOR_SLOT;
                auto hovered = ImGui::IsItemHovered();
                auto clicked = ImGui::IsItemClicked();
                auto active = ImGui::IsItemActivated();
                if (clicked || hovered || (selected_slot_name == slot_name)) {
                    slot_color = slot_highl_color;
                }
                if (clicked) {
                    this->graph.SetSelectedCallSlot(mod->full_name, slot_name);
                }

                ImGui::SetCursorScreenPos(slot_position);
                draw_list->AddCircleFilled(slot_position, slot_radius, slot_color);
                draw_list->AddCircle(slot_position, slot_radius, COLOR_SLOT_BORDER);

                ImVec2 text_pos;
                text_pos.y = slot_position.y - io.FontDefault->FontSize / 2.0f;
                if (slot_pair.first == Graph::CallSlotType::CALLER) {
                    text_pos.x =
                        slot_position.x - this->utils.TextWidgetWidth(slot_name) - slot_radius - slot_label_offset;
                } else if (slot_pair.first == Graph::CallSlotType::CALLEE) {
                    text_pos.x = slot_position.x + slot_radius + slot_label_offset;
                }
                draw_list->AddText(text_pos, slot_label_color, slot_name.c_str());
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


bool megamol::gui::Configurator::draw_canvas_selected_call(ImVec2 position_offset) {

    if (this->graph.IsCallSlotSelected()) {
        auto selected_call_slot = this->graph.GetSelectedCallSlot();
        ImGuiIO& io = ImGui::GetIO();
        const auto COLOR_CALL_CURVE = IM_COL32(200, 200, 100, 255);
        try {
            if (io.MouseDown[0]) {
                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                ImVec2 p1 = position_offset + selected_call_slot->GetGuiPos();
                ImVec2 p2 = io.MousePos;
                if (selected_call_slot->type == Graph::CallSlotType::CALLEE) {
                    ImVec2 tmp = p1;
                    p1 = p2;
                    p2 = tmp;
                }
                draw_list->AddBezierCurve(p1, p1 + ImVec2(+50, 0), p2 + ImVec2(-50, 0), p2, COLOR_CALL_CURVE, 3.0f);
            } else if (io.MouseReleased[0]) {
                this->graph.ResetSelectedCallSlot();
            }

        } catch (std::exception e) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
            return false;
        } catch (...) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
    }
    return false;
}

/*
void megamol::gui::Configurator::demo_dummy(void) {

    struct Node {
        int ID;
        char Name[32];
        ImVec2 Pos, Size;
        float Value;
        ImVec4 Color;
        int InputsCount, OutputsCount;

        Node(int id, const char* name, const ImVec2& pos, float value, const ImVec4& color, int inputs_count,
            int outputs_count) {
            ID = id;
            strncpy(Name, name, 31);
            Name[31] = 0;
            Pos = pos;
            Value = value;
            Color = color;
            InputsCount = inputs_count;
            OutputsCount = outputs_count;
        }

        ImVec2 GetInputSlotPos(int slot_no) const {
            return ImVec2(Pos.x, Pos.y + Size.y * ((float)slot_no + 1) / ((float)InputsCount + 1));
        }
        ImVec2 GetOutputSlotPos(int slot_no) const {
            return ImVec2(Pos.x + Size.x, Pos.y + Size.y * ((float)slot_no + 1) / ((float)OutputsCount + 1));
        }
    };
    struct NodeLink {
        int InputIdx, InputSlot, OutputIdx, OutputSlot;

        NodeLink(int input_idx, int input_slot, int output_idx, int output_slot) {
            InputIdx = input_idx;
            InputSlot = input_slot;
            OutputIdx = output_idx;
            OutputSlot = output_slot;
        }
    };

    static ImVector<Node> nodes;
    static ImVector<NodeLink> links;
    static bool inited = false;
    static ImVec2 scrolling = ImVec2(0.0f, 0.0f);
    static bool show_grid = true;
    static int selected_module_graph = -1;
    if (!inited) {
        nodes.push_back(Node(0, "MainTex", ImVec2(40, 50), 0.5f, ImColor(255, 100, 100), 1, 1));
        nodes.push_back(Node(1, "BumpMap", ImVec2(40, 150), 0.42f, ImColor(200, 100, 200), 1, 1));
        nodes.push_back(Node(2, "Combine", ImVec2(270, 80), 1.0f, ImColor(0, 200, 100), 2, 2));
        links.push_back(NodeLink(0, 0, 2, 0));
        links.push_back(NodeLink(1, 0, 2, 1));
        inited = true;
    }

    // Draw a list of nodes on the left side
    bool open_context_menu = false;
    int module_hovered_in_list = -1;
    int module_hovered_in_scene = -1;
    ImGui::BeginChild("module_list", ImVec2(100, 0));
    ImGui::Text("Nodes");
    ImGui::Separator();
    for (int module_idx = 0; module_idx < nodes.Size; module_idx++) {
        Node* node = &nodes[module_idx];
        ImGui::PushID(node->ID);
        if (ImGui::Selectable(node->Name, node->ID == selected_module_graph)) selected_module_graph = node->ID;
        if (ImGui::IsItemHovered()) {
            module_hovered_in_list = node->ID;
            open_context_menu |= ImGui::IsMouseClicked(1);
        }
        ImGui::PopID();
    }
    ImGui::EndChild();
    ImGui::SameLine();

    ImGui::BeginGroup();
    const float MODULE_SLOT_RADIUS = 4.0f;
    const ImVec2 MODULE_WINDOW_PADDING(8.0f, 8.0f);

    // Create our child canvas
    ImGui::Text("Hold middle mouse button to scroll (%.2f,%.2f)", scrolling.x, scrolling.y);
    ImGui::SameLine(ImGui::GetContentRegionAvailWidth() - 100);
    ImGui::Checkbox("Show grid", &show_grid);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(1, 1));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleColor(ImGuiCol_ChildWindowBg, IM_COL32(60, 60, 70, 200));
    ImGui::BeginChild("scrolling_region", ImVec2(0, 0), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);
    ImGui::PushItemWidth(120.0f);
    ImVec2 offset = ImGui::GetCursorScreenPos() + scrolling;
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Display grid
    if (show_grid) {
        ImU32 GRID_COLOR = IM_COL32(200, 200, 200, 40);
        float GRID_SZ = 64.0f;
        ImVec2 win_pos = ImGui::GetCursorScreenPos();
        ImVec2 canvas_sz = ImGui::GetWindowSize();
        for (float x = fmodf(scrolling.x, GRID_SZ); x < canvas_sz.x; x += GRID_SZ)
            draw_list->AddLine(ImVec2(x, 0.0f) + win_pos, ImVec2(x, canvas_sz.y) + win_pos, GRID_COLOR);
        for (float y = fmodf(scrolling.y, GRID_SZ); y < canvas_sz.y; y += GRID_SZ)
            draw_list->AddLine(ImVec2(0.0f, y) + win_pos, ImVec2(canvas_sz.x, y) + win_pos, GRID_COLOR);
    }

    // Display links
    draw_list->ChannelsSplit(2);
    draw_list->ChannelsSetCurrent(0); // Background
    for (int link_idx = 0; link_idx < links.Size; link_idx++) {
        NodeLink* link = &links[link_idx];
        Node* module_inp = &nodes[link->InputIdx];
        Node* module_out = &nodes[link->OutputIdx];
        ImVec2 p1 = offset + module_inp->GetOutputSlotPos(link->InputSlot);
        ImVec2 p2 = offset + module_out->GetInputSlotPos(link->OutputSlot);
        draw_list->AddBezierCurve(p1, p1 + ImVec2(+50, 0), p2 + ImVec2(-50, 0), p2, IM_COL32(200, 200, 100, 255), 3.0f);
    }

    // Display nodes
    for (int module_idx = 0; module_idx < nodes.Size; module_idx++) {
        Node* node = &nodes[module_idx];
        ImGui::PushID(node->ID);
        ImVec2 module_rect_min = offset + node->Pos;

        // Display node contents first
        draw_list->ChannelsSetCurrent(1); // Foreground
        bool old_any_active = ImGui::IsAnyItemActive();
        ImGui::SetCursorScreenPos(module_rect_min + MODULE_WINDOW_PADDING);
        ImGui::BeginGroup(); // Lock horizontal position
        ImGui::Text("%s", node->Name);
        ImGui::SliderFloat("##value", &node->Value, 0.0f, 1.0f, "Alpha %.2f");
        ImGui::ColorEdit3("##color", &node->Color.x);
        ImGui::EndGroup();

        // Save the size of what we have emitted and whether any of the widgets are being used
        bool module_widgets_active = (!old_any_active && ImGui::IsAnyItemActive());
        node->Size = ImGui::GetItemRectSize() + MODULE_WINDOW_PADDING + MODULE_WINDOW_PADDING;
        ImVec2 module_rect_max = module_rect_min + node->Size;

        // Display node box
        draw_list->ChannelsSetCurrent(0); // Background
        ImGui::SetCursorScreenPos(module_rect_min);
        ImGui::InvisibleButton("node", node->Size);
        if (ImGui::IsItemHovered()) {
            module_hovered_in_scene = node->ID;
            open_context_menu |= ImGui::IsMouseClicked(1);
        }
        bool module_moving_active = ImGui::IsItemActive();
        if (module_widgets_active || module_moving_active) selected_module_graph = node->ID;
        if (module_moving_active && ImGui::IsMouseDragging(0)) node->Pos = node->Pos + ImGui::GetIO().MouseDelta;

        ImU32 module_bg_color = (module_hovered_in_list == node->ID || module_hovered_in_scene == node->ID ||
                                    (module_hovered_in_list == -1 && selected_module_graph == node->ID))
                                    ? IM_COL32(75, 75, 75, 255)
                                    : IM_COL32(60, 60, 60, 255);
        draw_list->AddRectFilled(module_rect_min, module_rect_max, module_bg_color, 4.0f);
        draw_list->AddRect(module_rect_min, module_rect_max, IM_COL32(100, 100, 100, 255), 4.0f);
        for (int slot_idx = 0; slot_idx < node->InputsCount; slot_idx++)
            draw_list->AddCircleFilled(
                offset + node->GetInputSlotPos(slot_idx), MODULE_SLOT_RADIUS, IM_COL32(150, 150, 150, 150));
        for (int slot_idx = 0; slot_idx < node->OutputsCount; slot_idx++)
            draw_list->AddCircleFilled(
                offset + node->GetOutputSlotPos(slot_idx), MODULE_SLOT_RADIUS, IM_COL32(150, 150, 150, 150));

        ImGui::PopID();
    }
    draw_list->ChannelsMerge();

    // Open context menu
    if (!ImGui::IsAnyItemHovered() && ImGui::IsMouseHoveringWindow() && ImGui::IsMouseClicked(1)) {
        selected_module_graph = module_hovered_in_list = module_hovered_in_scene = -1;
        open_context_menu = true;
    }
    if (open_context_menu) {
        ImGui::OpenPopup("context_menu");
        if (module_hovered_in_list != -1) selected_module_graph = module_hovered_in_list;
        if (module_hovered_in_scene != -1) selected_module_graph = module_hovered_in_scene;
    }

    // Draw context menu
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
    if (ImGui::BeginPopup("context_menu")) {
        Node* node = selected_module_graph != -1 ? &nodes[selected_module_graph] : NULL;
        ImVec2 scene_pos = ImGui::GetMousePosOnOpeningCurrentPopup() - offset;
        if (node) {
            ImGui::Text("Node '%s'", node->Name);
            ImGui::Separator();
            if (ImGui::MenuItem("Rename..", NULL, false, false)) {
            }
            if (ImGui::MenuItem("Delete", NULL, false, false)) {
            }
            if (ImGui::MenuItem("Copy", NULL, false, false)) {
            }
        } else {
            if (ImGui::MenuItem("Add")) {
                nodes.push_back(Node(nodes.Size, "New node", scene_pos, 0.5f, ImColor(100, 100, 200), 2, 2));
            }
            if (ImGui::MenuItem("Paste", NULL, false, false)) {
            }
        }
        ImGui::EndPopup();
    }
    ImGui::PopStyleVar();

    // Scrolling
    if (ImGui::IsWindowHovered() && !ImGui::IsAnyItemActive() && ImGui::IsMouseDragging(2, 0.0f))
        scrolling = scrolling + ImGui::GetIO().MouseDelta;

    ImGui::PopItemWidth();
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
    ImGui::EndGroup();
}
*/