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


#define CONFIGURATOR_SLOT_RADIUS (8.0f)


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
    this->state.any_hovered_slot = false;
    this->state.selected_slot.reset();
    this->state.processing_selected_slot = 0;
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

    int id = 1; // Start with 1 because it is used as enumeration
    for (auto& mod : this->graph.GetModulesStock()) {
        if (search_string.empty() || this->utils.FindCaseInsensitiveSubstring(mod.class_name, search_string)) {
            ImGui::PushID(id);
            if (mod.is_view) {
                ImGui::PushStyleColor(ImGuiCol_Text, COLOR_MODULE_VIEW);
            }
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

            if (mod.is_view) {
                ImGui::PopStyleColor();
            }
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

    const ImU32 COLOR_MODULE_BACKGROUND = IM_COL32(60, 60, 70, 255);

    // Process module deletion
    if (std::get<1>(this->hotkeys[HotkeyIndex::DELETE_MODULE])) {
        std::get<1>(this->hotkeys[HotkeyIndex::DELETE_MODULE]) = false;
        this->graph.DeleteModule(this->state.selected_module_graph);
    }

    // Selected call slot processing state
    if (this->state.processing_selected_slot > 0) {
        this->state.processing_selected_slot--;
    }
    if ((this->state.selected_slot != nullptr) && (io.MouseReleased[0])) {
        // Needs one additional frame to be completely processed
        this->state.processing_selected_slot = 2;
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
    ImGui::PushStyleColor(ImGuiCol_ChildWindowBg, COLOR_MODULE_BACKGROUND);
    ImGui::BeginChild("region", ImVec2(0.0f, 0.0f), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);
    ImGui::PushItemWidth(120.0f);
    ImVec2 position_offset = ImGui::GetCursorScreenPos() + this->state.scrolling;
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    draw_list->ChannelsSplit(2);
    draw_list->ChannelsSetCurrent(0); // Background

    // Display grid -------------------------------------------------------
    if (this->state.show_grid) {
        this->draw_canvas_grid(this->state.scrolling, this->state.zooming);
    }

    // Draw modules -----------------------------------------------------
    this->draw_canvas_modules(position_offset);

    // Draw call --------------------------------------------------------
    this->draw_canvas_calls(position_offset);

    // Draw dnd call --------------------------------------------------------
    this->draw_canvas_selected_dnd_call(position_offset);

    draw_list->ChannelsMerge();

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

    ImGui::EndGroup();

    // Reset selected call slot processing state
    if (this->state.processing_selected_slot == 0) {
        this->state.selected_slot = nullptr;
    }

    return true;
}


bool megamol::gui::Configurator::draw_canvas_grid(ImVec2 scrolling, float zooming) {

    const ImU32 COLOR_GRID = IM_COL32(200, 200, 200, 40);
    const float GRID_SIZE = 64.0f;

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_size = ImGui::GetWindowSize();
    ImVec2 win_pos = ImGui::GetCursorScreenPos();
    float grid_size = GRID_SIZE * zooming;

    try {
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


bool megamol::gui::Configurator::draw_canvas_calls(ImVec2 position_offset) {

    const ImU32 COLOR_CALL_CURVE = IM_COL32(200, 200, 100, 255);

    try {
        int id = 0;
        for (auto& call : this->graph.GetGraphCalls()) {
            ImGui::PushID(id);
            // Update of call->gui.id is crucial -> see graph delete call function.
            call->gui.id = id;

            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            draw_list->ChannelsSetCurrent(1); // Foreground

            if (call->IsConnected()) {
                ImVec2 p1 = position_offset + call->GetCallSlot(Graph::CallSlotType::CALLER)->GetGuiPos();
                ImVec2 p2 = position_offset + call->GetCallSlot(Graph::CallSlotType::CALLEE)->GetGuiPos();
                draw_list->AddBezierCurve(
                    p1, p1 + ImVec2(50.0f, 0.0f), p2 + ImVec2(-50.0f, 0.0f), p2, COLOR_CALL_CURVE, 3.0f);
            }

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


bool megamol::gui::Configurator::draw_canvas_modules(ImVec2 position_offset) {

    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    const ImU32 COLOR_MODULE = IM_COL32(60, 60, 60, 255);
    const ImU32 COLOR_MODULE_HIGHTL = IM_COL32(75, 75, 75, 255);
    const ImU32 COLOR_MODULE_BORDER = IM_COL32(100, 100, 100, 255);
    const ImVec4 COLOR_VIEW_LABEL = ImVec4(0.5f, 0.5f, 0.0f, 1.0f);

    try {

        int id = 0;
        int module_hovered_in_scene = -1;
        for (auto& mod : this->graph.GetGraphModules()) {

            // Draw CALL SLOTS ----------------------------------------------------
            /// Draw call slots before modules to catch mouse clicks for slot area lying over module box.
            this->draw_canvas_module_call_slots(mod, position_offset);

            // Draw MODULE --------------------------------------------------------
            ImGui::PushID(id);
            // Update of mod->id is crucial -> see graph delete module function.
            mod->gui.id = id;

            if (!mod->gui.initialized) {
                this->init_module_gui_params(mod);
            }

            bool old_any_active = ImGui::IsAnyItemActive();

            // Draw text ------------------------------------------------------
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
                std::string view_label = "[view]";
                name_width = this->utils.TextWidgetWidth(view_label);
                ImGui::SetCursorScreenPos(module_center + ImVec2(-(name_width / 2.0f), -line_offset));
                ImGui::TextColored(COLOR_VIEW_LABEL, view_label.c_str());
            }

            ImGui::EndGroup();

            // Draw box -------------------------------------------------------
            draw_list->ChannelsSetCurrent(0); // Background

            ImGui::SetCursorScreenPos(module_rect_min);
            std::string label = "module###" + mod->full_name;
            ImGui::InvisibleButton(label.c_str(), mod->gui.size);
            // Gives slots which overlap modules priority for ToolTip and Context Menu.
            if (!this->state.any_hovered_slot) {
                this->utils.HoverToolTip(mod->description, ImGui::GetID(label.c_str()), 0.5f, 5.0f);
                // Context menu
                if (ImGui::BeginPopupContextItem()) {
                    if (ImGui::MenuItem("Delete Module",
                            std::get<0>(this->hotkeys[HotkeyIndex::DELETE_MODULE]).ToString().c_str())) {
                        std::get<1>(this->hotkeys[HotkeyIndex::DELETE_MODULE]) = true;
                        this->state.selected_module_graph = mod->gui.id;
                    }
                    ImGui::EndPopup();
                }
            }
            bool module_widgets_active = (!old_any_active && ImGui::IsAnyItemActive());
            bool module_moving_active = ImGui::IsItemActive();
            if (module_widgets_active || module_moving_active) {
                this->state.selected_module_graph = id;
            }
            if (module_moving_active && ImGui::IsMouseDragging(0)) {
                mod->gui.position = mod->gui.position + ImGui::GetIO().MouseDelta;
            }
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


bool megamol::gui::Configurator::draw_canvas_module_call_slots(Graph::ModulePtr mod, ImVec2 position_offset) {

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

    try {
        // Draw module call slots.
        int id = 0;
        this->state.any_hovered_slot = false;
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
                ImGui::PushID(id);

                ImVec2 slot_position = position_offset + slot->GetGuiPos();
                std::string slot_name = slot->name;
                slot_color = COLOR_SLOT;

                ImGui::SetCursorScreenPos(slot_position - ImVec2(CONFIGURATOR_SLOT_RADIUS, CONFIGURATOR_SLOT_RADIUS));
                std::string label = "slot###" + mod->full_name + slot_name + std::to_string(id);
                ImGui::InvisibleButton(
                    label.c_str(), ImVec2(CONFIGURATOR_SLOT_RADIUS * 2.0f, CONFIGURATOR_SLOT_RADIUS * 2.0f));
                std::string tooltip = slot->description + " | ";
                for (auto& cs : slot->compatible_call_idxs) {
                    tooltip += std::to_string(cs) + " | ";
                }
                this->utils.HoverToolTip(tooltip, ImGui::GetID(label.c_str()), 0.5f, 5.0f);
                auto hovered = ImGui::IsItemHovered();
                auto clicked = ImGui::IsItemClicked();
                if (hovered) {
                    this->state.any_hovered_slot = true;
                    if (this->state.processing_selected_slot && (slot != this->state.selected_slot)) {
                        // Check for compatible calls
                        size_t index;
                        bool found = false;
                        for (auto& selected_comp_call_slots : this->state.selected_slot->compatible_call_idxs) {
                            for (auto& current_comp_call_slots : slot->compatible_call_idxs) {
                                if (selected_comp_call_slots == current_comp_call_slots) {
                                    index = current_comp_call_slots;
                                    found = true;
                                }
                            }
                        }
                        if (!found) {
                            vislib::sys::Log::DefaultLog.WriteWarn(
                                "Found no compatible call. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                        } else {
                            this->graph.AddCall(index, this->state.selected_slot, slot);
                        }
                    }
                }
                if (clicked) {
                    this->state.selected_slot = slot;
                }
                if (hovered || (this->state.selected_slot == slot)) {
                    slot_color = slot_highl_color;
                }

                ImGui::SetCursorScreenPos(slot_position);
                draw_list->AddCircleFilled(slot_position, CONFIGURATOR_SLOT_RADIUS, slot_color);
                draw_list->AddCircle(slot_position, CONFIGURATOR_SLOT_RADIUS, COLOR_SLOT_BORDER);

                ImVec2 text_pos;
                text_pos.y = slot_position.y - io.FontDefault->FontSize / 2.0f;
                if (slot_pair.first == Graph::CallSlotType::CALLER) {
                    text_pos.x =
                        slot_position.x - this->utils.TextWidgetWidth(slot_name) - (2.0f * CONFIGURATOR_SLOT_RADIUS);
                } else if (slot_pair.first == Graph::CallSlotType::CALLEE) {
                    text_pos.x = slot_position.x + (2.0f * CONFIGURATOR_SLOT_RADIUS);
                }
                draw_list->AddText(text_pos, slot_label_color, slot_name.c_str());

                ImGui::PopID();
                id++;
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


bool megamol::gui::Configurator::draw_canvas_selected_dnd_call(ImVec2 position_offset) {

    const auto COLOR_CALL_CURVE = IM_COL32(200, 200, 100, 255);

    if (this->state.selected_slot != nullptr) {
        ImGuiIO& io = ImGui::GetIO();
        try {
            if (io.MouseDown[0]) {
                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                ImVec2 p1 = position_offset + this->state.selected_slot->GetGuiPos();
                ImVec2 p2 = io.MousePos;
                if (this->state.selected_slot->type == Graph::CallSlotType::CALLEE) {
                    ImVec2 tmp = p1;
                    p1 = p2;
                    p2 = tmp;
                }
                draw_list->AddBezierCurve(p1, p1 + ImVec2(+50, 0), p2 + ImVec2(-50, 0), p2, COLOR_CALL_CURVE, 3.0f);
                this->state.processing_selected_slot = 1;
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
    return true;
}

bool megamol::gui::Configurator::init_module_gui_params(Graph::ModulePtr mod) {

    // Init SIZE of module (prior to position) --------------------------------
    /// Assuming mod->name is longest used string for module label
    float max_label_length = this->utils.TextWidgetWidth(mod->class_name); /// TODO change to full_name
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

    // Init POSIITON ----------------------------------------------------------
    ImVec2 canvas_size = ImGui::GetWindowSize();
    mod->gui.position = ImVec2((canvas_size.x - mod->gui.size.x) / 2.0f, (canvas_size.y - mod->gui.size.y) / 2.0f);

    mod->gui.initialized = true;

    return true;
}
