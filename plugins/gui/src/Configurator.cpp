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

    // DEMO DUMMY:
    // this->demo_dummy();
    // return true;

    bool retval = true;

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
        // 2] Load available modules and calls once

        this->graph.UpdateAvailableModulesCallsOnce(core_instance);
        this->window_rendering_state++;
    } else {
        // 3] Render final configurator content
        bool state_draw_menu = this->draw_window_menu(core_instance);
        bool state_draw_module_list = this->draw_window_module_list();
        ImGui::SameLine(); // Draws module list and graph canvas next to each other
        bool state_draw_graph_canvas = this->draw_window_graph_canvas();

        retval = (state_draw_menu && state_draw_module_list && state_draw_graph_canvas);
    }

    return retval;
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
            /// Not supported so far
            // if (ImGui::MenuItem("Load Project", "no hotkey set")) {
/// TODO:  Load parameter file
//    std::string projectFilename;
//    this->GetCoreInstance()->LoadProject(vislib::StringA(projectFilename.c_str()));
//}
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
        this->utils.SetSearchFocus(true);
        std::get<1>(this->hotkeys[HotkeyIndex::MODULE_SEARCH]) = false;
    }
    std::string help_text = "[" + std::get<0>(this->hotkeys[HotkeyIndex::MODULE_SEARCH]).ToString() +
                            "] Set keyboard focus to search input field.\n"
                            "Case insensitive substring search in module names.";
    this->utils.StringSearch("Search Modules", help_text);
    auto search_string = this->utils.GetSearchString();

    ImGui::EndChild();

    ImGui::BeginChild("module_list", ImVec2(child_width, 0.0f), true, ImGuiWindowFlags_HorizontalScrollbar);

    int id = 1; // Start with 1 because it is used as enumeration
    for (auto& m : this->graph.GetAvailableModulesList()) {
        if (search_string.empty() || this->utils.FindCaseInsensitiveSubstring(m.class_name, search_string)) {
            ImGui::PushID(id);
            std::string label = std::to_string(id) + " " + m.class_name + " (" + m.plugin_name + ")";
            if (ImGui::Selectable(label.c_str(), id == this->state.selected_module_list)) {
                this->state.selected_module_list = id;
            }
            // Left mouse button double click action
            if (ImGui::IsMouseDoubleClicked(0) && ImGui::IsItemHovered()) {
                this->graph.AddModule(m.class_name);
            }
            // Context menu
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Add Module")) {
                    this->graph.AddModule(m.class_name);
                }
                ImGui::EndPopup();
            }
            // Hover tool tip
            this->utils.HoverToolTip(m.description, id, 0.5f, 5.0f);
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
    ImGuiStyle& style = ImGui::GetStyle();
    /// Font scaling with zooming factor is not possible locally within window.
    /// io.FontDefault->Scale = this->state.zooming;

    ImGui::BeginGroup();

    int module_hovered_in_scene = -1;
    std::string hovered_desc;

    // CONSTS -----------------------------------------------------------------

    // Colors
    const auto COLOR_SLOT_CALLER_LABEL = IM_COL32(0, 192, 192, 255);
    const auto COLOR_SLOT_CALLER_HIGHTL = IM_COL32(0, 192, 192, 255);
    const auto COLOR_SLOT_CALLEE_LABEL = IM_COL32(192, 192, 0, 255);
    const auto COLOR_SLOT_CALLEE_HIGHTL = IM_COL32(192, 192, 0, 255);

    const auto COLOR_SLOT = IM_COL32(175, 175, 175, 255);
    const auto COLOR_SLOT_BORDER = IM_COL32(225, 225, 225, 255);

    const auto COLOR_MODULE = IM_COL32(60, 60, 60, 255);
    const auto COLOR_MODULE_HIGHTL = IM_COL32(75, 75, 75, 255);
    const auto COLOR_MODULE_BORDER = IM_COL32(100, 100, 100, 255);

    //  Misc
    const float SLOT_LABEL_OFFSET = 5.0f;
    const float MODULE_SLOT_RADIUS = 8.0f;
    const float MODULE_SLOT_DIAMETER = MODULE_SLOT_RADIUS * 2.0f;
    const ImVec2 MODULE_WINDOW_PADDING(10.0f, 10.0f);

    // Info -------------------------------------------------------------------
    ImGui::BeginChild("info", ImVec2(0.0f, (2.0f * ImGui::GetItemsLineHeightWithSpacing())), true,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);
    std::string label =
        "This is a PROTOTYPE. Any changes will NOT EFFECT the currently loaded project.\n"
        "You can save the modified graph to a SEPARATE PROJECT FILE (parameters are not considered yet).";
    ImGui::TextColored(ImVec4(0.9f, 0.2f, 0.2f, 1.0f), label.c_str());
    ImGui::EndChild();

    // Create child canvas ----------------------------------------------------
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(1.0f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_ChildWindowBg, IM_COL32(60, 60, 70, 200));
    ImGui::BeginChild("region", ImVec2(0.0f, 0.0f), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);
    ImGui::PushItemWidth(120.0f);
    ImVec2 offset = ImGui::GetCursorScreenPos() + this->state.scrolling;
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Display grid
    if (this->state.show_grid) {
        this->draw_canvas_grid(this->state.scrolling, this->state.zooming);
    }

    // Display links
    /*
    for (int link_idx = 0; link_idx < links.Size; link_idx++) {
        NodeLink* link = &links[link_idx];
        Node* module_inp = &nodes[link->InputIdx];
        Node* module_out = &nodes[link->OutputIdx];
        ImVec2 p1 = offset + module_inp->GetOutputSlotPos(link->InputSlot);
        ImVec2 p2 = offset + module_out->GetInputSlotPos(link->OutputSlot);
        draw_list->AddBezierCurve(p1, p1 + ImVec2(+50, 0), p2 + ImVec2(-50, 0), p2, IM_COL32(200, 200, 100, 255), 3.0f);
    }
    */

    // Display Modules
    /// TODO Automatic layouting of modules
    int id = 0;
    draw_list->ChannelsSplit(2);
    draw_list->ChannelsSetCurrent(0); // Background
    for (auto& mod : this->graph.GetGraphModules()) {
        ImGui::PushID(id);
        ImVec2 module_rect_min = offset + mod.gui.position;

        // Draw MODULE text ---------------------------------------------------
        draw_list->ChannelsSetCurrent(1); // Foreground

        bool old_any_active = ImGui::IsAnyItemActive();
        ImGui::SetCursorScreenPos(module_rect_min + MODULE_WINDOW_PADDING);
        ImGui::BeginGroup(); // Lock horizontal position
        std::string module_class_name = mod.class_name;
        if (mod.is_view) {
            module_class_name += "[VIEW]";
        }
        ImGui::Text(module_class_name.c_str());
        ImGui::Text(mod.name.c_str());
        ImGui::EndGroup();

        // Save the size of what we have emitted and whether any of the widgets are being used
        bool module_widgets_active = (!old_any_active && ImGui::IsAnyItemActive());
        ImVec2 module_rect_max = module_rect_min + mod.gui.size;

        // Draw MODULE box ----------------------------------------------------
        draw_list->ChannelsSetCurrent(0); // Background

        ImGui::SetCursorScreenPos(module_rect_min);
        ImGui::InvisibleButton("module", mod.gui.size);
        if (ImGui::IsItemHovered()) {
            module_hovered_in_scene = id;
            hovered_desc = mod.description;
        }

        bool module_moving_active = ImGui::IsItemActive();
        if (module_widgets_active || module_moving_active) {
            this->state.selected_module_graph = id;
        }
        if (module_moving_active && ImGui::IsMouseDragging(0)) {
            mod.gui.position = mod.gui.position + ImGui::GetIO().MouseDelta;
        }

        ImU32 module_bg_color = (module_hovered_in_scene == id || this->state.selected_module_graph == id)
                                    ? COLOR_MODULE_HIGHTL
                                    : COLOR_MODULE;
        draw_list->AddRectFilled(module_rect_min, module_rect_max, module_bg_color, 4.0f);
        draw_list->AddRect(module_rect_min, module_rect_max, COLOR_MODULE_BORDER, 4.0f);

        // Draw SLOTS ---------------------------------------------------------
        auto current_slot_color = COLOR_SLOT;

        // Caller Slots
        size_t caller_slots_count = mod.caller_slots.size();
        for (int slot_idx = 0; slot_idx < caller_slots_count; slot_idx++) {
            draw_list->ChannelsSetCurrent(0); // Background

            ImVec2 slot_position = offset + mod.GetCallerSlotPos(slot_idx);

            ImGui::SetCursorScreenPos(slot_position - ImVec2(MODULE_SLOT_RADIUS, MODULE_SLOT_RADIUS));
            std::string slot_button = "caller_slot###" + std::to_string(slot_idx);
            ImGui::InvisibleButton(slot_button.c_str(), ImVec2(MODULE_SLOT_DIAMETER, MODULE_SLOT_DIAMETER));
            current_slot_color = COLOR_SLOT;
            if (ImGui::IsItemHovered()) {
                current_slot_color = COLOR_SLOT_CALLER_HIGHTL;
                hovered_desc = mod.caller_slots[slot_idx].description;
            }
            ImGui::SetCursorScreenPos(slot_position);
            draw_list->AddCircleFilled(slot_position, MODULE_SLOT_RADIUS, current_slot_color);
            draw_list->AddCircle(slot_position, MODULE_SLOT_RADIUS, COLOR_SLOT_BORDER);

            draw_list->ChannelsSetCurrent(1); // Foreground

            std::string label = mod.caller_slots[slot_idx].name;
            ImVec2 text_pos = slot_position;
            text_pos.x = text_pos.x - this->utils.TextWidgetWidth(label) - MODULE_SLOT_RADIUS - SLOT_LABEL_OFFSET;
            text_pos.y = text_pos.y - io.FontDefault->FontSize / 2.0f;
            draw_list->AddText(text_pos, COLOR_SLOT_CALLER_LABEL, label.c_str());
        }

        // Callee Slots
        size_t callee_slots_count = mod.callee_slots.size();
        for (int slot_idx = 0; slot_idx < callee_slots_count; slot_idx++) {
            draw_list->ChannelsSetCurrent(0); // Background

            ImVec2 slot_position = offset + mod.GetCalleeSlotPos(slot_idx);

            ImGui::SetCursorScreenPos(slot_position - ImVec2(MODULE_SLOT_RADIUS, MODULE_SLOT_RADIUS));
            std::string slot_button = "callee_slot###" + std::to_string(slot_idx);
            ImGui::InvisibleButton(slot_button.c_str(), ImVec2(MODULE_SLOT_DIAMETER, MODULE_SLOT_DIAMETER));
            current_slot_color = COLOR_SLOT;
            if (ImGui::IsItemHovered()) {
                current_slot_color = COLOR_SLOT_CALLEE_HIGHTL;
                hovered_desc = mod.callee_slots[slot_idx].description;
            }
            ImGui::SetCursorScreenPos(slot_position);
            draw_list->AddCircleFilled(slot_position, MODULE_SLOT_RADIUS, current_slot_color);
            draw_list->AddCircle(slot_position, MODULE_SLOT_RADIUS, COLOR_SLOT_BORDER);

            draw_list->ChannelsSetCurrent(1); // Foreground

            std::string label = mod.callee_slots[slot_idx].name;
            ImVec2 text_pos = slot_position;
            text_pos.x = text_pos.x + MODULE_SLOT_RADIUS + SLOT_LABEL_OFFSET;
            text_pos.y = text_pos.y - io.FontDefault->FontSize / 2.0f;
            draw_list->AddText(text_pos, COLOR_SLOT_CALLEE_LABEL, label.c_str());
        }

        // --------------------------------------------------------------------

        ImGui::PopID();
        id++;
    }
    draw_list->ChannelsMerge();

    if (ImGui::IsWindowHovered()) { // && !ImGui::IsAnyItemActive()) {
        // Scrolling (2 = Middle Mouse Button)
        if (ImGui::IsMouseDragging(2, 0.0f)) {
            this->state.scrolling = this->state.scrolling + ImGui::GetIO().MouseDelta;
        }
        // Zooming (Mouse Wheel)
        float last_zooming = this->state.zooming;
        this->state.zooming = this->state.zooming + io.MouseWheel / 10.0f;
        this->state.zooming = (this->state.zooming < 0.1f) ? (0.1f) : (this->state.zooming);
        if (last_zooming != this->state.zooming) {
            /// TODO Adapt scrolling for zooming at current mouse position
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

    for (float x = std::fmodf(this->state.scrolling.x, grid_size); x < canvas_size.x; x += grid_size) {
        draw_list->AddLine(ImVec2(x, 0.0f) + win_pos, ImVec2(x, canvas_size.y) + win_pos, grid_color);
    }

    for (float y = std::fmodf(this->state.scrolling.y, grid_size); y < canvas_size.y; y += grid_size) {
        draw_list->AddLine(ImVec2(0.0f, y) + win_pos, ImVec2(canvas_size.x, y) + win_pos, grid_color);
    }

    return true;
}


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