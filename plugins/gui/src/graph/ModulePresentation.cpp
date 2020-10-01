/*
 * Module.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ModulePresentation.h"

#include "Call.h"
#include "CallSlot.h"
#include "InterfaceSlot.h"
#include "Module.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::ModulePresentation::ModulePresentation(void)
    : group()
    , label_visible(true)
    , position(ImVec2(FLT_MAX, FLT_MAX))
    , param_groups()
    , size(ImVec2(0.0f, 0.0f))
    , selected(false)
    , update(true)
    , param_child_show(false)
    , param_child_height(1.0f)
    , set_screen_position(ImVec2(FLT_MAX, FLT_MAX))
    , set_selected_slot_position(false)
    , tooltip()
    , rename_popup() {

    this->group.uid = GUI_INVALID_ID;
    this->group.visible = false;
    this->group.name = "";
}


megamol::gui::ModulePresentation::~ModulePresentation(void) {}


void megamol::gui::ModulePresentation::Present(
    megamol::gui::PresentPhase phase, megamol::gui::Module& inout_module, megamol::gui::GraphItemsState_t& state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    ImGuiStyle& style = ImGui::GetStyle();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    try {
        // Update size
        if (this->update || (this->size.x <= 0.0f) || (this->size.y <= 0.0f)) {
            this->Update(inout_module, state.canvas);
            this->update = false;
        }

        // Init position of newly created module (check after size update)
        if ((this->set_screen_position.x != FLT_MAX) && (this->set_screen_position.y != FLT_MAX)) {
            this->position = (this->set_screen_position - state.canvas.offset) / state.canvas.zooming;
            this->set_screen_position = ImVec2(FLT_MAX, FLT_MAX);
        }
        // Init position using current compatible slot
        if (this->set_selected_slot_position) {
            for (auto& callslot_map : inout_module.GetCallSlots()) {
                for (auto& callslot_ptr : callslot_map.second) {
                    CallSlotType callslot_type =
                        (callslot_ptr->type == CallSlotType::CALLEE) ? (CallSlotType::CALLER) : (CallSlotType::CALLEE);
                    for (auto& call_ptr : callslot_ptr->GetConnectedCalls()) {
                        auto connected_callslot_ptr = call_ptr->GetCallSlot(callslot_type);
                        float call_width =
                            (4.0f * GUI_GRAPH_BORDER + ImGui::CalcTextSize(call_ptr->class_name.c_str()).x);
                        if (state.interact.callslot_selected_uid != GUI_INVALID_ID) {
                            if ((connected_callslot_ptr->uid == state.interact.callslot_selected_uid) &&
                                connected_callslot_ptr->IsParentModuleConnected()) {
                                ImVec2 module_size = connected_callslot_ptr->GetParentModule()->present.GetSize();
                                ImVec2 module_pos = connected_callslot_ptr->GetParentModule()->present.position;
                                if (connected_callslot_ptr->type == CallSlotType::CALLEE) {
                                    this->position = module_pos - ImVec2((call_width + this->size.x), 0.0f);
                                } else {
                                    this->position = module_pos + ImVec2((call_width + module_size.x), 0.0f);
                                }
                                break;
                            }
                        } else if ((state.interact.interfaceslot_selected_uid != GUI_INVALID_ID) &&
                                   (connected_callslot_ptr->present.group.interfaceslot_ptr != nullptr)) {
                            if (state.interact.interfaceslot_selected_uid ==
                                connected_callslot_ptr->present.group.interfaceslot_ptr->uid) {
                                ImVec2 interfaceslot_position =
                                    (connected_callslot_ptr->present.group.interfaceslot_ptr->GetGUIPosition() -
                                        state.canvas.offset) /
                                    state.canvas.zooming;
                                if (connected_callslot_ptr->type == CallSlotType::CALLEE) {
                                    this->position = interfaceslot_position - ImVec2((call_width + this->size.x), 0.0f);
                                } else {
                                    this->position = interfaceslot_position + ImVec2(call_width, 0.0f);
                                }
                                break;
                            }
                        }
                    }
                }
            }
            this->set_selected_slot_position = false;
        }
        if ((this->position.x == FLT_MAX) && (this->position.y == FLT_MAX)) {
            // See layout border_offset in Graph::Presentation::layout_graph
            this->position = this->GetDefaultModulePosition(state.canvas);
        }

        // Check if module and call slots are visible
        bool visible =
            (this->group.uid == GUI_INVALID_ID) || ((this->group.uid != GUI_INVALID_ID) && this->group.visible);
        for (auto& callslots_map : inout_module.GetCallSlots()) {
            for (auto& callslot_ptr : callslots_map.second) {
                callslot_ptr->present.visible = visible;
            }
        }

        if (visible) {
            bool mouse_clicked_anywhere = ImGui::IsWindowHovered() && ImGui::GetIO().MouseClicked[0];

            ImGui::PushID(inout_module.uid);

            // Get current module information
            ImVec2 module_size = this->size * state.canvas.zooming;
            ImVec2 module_rect_min = state.canvas.offset + (this->position * state.canvas.zooming);
            ImVec2 module_rect_max = module_rect_min + module_size;
            ImVec2 module_center = module_rect_min + ImVec2(module_size.x / 2.0f, module_size.y / 2.0f);

            // Clip module if lying ouside the canvas
            /// Is there a benefit since ImGui::PushClipRect is used?
            ImVec2 canvas_rect_min = state.canvas.position;
            ImVec2 canvas_rect_max = state.canvas.position + state.canvas.size;
            if (!((canvas_rect_min.x < module_rect_max.x) && (canvas_rect_max.x > module_rect_min.x) &&
                    (canvas_rect_min.y < module_rect_max.y) && (canvas_rect_max.y > module_rect_min.y))) {
                if (mouse_clicked_anywhere) {
                    this->selected = false;
                    if (this->found_uid(state.interact.modules_selected_uids, inout_module.uid)) {
                        this->erase_uid(state.interact.modules_selected_uids, inout_module.uid);
                    }
                }
            } else {
                // MODULE ------------------------------------------------------
                std::string button_label = "module_" + std::to_string(inout_module.uid);

                if (phase == megamol::gui::PresentPhase::INTERACTION) {

                    // Button
                    ImGui::SetCursorScreenPos(module_rect_min);
                    ImGui::SetItemAllowOverlap();
                    ImGui::InvisibleButton(button_label.c_str(), module_size);
                    ImGui::SetItemAllowOverlap();
                    if (ImGui::IsItemActivated()) {
                        state.interact.button_active_uid = inout_module.uid;
                    }
                    if (ImGui::IsItemHovered()) {
                        state.interact.button_hovered_uid = inout_module.uid;
                    }

                    // Context menu
                    bool popup_rename = false;
                    if (ImGui::BeginPopupContextItem("invisible_button_context")) {
                        state.interact.button_active_uid = inout_module.uid;
                        bool singleselect = ((state.interact.modules_selected_uids.size() == 1) &&
                                             (this->found_uid(state.interact.modules_selected_uids, inout_module.uid)));

                        ImGui::TextDisabled("Module");
                        ImGui::Separator();

                        if (ImGui::MenuItem(
                                "Delete", std::get<0>(state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM])
                                              .ToString()
                                              .c_str())) {
                            state.interact.process_deletion = true;
                        }
                        if (ImGui::MenuItem("Layout", nullptr, false, !singleselect)) {
                            state.interact.modules_layout = true;
                        }
                        if (ImGui::MenuItem("Rename", nullptr, false, singleselect)) {
                            popup_rename = true;
                        }
                        if (ImGui::BeginMenu("Add to Group", true)) {
                            if (ImGui::MenuItem("New")) {
                                state.interact.modules_add_group_uids.clear();
                                if (this->selected) {
                                    for (auto& module_uid : state.interact.modules_selected_uids) {
                                        state.interact.modules_add_group_uids.emplace_back(
                                            UIDPair_t(module_uid, GUI_INVALID_ID));
                                    }
                                } else {
                                    state.interact.modules_add_group_uids.emplace_back(
                                        UIDPair_t(inout_module.uid, GUI_INVALID_ID));
                                }
                            }
                            if (!state.groups.empty()) {
                                ImGui::Separator();
                            }
                            for (auto& group_pair : state.groups) {
                                if (ImGui::MenuItem(group_pair.second.c_str())) {
                                    state.interact.modules_add_group_uids.clear();
                                    if (this->selected) {
                                        for (auto& module_uid : state.interact.modules_selected_uids) {
                                            state.interact.modules_add_group_uids.emplace_back(
                                                UIDPair_t(module_uid, group_pair.first));
                                        }
                                    } else {
                                        state.interact.modules_add_group_uids.emplace_back(
                                            UIDPair_t(inout_module.uid, group_pair.first));
                                    }
                                }
                            }
                            ImGui::EndMenu();
                        }
                        if (ImGui::MenuItem("Remove from Group", nullptr, false, (this->group.uid != GUI_INVALID_ID))) {
                            state.interact.modules_remove_group_uids.clear();
                            if (this->selected) {
                                for (auto& module_uid : state.interact.modules_selected_uids) {
                                    state.interact.modules_remove_group_uids.emplace_back(module_uid);
                                }
                            } else {
                                state.interact.modules_remove_group_uids.emplace_back(inout_module.uid);
                            }
                        }

                        if (singleselect) {
                            ImGui::Separator();
                            ImGui::TextDisabled("Description");
                            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 13.0f);
                            ImGui::TextUnformatted(inout_module.description.c_str());
                            ImGui::PopTextWrapPos();
                        }
                        ImGui::EndPopup();
                    }

                    // Hover Tooltip
                    if ((state.interact.module_hovered_uid == inout_module.uid) && !this->label_visible) {
                        this->tooltip.ToolTip(inout_module.name, ImGui::GetID(button_label.c_str()), 0.5f, 5.0f);
                    } else {
                        this->tooltip.Reset();
                    }

                    // Rename pop-up
                    if (this->rename_popup.PopUp("Rename Project", popup_rename, inout_module.name)) {
                        this->Update(inout_module, state.canvas);
                    }
                } else if (phase == megamol::gui::PresentPhase::RENDERING) {

                    bool active = (state.interact.button_active_uid == inout_module.uid);
                    bool hovered = (state.interact.button_hovered_uid == inout_module.uid);

                    // Selection
                    if (!this->selected &&
                        (active || this->found_uid(state.interact.modules_selected_uids, inout_module.uid))) {
                        if (!this->found_uid(state.interact.modules_selected_uids, inout_module.uid)) {
                            if (GUI_MULTISELECT_MODIFIER) {
                                // Multiple Selection
                                this->add_uid(state.interact.modules_selected_uids, inout_module.uid);
                            } else {
                                // Single Selection
                                state.interact.modules_selected_uids.clear();
                                state.interact.modules_selected_uids.emplace_back(inout_module.uid);
                            }
                        }
                        this->selected = true;
                        state.interact.callslot_selected_uid = GUI_INVALID_ID;
                        state.interact.call_selected_uid = GUI_INVALID_ID;
                        state.interact.group_selected_uid = GUI_INVALID_ID;
                        state.interact.interfaceslot_selected_uid = GUI_INVALID_ID;
                    }
                    // Deselection
                    else if (this->selected &&
                             ((mouse_clicked_anywhere && (state.interact.module_hovered_uid == GUI_INVALID_ID) &&
                                  !GUI_MULTISELECT_MODIFIER) ||
                                 (active && GUI_MULTISELECT_MODIFIER) ||
                                 (!this->found_uid(state.interact.modules_selected_uids, inout_module.uid)))) {
                        this->selected = false;
                        this->erase_uid(state.interact.modules_selected_uids, inout_module.uid);
                    }

                    // Dragging
                    if (this->selected && ImGui::IsWindowHovered() && ImGui::IsMouseDragging(0)) {
                        this->position += (ImGui::GetIO().MouseDelta / state.canvas.zooming);
                        this->Update(inout_module, state.canvas);
                    }

                    // Hovering
                    if (hovered) {
                        state.interact.module_hovered_uid = inout_module.uid;
                    }
                    if (!hovered && (state.interact.module_hovered_uid == inout_module.uid)) {
                        state.interact.module_hovered_uid = GUI_INVALID_ID;
                    }

                    // Colors
                    ImVec4 tmpcol = style.Colors[ImGuiCol_FrameBg];
                    tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
                    const ImU32 COLOR_MODULE_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);

                    tmpcol = style.Colors[ImGuiCol_FrameBgActive];
                    tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
                    const ImU32 COLOR_MODULE_HIGHTLIGHT = ImGui::ColorConvertFloat4ToU32(tmpcol);

                    tmpcol = style.Colors[ImGuiCol_ScrollbarGrabActive];
                    tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
                    const ImU32 COLOR_MODULE_BORDER = ImGui::ColorConvertFloat4ToU32(tmpcol);

                    const ImU32 COLOR_TEXT = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Text]);

                    const ImU32 COLOR_HEADER = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_FrameBgHovered]);

                    const ImU32 COLOR_HEADER_HIGHLIGHT =
                        ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonActive]);

                    // Draw Background
                    ImU32 module_bg_color = (this->selected) ? (COLOR_MODULE_HIGHTLIGHT) : (COLOR_MODULE_BACKGROUND);
                    draw_list->AddRectFilled(module_rect_min, module_rect_max, module_bg_color, GUI_RECT_CORNER_RADIUS,
                        ImDrawCornerFlags_All);

                    // Draw Text and Option Buttons
                    float text_width;
                    ImVec2 text_pos_left_upper;
                    const float line_height = ImGui::GetTextLineHeightWithSpacing();
                    bool other_item_hovered = false;

                    if (this->label_visible) {
                        bool main_view_button = inout_module.is_view;
                        bool parameter_button = (inout_module.parameters.size() > 0);
                        bool any_option_button = (main_view_button || parameter_button);

                        auto header_color = (this->selected) ? (COLOR_HEADER_HIGHLIGHT) : (COLOR_HEADER);
                        ImVec2 header_rect_max =
                            module_rect_min + ImVec2(module_size.x, ImGui::GetTextLineHeightWithSpacing());
                        draw_list->AddRectFilled(module_rect_min, header_rect_max, header_color, GUI_RECT_CORNER_RADIUS,
                            (ImDrawCornerFlags_TopLeft | ImDrawCornerFlags_TopRight));

                        text_width = ImGui::CalcTextSize(inout_module.class_name.c_str()).x;
                        text_pos_left_upper = ImVec2(
                            module_center.x - (text_width / 2.0f), module_rect_min.y + (style.ItemSpacing.y / 2.0f));
                        draw_list->AddText(text_pos_left_upper, COLOR_TEXT, inout_module.class_name.c_str());

                        text_width = ImGui::CalcTextSize(inout_module.name.c_str()).x;
                        text_pos_left_upper =
                            module_center -
                            ImVec2((text_width / 2.0f), ((any_option_button) ? (line_height * 0.6f) : (0.0f)));
                        draw_list->AddText(text_pos_left_upper, COLOR_TEXT, inout_module.name.c_str());

                        if (any_option_button) {
                            float item_y_offset = (line_height / 2.0f);
                            float item_x_offset = (ImGui::GetFrameHeight() / 2.0f);
                            if (main_view_button && parameter_button) {
                                item_x_offset =
                                    ImGui::GetFrameHeight() + (0.5f * style.ItemSpacing.x * state.canvas.zooming);
                            }
                            ImGui::SetCursorScreenPos(module_center + ImVec2(-item_x_offset, item_y_offset));

                            if (main_view_button) {
                                if (ImGui::RadioButton("###main_view_switch", inout_module.is_view_instance)) {
                                    if (hovered) {
                                        state.interact.module_mainview_uid = inout_module.uid;
                                        inout_module.is_view_instance = !inout_module.is_view_instance;
                                    }
                                }
                                ImGui::SetItemAllowOverlap();
                                if (hovered) {
                                    other_item_hovered = other_item_hovered || this->tooltip.ToolTip("Main View");
                                }
                                ImGui::SameLine(0.0f, style.ItemSpacing.x * state.canvas.zooming);
                            }

                            if (parameter_button) {
                                bool param_child_hovered = false;
                                ImVec2 param_button_pos = ImGui::GetCursorScreenPos();

                                // Parameter Child Window
                                if (this->param_child_show) {
                                    ImGui::PushStyleColor(ImGuiCol_ChildBg, COLOR_MODULE_BACKGROUND);
                                    const ImGuiID last_active_id = ImGui::GetActiveID();

                                    const float param_child_width = 325.0f * state.canvas.zooming;
                                    ImVec2 param_child_pos = param_button_pos;
                                    param_child_pos.x +=
                                        ImGui::GetFrameHeight(); // Fix x position to right side of button
                                    float avail_height =
                                        (state.canvas.position.y + state.canvas.size.y) - param_child_pos.y;
                                    this->param_child_height = std::min(state.canvas.size.y, this->param_child_height);
                                    if (this->param_child_height > avail_height) {
                                        param_child_pos.y -= (this->param_child_height - avail_height);
                                    }

                                    ImGui::SetCursorScreenPos(param_child_pos);

                                    auto child_flags = ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoMove |
                                                       ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                                       ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NavFlattened;
                                    ImGui::BeginChild("module_parameter_child",
                                        ImVec2(param_child_width, this->param_child_height), true, child_flags);

                                    float cursor_pos_y = ImGui::GetCursorPosY();

                                    // Draw parameters
                                    this->param_groups.PresentGUI(inout_module.parameters, inout_module.FullName(), "",
                                        vislib::math::Ternary(vislib::math::Ternary::TRI_UNKNOWN), false,
                                        ParameterPresentation::WidgetScope::LOCAL, nullptr, nullptr);

                                    this->param_child_height = ImGui::GetCursorPosY() - cursor_pos_y +
                                                               ImGui::GetFrameHeight() + ImGui::GetFrameHeight();

                                    ImGui::EndChild();
                                    ImGui::PopStyleColor();

                                    // Also check for active items because combo box might fold out below the child
                                    // window's border
                                    bool param_active = last_active_id != ImGui::GetActiveID();
                                    if (((ImGui::GetMousePos().x >= param_child_pos.x) &&
                                            (ImGui::GetMousePos().x <= (param_child_pos.x + param_child_width)) &&
                                            (ImGui::GetMousePos().y >= param_child_pos.y) &&
                                            (ImGui::GetMousePos().y <=
                                                (param_child_pos.y + this->param_child_height))) ||
                                        param_active) {
                                        param_child_hovered = true;
                                    }
                                }

                                // Param Button
                                ImGui::SetCursorScreenPos(param_button_pos);
                                if (ImGui::ArrowButton("###parameter_toggle",
                                        ((this->param_child_show) ? (ImGuiDir_Down) : (ImGuiDir_Up))) &&
                                    hovered) {
                                    this->param_child_show = !this->param_child_show;
                                } else if (ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape)) ||
                                           (ImGui::IsMouseClicked(0) && !param_child_hovered &&
                                               !ImGui::IsItemHovered())) { /// Ignore if button is hovered
                                    // Close child window: 'Escape' and 'Mouse Click' outside param window
                                    this->param_child_show = false;
                                }
                                ImGui::SetItemAllowOverlap();
                                if (hovered) {
                                    other_item_hovered |= this->tooltip.ToolTip("Parameters");
                                }
                            }
                        }
                    }

                    // Draw Outline
                    float border = ((inout_module.is_view_instance) ? (4.0f) : (1.0f)) * state.canvas.zooming;
                    draw_list->AddRect(module_rect_min, module_rect_max, COLOR_MODULE_BORDER, GUI_RECT_CORNER_RADIUS,
                        ImDrawCornerFlags_All, border);
                }
            }

            ImGui::PopID();

            // CALL SLOTS ------------------------------------------------------
            for (auto& callslots_map : inout_module.GetCallSlots()) {
                for (auto& callslot_ptr : callslots_map.second) {
                    callslot_ptr->PresentGUI(phase, state);
                }
            }
        }

    } catch (std::exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


ImVec2 megamol::gui::ModulePresentation::GetDefaultModulePosition(const GraphCanvas_t& canvas) {

    return ((ImVec2((2.0f * GUI_GRAPH_BORDER), (2.0f * GUI_GRAPH_BORDER)) + // ImGui::GetTextLineHeightWithSpacing()) +
                (canvas.position - canvas.offset)) /
            canvas.zooming);
}


void megamol::gui::ModulePresentation::Update(megamol::gui::Module& inout_module, const GraphCanvas_t& in_canvas) {

    ImGuiStyle& style = ImGui::GetStyle();

    // WIDTH
    float class_width = 0.0f;
    float max_label_length = 0.0f;
    if (this->label_visible) {
        class_width = ImGui::CalcTextSize(inout_module.class_name.c_str()).x;
        float name_length = ImGui::CalcTextSize(inout_module.name.c_str()).x;
        float button_width =
            ((inout_module.is_view) ? (2.0f) : (1.0f)) * ImGui::GetTextLineHeightWithSpacing() + style.ItemSpacing.x;
        max_label_length = std::max(name_length, button_width);
    }
    max_label_length /= in_canvas.zooming;
    float max_slot_name_length = 0.0f;
    for (auto& callslots_map : inout_module.GetCallSlots()) {
        for (auto& callslot_ptr : callslots_map.second) {
            if (callslot_ptr->present.label_visible) {
                max_slot_name_length =
                    std::max(ImGui::CalcTextSize(callslot_ptr->name.c_str()).x, max_slot_name_length);
            }
        }
    }
    if (max_slot_name_length > 0.0f) {
        max_slot_name_length = (2.0f * max_slot_name_length / in_canvas.zooming) + (1.0f * GUI_SLOT_RADIUS);
    }
    float module_width = std::max((class_width / in_canvas.zooming), (max_label_length + max_slot_name_length)) +
                         (3.0f * GUI_SLOT_RADIUS);

    // HEIGHT
    float line_height = (ImGui::GetTextLineHeightWithSpacing() / in_canvas.zooming);
    auto max_slot_count = std::max(
        inout_module.GetCallSlots(CallSlotType::CALLEE).size(), inout_module.GetCallSlots(CallSlotType::CALLER).size());
    float module_slot_height =
        line_height + (static_cast<float>(max_slot_count) * (GUI_SLOT_RADIUS * 2.0f) * 1.5f) + GUI_SLOT_RADIUS;
    float text_button_height = (line_height * ((this->label_visible) ? (4.0f) : (1.0f)));
    float module_height = std::max(module_slot_height, text_button_height);

    // Clamp to minimum size
    this->size = ImVec2(std::max(module_width, 100.0f), std::max(module_height, 50.0f));

    // UPDATE all Call Slots ---------------------
    for (auto& slot_pair : inout_module.GetCallSlots()) {
        for (auto& slot : slot_pair.second) {
            slot->UpdateGUI(in_canvas);
        }
    }
}
