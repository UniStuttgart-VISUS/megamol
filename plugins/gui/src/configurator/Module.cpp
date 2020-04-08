/*
 * Module.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "Call.h"
#include "CallSlot.h"
#include "Module.h"


using namespace megamol;
using namespace megamol::gui;
using namespace megamol::gui::configurator;


megamol::gui::configurator::Module::Module(ImGuiID uid)
    : uid(uid)
    , class_name()
    , description()
    , plugin_name()
    , is_view(false)
    , parameters()
    , name()
    , is_view_instance(false)
    , call_slots()
    , present() {

    this->call_slots.emplace(megamol::gui::configurator::CallSlotType::CALLER, std::vector<CallSlotPtrType>());

    this->call_slots.emplace(megamol::gui::configurator::CallSlotType::CALLEE, std::vector<CallSlotPtrType>());
}


megamol::gui::configurator::Module::~Module() { this->RemoveAllCallSlots(); }


bool megamol::gui::configurator::Module::AddCallSlot(megamol::gui::configurator::CallSlotPtrType call_slot) {

    if (call_slot == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to given call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    auto type = call_slot->type;
    for (auto& call_slot_ptr : this->call_slots[type]) {
        if (call_slot_ptr == call_slot) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Pointer to call slot already registered in modules call slot list. [%s, %s, line %d]\n", __FILE__,
                __FUNCTION__, __LINE__);
            return false;
        }
    }
    this->call_slots[type].emplace_back(call_slot);
    return true;
}


bool megamol::gui::configurator::Module::RemoveAllCallSlots(void) {

    try {
        for (auto& call_slots_map : this->call_slots) {
            for (auto& call_slot_ptr : call_slots_map.second) {
                if (call_slot_ptr == nullptr) {
                    // vislib::sys::Log::DefaultLog.WriteWarn(
                    //     "Call slot is already disconnected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                } else {
                    call_slot_ptr->DisConnectCalls();
                    call_slot_ptr->DisConnectParentModule();

                    if (call_slot_ptr.use_count() > 1) {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "Unclean deletion. Found %i references pointing to call slot. [%s, %s, line %d]\n",
                            call_slot_ptr.use_count(), __FILE__, __FUNCTION__, __LINE__);
                    }

                    call_slot_ptr.reset();
                }
            }
            call_slots_map.second.clear();
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


bool megamol::gui::configurator::Module::GetCallSlot(
    ImGuiID callslot_uid, megamol::gui::configurator::CallSlotPtrType& out_callslot_ptr) {

    if (callslot_uid != GUI_INVALID_ID) {
        for (auto& call_slot_map : this->GetCallSlots()) {
            for (auto& callslot : call_slot_map.second) {
                if (callslot->uid == callslot_uid) {
                    out_callslot_ptr = callslot;
                    return true;
                }
            }
        }
    }
    return false;
}


const std::vector<megamol::gui::configurator::CallSlotPtrType>& megamol::gui::configurator::Module::GetCallSlots(
    megamol::gui::configurator::CallSlotType type) {

    // if (this->call_slots[type].empty()) {
    //    vislib::sys::Log::DefaultLog.WriteWarn(
    //        "Returned call slot list is empty. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    //}
    return this->call_slots[type];
}


const std::map<megamol::gui::configurator::CallSlotType, std::vector<megamol::gui::configurator::CallSlotPtrType>>&
megamol::gui::configurator::Module::GetCallSlots(void) {

    return this->call_slots;
}


// MODULE PRESENTATION ####################################################

megamol::gui::configurator::Module::Presentation::Presentation(void)
    : group()
    , presentations(Module::Presentations::DEFAULT)
    , label_visible(true)
    , position(ImVec2(FLT_MAX, FLT_MAX))
    , size(ImVec2(0.0f, 0.0f))
    , utils()
    , selected(false)
    , update(true)
    , other_item_hovered(false)
    , show_params(false) {

    this->group.member = GUI_INVALID_ID;
    this->group.visible = false;
    this->group.name = "";
}


megamol::gui::configurator::Module::Presentation::~Presentation(void) {}


void megamol::gui::configurator::Module::Presentation::Present(
    megamol::gui::configurator::Module& inout_module, megamol::gui::GraphItemsStateType& state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    ImGuiStyle& style = ImGui::GetStyle();
    ImGuiIO& io = ImGui::GetIO();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    bool popup_rename = false;

    try {

        // Update size
        if (this->update || (this->size.x <= 0.0f) || (this->size.y <= 0.0f)) {
            this->UpdateSize(inout_module, state.canvas);
            this->update = false;
        }

        // Init position of newly created module (check after size update)
        /// There are the following possibilities to create a module:
        /// 1) Load from file: Either all modules have the confPos tag or non of the modules have initial positions.
        /// 2) Load from running project: No inital position.
        /// 3) Add module from stock list 'stand alone' (no call is created): No inital position.
        /// 4) Add module from stock list while compatible call slot is selected and a new call is created between
        /// modules: No inital position.
        if ((this->position.x == FLT_MAX) && (this->position.y == FLT_MAX)) {
            unsigned int connected_callslot_count = 0;
            for (auto& callslot_map : inout_module.GetCallSlots()) {
                for (auto callslot_ptr : callslot_map.second) {
                    if (callslot_ptr->CallsConnected()) {
                        connected_callslot_count++;
                    }
                }
            }
            bool one_callslot_connected = (connected_callslot_count == 1);
            // Position for modules added while compatible call slot is selected and a new call is created between
            // modules
            if (one_callslot_connected) {
                for (auto& callslot_map : inout_module.GetCallSlots()) {
                    for (auto callslot_ptr : callslot_map.second) {
                        if (callslot_ptr->CallsConnected()) {
                            CallSlotPtrType connected_callslot_ptr;
                            if (callslot_map.first == CallSlotType::CALLER) {
                                connected_callslot_ptr =
                                    callslot_ptr->GetConnectedCalls()[0]->GetCallSlot(CallSlotType::CALLEE);
                            } else if (callslot_map.first == CallSlotType::CALLEE) {
                                connected_callslot_ptr =
                                    callslot_ptr->GetConnectedCalls()[0]->GetCallSlot(CallSlotType::CALLER);
                            }
                            if ((connected_callslot_ptr != nullptr) &&
                                (connected_callslot_ptr->ParentModuleConnected())) {
                                float call_name_width =
                                    GUIUtils::TextWidgetWidth(callslot_ptr->GetConnectedCalls()[0]->class_name);
                                ImVec2 module_size = connected_callslot_ptr->GetParentModule()->GUI_GetSize();
                                ImVec2 module_pos = connected_callslot_ptr->GetParentModule()->GUI_GetPosition();
                                float call_width = (2.0f * GUI_GRAPH_BORDER + call_name_width * 1.5f);
                                if (callslot_map.first == CallSlotType::CALLER) {
                                    // Left of connected module
                                    this->position = module_pos - ImVec2((call_width + this->size.x), 0.0f);
                                } else if (callslot_map.first == CallSlotType::CALLEE) {
                                    // Right of connected module
                                    this->position = module_pos + ImVec2((call_width + module_size.x), 0.0f);
                                }
                            }
                            break;
                        }
                    }
                }
            } else {
                // See layout border_offset in Graph::Presentation::layout_graph
                this->position = this->GetInitModulePosition(state.canvas);
            }
        }

        // Draw module --------------------------------------------------------
        if ((this->group.member == GUI_INVALID_ID) || ((this->group.member != GUI_INVALID_ID) && this->group.visible)) {

            // Get current module information
            ImVec2 module_size = this->size * state.canvas.zooming;
            ImVec2 module_rect_min = state.canvas.offset + this->position * state.canvas.zooming;
            ImVec2 module_rect_max = module_rect_min + module_size;
            ImVec2 module_center = module_rect_min + ImVec2(module_size.x / 2.0f, module_size.y / 2.0f);

            // Clip module if lying ouside the canvas
            /// XXX Is there a benefit since ImGui::PushClipRect is used?
            /*
            ImVec2 canvas_rect_min = state.canvas.position;
            ImVec2 canvas_rect_max = state.canvas.position + state.canvas.size;
            if (!((canvas_rect_min.x < module_rect_max.x) && (canvas_rect_max.x > module_rect_min.x) &&
                    (canvas_rect_min.y < module_rect_max.y) && (canvas_rect_max.y > module_rect_min.y))) {
                if (mouse_clicked) {
                    this->selected = false;
                    if (this->found_uid(state.interact.modules_selected_uids, inout_module.uid)) {
                        this->erase_uid(state.interact.modules_selected_uids, inout_module.uid);
                    }
                }
                if (this->selected) {
                    if (!this->found_uid(state.interact.modules_selected_uids, inout_module.uid)) {
                        state.interact.modules_selected_uids.emplace_back(inout_module.uid);
                    }
                }
                return;
            }
            */

            ImGui::PushID(inout_module.uid);

            // Colors
            ImVec4 tmpcol = style.Colors[ImGuiCol_FrameBg]; // ImGuiCol_FrameBg ImGuiCol_Button
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_MODULE_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);

            tmpcol = style.Colors[ImGuiCol_FrameBgActive]; // ImGuiCol_FrameBgActive ImGuiCol_ButtonActive
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_MODULE_HIGHTLIGHT = ImGui::ColorConvertFloat4ToU32(tmpcol);

            tmpcol = style.Colors[ImGuiCol_ScrollbarGrabActive]; // ImGuiCol_Border ImGuiCol_ScrollbarGrabActive
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_MODULE_GROUP_BORDER = ImGui::ColorConvertFloat4ToU32(tmpcol);

            const ImU32 COLOR_TEXT = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Text]);
            const ImU32 COLOR_HEADER =
                ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_FrameBgHovered]); // ImGuiCol_MenuBarBg
            const ImU32 COLOR_HEADER_HIGHLIGHT = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonActive]);

            // Button
            ImGui::SetCursorScreenPos(module_rect_min);
            std::string label = "module_" + inout_module.name;
            ImGui::SetItemAllowOverlap();
            ImGui::InvisibleButton(label.c_str(), module_size);
            ImGui::SetItemAllowOverlap();

            bool active = ImGui::IsItemActive();
            bool mouse_clicked = ImGui::IsWindowHovered() && ImGui::GetIO().MouseClicked[0];
            bool hovered = (ImGui::IsItemHovered() && (state.interact.callslot_hovered_uid == GUI_INVALID_ID) &&
                            ((state.interact.module_hovered_uid == GUI_INVALID_ID) ||
                                (state.interact.module_hovered_uid == inout_module.uid)));

            // Context menu
            if (state.interact.callslot_hovered_uid == GUI_INVALID_ID) {
                if (ImGui::BeginPopupContextItem("invisible_button_context")) {
                    active = true; // Force selection

                    ImGui::TextUnformatted("Module");
                    ImGui::Separator();
                    if (ImGui::MenuItem(
                            "Delete", std::get<0>(state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM])
                                          .ToString()
                                          .c_str())) {
                        std::get<1>(state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM]) = true;
                    }
                    bool rename_valid = ((state.interact.modules_selected_uids.size() == 1) &&
                                         (this->found_uid(state.interact.modules_selected_uids, inout_module.uid)));
                    if (ImGui::MenuItem("Rename", nullptr, false, rename_valid)) {
                        popup_rename = true;
                    }
                    if (ImGui::BeginMenu("Add to Group", true)) {
                        if (ImGui::MenuItem("New")) {
                            state.interact.modules_add_group_uids.clear();
                            if (this->selected) {
                                for (auto& module_uid : state.interact.modules_selected_uids) {
                                    state.interact.modules_add_group_uids.emplace_back(
                                        UIDPairType(module_uid, GUI_INVALID_ID));
                                }
                            } else {
                                state.interact.modules_add_group_uids.emplace_back(
                                    UIDPairType(inout_module.uid, GUI_INVALID_ID));
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
                                            UIDPairType(module_uid, group_pair.first));
                                    }
                                } else {
                                    state.interact.modules_add_group_uids.emplace_back(
                                        UIDPairType(inout_module.uid, group_pair.first));
                                }
                            }
                        }
                        ImGui::EndMenu();
                    }
                    if (ImGui::MenuItem("Remove from Group", nullptr, false, (this->group.member != GUI_INVALID_ID))) {
                        state.interact.modules_remove_group_uids.clear();
                        if (this->selected) {
                            for (auto& module_uid : state.interact.modules_selected_uids) {
                                state.interact.modules_remove_group_uids.emplace_back(module_uid);
                            }
                        } else {
                            state.interact.modules_remove_group_uids.emplace_back(inout_module.uid);
                        }
                    }
                    ImGui::EndPopup();
                }
            }

            // Hover Tooltip
            if (hovered && !this->other_item_hovered) {
                std::string hover_text = inout_module.description;
                if (!this->label_visible) {
                    hover_text = "[" + inout_module.name + "] " + hover_text;
                }
                this->utils.HoverToolTip(hover_text.c_str(), ImGui::GetID(label.c_str()), 0.75f, 5.0f);
            } else {
                this->utils.ResetHoverToolTip();
            }
            this->other_item_hovered = false;

            // Draw Background
            ImU32 module_bg_color = (this->selected) ? (COLOR_MODULE_HIGHTLIGHT) : (COLOR_MODULE_BACKGROUND);
            draw_list->AddRectFilled(
                module_rect_min, module_rect_max, module_bg_color, GUI_RECT_CORNER_RADIUS, ImDrawCornerFlags_All);

            // Draw Text and Option Buttons
            float text_width;
            ImVec2 text_pos_left_upper;
            const float line_height = ImGui::GetTextLineHeightWithSpacing();
            ImVec2 param_child_pos;
            if (this->label_visible) {

                bool main_view_button = inout_module.is_view;
                bool parameter_button = (inout_module.parameters.size() > 0);
                bool any_button = (main_view_button || parameter_button);

                auto header_color = (this->selected) ? (COLOR_HEADER_HIGHLIGHT) : (COLOR_HEADER);
                ImVec2 header_rect_max = module_rect_min + ImVec2(module_size.x, ImGui::GetTextLineHeightWithSpacing());
                draw_list->AddRectFilled(module_rect_min, header_rect_max, header_color, GUI_RECT_CORNER_RADIUS,
                    (ImDrawCornerFlags_TopLeft | ImDrawCornerFlags_TopRight));

                text_width = GUIUtils::TextWidgetWidth(inout_module.class_name);
                text_pos_left_upper =
                    ImVec2(module_center.x - (text_width / 2.0f), module_rect_min.y + (style.ItemSpacing.y / 2.0f));
                draw_list->AddText(text_pos_left_upper, COLOR_TEXT, inout_module.class_name.c_str());

                text_width = GUIUtils::TextWidgetWidth(inout_module.name);
                text_pos_left_upper =
                    module_center - ImVec2((text_width / 2.0f), ((any_button) ? (line_height * 0.6f) : (0.0f)));
                draw_list->AddText(text_pos_left_upper, COLOR_TEXT, inout_module.name.c_str());

                if (any_button) {
                    float item_y_offset = (line_height / 2.0f);
                    float item_x_offset = (ImGui::GetFrameHeight() / 2.0f);
                    if (main_view_button && parameter_button) {
                        item_x_offset = ImGui::GetFrameHeight() + (0.5f * style.ItemSpacing.x * state.canvas.zooming);
                    }
                    ImGui::SetCursorScreenPos(module_center + ImVec2(-item_x_offset, item_y_offset));

                    if (main_view_button) {
                        if (ImGui::RadioButton("###main_view_switch", inout_module.is_view_instance)) {
                            state.interact.module_mainview_uid = inout_module.uid;
                            inout_module.is_view_instance = !inout_module.is_view_instance;
                            this->add_uid(state.interact.modules_selected_uids,
                                inout_module.uid); // Force selection (must be set in same frame)
                        }
                        if (hovered) {
                            this->other_item_hovered = this->utils.HoverToolTip("Main View");
                        }
                        ImGui::SameLine(0.0f, style.ItemSpacing.x * state.canvas.zooming);
                    }

                    if (parameter_button) {
                        param_child_pos = ImGui::GetCursorScreenPos();
                        param_child_pos.y += ImGui::GetFrameHeight();
                        if (ImGui::ArrowButton(
                                "###parameter_toggle", ((this->show_params) ? (ImGuiDir_Down) : (ImGuiDir_Up)))) {
                            this->show_params = !this->show_params;
                            this->add_uid(state.interact.modules_selected_uids,
                                inout_module.uid); // Force selection (must be set in same frame)
                        }
                        if (hovered) {
                            this->other_item_hovered =
                                this->other_item_hovered || this->utils.HoverToolTip("Parameters");
                        }
                    }
                }
            }

            // Draw Outline
            float border = ((inout_module.is_view_instance) ? (4.0f) : (1.0f)) * state.canvas.zooming;
            draw_list->AddRect(module_rect_min, module_rect_max, COLOR_MODULE_GROUP_BORDER, GUI_RECT_CORNER_RADIUS,
                ImDrawCornerFlags_All, border);

            // Hovering
            if (!hovered && (state.interact.module_hovered_uid == inout_module.uid)) {
                state.interact.module_hovered_uid = GUI_INVALID_ID;
            }
            if (hovered) {
                state.interact.module_hovered_uid = inout_module.uid;
            }

            // Selection
            if (this->found_uid(state.interact.modules_selected_uids, inout_module.uid)) {
                /// Call before "active" if-statement for one frame delayed check for last valid candidate for selection
                this->selected = true;
                state.interact.callslot_selected_uid = GUI_INVALID_ID;
                state.interact.call_selected_uid = GUI_INVALID_ID;
                state.interact.group_selected_uid = GUI_INVALID_ID;
            }
            if (active && !this->selected) {
                // Multiple Selection
                if (io.KeyShift) {
                    this->add_uid(state.interact.modules_selected_uids, inout_module.uid);
                } else {
                    // Single Selection
                    state.interact.modules_selected_uids.clear();
                    state.interact.modules_selected_uids.emplace_back(inout_module.uid);
                }
            }
            if ((mouse_clicked && (!hovered || state.interact.callslot_hovered_uid != GUI_INVALID_ID) &&
                    (state.interact.module_hovered_uid == GUI_INVALID_ID)) ||
                (!this->found_uid(state.interact.modules_selected_uids, inout_module.uid))) {
                this->selected = false;
                if (this->found_uid(state.interact.modules_selected_uids, inout_module.uid)) {
                    this->erase_uid(state.interact.modules_selected_uids, inout_module.uid);
                }
            }

            // Dragging
            if (this->selected && ImGui::IsWindowHovered() && ImGui::IsMouseDragging(0)) {
                this->position += (ImGui::GetIO().MouseDelta / state.canvas.zooming);
                this->UpdateSize(inout_module, state.canvas);
            }

            // Rename pop-up
            if (this->utils.RenamePopUp("Rename Project", popup_rename, inout_module.name)) {
                this->UpdateSize(inout_module, state.canvas);
            }

            // Parameter Child Window
            if (this->label_visible && this->show_params) {
                ImGui::PushStyleColor(ImGuiCol_ChildBg, COLOR_MODULE_BACKGROUND);
                ImGui::SetCursorScreenPos(param_child_pos);

                float param_height = 0.0f;
                for (auto& param : inout_module.parameters) {
                    param_height += param.GUI_GetHeight();
                }
                param_height += style.ScrollbarSize;
                float child_width = 325.0f * state.canvas.zooming;
                float child_height = std::min((ImGui::GetContentRegionAvail().y), param_height);

                auto child_flags = ImGuiWindowFlags_AlwaysVerticalScrollbar |
                                   ImGuiWindowFlags_AlwaysHorizontalScrollbar | ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_NavFlattened;
                ImGui::BeginChild("module_parameter_child", ImVec2(child_width, child_height), true, child_flags);

                for (auto& param : inout_module.parameters) {
                    param.GUI_Present();
                }

                ImGui::EndChild();
                ImGui::PopStyleColor();
                if (ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape))) {
                    this->show_params = false;
                }
            }

            ImGui::PopID();
        }

        // Draw call slots ----------------------------------------------------
        for (auto& slot_pair : inout_module.GetCallSlots()) {
            for (auto& slot : slot_pair.second) {
                slot->GUI_Present(state);
            }
        }

    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


ImVec2 megamol::gui::configurator::Module::Presentation::GetInitModulePosition(const GraphCanvasType& canvas) {

    return ImVec2((GUI_GRAPH_BORDER), (GUI_GRAPH_BORDER) + ImGui::GetTextLineHeightWithSpacing()) +
           (canvas.position - canvas.offset) / canvas.zooming;
}


void megamol::gui::configurator::Module::Presentation::UpdateSize(
    megamol::gui::configurator::Module& inout_module, const GraphCanvasType& in_canvas) {

    ImGuiStyle& style = ImGui::GetStyle();

    // WIDTH
    float max_label_length = 0.0f;
    if (this->label_visible) {
        float class_width = GUIUtils::TextWidgetWidth(inout_module.class_name);
        float name_length = GUIUtils::TextWidgetWidth(inout_module.name);
        float button_width =
            ((this->label_visible) ? (2.0f) : (1.0f)) * ImGui::GetTextLineHeightWithSpacing() + style.ItemSpacing.x;
        max_label_length = std::max(class_width, name_length);
        max_label_length = std::max(max_label_length, button_width);
    }
    max_label_length /= in_canvas.zooming;
    float max_slot_name_length = 0.0f;
    for (auto& call_slot_type_list : inout_module.GetCallSlots()) {
        for (auto& call_slot : call_slot_type_list.second) {
            if (call_slot->GUI_GetLabelVisibility()) {
                max_slot_name_length = std::max(GUIUtils::TextWidgetWidth(call_slot->name), max_slot_name_length);
            }
        }
    }
    if (max_slot_name_length > 0.0f) {
        max_slot_name_length = (2.0f * max_slot_name_length / in_canvas.zooming) + (1.0f * GUI_CALL_SLOT_RADIUS);
    }
    float module_width = (max_label_length + max_slot_name_length) + (3.0f * GUI_CALL_SLOT_RADIUS);

    // HEIGHT
    float line_height = (ImGui::GetTextLineHeightWithSpacing() / in_canvas.zooming);
    auto max_slot_count = std::max(
        inout_module.GetCallSlots(CallSlotType::CALLEE).size(), inout_module.GetCallSlots(CallSlotType::CALLER).size());
    float module_slot_height = line_height +
                               (static_cast<float>(max_slot_count) * (GUI_CALL_SLOT_RADIUS * 2.0f) * 1.5f) +
                               GUI_CALL_SLOT_RADIUS;
    float text_button_height = (line_height * ((this->label_visible) ? (4.0f) : (1.0f)));
    float module_height = std::max(module_slot_height, text_button_height);

    // Clamp to minimum size
    this->size = ImVec2(std::max(module_width, 100.0f), std::max(module_height, 50.0f));

    // UPDATE all Call Slots ---------------------
    for (auto& slot_pair : inout_module.GetCallSlots()) {
        for (auto& slot : slot_pair.second) {
            slot->GUI_Update(in_canvas);
        }
    }
}
