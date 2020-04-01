/*
 * CallSlot.cpp
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


megamol::gui::configurator::CallSlot::CallSlot(ImGuiID uid)
    : uid(uid), name(), description(), compatible_call_idxs(), type(), parent_module(), connected_calls(), present() {}


megamol::gui::configurator::CallSlot::~CallSlot() {

    // Call separately and check for reference count
    this->DisConnectCalls();
    this->DisConnectParentModule();
}


bool megamol::gui::configurator::CallSlot::CallsConnected(void) const {

    /// Check for unclean references
    for (auto& call_ptr : this->connected_calls) {
        if (call_ptr == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Pointer to one of the connected calls is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
        }
    }

    return (!this->connected_calls.empty());
}


bool megamol::gui::configurator::CallSlot::ConnectCall(megamol::gui::configurator::CallPtrType call) {

    if (call == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to given call is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (this->type == CallSlotType::CALLER) {
        if (this->connected_calls.size() > 0) {
            vislib::sys::Log::DefaultLog.WriteWarn(
                "Caller slots can only be connected to one call. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
            return false;
        }
    }
    this->connected_calls.emplace_back(call);
    return true;
}


bool megamol::gui::configurator::CallSlot::DisConnectCall(ImGuiID call_uid, bool called_by_call) {

    try {
        for (auto call_iter = this->connected_calls.begin(); call_iter != this->connected_calls.end(); call_iter++) {
            if ((*call_iter) == nullptr) {
                // vislib::sys::Log::DefaultLog.WriteWarn(
                //     "Call is already disconnected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            } else {
                if ((*call_iter)->uid == call_uid) {
                    if (!called_by_call) {
                        (*call_iter)->DisConnectCallSlots();
                    }
                    (*call_iter).reset();
                    this->connected_calls.erase(call_iter);
                    return true;
                }
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
    return false;
}


bool megamol::gui::configurator::CallSlot::DisConnectCalls(void) {

    try {
        // Since connected calls operate on this list for disconnecting slots
        // a local copy of the connected calls is required.
        auto connected_calls_copy = this->connected_calls;
        for (auto& call_ptr : connected_calls_copy) {
            if (call_ptr == nullptr) {
                // vislib::sys::Log::DefaultLog.WriteWarn(
                //     "Call is already disconnected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            } else {
                call_ptr->DisConnectCallSlots();
            }
        }
        this->connected_calls.clear();
        connected_calls_copy.clear();
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


const std::vector<megamol::gui::configurator::CallPtrType>& megamol::gui::configurator::CallSlot::GetConnectedCalls(
    void) {

    /// Check for unclean references
    for (auto& call_ptr : this->connected_calls) {
        if (call_ptr == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Pointer to one of the connected calls is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
        }
    }

    return this->connected_calls;
}


bool megamol::gui::configurator::CallSlot::ParentModuleConnected(void) const {
    return (this->parent_module != nullptr);
}


bool megamol::gui::configurator::CallSlot::ConnectParentModule(
    megamol::gui::configurator::ModulePtrType parent_module) {

    if (parent_module == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to given parent module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    if (this->parent_module != nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to parent module is already set. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->parent_module = parent_module;
    return true;
}


bool megamol::gui::configurator::CallSlot::DisConnectParentModule(void) {

    if (parent_module == nullptr) {
        // vislib::sys::Log::DefaultLog.WriteWarn(
        //      "Pointer to parent module is already nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->parent_module.reset();
    return true;
}


const megamol::gui::configurator::ModulePtrType megamol::gui::configurator::CallSlot::GetParentModule(void) {

    if (this->parent_module == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Returned pointer to parent module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    return this->parent_module;
}


ImGuiID megamol::gui::configurator::CallSlot::CheckCompatibleAvailableCallIndex(
    const megamol::gui::configurator::CallSlotPtrType call_slot_ptr, megamol::gui::configurator::CallSlot& call_slot) {

    if (call_slot_ptr != nullptr) {
        if (call_slot_ptr->GetParentModule() != call_slot.GetParentModule() &&
            (call_slot_ptr->type != call_slot.type)) {
            // Return first found compatible call index
            for (auto& selected_comp_call_slot : call_slot_ptr->compatible_call_idxs) {
                for (auto& current_comp_call_slots : call_slot.compatible_call_idxs) {
                    if (selected_comp_call_slot == current_comp_call_slots) {
                        return static_cast<ImGuiID>(current_comp_call_slots);
                    }
                }
            }
        }
    }
    return GUI_INVALID_ID;
}


ImGuiID megamol::gui::configurator::CallSlot::GetCompatibleCallIndex(
    const CallSlotPtrType call_slot_1, const CallSlotPtrType call_slot_2) {

    if ((call_slot_1 != nullptr) && (call_slot_2 != nullptr)) {
        if (call_slot_1->GetParentModule() != call_slot_2->GetParentModule() &&
            (call_slot_1->type != call_slot_2->type)) {
            // Return first found compatible call index
            for (auto& comp_call_idx_1 : call_slot_1->compatible_call_idxs) {
                for (auto& comp_call_idx_2 : call_slot_2->compatible_call_idxs) {
                    if (comp_call_idx_1 == comp_call_idx_2) {
                        return static_cast<ImGuiID>(comp_call_idx_1);
                    }
                }
            }
        }
    }
    return GUI_INVALID_ID;
}


ImGuiID megamol::gui::configurator::CallSlot::GetCompatibleCallIndex(
    const CallSlotPtrType call_slot, const CallSlot::StockCallSlot& stock_call_slot) {

    if (call_slot != nullptr) {
        if (call_slot->type != stock_call_slot.type) {
            // Return first found compatible call index
            for (auto& comp_call_idx_1 : call_slot->compatible_call_idxs) {
                for (auto& comp_call_idx_2 : stock_call_slot.compatible_call_idxs) {
                    if (comp_call_idx_1 == comp_call_idx_2) {
                        return static_cast<ImGuiID>(comp_call_idx_1);
                    }
                }
            }
        }
    }
    return GUI_INVALID_ID;
}


// CALL SLOT PRESENTATION ####################################################

megamol::gui::configurator::CallSlot::Presentation::Presentation(void)
    : group()
    , presentations(CallSlot::Presentations::DEFAULT)
    , label_visible(false)
    , position()
    , utils()
    , selected(false)
    , update_once(true) {

    this->group.is_interface = false;
    this->group.position = ImVec2(FLT_MAX, FLT_MAX);
}


megamol::gui::configurator::CallSlot::Presentation::~Presentation(void) {}


void megamol::gui::configurator::CallSlot::Presentation::Present(
    megamol::gui::configurator::CallSlot& inout_call_slot, megamol::gui::GraphItemsStateType& state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
    ImGuiStyle& style = ImGui::GetStyle();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    try {

        if (this->update_once) {
            this->UpdatePosition(inout_call_slot, state.canvas);
            this->update_once = false;
        }

        std::string module_label;
        ImGuiID is_parent_module_group_member = GUI_INVALID_ID;
        bool is_parent_module_group_visible = GUI_INVALID_ID;
        ImGuiID parent_module_uid = GUI_INVALID_ID;
        if (inout_call_slot.ParentModuleConnected()) {
            is_parent_module_group_member = inout_call_slot.GetParentModule()->GUI_GetGroupMembership();
            is_parent_module_group_visible = inout_call_slot.GetParentModule()->GUI_GetGroupVisibility();
            module_label = "[" + inout_call_slot.GetParentModule()->name + "]";
            parent_module_uid = inout_call_slot.GetParentModule()->uid;
        }

        // Check for slot invisibility
        if (!this->group.is_interface && (is_parent_module_group_member != GUI_INVALID_ID) && !is_parent_module_group_visible) {
            return;
        }

        ImVec2 slot_position = this->GetPosition(inout_call_slot);
        float radius = GUI_CALL_SLOT_RADIUS * state.canvas.zooming;
        std::string slot_label = "[" + inout_call_slot.name + "]";

        ImVec2 text_pos_left_upper = ImVec2(0.0f, 0.0f);
        if (this->label_visible) {
            text_pos_left_upper.y = slot_position.y - ImGui::GetTextLineHeightWithSpacing() / 2.0f;
            if (inout_call_slot.type == CallSlotType::CALLER) {
                text_pos_left_upper.x =
                    slot_position.x - GUIUtils::TextWidgetWidth(inout_call_slot.name) - (1.5f * radius);
            } else if (inout_call_slot.type == CallSlotType::CALLEE) {
                text_pos_left_upper.x = slot_position.x + (1.5f * radius);
            }
        }

        // Clip call slots if lying ouside the canvas
        /// XXX Is there a benefit since ImGui::PushClipRect is used?
        /*
        ImVec2 canvas_rect_min = state.canvas.position;
        ImVec2 canvas_rect_max = state.canvas.position + state.canvas.size;
        ImVec2 slot_rect_min = ImVec2(slot_position.x - radius, slot_position.y - radius);
        ImVec2 slot_rect_max = ImVec2(slot_position.x + radius, slot_position.y + radius);
        if (this->label_visible) {
            ImVec2 text_clip_pos = text_pos_left_upper;
            if (inout_call_slot.type == CallSlotType::CALLEE) {
                text_clip_pos = ImVec2(
                    text_pos_left_upper.x + GUIUtils::TextWidgetWidth(inout_call_slot.name), text_pos_left_upper.y);
            }
            if (text_clip_pos.x < slot_rect_min.x) slot_rect_min.x = text_clip_pos.x;
            if (text_clip_pos.x > slot_rect_max.x) slot_rect_max.x = text_clip_pos.x;
            if (text_clip_pos.y < slot_rect_min.y) slot_rect_min.y = text_clip_pos.y;
            if (text_clip_pos.y > slot_rect_max.y) slot_rect_max.y = text_clip_pos.y;
        }
        if (!((canvas_rect_min.x < (slot_rect_max.x)) && (canvas_rect_max.x > (slot_rect_min.x)) &&
                (canvas_rect_min.y < (slot_rect_max.y)) && (canvas_rect_max.y > (slot_rect_min.y)))) {
            if (mouse_clicked) {
                this->selected = false;
                if (state.interact.callslot_selected_uid == inout_module.uid) {
                    state.interact.callslot_selected_uid = GUI_INVALID_ID;
                }
            }
            if (this->selected) {
                state.interact.callslot_selected_uid = inout_module.uid;
            }
            return;
        }
        */

        ImGui::PushID(inout_call_slot.uid);

        // Colors
        ImVec4 tmpcol = style.Colors[ImGuiCol_FrameBg]; // ImGuiCol_FrameBg ImGuiCol_Button
        tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
        const ImU32 COLOR_SLOT_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);

        tmpcol = style.Colors[ImGuiCol_ScrollbarGrabActive]; // ImGuiCol_Border ImGuiCol_ScrollbarGrabActive
        tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
        const ImU32 COLOR_SLOT_GROUP_BORDER = ImGui::ColorConvertFloat4ToU32(tmpcol);

        ImU32 COLOR_SLOT_CALLER_HIGHLIGHT = IM_COL32(0, 255, 192, 255);
        ImU32 COLOR_SLOT_CALLEE_HIGHLIGHT = IM_COL32(192, 0, 255, 255);
        ImU32 COLOR_SLOT_COMPATIBLE = IM_COL32(192, 255, 64, 255);

        ImU32 slot_color = COLOR_SLOT_BACKGROUND;

        ImU32 slot_highlight_color = COLOR_SLOT_BACKGROUND;
        if (inout_call_slot.type == CallSlotType::CALLER) {
            slot_highlight_color = COLOR_SLOT_CALLER_HIGHLIGHT;
        } else if (inout_call_slot.type == CallSlotType::CALLEE) {
            slot_highlight_color = COLOR_SLOT_CALLEE_HIGHLIGHT;
        }

        ImGui::SetCursorScreenPos(slot_position - ImVec2(radius, radius));
        std::string label = "slot_" + inout_call_slot.name + std::to_string(inout_call_slot.uid);
        ImGui::SetItemAllowOverlap();
        ImGui::InvisibleButton(label.c_str(), ImVec2(radius * 2.0f, radius * 2.0f));
        ImGui::SetItemAllowOverlap();

        bool active = ImGui::IsItemActive(); 
        bool mouse_clicked = ImGui::IsWindowHovered() && ImGui::GetIO().MouseClicked[0];                                     
        bool hovered = (ImGui::IsItemHovered() && 
            ((state.interact.module_hovered_uid == GUI_INVALID_ID) || (state.interact.module_hovered_uid == parent_module_uid)) && 
            ((state.interact.callslot_hovered_uid == GUI_INVALID_ID) || (state.interact.callslot_hovered_uid == inout_call_slot.uid)));

        // Context menu
        if (state.interact.module_hovered_uid == GUI_INVALID_ID) {
            if (ImGui::BeginPopupContextItem("invisible_button_context")) {
                active = true; // Force selection                
                ImGui::Text("Call Slot");
                ImGui::Separator();
                /// Menu items is only active depending on group membership of  parent module 
                if (ImGui::MenuItem("Add to Group Interface ", nullptr, false,
                        (!this->group.is_interface && (is_parent_module_group_member != GUI_INVALID_ID)))) {
                    state.interact.callslot_add_group_uid.first = inout_call_slot.uid;
                    state.interact.callslot_add_group_uid.second = inout_call_slot.GetParentModule()->uid;
                }
                if (ImGui::MenuItem("Remove Group Interface", nullptr, false,
                        (this->group.is_interface && (is_parent_module_group_member != GUI_INVALID_ID)))) {
                    state.interact.callslot_remove_group_uid = inout_call_slot.uid;
                }
                ImGui::EndPopup();
            }
        }
        
        // Hover Tooltip
        if (hovered) {
            std::string tooltip = inout_call_slot.description;
            if (this->group.is_interface && (is_parent_module_group_member != GUI_INVALID_ID) && !is_parent_module_group_visible) {
                tooltip = module_label + " > " + slot_label + " " + inout_call_slot.description;
            }
            else if (!this->label_visible) {
                tooltip = slot_label + " " + inout_call_slot.description;
            }
            this->utils.HoverToolTip(tooltip, ImGui::GetID(label.c_str()), 0.75f, 5.0f);        
        }
        else {
            this->utils.ResetHoverToolTip();
        }
        
        // State
        if (CallSlot::CheckCompatibleAvailableCallIndex(state.interact.callslot_compat_ptr, inout_call_slot) !=
            GUI_INVALID_ID) {
            slot_color = COLOR_SLOT_COMPATIBLE;
        }
        if ((mouse_clicked && !hovered) || (state.interact.callslot_selected_uid != inout_call_slot.uid)) {
            this->selected = false;
            if (state.interact.callslot_selected_uid == inout_call_slot.uid) {
                state.interact.callslot_selected_uid = GUI_INVALID_ID;
            }
        }
        if (active) {
            this->selected = true;
            state.interact.callslot_selected_uid = inout_call_slot.uid;
            state.interact.call_selected_uid = GUI_INVALID_ID;
            state.interact.module_selected_uid = GUI_INVALID_ID;
            state.interact.group_selected_uid = GUI_INVALID_ID;
        }
        if (hovered || this->selected) {
            slot_color = slot_highlight_color;
        }
        if (hovered) {
            state.interact.callslot_hovered_uid = inout_call_slot.uid;
        }
        if (!hovered && (state.interact.callslot_hovered_uid == inout_call_slot.uid)) {
            state.interact.callslot_hovered_uid = GUI_INVALID_ID;
        }
        if (ImGui::BeginDragDropTarget()) {
            if (ImGui::AcceptDragDropPayload(GUI_DND_CALL_UID_TYPE) != nullptr) {
                state.interact.callslot_dropped_uid = inout_call_slot.uid;
            }
            ImGui::EndDragDropTarget();
        }
        if (this->selected) {
            auto dnd_flags = ImGuiDragDropFlags_AcceptNoDrawDefaultRect; // | ImGuiDragDropFlags_SourceNoPreviewTooltip;
            if (ImGui::BeginDragDropSource(dnd_flags)) {
                ImGui::SetDragDropPayload(GUI_DND_CALL_UID_TYPE, &inout_call_slot.uid, sizeof(ImGuiID));
                ImGui::Text(slot_label.c_str());
                ImGui::EndDragDropSource();
            }
        }
        
        // Slot
        if (this->group.is_interface && (is_parent_module_group_member != GUI_INVALID_ID) &&
            is_parent_module_group_visible) {
            draw_list->AddLine(this->position, slot_position,
                ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Border]),
                (GUI_CALL_SLOT_RADIUS / 3.0f) * state.canvas.zooming);
        }
        draw_list->AddCircleFilled(slot_position, radius, slot_color);
        draw_list->AddCircle(slot_position, radius, COLOR_SLOT_GROUP_BORDER);
        
        // Text
        if (this->label_visible && !(this->group.is_interface && (is_parent_module_group_member != GUI_INVALID_ID) && !is_parent_module_group_visible)) {
            draw_list->AddText(text_pos_left_upper, slot_highlight_color, inout_call_slot.name.c_str());
        }

        ImGui::PopID();

    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


void megamol::gui::configurator::CallSlot::Presentation::UpdatePosition(
    megamol::gui::configurator::CallSlot& inout_call_slot, const GraphCanvasType& in_canvas) {

    if (inout_call_slot.ParentModuleConnected()) {
        auto slot_count = inout_call_slot.GetParentModule()->GetCallSlots(inout_call_slot.type).size();
        size_t slot_idx = 0;
        for (size_t idx = 0; idx < slot_count; idx++) {
            if (inout_call_slot.name ==
                inout_call_slot.GetParentModule()->GetCallSlots(inout_call_slot.type)[idx]->name) {
                slot_idx = idx;
            }
        }
        
        float y_offset = 0.0f;
        if (inout_call_slot.GetParentModule()->GUI_GetLabelVisibility()) {
            y_offset = ImGui::GetTextLineHeightWithSpacing() / in_canvas.zooming;
        }
        auto module_pos = inout_call_slot.GetParentModule()->GUI_GetPosition();
        module_pos.y += y_offset;
        ImVec2 pos = in_canvas.offset +  module_pos * in_canvas.zooming;
        auto module_size = inout_call_slot.GetParentModule()->GUI_GetSize();
        module_size.y -= y_offset;
        ImVec2 size = module_size * in_canvas.zooming;
        this->position = ImVec2(pos.x + ((inout_call_slot.type == CallSlotType::CALLER) ? (size.x) : (0.0f)),
            pos.y + size.y * ((float)slot_idx + 1) / ((float)slot_count + 1));
    }
}


ImVec2 megamol::gui::configurator::CallSlot::Presentation::GetPosition(CallSlot& inout_call_slot) {

    ImGuiID is_parent_module_group_member = GUI_INVALID_ID;
    bool is_parent_module_group_visible = false;
    if (inout_call_slot.ParentModuleConnected()) {
        is_parent_module_group_member = inout_call_slot.GetParentModule()->GUI_GetGroupMembership();
        is_parent_module_group_visible = inout_call_slot.GetParentModule()->GUI_GetGroupVisibility();
    }

    // Actual position to use
    ImVec2 actual_position = this->position;
    if (this->group.is_interface) {
        actual_position = this->group.position;
        if ((is_parent_module_group_member != GUI_INVALID_ID) && is_parent_module_group_visible) {
            actual_position.y = this->position.y;
        }
    }
    return actual_position;
}
