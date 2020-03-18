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


megamol::gui::configurator::CallSlot::CallSlot(ImGuiID uid) : uid(uid), present() {
    this->parent_module.reset();
    connected_calls.clear();
}


megamol::gui::configurator::CallSlot::~CallSlot() {

    // Call separately and check for reference count
    this->DisConnectCalls();
    this->DisConnectParentModule();
}


bool megamol::gui::configurator::CallSlot::CallsConnected(void) const {

    /// TEMP Check for unclean references
    for (auto& call_ptr : this->connected_calls) {
        if (call_ptr == nullptr) {
            throw std::invalid_argument("Pointer to connected call is nullptr.");
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
    if (this->type == CallSlot::CallSlotType::CALLER) {
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
                vislib::sys::Log::DefaultLog.WriteWarn(
                    "Call is already disconnected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
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
                vislib::sys::Log::DefaultLog.WriteWarn(
                    "Call is already disconnected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
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

    /// TEMP Check for unclean references
    for (auto& call_ptr : this->connected_calls) {
        if (call_ptr == nullptr) {
            throw std::invalid_argument("Pointer to connected call is nullptr.");
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
                        ///XXX Disabled Feature
                        // Show only comaptible calls for unconnected caller slots
                        // if ((call_slot_ptr->type == CallSlot::CallSlotType::CALLER) &&
                        //     (call_slot_ptr->CallsConnected())) {
                        //     return GUI_INVALID_ID;
                        // } else if ((call_slot.type == CallSlot::CallSlotType::CALLER) && (call_slot.CallsConnected())) {
                        //     return GUI_INVALID_ID;
                        // }
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
    : presentations(CallSlot::Presentations::DEFAULT), label_visible(false), position(), utils(), selected(false) {}

megamol::gui::configurator::CallSlot::Presentation::~Presentation(void) {}


ImGuiID megamol::gui::configurator::CallSlot::Presentation::Present(
    megamol::gui::configurator::CallSlot& inout_call_slot, const Canvas& in_canvas, ImGuiID& out_hovered_call_slot_uid,
    const CallSlotPtrType compatible_call_slot_ptr) {

    ImGuiID retval_id = GUI_INVALID_ID;
    ImGuiStyle& style = ImGui::GetStyle();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);
    if (ImGui::GetCurrentContext() == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    try {
        // Trigger only when canvas was updated
        // Always update position before clipping -> calls need updated slot positions.
        if (in_canvas.updated) {
            this->UpdatePosition(inout_call_slot, in_canvas.offset, in_canvas.zooming);
        }
        ImVec2 slot_position = this->position;
        float radius = GUI_CALL_SLOT_RADIUS * in_canvas.zooming;

        // Draw text
        ImVec2 text_pos_left_upper = ImVec2(0.0f, 0.0f);
        if (this->label_visible) {
            text_pos_left_upper.y = slot_position.y - ImGui::GetTextLineHeightWithSpacing() / 2.0f;
            if (inout_call_slot.type == CallSlot::CallSlotType::CALLER) {
                text_pos_left_upper.x =
                    slot_position.x - this->utils.TextWidgetWidth(inout_call_slot.name) - (2.0f * radius);
            } else if (inout_call_slot.type == CallSlot::CallSlotType::CALLEE) {
                text_pos_left_upper.x = slot_position.x + (2.0f * radius);
            }
        }

        // Clip call slots if lying ouside the canvas
        ImVec2 canvas_rect_min = in_canvas.position;
        ImVec2 canvas_rect_max = in_canvas.position + in_canvas.size;
        ImVec2 slot_rect_min = ImVec2(slot_position.x - radius, slot_position.y - radius);
        ImVec2 slot_rect_max = ImVec2(slot_position.x + radius, slot_position.y + radius);
        if (this->label_visible) {
            ImVec2 text_clip_pos = text_pos_left_upper;
            if (inout_call_slot.type == CallSlot::CallSlotType::CALLEE) {
                text_clip_pos = ImVec2(
                    text_pos_left_upper.x + this->utils.TextWidgetWidth(inout_call_slot.name), text_pos_left_upper.y);
            }
            if (text_clip_pos.x < slot_rect_min.x) slot_rect_min.x = text_clip_pos.x;
            if (text_clip_pos.x > slot_rect_max.x) slot_rect_max.x = text_clip_pos.x;
            if (text_clip_pos.y < slot_rect_min.y) slot_rect_min.y = text_clip_pos.y;
            if (text_clip_pos.y > slot_rect_max.y) slot_rect_max.y = text_clip_pos.y;
        }
        if (!((canvas_rect_min.x < (slot_rect_max.x)) && (canvas_rect_max.x > (slot_rect_min.x)) &&
                (canvas_rect_min.y < (slot_rect_max.y)) && (canvas_rect_max.y > (slot_rect_min.y)))) {
            this->selected = false;
            return GUI_INVALID_ID;
        }

        ImGui::PushID(inout_call_slot.uid);

        draw_list->ChannelsSetCurrent(1); // Foreground

        ImVec4 tmpcol = style.Colors[ImGuiCol_Button];
        tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
        const ImU32 COLOR_SLOT_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);
        const ImU32 COLOR_SLOT_BORDER = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_PopupBg]);
        const ImU32 COLOR_SLOT_CALLER = IM_COL32(0, 255, 192, 255);
        const ImU32 COLOR_SLOT_CALLEE = IM_COL32(192, 255, 64, 255);
        const ImU32 COLOR_SLOT_COMPATIBLE = IM_COL32(0, 192, 0, 255);

        ImU32 slot_color = COLOR_SLOT_BACKGROUND;

        ImU32 slot_highlight_color = COLOR_SLOT_BACKGROUND;
        if (inout_call_slot.type == CallSlot::CallSlotType::CALLER) {
            slot_highlight_color = COLOR_SLOT_CALLER;
        } else if (inout_call_slot.type == CallSlot::CallSlotType::CALLEE) {
            slot_highlight_color = COLOR_SLOT_CALLEE;
        }

        ImGui::SetCursorScreenPos(slot_position - ImVec2(radius, radius));
        std::string label = "slot_" + inout_call_slot.name + std::to_string(inout_call_slot.uid);
        ImGui::InvisibleButton(label.c_str(), ImVec2(radius * 2.0f, radius * 2.0f));
        bool active = ImGui::IsItemActive();
        bool hovered = ImGui::IsItemHovered();
        bool mouse_clicked = ImGui::GetIO().MouseClicked[0];

        std::string tooltip = inout_call_slot.description;
        if (!this->label_visible) {
            tooltip = "[" + inout_call_slot.name + "] " + tooltip;
        }
        this->utils.HoverToolTip(tooltip, ImGui::GetID(label.c_str()), 0.5f, 5.0f);

        // Highlight if compatible to given call slot
        if (CallSlot::CheckCompatibleAvailableCallIndex(compatible_call_slot_ptr, inout_call_slot) != GUI_INVALID_ID) {
            slot_color = COLOR_SLOT_COMPATIBLE;
        }
        if (mouse_clicked && !hovered) {
            this->selected = false;
        }
        if (active) {
            this->selected = true;
        }
        if (hovered || this->selected) {
            slot_color = slot_highlight_color;
        }
        if (hovered) {
            out_hovered_call_slot_uid = inout_call_slot.uid;
        }
        if (this->selected) {
            retval_id = inout_call_slot.uid;
        }

        ImGui::SetCursorScreenPos(slot_position);
        draw_list->AddCircleFilled(slot_position, radius, slot_color);
        draw_list->AddCircle(slot_position, radius, COLOR_SLOT_BORDER);

        if (this->label_visible) {
            draw_list->AddText(text_pos_left_upper, slot_highlight_color, inout_call_slot.name.c_str());
        }

        ImGui::PopID();

    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    }

    return retval_id;
}


void megamol::gui::configurator::CallSlot::Presentation::UpdatePosition(
    megamol::gui::configurator::CallSlot& call_slot, ImVec2 canvas_offset, float canvas_zooming) {

    if (call_slot.ParentModuleConnected()) {
        auto slot_count = call_slot.GetParentModule()->GetCallSlots(call_slot.type).size();
        size_t slot_idx = 0;
        for (size_t idx = 0; idx < slot_count; idx++) {
            if (call_slot.name == call_slot.GetParentModule()->GetCallSlots(call_slot.type)[idx]->name) {
                slot_idx = idx;
            }
        }
        auto pos = canvas_offset + call_slot.GetParentModule()->GUI_GetPosition() * canvas_zooming;
        auto size = call_slot.GetParentModule()->GUI_GetSize() * canvas_zooming;
        this->position = ImVec2(pos.x + ((call_slot.type == CallSlot::CallSlotType::CALLER) ? (size.x) : (0.0f)),
            pos.y + size.y * ((float)slot_idx + 1) / ((float)slot_count + 1));
    }
}
