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
using namespace megamol::gui::configurator;


megamol::gui::configurator::CallSlot::CallSlot(int uid) : uid(uid), present() {
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


bool megamol::gui::configurator::CallSlot::DisConnectCall(int call_uid, bool called_by_call) {

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
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to parent module is already nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
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


int megamol::gui::configurator::CallSlot::GetCompatibleCallIndex(
    megamol::gui::configurator::CallSlotPtrType call_slot_1, megamol::gui::configurator::CallSlotPtrType call_slot_2) {

    if ((call_slot_1 != nullptr) && (call_slot_2 != nullptr)) {
        if ((call_slot_1 != call_slot_2) && (call_slot_1->GetParentModule() != call_slot_2->GetParentModule()) &&
            (call_slot_1->type != call_slot_2->type)) {
            // Return first found compatible call index
            for (auto& selected_comp_call_slots : call_slot_1->compatible_call_idxs) {
                for (auto& current_comp_call_slots : call_slot_2->compatible_call_idxs) {
                    if (selected_comp_call_slots == current_comp_call_slots) {
                        // Show only comaptible calls for unconnected caller slots
                        if ((call_slot_1->type == CallSlot::CallSlotType::CALLER) && (call_slot_1->CallsConnected())) {
                            return GUI_INVALID_ID;
                        } else if ((call_slot_2->type == CallSlot::CallSlotType::CALLER) &&
                                   (call_slot_2->CallsConnected())) {
                            return GUI_INVALID_ID;
                        }
                        return static_cast<int>(current_comp_call_slots);
                    }
                }
            }
        }
    }
    return GUI_INVALID_ID;
}


int megamol::gui::configurator::CallSlot::GetCompatibleCallIndex(megamol::gui::configurator::CallSlotPtrType call_slot,
    megamol::gui::configurator::CallSlot::StockCallSlot stock_call_slot) {

    if (call_slot != nullptr) {
        if (call_slot->type != stock_call_slot.type) {
            // Return first found compatible call index
            for (auto& selected_comp_call_slots : call_slot->compatible_call_idxs) {
                for (auto& current_comp_call_slots : stock_call_slot.compatible_call_idxs) {
                    if (selected_comp_call_slots == current_comp_call_slots) {
                        return static_cast<int>(current_comp_call_slots);
                    }
                }
            }
        }
    }
    return GUI_INVALID_ID;
}


// CALL SLOT PRESENTATION ####################################################

megamol::gui::configurator::CallSlot::Presentation::Presentation(void)
    : presentations(Presentation::DEFAULT), label_visible(true), position(), slot_radius(8.0f), utils() {}

megamol::gui::configurator::CallSlot::Presentation::~Presentation(void) {}


ImGuiID megamol::gui::configurator::CallSlot::Presentation::GUI_Present(
    megamol::gui::configurator::CallSlot& call_slot, ImVec2 canvas_offset, float canvas_zooming) {

    int retval_id = GUI_INVALID_ID;

    try {

        if (ImGui::GetCurrentContext() == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        /// XXX Trigger only when necessary
        this->UpdatePosition(call_slot, canvas_offset, canvas_zooming);

        ImGui::PushID(call_slot.uid);

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

        if (call_slot.type == CallSlot::CallSlotType::CALLER) {
            slot_highl_color = COLOR_SLOT_CALLER_HIGHTLIGHT;
            slot_label_color = COLOR_SLOT_CALLER_LABEL;
        } else if (call_slot.type == CallSlot::CallSlotType::CALLEE) {
            slot_highl_color = COLOR_SLOT_CALLEE_HIGHTLIGHT;
            slot_label_color = COLOR_SLOT_CALLEE_LABEL;
        }

        ImVec2 slot_position = this->position;
        float radius = this->slot_radius * canvas_zooming;
        std::string slot_name = call_slot.name;
        slot_color = COLOR_SLOT;

        ImGui::SetCursorScreenPos(slot_position - ImVec2(radius, radius));
        std::string label = "slot_" + slot_name + std::to_string(call_slot.uid);
        ImGui::InvisibleButton(label.c_str(), ImVec2(radius * 2.0f, radius * 2.0f));

        std::string tooltip = call_slot.description;
        if (!this->label_visible) {
            tooltip = call_slot.name + " | " + tooltip;
        }
        this->utils.HoverToolTip(tooltip, ImGui::GetID(label.c_str()), 0.5f, 5.0f);
        auto hovered = ImGui::IsItemHovered();
        auto clicked = ImGui::IsItemClicked();
        /// XXX
        /*
         int compat_call_idx = CallSlot::GetCompatibleCallIndex(selected_slot_ptr, call_slot);
         if (hovered) {
            retval_id = call_slot.uid;
            // Check if selected call slot should be connected with current slot
            if (process_selected_slot > 0) {
                if (graph->AddCall(this->graph_manager.GetCallsStock(), compat_call_idx, selected_slot_ptr, call_slot))
        { selected_slot_ptr = nullptr;
                }
            }
        }
         if (clicked) {
            // Select / Unselect call slot
            if (selected_slot_ptr != call_slot) {
                selected_slot_ptr = call_slot;
            } else {
                selected_slot_ptr = nullptr;
            }
        }
         if (hovered || (selected_slot_ptr == call_slot)) {
            slot_color = slot_highl_color;
        }
         //Highlight if compatible to selected slot

         if (compat_call_idx > 0) {
            slot_color = COLOR_SLOT_COMPATIBLE;
        }
        */

        ImGui::SetCursorScreenPos(slot_position);
        draw_list->AddCircleFilled(slot_position, radius, slot_color);
        draw_list->AddCircle(slot_position, radius, COLOR_SLOT_BORDER);

        // Draw text
        if (this->label_visible) {
            ImVec2 text_pos;
            text_pos.y = slot_position.y - ImGui::GetFontSize() / 2.0f;
            if (call_slot.type == CallSlot::CallSlotType::CALLER) {
                text_pos.x = slot_position.x - this->utils.TextWidgetWidth(slot_name) - (2.0f * radius);
            } else if (call_slot.type == CallSlot::CallSlotType::CALLEE) {
                text_pos.x = slot_position.x + (2.0f * radius);
            }
            draw_list->AddText(text_pos, slot_label_color, slot_name.c_str());
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
        auto size = call_slot.GetParentModule()->GUI_GetSize();
        this->position = ImVec2(pos.x + ((call_slot.type == CallSlot::CallSlotType::CALLER) ? (size.x) : (0.0f)),
            pos.y + size.y * ((float)slot_idx + 1) / ((float)slot_count + 1));
    }
}
