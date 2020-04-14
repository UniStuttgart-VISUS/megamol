/*
 * InterfaceSlot.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "CallSlot.h"
#include "InterfaceSlot.h"


using namespace megamol;
using namespace megamol::gui;
using namespace megamol::gui::configurator;


megamol::gui::configurator::InterfaceSlot::InterfaceSlot() {}


megamol::gui::configurator::InterfaceSlot::~InterfaceSlot(void) {

    // Remove all call slots from interface slot
    std::vector<ImGuiID> callslots_uids;
    for (auto& callslot_ptr : callslots) {
        callslots_uids.emplace_back(callslot_ptr->uid);
    }
    for (auto& callslots_uid : callslots_uids) {
        this->RemoveCallSlot(callslots_uid);
    }
    this->callslots.clear();
}


bool megamol::gui::configurator::InterfaceSlot::AddCallSlot(CallSlotPtrType callslot_ptr) {

    try {
        if (callslot_ptr == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        if (!(this->ContainsCallSlot(callslot_ptr->uid)) && (this->IsCallSlotCompatible(callslot_ptr))) {
            this->callslots.emplace_back(callslot_ptr);

            InterfaceSlotPtrType interface_ptr = std::make_shared<InterfaceSlot>(*this);
            if (interface_ptr != nullptr) {
                callslot_ptr->GUI_SetGroupInterface(interface_ptr);

                vislib::sys::Log::DefaultLog.WriteInfo(
                    "Added call slot '%s' to interface slot of group.\n", callslot_ptr->name.c_str());
                return true;
            }
            vislib::sys::Log::DefaultLog.WriteError(
                "Unable to create pointer to interface slot of group. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
        }
    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    vislib::sys::Log::DefaultLog.WriteError(
        "Unable to add call slot '%s' to interface slot of group. [%s, %s, line %d]\n", callslot_ptr->name.c_str(),
        __FILE__, __FUNCTION__, __LINE__);
    return false;
}


bool megamol::gui::configurator::InterfaceSlot::RemoveCallSlot(ImGuiID callslot_uid) {

    try {
        for (auto iter = this->callslots.begin(); iter != this->callslots.end(); iter++) {
            if ((*iter)->uid == callslot_uid) {

                (*iter)->GUI_SetGroupInterface(nullptr);

                vislib::sys::Log::DefaultLog.WriteInfo(
                    "Removed call slot '%s' from interface slot of group.\n", (*iter)->name.c_str());
                (*iter).reset();
                this->callslots.erase(iter);

                return true;
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


bool megamol::gui::configurator::InterfaceSlot::ContainsCallSlot(ImGuiID callslot_uid) {

    for (auto& callslot : this->callslots) {
        if (callslot_uid == callslot->uid) {
            return true;
        }
    }
    return false;
}


bool megamol::gui::configurator::InterfaceSlot::IsCallSlotCompatible(CallSlotPtrType callslot_ptr) {

    if (callslot_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Check for compatibility (with all available call slots...)
    size_t compatible = 0;
    for (auto& callslot : this->callslots) {
        if ((callslot_ptr->type == callslot->type) &&
            (callslot_ptr->compatible_call_idxs == callslot->compatible_call_idxs)) {
            compatible++;
        }
    }

    bool retval = (compatible == this->callslots.size());

    if ((compatible > 0) && (compatible != this->callslots.size())) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Interface slot contains incompatible call slots. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    return retval;
}


bool megamol::gui::configurator::InterfaceSlot::IsEmpty(void) { return (this->callslots.empty()); }


// GROUP INTERFACE SLOT PRESENTATION ###########################################

megamol::gui::configurator::InterfaceSlot::Presentation::Presentation(void)
    : position(ImVec2(FLT_MAX, FLT_MAX)), utils(), selected(false) {}


megamol::gui::configurator::InterfaceSlot::Presentation::~Presentation(void) {}


void megamol::gui::configurator::InterfaceSlot::Presentation::Present(
    megamol::gui::configurator::InterfaceSlot& inout_interfaceslot, megamol::gui::GraphItemsStateType& state,
    bool collapsed_view) {

    if (ImGui::GetCurrentContext() == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    try {

        CallSlotType type = inout_interfaceslot.GetCallSlots().front()->type;
        bool compatible = (CallSlot::CheckCompatibleAvailableCallIndex(state.interact.callslot_compat_ptr,
                               (*inout_interfaceslot.GetCallSlots().front())) != GUI_INVALID_ID);
        // Generate UID sting (interface slot otherwise does not need one ...)
        std::string interface_uid_str;
        for (auto& callslot_ptr : inout_interfaceslot.GetCallSlots()) {
            interface_uid_str += std::to_string(callslot_ptr->uid);
        }

        float radius = GUI_SLOT_RADIUS * state.canvas.zooming;

        // Button
        ImGui::SetCursorScreenPos(this->position - ImVec2(radius, radius));
        std::string label = "interface_slot" + interface_uid_str;
        ImGui::SetItemAllowOverlap();
        ImGui::InvisibleButton(label.c_str(), ImVec2(radius * 2.0f, radius * 2.0f));
        ImGui::SetItemAllowOverlap();

        bool button_active = ImGui::IsItemActive();
        bool mouse_clicked_anywhere = ImGui::IsWindowHovered() && ImGui::GetIO().MouseClicked[0];
        bool button_hovered = (ImGui::IsItemHovered() && (state.interact.module_hovered_uid == GUI_INVALID_ID) &&
                               (state.interact.callslot_hovered_uid == GUI_INVALID_ID));

        // Hover Tooltip
        if (button_hovered) {
            std::string tooltip;
            for (auto& callslot_ptr : inout_interfaceslot.GetCallSlots()) {
                tooltip += (callslot_ptr->name + "\n");
            }
            this->utils.HoverToolTip(tooltip, ImGui::GetID(label.c_str()), 0.5f, 5.0f);
        } else {
            this->utils.ResetHoverToolTip();
        }

        // Colors
        ImVec4 tmpcol = style.Colors[ImGuiCol_FrameBg];
        tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
        const ImU32 COLOR_INTERFACE_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);

        tmpcol = style.Colors[ImGuiCol_ScrollbarGrab];
        tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
        const ImU32 COLOR_INTERFACE_LINE = ImGui::ColorConvertFloat4ToU32(tmpcol);

        const ImU32 COLOR_INTERFACE_HIGHTLIGHT = ImGui::ColorConvertFloat4ToU32(tmpcol);

        tmpcol = style.Colors[ImGuiCol_ScrollbarGrabActive];
        tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
        const ImU32 COLOR_INTERFACE_GROUP_BORDER = ImGui::ColorConvertFloat4ToU32(tmpcol);

        tmpcol = style.Colors[ImGuiCol_FrameBg];
        tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
        const ImU32 COLOR_SLOT_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);

        tmpcol = style.Colors[ImGuiCol_ScrollbarGrabActive];
        tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
        const ImU32 COLOR_SLOT_GROUP_BORDER = ImGui::ColorConvertFloat4ToU32(tmpcol);

        ImU32 COLOR_SLOT_CALLER_HIGHLIGHT = IM_COL32(0, 255, 192, 255);

        ImU32 COLOR_SLOT_CALLEE_HIGHLIGHT = IM_COL32(192, 0, 255, 255);

        ImU32 COLOR_SLOT_COMPATIBLE = IM_COL32(192, 255, 64, 255);

        // Color modification
        ImU32 slot_highlight_color = COLOR_SLOT_BACKGROUND;
        if (type == CallSlotType::CALLER) {
            slot_highlight_color = COLOR_SLOT_CALLER_HIGHLIGHT;
        } else if (type == CallSlotType::CALLEE) {
            slot_highlight_color = COLOR_SLOT_CALLEE_HIGHLIGHT;
        }
        ImU32 slot_color = COLOR_SLOT_BACKGROUND;
        if (compatible) {
            slot_color = COLOR_SLOT_COMPATIBLE;
        }
        if (button_hovered || this->selected) {
            slot_color = slot_highlight_color;
        }

        // Draw Slot
        const float segment_numer = 20.0f;
        draw_list->AddCircleFilled(this->position, radius, slot_color, segment_numer);
        draw_list->AddCircle(this->position, radius, COLOR_SLOT_GROUP_BORDER, segment_numer);

        // Draw Curves
        if (!collapsed_view) {
            for (auto& callslot_ptr : inout_interfaceslot.GetCallSlots()) {
                draw_list->AddLine(this->position, callslot_ptr->GUI_GetPosition(), COLOR_INTERFACE_LINE,
                    GUI_LINE_THICKNESS * state.canvas.zooming);
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
