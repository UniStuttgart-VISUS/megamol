/*
 * Call.cpp
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


megamol::gui::configurator::Call::Call(int uid) : uid(uid), present() {

    this->connected_call_slots.clear();
    this->connected_call_slots.emplace(CallSlot::CallSlotType::CALLER, nullptr);
    this->connected_call_slots.emplace(CallSlot::CallSlotType::CALLEE, nullptr);
}


megamol::gui::configurator::Call::~Call() { this->DisConnectCallSlots(); }


bool megamol::gui::configurator::Call::IsConnected(void) {

    unsigned int not_connected = 0;
    for (auto& call_slot_map : this->connected_call_slots) {
        if (call_slot_map.second != nullptr) {
            not_connected++;
        }
    }
    if (not_connected == 1) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Only one call slot is connected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return (not_connected == 2);
}


bool megamol::gui::configurator::Call::ConnectCallSlots(
    megamol::gui::configurator::CallSlotPtrType call_slot_1, megamol::gui::configurator::CallSlotPtrType call_slot_2) {

    if ((call_slot_1 == nullptr) || (call_slot_2 == nullptr)) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to given call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (call_slot_1->type == call_slot_2->type) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Call slots must have different type. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (call_slot_1->GetParentModule() == call_slot_2->GetParentModule()) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Call slots must have different parent module. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if ((this->connected_call_slots[call_slot_1->type] != nullptr) ||
        (this->connected_call_slots[call_slot_2->type] != nullptr)) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Call is already connected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    this->connected_call_slots[call_slot_1->type] = call_slot_1;
    this->connected_call_slots[call_slot_2->type] = call_slot_2;

    return true;
}


bool megamol::gui::configurator::Call::DisConnectCallSlots(void) {

    try {
        for (auto& call_slot_map : this->connected_call_slots) {
            if (call_slot_map.second == nullptr) {
                vislib::sys::Log::DefaultLog.WriteWarn(
                    "Call slot is already disconnected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            } else {
                call_slot_map.second->DisConnectCall(this->uid, true);
                call_slot_map.second.reset();
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


const megamol::gui::configurator::CallSlotPtrType megamol::gui::configurator::Call::GetCallSlot(
    megamol::gui::configurator::CallSlot::CallSlotType type) {

    if (this->connected_call_slots[type] == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Returned pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    return this->connected_call_slots[type];
}


// CALL PRESENTATION #########################################################

megamol::gui::configurator::Call::Presentation::Presentation(void)
    : presentations(Call::Presentations::DEFAULT), label_visible(true), utils(), selected(false) {}


megamol::gui::configurator::Call::Presentation::~Presentation(void) {}


int megamol::gui::configurator::Call::Presentation::Present(
    megamol::gui::configurator::Call& call, ImVec2 canvas_offset, float canvas_zooming, HotKeyArrayType& hotkeys) {

    int retval_id = GUI_INVALID_ID;

    try {

        if (ImGui::GetCurrentContext() == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        ImGui::PushID(call.uid);

        ImGuiStyle& style = ImGui::GetStyle();
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        assert(draw_list != nullptr);

        const ImU32 COLOR_CALL_CURVE = IM_COL32(225, 225, 0, 255);
        const ImU32 COLOR_CALL_BACKGROUND = IM_COL32(64, 61, 64, 255);
        const ImU32 COLOR_CALL_HIGHTLIGHT = IM_COL32(92, 116, 92, 255);
        const ImU32 COLOR_CALL_BORDER = IM_COL32(128, 128, 128, 255);

        const float CURVE_THICKNESS = 3.0f;

        if (call.IsConnected()) {

            ImVec2 p1 = call.GetCallSlot(CallSlot::CallSlotType::CALLER)->GUI_GetPosition();
            ImVec2 p2 = call.GetCallSlot(CallSlot::CallSlotType::CALLEE)->GUI_GetPosition();

            // Draw simple line if zooming is too small for nice bezier curves
            draw_list->ChannelsSetCurrent(0); // Background
            const float zooming_switch_curve = 0.4f;
            if (canvas_zooming < zooming_switch_curve) {
                draw_list->AddLine(p1, p2, COLOR_CALL_CURVE, CURVE_THICKNESS * canvas_zooming);
            } else {
                draw_list->AddBezierCurve(p1, p1 + ImVec2(50.0f, 0.0f), p2 + ImVec2(-50.0f, 0.0f), p2, COLOR_CALL_CURVE,
                    CURVE_THICKNESS * canvas_zooming);
            }

            if (this->label_visible) {
                draw_list->ChannelsSetCurrent(1); // Foreground

                ImVec2 call_center = ImVec2(p1.x + (p2.x - p1.x) / 2.0f, p1.y + (p2.y - p1.y) / 2.0f);
                auto call_name_width = this->utils.TextWidgetWidth(call.class_name);

                // Draw box
                ImVec2 rect_size = ImVec2(call_name_width + (2.0f * style.ItemSpacing.x),
                    ImGui::GetFontSize() + (2.0f * style.ItemSpacing.y));
                ImVec2 call_rect_min =
                    ImVec2(call_center.x - (rect_size.x / 2.0f), call_center.y - (rect_size.y / 2.0f));
                ImVec2 call_rect_max = ImVec2((call_rect_min.x + rect_size.x), (call_rect_min.y + rect_size.y));
                ImGui::SetCursorScreenPos(call_rect_min);
                std::string label = "call_" + call.class_name + std::to_string(call.uid);
                ImGui::InvisibleButton(label.c_str(), rect_size);
                // Context menu
                if (ImGui::BeginPopupContextItem()) {
                    if (ImGui::MenuItem(
                            "Delete", std::get<0>(hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM]).ToString().c_str())) {
                        std::get<1>(hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM]) = true;
                        retval_id = call.uid;
                    }
                    ImGui::EndPopup();
                }
                bool active = ImGui::IsItemActive();
                bool hovered = ImGui::IsItemHovered();
                bool mouse_clicked = ImGui::IsMouseClicked(0);
                if (mouse_clicked && !hovered) { //  && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows)) {
                    this->selected = false;
                }
                if (active) {
                    this->selected = true;
                }
                if (this->selected) {
                    retval_id = call.uid;
                }   
                ImU32 call_bg_color =
                    (hovered || this->selected) ? COLOR_CALL_HIGHTLIGHT : COLOR_CALL_BACKGROUND;
                draw_list->AddRectFilled(call_rect_min, call_rect_max, call_bg_color, 4.0f);
                draw_list->AddRect(call_rect_min, call_rect_max, COLOR_CALL_BORDER, 4.0f);

                // Draw text
                ImGui::SetCursorScreenPos(
                    call_center + ImVec2(-(call_name_width / 2.0f), -0.5f * ImGui::GetFontSize()));
                ImGui::Text(call.class_name.c_str());
            }
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