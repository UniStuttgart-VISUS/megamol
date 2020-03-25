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
using namespace megamol::gui;
using namespace megamol::gui::configurator;


megamol::gui::configurator::Call::Call(ImGuiID uid) : uid(uid), present() {

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
                // vislib::sys::Log::DefaultLog.WriteWarn(
                //    "Call slot is already disconnected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
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


void megamol::gui::configurator::Call::Presentation::Present(
    megamol::gui::configurator::Call& inout_call, const megamol::gui::CanvasType& in_canvas, megamol::gui::HotKeyArrayType& inout_hotkeys, megamol::gui::InteractType& interact_state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
    ImGuiStyle& style = ImGui::GetStyle();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    try {
        if (inout_call.IsConnected()) {

            const float CURVE_THICKNESS = 3.0f;
            ImVec2 p1 = inout_call.GetCallSlot(CallSlot::CallSlotType::CALLER)->GUI_GetPosition();
            ImVec2 p2 = inout_call.GetCallSlot(CallSlot::CallSlotType::CALLEE)->GUI_GetPosition();

            // Clip calls if lying ouside the canvas
            /// XXX Check is too expensive
            // ImVec2 canvas_rect_min = in_canvas.position;
            // ImVec2 canvas_rect_max = in_canvas.position + in_canvas.size;
            // if (...) {
            //    return GUI_INVALID_ID;
            //}

            // Draw simple line if zooming is too small for nice bezier curves
            ImGui::PushID(inout_call.uid);

            ImVec4 tmpcol = style.Colors[ImGuiCol_FrameBg]; // ImGuiCol_FrameBg ImGuiCol_Button
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_CALL_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);
            tmpcol = style.Colors[ImGuiCol_FrameBgActive]; // ImGuiCol_FrameBgActive ImGuiCol_ButtonActive
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);            
            const ImU32 COLOR_CALL_CURVE = ImGui::ColorConvertFloat4ToU32(tmpcol);
            //tmpcol = style.Colors[ImGuiCol_ButtonActive];
            //tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);            
            const ImU32 COLOR_CALL_HIGHTLIGHT = ImGui::ColorConvertFloat4ToU32(tmpcol);
            tmpcol = style.Colors[ImGuiCol_ScrollbarGrabActive]; // ImGuiCol_Border ImGuiCol_ScrollbarGrabActive
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);            
            const ImU32 COLOR_CALL_BORDER = ImGui::ColorConvertFloat4ToU32(tmpcol);

            // LEVEL OF DETAIL depending on zooming
            if (in_canvas.zooming < 0.25f) {
                draw_list->AddLine(p1, p2, COLOR_CALL_CURVE, CURVE_THICKNESS * in_canvas.zooming);
            } else {
                draw_list->AddBezierCurve(p1, p1 + ImVec2(50.0f, 0.0f), p2 + ImVec2(-50.0f, 0.0f), p2, COLOR_CALL_CURVE,
                    CURVE_THICKNESS * in_canvas.zooming);
            }

            if (this->label_visible) {

                ImVec2 call_center = ImVec2(p1.x + (p2.x - p1.x) / 2.0f, p1.y + (p2.y - p1.y) / 2.0f);
                auto call_name_width = this->utils.TextWidgetWidth(inout_call.class_name);

                // Draw box
                ImVec2 rect_size = ImVec2(call_name_width + (2.0f * style.ItemSpacing.x),
                    ImGui::GetFontSize() + (2.0f * style.ItemSpacing.y));
                ImVec2 call_rect_min =
                    ImVec2(call_center.x - (rect_size.x / 2.0f), call_center.y - (rect_size.y / 2.0f));
                ImVec2 call_rect_max = ImVec2((call_rect_min.x + rect_size.x), (call_rect_min.y + rect_size.y));
                ImGui::SetCursorScreenPos(call_rect_min);
                std::string label = "call_" + inout_call.class_name + std::to_string(inout_call.uid);
                ImGui::InvisibleButton(label.c_str(), rect_size);
                // Context menu
                if (ImGui::BeginPopupContextItem()) {
                    if (ImGui::MenuItem(
                            "Delete", std::get<0>(inout_hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM]).ToString().c_str())) {
                        std::get<1>(inout_hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM]) = true;
                        this->selected = true;
                    }
                    ImGui::EndPopup();
                }
                bool active = ImGui::IsItemActive();
                bool hovered = ImGui::IsItemHovered();
                bool mouse_clicked = ImGui::IsWindowHovered() && ImGui::GetIO().MouseClicked[0];
                if ((mouse_clicked && !hovered) || (interact_state.call_selected_uid != inout_call.uid)) {
                    this->selected = false;
                    if (interact_state.call_selected_uid == inout_call.uid) {
                        interact_state.call_selected_uid = GUI_INVALID_ID;
                    }
                }
                if (active) {
                    this->selected = true;
                    interact_state.call_selected_uid = inout_call.uid;
                    interact_state.module_selected_uid = GUI_INVALID_ID;
                }

                ImU32 call_bg_color = (hovered || this->selected) ? COLOR_CALL_HIGHTLIGHT : COLOR_CALL_BACKGROUND;
                draw_list->AddRectFilled(call_rect_min, call_rect_max, call_bg_color, 4.0f);
                draw_list->AddRect(call_rect_min, call_rect_max, COLOR_CALL_BORDER, 4.0f);

                // Draw text
                ImGui::SetCursorScreenPos(
                    call_center + ImVec2(-(call_name_width / 2.0f), -0.5f * ImGui::GetFontSize()));
                ImGui::Text(inout_call.class_name.c_str());
            }

            ImGui::PopID();
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