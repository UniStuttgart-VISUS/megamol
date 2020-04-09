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


megamol::gui::configurator::Call::Call(ImGuiID uid)
    : uid(uid), class_name(), description(), plugin_name(), functions(), connected_call_slots(), present() {

    this->connected_call_slots.emplace(CallSlotType::CALLER, nullptr);
    this->connected_call_slots.emplace(CallSlotType::CALLEE, nullptr);
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


const megamol::gui::configurator::CallSlotPtrType& megamol::gui::configurator::Call::GetCallSlot(
    megamol::gui::configurator::CallSlotType type) {

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
    megamol::gui::configurator::Call& inout_call, megamol::gui::GraphItemsStateType& state) {

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
            // Clip calls if lying ouside the canvas
            /// XXX Check is too expensive
            // ImVec2 canvas_rect_min = state.canvas.position;
            // ImVec2 canvas_rect_max = state.canvas.position + state.canvas.size;
            // if (...) {
            //    return GUI_INVALID_ID;
            //}
                        
            const float CURVE_THICKNESS = 3.0f;
                        
            auto callerslot_ptr = inout_call.GetCallSlot(CallSlotType::CALLER);
            auto calleeslot_ptr = inout_call.GetCallSlot(CallSlotType::CALLEE);
            if ((callerslot_ptr == nullptr) || (calleeslot_ptr == nullptr)) {
                return;
            }
            ImVec2 caller_position = callerslot_ptr->GUI_GetPosition();
            if (callerslot_ptr->GUI_IsGroupInterface()) {
                /// XXX caller_position = callerslot_ptr->GUI_GetGroupInterfacePosition();
            }
            ImVec2 callee_position = calleeslot_ptr->GUI_GetPosition();
            if (calleeslot_ptr->GUI_IsGroupInterface()) {
                /// XXX callee_position = calleeslot_ptr->GUI_GetGroupInterfacePosition();
            }
            
            ImGui::PushID(inout_call.uid);

            // Colors
            ImVec4 tmpcol = style.Colors[ImGuiCol_FrameBg]; // ImGuiCol_FrameBg ImGuiCol_Button
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_CALL_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);

            tmpcol = style.Colors[ImGuiCol_FrameBgActive]; // ImGuiCol_FrameBgActive ImGuiCol_ButtonActive
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_CALL_CURVE = ImGui::ColorConvertFloat4ToU32(tmpcol);

            const ImU32 COLOR_CALL_HIGHTLIGHT = ImGui::ColorConvertFloat4ToU32(tmpcol);

            tmpcol = style.Colors[ImGuiCol_ScrollbarGrabActive]; // ImGuiCol_Border ImGuiCol_ScrollbarGrabActive
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_CALL_GROUP_BORDER = ImGui::ColorConvertFloat4ToU32(tmpcol);

            // Draw Curve
            ImVec2 p1 = caller_position;
            ImVec2 p2 = callee_position;
            /// Draw simple line if zooming is too small for nice bezier curves         
            if (state.canvas.zooming < 0.25f) {
                draw_list->AddLine(p1, p2, COLOR_CALL_CURVE, CURVE_THICKNESS * state.canvas.zooming);
            } else {
                draw_list->AddBezierCurve(p1, p1 + ImVec2(50.0f, 0.0f), p2 + ImVec2(-50.0f, 0.0f), p2,
                    COLOR_CALL_CURVE, CURVE_THICKNESS * state.canvas.zooming);
            }

            if (this->label_visible) {
                ImVec2 call_center = ImVec2(p1.x + (p2.x - p1.x) / 2.0f, p1.y + (p2.y - p1.y) / 2.0f);
                auto call_name_width = this->utils.TextWidgetWidth(inout_call.class_name);

                // Button
                ImVec2 rect_size = ImVec2(call_name_width + (2.0f * style.ItemSpacing.x),
                    ImGui::GetFontSize() + (2.0f * style.ItemSpacing.y));
                ImVec2 call_rect_min =
                    ImVec2(call_center.x - (rect_size.x / 2.0f), call_center.y - (rect_size.y / 2.0f));
                ImVec2 call_rect_max = ImVec2((call_rect_min.x + rect_size.x), (call_rect_min.y + rect_size.y));

                ImGui::SetCursorScreenPos(call_rect_min);
                std::string label = "call_" + inout_call.class_name + std::to_string(inout_call.uid);
                ImGui::SetItemAllowOverlap();
                ImGui::InvisibleButton(label.c_str(), rect_size);
                ImGui::SetItemAllowOverlap();

                bool button_active = ImGui::IsItemActive();
                bool mouse_clicked_anywhere = ImGui::IsWindowHovered() && ImGui::GetIO().MouseClicked[0];                
                bool button_hovered = ImGui::IsItemHovered();

                // Context Menu
                if (ImGui::BeginPopupContextItem()) {
                    button_active = true; // Force selection (next frame)

                    ImGui::TextUnformatted("Call");
                    ImGui::Separator();
                    if (ImGui::MenuItem(
                            "Delete", std::get<0>(state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM])
                                          .ToString()
                                          .c_str())) {
                        std::get<1>(state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM]) = true;
                    }
                    ImGui::EndPopup();
                }

                // Draw Background
                ImU32 call_bg_color = (this->selected) ? (COLOR_CALL_HIGHTLIGHT) : (COLOR_CALL_BACKGROUND);
                draw_list->AddRectFilled(call_rect_min, call_rect_max, call_bg_color, GUI_RECT_CORNER_RADIUS);
                draw_list->AddRect(call_rect_min, call_rect_max, COLOR_CALL_GROUP_BORDER, GUI_RECT_CORNER_RADIUS);

                // Draw Text
                ImVec2 text_pos_left_upper =
                    (call_center + ImVec2(-(call_name_width / 2.0f), -0.5f * ImGui::GetFontSize()));
                draw_list->AddText(text_pos_left_upper, ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Text]),
                    inout_call.class_name.c_str());
                    
                // Selection
                if (button_active) {
                    state.interact.call_selected_uid = inout_call.uid;
                    this->selected = true;
                    state.interact.callslot_selected_uid = GUI_INVALID_ID;
                    state.interact.modules_selected_uids.clear();
                    state.interact.group_selected_uid = GUI_INVALID_ID;                    
                }
                // Deselection
                if ((mouse_clicked_anywhere && !button_hovered) || (state.interact.call_selected_uid != inout_call.uid)) {
                    this->selected = false;
                    if (state.interact.call_selected_uid == inout_call.uid) {
                        state.interact.call_selected_uid = GUI_INVALID_ID;
                    }
                }                    
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
