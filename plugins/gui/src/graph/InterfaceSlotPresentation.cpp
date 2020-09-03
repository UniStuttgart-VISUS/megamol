/*
 * InterfaceSlot.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "InterfaceSlotPresentation.h"

#include "Call.h"
#include "CallSlot.h"
#include "InterfaceSlot.h"
#include "Module.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::InterfaceSlotPresentation::InterfaceSlotPresentation(void)
    : group()
    , label_visible(false)
    , tooltip()
    , selected(false)
    , label()
    , last_compat_callslot_uid(GUI_INVALID_ID)
    , last_compat_interface_uid(GUI_INVALID_ID)
    , compatible(false)
    , position(ImVec2(FLT_MAX, FLT_MAX)) {

    this->group.uid = GUI_INVALID_ID;
    this->group.collapsed_view = false;
    this->group.collapsed_view = false;
}


megamol::gui::InterfaceSlotPresentation::~InterfaceSlotPresentation(void) {}


void megamol::gui::InterfaceSlotPresentation::Present(
    PresentPhase phase, megamol::gui::InterfaceSlot& inout_interfaceslot, megamol::gui::GraphItemsState_t& state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    ImGuiStyle& style = ImGui::GetStyle();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    try {
        ImVec2 actual_position = this->GetPosition(inout_interfaceslot);
        float radius = GUI_SLOT_RADIUS * state.canvas.zooming;
        this->label.clear();
        for (auto& callslot_ptr : inout_interfaceslot.GetCallSlots()) {
            this->label += (callslot_ptr->name + " ");
        }
        auto callslot_count = inout_interfaceslot.GetCallSlots().size();
        if (callslot_count > 1) {
            this->label += ("[" + std::to_string(callslot_count) + "]");
        }
        std::string button_label = "interfaceslot_" + std::to_string(inout_interfaceslot.uid);

        ImGui::PushID(inout_interfaceslot.uid);

        if (phase == megamol::gui::PresentPhase::INTERACTION) {

            // Button
            ImGui::SetCursorScreenPos(actual_position - ImVec2(radius, radius));
            ImGui::SetItemAllowOverlap();
            ImGui::InvisibleButton(button_label.c_str(), ImVec2(radius * 2.0f, radius * 2.0f));
            ImGui::SetItemAllowOverlap();
            if (ImGui::IsItemActivated()) {
                state.interact.button_active_uid = inout_interfaceslot.uid;
            }
            if (ImGui::IsItemHovered()) {
                state.interact.button_hovered_uid = inout_interfaceslot.uid;
            }

            // Context Menu
            if (ImGui::BeginPopupContextItem("invisible_button_context")) {
                state.interact.button_active_uid = inout_interfaceslot.uid;

                ImGui::TextDisabled("Interface Slot");
                ImGui::Separator();

                if (ImGui::MenuItem("Delete",
                        std::get<0>(state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM]).ToString().c_str())) {
                    state.interact.process_deletion = true;
                }

                ImGui::EndPopup();
            }

            // Drag & Drop
            if (ImGui::BeginDragDropTarget()) {
                if (ImGui::AcceptDragDropPayload(GUI_DND_CALLSLOT_UID_TYPE) != nullptr) {
                    state.interact.slot_dropped_uid = inout_interfaceslot.uid;
                }
                ImGui::EndDragDropTarget();
            }
            if (this->selected) {
                auto dnd_flags =
                    ImGuiDragDropFlags_AcceptNoDrawDefaultRect; // | ImGuiDragDropFlags_SourceNoPreviewTooltip;
                if (ImGui::BeginDragDropSource(dnd_flags)) {
                    ImGui::SetDragDropPayload(GUI_DND_CALLSLOT_UID_TYPE, &inout_interfaceslot.uid, sizeof(ImGuiID));
                    std::string drag_str;
                    for (auto& callslot_ptr : inout_interfaceslot.GetCallSlots()) {
                        drag_str += (callslot_ptr->name + "\n");
                    }
                    ImGui::TextUnformatted(drag_str.c_str());
                    ImGui::EndDragDropSource();
                }
            }

            // Hover Tooltip
            if ((state.interact.interfaceslot_hovered_uid == inout_interfaceslot.uid) && !this->label_visible) {
                this->tooltip.ToolTip(this->label, ImGui::GetID(button_label.c_str()), 0.5f, 5.0f);
            } else {
                this->tooltip.Reset();
            }
        } else if (phase == megamol::gui::PresentPhase::RENDERING) {

            bool active = (state.interact.button_active_uid == inout_interfaceslot.uid);
            bool hovered = (state.interact.button_hovered_uid == inout_interfaceslot.uid);
            bool mouse_clicked_anywhere = ImGui::IsWindowHovered() && ImGui::GetIO().MouseClicked[0];

            // Compatibility
            if (state.interact.callslot_compat_ptr != nullptr) {
                if (state.interact.callslot_compat_ptr->uid != this->last_compat_callslot_uid) {
                    this->compatible = inout_interfaceslot.IsConnectionValid((*state.interact.callslot_compat_ptr));
                    this->last_compat_callslot_uid = state.interact.callslot_compat_ptr->uid;
                }
            } else if (state.interact.interfaceslot_compat_ptr != nullptr) {
                if (state.interact.interfaceslot_compat_ptr->uid != this->last_compat_interface_uid) {
                    this->compatible =
                        inout_interfaceslot.IsConnectionValid((*state.interact.interfaceslot_compat_ptr));
                    this->last_compat_interface_uid = state.interact.interfaceslot_compat_ptr->uid;
                }
            } else { /// (state.interact.callslot_compat_ptr == nullptr) && (state.interact.interfaceslot_compat_ptr ==
                     /// nullptr)
                this->compatible = false;
                this->last_compat_callslot_uid = GUI_INVALID_ID;
                this->last_compat_interface_uid = GUI_INVALID_ID;
            }

            // Selection
            if (!this->selected && active) {
                state.interact.interfaceslot_selected_uid = inout_interfaceslot.uid;
                this->selected = true;
                state.interact.callslot_selected_uid = GUI_INVALID_ID;
                state.interact.call_selected_uid = GUI_INVALID_ID;
                state.interact.modules_selected_uids.clear();
                state.interact.group_selected_uid = GUI_INVALID_ID;
            }
            // Deselection
            else if (this->selected && ((mouse_clicked_anywhere && !hovered) ||
                                           (state.interact.interfaceslot_selected_uid != inout_interfaceslot.uid))) {
                this->selected = false;
                if (state.interact.interfaceslot_selected_uid == inout_interfaceslot.uid) {
                    state.interact.interfaceslot_selected_uid = GUI_INVALID_ID;
                }
            }

            // Hovering
            if (hovered) {
                state.interact.interfaceslot_hovered_uid = inout_interfaceslot.uid;
            }
            if (!hovered && (state.interact.interfaceslot_hovered_uid == inout_interfaceslot.uid)) {
                state.interact.interfaceslot_hovered_uid = GUI_INVALID_ID;
            }

            // Colors
            ImVec4 tmpcol = style.Colors[ImGuiCol_FrameBg];
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_INTERFACE_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);

            tmpcol = style.Colors[ImGuiCol_ScrollbarGrabActive];
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_INTERFACE_BORDER = ImGui::ColorConvertFloat4ToU32(tmpcol);

            tmpcol = style.Colors[ImGuiCol_FrameBgHovered];
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_INTERFACE_CURVE = ImGui::ColorConvertFloat4ToU32(tmpcol);

            // Color modification
            ImU32 slot_highlight_color = ImGui::ColorConvertFloat4ToU32(GUI_COLOR_SLOT_CALLER);
            ;
            if (inout_interfaceslot.GetCallSlotType() == CallSlotType::CALLEE) {
                slot_highlight_color = ImGui::ColorConvertFloat4ToU32(GUI_COLOR_SLOT_CALLEE);
            }
            ImU32 slot_color = COLOR_INTERFACE_BACKGROUND;
            if (this->compatible) {
                slot_color = ImGui::ColorConvertFloat4ToU32(GUI_COLOR_SLOT_COMPATIBLE);
            }
            if (hovered || this->selected) {
                slot_color = slot_highlight_color;
            }

            // Draw Slot
            const float segment_numer = 20.0f;
            draw_list->AddCircleFilled(actual_position, radius, slot_color, segment_numer);
            draw_list->AddCircle(actual_position, radius, COLOR_INTERFACE_BORDER, segment_numer);

            // Draw Curves
            if (!this->group.collapsed_view) {
                for (auto& callslot_ptr : inout_interfaceslot.GetCallSlots()) {
                    draw_list->AddLine(actual_position, callslot_ptr->present.GetPosition(), COLOR_INTERFACE_CURVE,
                        GUI_LINE_THICKNESS * state.canvas.zooming);
                }
            }

            // Text
            if (this->group.collapsed_view) {
                auto type = inout_interfaceslot.GetCallSlotType();
                ImVec2 text_pos_left_upper = ImVec2(0.0f, 0.0f);
                text_pos_left_upper.y = actual_position.y - ImGui::GetTextLineHeightWithSpacing() / 2.0f;
                text_pos_left_upper.x =
                    actual_position.x - ImGui::CalcTextSize(this->label.c_str()).x - (1.5f * radius);
                if (type == CallSlotType::CALLEE) {
                    text_pos_left_upper.x = actual_position.x + (1.5f * radius);
                }
                ImU32 slot_text_color = ImGui::ColorConvertFloat4ToU32(GUI_COLOR_SLOT_CALLER);
                if (type == CallSlotType::CALLEE) {
                    slot_text_color = ImGui::ColorConvertFloat4ToU32(GUI_COLOR_SLOT_CALLEE);
                }
                draw_list->AddText(text_pos_left_upper, slot_text_color, this->label.c_str());
            }
        }

        ImGui::PopID();

    } catch (std::exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


ImVec2 megamol::gui::InterfaceSlotPresentation::GetPosition(InterfaceSlot& inout_interfaceslot) {

    ImVec2 ret_position = this->position;
    if ((!this->group.collapsed_view) && (inout_interfaceslot.GetCallSlots().size() > 0)) {
        auto only_callslot_ptr = inout_interfaceslot.GetCallSlots().front();
        ret_position.x = this->position.x;
        ret_position.y = only_callslot_ptr->present.GetPosition().y;
    }
    return ret_position;
}
