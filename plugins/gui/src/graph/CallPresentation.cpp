/*
 * Call.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CallPresentation.h"

#include "Call.h"
#include "CallSlot.h"
#include "InterfaceSlot.h"
#include "Module.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::CallPresentation::CallPresentation(void) : label_visible(true), selected(false), tooltip() {}


megamol::gui::CallPresentation::~CallPresentation(void) {}


void megamol::gui::CallPresentation::Present(
    megamol::gui::PresentPhase phase, megamol::gui::Call& inout_call, megamol::gui::GraphItemsState_t& state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    ImGuiStyle& style = ImGui::GetStyle();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    try {
        if (inout_call.IsConnected()) {
            auto callerslot_ptr = inout_call.GetCallSlot(CallSlotType::CALLER);
            auto calleeslot_ptr = inout_call.GetCallSlot(CallSlotType::CALLEE);
            if ((callerslot_ptr == nullptr) || (calleeslot_ptr == nullptr)) {
                return;
            }
            bool visible =
                ((callerslot_ptr->present.visible || (callerslot_ptr->present.group.interfaceslot_ptr != nullptr)) &&
                    (calleeslot_ptr->present.visible || (calleeslot_ptr->present.group.interfaceslot_ptr != nullptr)));
            if (visible) {

                ImVec2 caller_position = callerslot_ptr->present.GetPosition();
                ImVec2 callee_position = calleeslot_ptr->present.GetPosition();
                bool connect_interface_slot = true;
                if (callerslot_ptr->IsParentModuleConnected() && calleeslot_ptr->IsParentModuleConnected()) {
                    if (callerslot_ptr->GetParentModule()->present.group.uid ==
                        calleeslot_ptr->GetParentModule()->present.group.uid) {
                        connect_interface_slot = false;
                    }
                }
                if (connect_interface_slot) {
                    if (callerslot_ptr->present.group.interfaceslot_ptr != nullptr) {
                        caller_position = callerslot_ptr->present.group.interfaceslot_ptr->GetGUIPosition();
                    }
                    if (calleeslot_ptr->present.group.interfaceslot_ptr != nullptr) {
                        callee_position = calleeslot_ptr->present.group.interfaceslot_ptr->GetGUIPosition();
                    }
                }
                ImVec2 p1 = caller_position;
                ImVec2 p2 = callee_position;

                ImGui::PushID(inout_call.uid);

                // Colors
                ImVec4 tmpcol = style.Colors[ImGuiCol_FrameBg];
                tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
                const ImU32 COLOR_CALL_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);

                tmpcol = style.Colors[ImGuiCol_FrameBgActive];
                tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
                const ImU32 COLOR_CALL_HIGHTLIGHT = ImGui::ColorConvertFloat4ToU32(tmpcol);

                tmpcol = style.Colors[ImGuiCol_FrameBgHovered];
                tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
                const ImU32 COLOR_CALL_CURVE = ImGui::ColorConvertFloat4ToU32(tmpcol);

                tmpcol = style.Colors[ImGuiCol_ButtonActive];
                tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
                const ImU32 COLOR_CALL_CURVE_HIGHLIGHT = ImGui::ColorConvertFloat4ToU32(tmpcol);

                tmpcol = style.Colors[ImGuiCol_ScrollbarGrabActive];
                tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
                const ImU32 COLOR_CALL_GROUP_BORDER = ImGui::ColorConvertFloat4ToU32(tmpcol);

                if (phase == megamol::gui::PresentPhase::RENDERING) {
                    bool hovered = (state.interact.button_hovered_uid == inout_call.uid);

                    // Draw Curve
                    ImU32 color_curve = COLOR_CALL_CURVE;
                    if (hovered || this->selected) {
                        color_curve = COLOR_CALL_CURVE_HIGHLIGHT;
                    }
                    /// Draw simple line if zooming is too small for nice bezier curves.
                    if (state.canvas.zooming < 0.25f) {
                        draw_list->AddLine(p1, p2, color_curve, GUI_LINE_THICKNESS * state.canvas.zooming);
                    } else {
                        draw_list->AddBezierCurve(p1, p1 + ImVec2(50.0f, 0.0f), p2 + ImVec2(-50.0f, 0.0f), p2,
                            color_curve, GUI_LINE_THICKNESS * state.canvas.zooming);
                    }
                }

                if (this->label_visible) {
                    ImVec2 call_center = ImVec2(p1.x + (p2.x - p1.x) / 2.0f, p1.y + (p2.y - p1.y) / 2.0f);
                    auto call_name_width = ImGui::CalcTextSize(inout_call.class_name.c_str()).x;
                    ImVec2 rect_size = ImVec2(call_name_width + (2.0f * style.ItemSpacing.x),
                        ImGui::GetFontSize() + (2.0f * style.ItemSpacing.y));
                    ImVec2 call_rect_min =
                        ImVec2(call_center.x - (rect_size.x / 2.0f), call_center.y - (rect_size.y / 2.0f));
                    ImVec2 call_rect_max = ImVec2((call_rect_min.x + rect_size.x), (call_rect_min.y + rect_size.y));

                    std::string button_label = "call_" + std::to_string(inout_call.uid);

                    if (phase == megamol::gui::PresentPhase::INTERACTION) {

                        // Button
                        ImGui::SetCursorScreenPos(call_rect_min);
                        ImGui::SetItemAllowOverlap();
                        ImGui::InvisibleButton(button_label.c_str(), rect_size);
                        ImGui::SetItemAllowOverlap();
                        if (ImGui::IsItemActivated()) {
                            state.interact.button_active_uid = inout_call.uid;
                        }
                        if (ImGui::IsItemHovered()) {
                            state.interact.button_hovered_uid = inout_call.uid;
                        }

                        // Context Menu
                        if (ImGui::BeginPopupContextItem()) {
                            state.interact.button_active_uid = inout_call.uid;

                            ImGui::TextDisabled("Call");
                            ImGui::Separator();

                            if (ImGui::MenuItem(
                                    "Delete", std::get<0>(state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM])
                                                  .ToString()
                                                  .c_str())) {
                                state.interact.process_deletion = true;
                            }
                            ImGui::Separator();

                            ImGui::TextDisabled("Description");
                            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 13.0f);
                            ImGui::TextUnformatted(inout_call.description.c_str());
                            ImGui::PopTextWrapPos();

                            ImGui::EndPopup();
                        }

                        // Hover Tooltip
                        if (state.interact.call_hovered_uid == inout_call.uid) {
                            std::string tooltip = callerslot_ptr->name + " > " + calleeslot_ptr->name;
                            this->tooltip.ToolTip(tooltip, ImGui::GetID(button_label.c_str()), 0.5f, 5.0f);
                        } else {
                            this->tooltip.Reset();
                        }

                    } else if (phase == megamol::gui::PresentPhase::RENDERING) {

                        bool active = (state.interact.button_active_uid == inout_call.uid);
                        bool hovered = (state.interact.button_hovered_uid == inout_call.uid);
                        bool mouse_clicked_anywhere = ImGui::IsWindowHovered() && ImGui::GetIO().MouseClicked[0];

                        // Selection
                        if (!this->selected && active) {
                            state.interact.call_selected_uid = inout_call.uid;
                            this->selected = true;
                            state.interact.callslot_selected_uid = GUI_INVALID_ID;
                            state.interact.modules_selected_uids.clear();
                            state.interact.group_selected_uid = GUI_INVALID_ID;
                            state.interact.interfaceslot_selected_uid = GUI_INVALID_ID;
                        }
                        // Deselection
                        else if (this->selected && ((mouse_clicked_anywhere && !hovered) ||
                                                       (state.interact.call_selected_uid != inout_call.uid))) {
                            this->selected = false;
                            if (state.interact.call_selected_uid == inout_call.uid) {
                                state.interact.call_selected_uid = GUI_INVALID_ID;
                            }
                        }

                        // Hovering
                        if (hovered) {
                            state.interact.call_hovered_uid = inout_call.uid;
                        }
                        if (!hovered && (state.interact.call_hovered_uid == inout_call.uid)) {
                            state.interact.call_hovered_uid = GUI_INVALID_ID;
                        }

                        // Draw Background
                        ImU32 call_bg_color =
                            (this->selected || hovered) ? (COLOR_CALL_HIGHTLIGHT) : (COLOR_CALL_BACKGROUND);
                        draw_list->AddRectFilled(call_rect_min, call_rect_max, call_bg_color, GUI_RECT_CORNER_RADIUS);
                        draw_list->AddRect(
                            call_rect_min, call_rect_max, COLOR_CALL_GROUP_BORDER, GUI_RECT_CORNER_RADIUS);

                        // Draw Text
                        ImVec2 text_pos_left_upper =
                            (call_center + ImVec2(-(call_name_width / 2.0f), -0.5f * ImGui::GetFontSize()));
                        draw_list->AddText(text_pos_left_upper,
                            ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Text]), inout_call.class_name.c_str());
                    }
                }

                ImGui::PopID();
            }
        }
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
