/*
 * Call.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Call.h"

#include "CallSlot.h"
#include "InterfaceSlot.h"
#include "Module.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::Call::Call(ImGuiID uid, const std::string& class_name, const std::string& description,
    const std::string& plugin_name, const std::vector<std::string>& functions)
        : uid(uid)
        , class_name(class_name)
        , description(description)
        , plugin_name(plugin_name)
        , functions(functions)
        , connected_callslots()
        , gui_selected(false)
        , gui_label_visible(true)
        , gui_slots_visible(false)
        , gui_tooltip() {

    this->connected_callslots.emplace(CallSlotType::CALLER, nullptr);
    this->connected_callslots.emplace(CallSlotType::CALLEE, nullptr);
}


megamol::gui::Call::~Call() {

    // Disconnect call slots
    this->DisconnectCallSlots();
}


bool megamol::gui::Call::IsConnected(void) {

    unsigned int connected = 0;
    for (auto& callslot_map : this->connected_callslots) {
        if (callslot_map.second != nullptr) {
            connected++;
        }
    }
    if (connected != 2) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteWarn("[GUI] Call has only one connected call slot. [%s,
        /// %s, line %d]\n",
        /// __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return (connected == 2);
}


bool megamol::gui::Call::ConnectCallSlots(
    megamol::gui::CallSlotPtr_t callslot_1, megamol::gui::CallSlotPtr_t callslot_2) {

    if ((callslot_1 == nullptr) || (callslot_2 == nullptr)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Pointer to given call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if ((this->connected_callslots[callslot_1->Type()] != nullptr) ||
        (this->connected_callslots[callslot_2->Type()] != nullptr)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Call is already connected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (callslot_1->IsConnectionValid((*callslot_2))) {
        this->connected_callslots[callslot_1->Type()] = callslot_1;
        this->connected_callslots[callslot_2->Type()] = callslot_2;
        return true;
    }
    return false;
}


bool megamol::gui::Call::DisconnectCallSlots(ImGuiID calling_callslot_uid) {

    try {
        for (auto& callslot_map : this->connected_callslots) {
            if (callslot_map.second != nullptr) {
                if (callslot_map.second->UID() != calling_callslot_uid) {
                    callslot_map.second->DisconnectCall(this->uid);
                }
                callslot_map.second.reset();
            }
        }
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


const megamol::gui::CallSlotPtr_t& megamol::gui::Call::CallSlotPtr(megamol::gui::CallSlotType type) {

    if (this->connected_callslots[type] == nullptr) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteWarn("[GUI] Returned pointer to call slot is nullptr. [%s,
        /// %s, line %d]\n",
        /// __FILE__, __FUNCTION__, __LINE__);
    }
    return this->connected_callslots[type];
}


void megamol::gui::Call::Draw(megamol::gui::PresentPhase phase, megamol::gui::GraphItemsState_t& state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    ImGuiStyle& style = ImGui::GetStyle();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    try {
        if (this->IsConnected()) {
            auto callerslot_ptr = this->CallSlotPtr(CallSlotType::CALLER);
            auto calleeslot_ptr = this->CallSlotPtr(CallSlotType::CALLEE);
            if ((callerslot_ptr == nullptr) || (calleeslot_ptr == nullptr)) {
                return;
            }
            bool visible = ((callerslot_ptr->IsVisible() || (callerslot_ptr->InterfaceSlotPtr() != nullptr)) &&
                            (calleeslot_ptr->IsVisible() || (calleeslot_ptr->InterfaceSlotPtr() != nullptr)));
            if (visible) {

                ImVec2 caller_position = callerslot_ptr->Position();
                ImVec2 callee_position = calleeslot_ptr->Position();
                bool connect_interface_slot = true;
                if (callerslot_ptr->IsParentModuleConnected() && calleeslot_ptr->IsParentModuleConnected()) {
                    if (callerslot_ptr->GetParentModule()->GroupUID() ==
                        calleeslot_ptr->GetParentModule()->GroupUID()) {
                        connect_interface_slot = false;
                    }
                }
                if (connect_interface_slot) {
                    if (callerslot_ptr->InterfaceSlotPtr() != nullptr) {
                        caller_position = callerslot_ptr->InterfaceSlotPtr()->Position();
                    }
                    if (calleeslot_ptr->InterfaceSlotPtr() != nullptr) {
                        callee_position = calleeslot_ptr->InterfaceSlotPtr()->Position();
                    }
                }
                ImVec2 p1 = caller_position;
                ImVec2 p2 = callee_position;

                ImGui::PushID(this->uid);

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
                    bool hovered = (state.interact.button_hovered_uid == this->uid);

                    // Draw Curve
                    ImU32 color_curve = COLOR_CALL_CURVE;
                    if (hovered || this->gui_selected) {
                        color_curve = COLOR_CALL_CURVE_HIGHLIGHT;
                    }
                    /// Draw simple line if zooming is too small for nice bezier curves.
                    if (state.canvas.zooming < 0.25f) {
                        draw_list->AddLine(p1, p2, color_curve, GUI_LINE_THICKNESS * state.canvas.zooming);
                    } else {
                        draw_list->AddBezierCurve(p1, p1 + ImVec2((50.0f * megamol::gui::gui_scaling.Get()), 0.0f),
                            p2 + ImVec2((-50.0f * megamol::gui::gui_scaling.Get()), 0.0f), p2, color_curve,
                            GUI_LINE_THICKNESS * state.canvas.zooming);
                    }
                }

                if (this->gui_label_visible || this->gui_slots_visible) {
                    std::string slots_label = this->SlotsLabel();
                    auto slots_label_width = ImGui::CalcTextSize(slots_label.c_str()).x;
                    auto class_name_width = ImGui::CalcTextSize(this->class_name.c_str()).x;
                    ImVec2 call_center = ImVec2(p1.x + (p2.x - p1.x) / 2.0f, p1.y + (p2.y - p1.y) / 2.0f);
                    auto call_name_width = 0.0f;
                    if (this->gui_label_visible) {
                        call_name_width = std::max(call_name_width, class_name_width);
                    }
                    if (this->gui_slots_visible) {
                        call_name_width = std::max(call_name_width, slots_label_width);
                    }
                    ImVec2 rect_size = ImVec2(call_name_width + (2.0f * style.ItemSpacing.x),
                        ImGui::GetFontSize() + (2.0f * style.ItemSpacing.y));
                    if (this->gui_label_visible && this->gui_slots_visible) {
                        rect_size.y += (ImGui::GetFontSize() + style.ItemSpacing.y);
                    }
                    ImVec2 call_rect_min =
                        ImVec2(call_center.x - (rect_size.x / 2.0f), call_center.y - (rect_size.y / 2.0f));
                    ImVec2 call_rect_max = ImVec2((call_rect_min.x + rect_size.x), (call_rect_min.y + rect_size.y));

                    std::string button_label = "call_" + std::to_string(this->uid);

                    if (phase == megamol::gui::PresentPhase::INTERACTION) {

                        // Button
                        ImGui::SetCursorScreenPos(call_rect_min);
                        ImGui::SetItemAllowOverlap();
                        ImGui::InvisibleButton(button_label.c_str(), rect_size);
                        ImGui::SetItemAllowOverlap();
                        if (ImGui::IsItemActivated()) {
                            state.interact.button_active_uid = this->uid;
                        }
                        if (ImGui::IsItemHovered()) {
                            state.interact.button_hovered_uid = this->uid;
                        }

                        // Context Menu
                        if (ImGui::BeginPopupContextItem()) {
                            state.interact.button_active_uid = this->uid;

                            ImGui::TextDisabled("Call");
                            ImGui::Separator();

                            if (ImGui::MenuItem("Delete", state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM]
                                                              .keycode.ToString()
                                                              .c_str())) {
                                state.interact.process_deletion = true;
                            }
                            ImGui::Separator();

                            ImGui::TextDisabled("Description");
                            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 13.0f);
                            ImGui::TextUnformatted(this->description.c_str());
                            ImGui::PopTextWrapPos();

                            ImGui::EndPopup();
                        }

                        // Hover Tooltip
                        if (!this->gui_slots_visible) {
                            if (state.interact.call_hovered_uid == this->uid) {
                                this->gui_tooltip.ToolTip(slots_label, ImGui::GetID(button_label.c_str()), 0.5f, 5.0f);
                            } else {
                                this->gui_tooltip.Reset();
                            }
                        }

                    } else if (phase == megamol::gui::PresentPhase::RENDERING) {

                        bool active = (state.interact.button_active_uid == this->uid);
                        bool hovered = (state.interact.button_hovered_uid == this->uid);
                        bool mouse_clicked_anywhere = ImGui::IsWindowHovered() && ImGui::GetIO().MouseClicked[0];

                        // Selection
                        if (!this->gui_selected && active) {
                            state.interact.call_selected_uid = this->uid;
                            this->gui_selected = true;
                            state.interact.callslot_selected_uid = GUI_INVALID_ID;
                            state.interact.modules_selected_uids.clear();
                            state.interact.group_selected_uid = GUI_INVALID_ID;
                            state.interact.interfaceslot_selected_uid = GUI_INVALID_ID;
                        }
                        // Deselection
                        else if (this->gui_selected && ((mouse_clicked_anywhere && !hovered) ||
                                                           (state.interact.call_selected_uid != this->uid))) {
                            this->gui_selected = false;
                            if (state.interact.call_selected_uid == this->uid) {
                                state.interact.call_selected_uid = GUI_INVALID_ID;
                            }
                        }

                        // Hovering
                        if (hovered) {
                            state.interact.call_hovered_uid = this->uid;
                        }
                        if (!hovered && (state.interact.call_hovered_uid == this->uid)) {
                            state.interact.call_hovered_uid = GUI_INVALID_ID;
                        }

                        // Draw Background
                        ImU32 call_bg_color =
                            (this->gui_selected || hovered) ? (COLOR_CALL_HIGHTLIGHT) : (COLOR_CALL_BACKGROUND);
                        draw_list->AddRectFilled(call_rect_min, call_rect_max, call_bg_color, GUI_RECT_CORNER_RADIUS);
                        draw_list->AddRect(
                            call_rect_min, call_rect_max, COLOR_CALL_GROUP_BORDER, GUI_RECT_CORNER_RADIUS);

                        // Draw Text
                        ImVec2 text_pos_left_upper =
                            (call_center + ImVec2(-(class_name_width / 2.0f), -0.5f * ImGui::GetFontSize()));
                        if (this->gui_label_visible && this->gui_slots_visible) {
                            text_pos_left_upper.y -= (0.5f * ImGui::GetFontSize());
                        }
                        if (this->gui_label_visible) {
                            draw_list->AddText(text_pos_left_upper,
                                ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Text]), this->class_name.c_str());
                        }
                        text_pos_left_upper =
                            (call_center + ImVec2(-(slots_label_width / 2.0f), -0.5f * ImGui::GetFontSize()));
                        if (this->gui_label_visible && this->gui_slots_visible) {
                            text_pos_left_upper.y += (0.5f * ImGui::GetFontSize());
                        }
                        if (this->gui_slots_visible) {
                            draw_list->AddText(text_pos_left_upper,
                                ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Text]), slots_label.c_str());
                        }
                    }
                }

                ImGui::PopID();
            }
        }
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


const std::string megamol::gui::Call::SlotsLabel(void) {

    std::string caller = "n/a";
    std::string callee = "n/a";
    auto callerslot_ptr = this->CallSlotPtr(CallSlotType::CALLER);
    auto calleeslot_ptr = this->CallSlotPtr(CallSlotType::CALLEE);
    if (callerslot_ptr != nullptr)
        caller = callerslot_ptr->Name();
    if (calleeslot_ptr != nullptr)
        callee = calleeslot_ptr->Name();
    return std::string("[" + caller + "] > [" + callee + "]");
}
