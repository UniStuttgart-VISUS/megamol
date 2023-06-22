/*
 * InterfaceSlot.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "InterfaceSlot.h"
#include "Call.h"
#include "CallSlot.h"
#include "Module.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::InterfaceSlot::InterfaceSlot(ImGuiID uid, bool auto_create)
        : uid(uid)
        , auto_created(auto_create)
        , callslots()
        , group_uid(GUI_INVALID_ID)
        , gui_selected(false)
        , gui_position(ImVec2(FLT_MAX, FLT_MAX))
        , gui_label()
        , gui_last_compat_callslot_uid(GUI_INVALID_ID)
        , gui_last_compat_interface_uid(GUI_INVALID_ID)
        , gui_compatible(false)
        , gui_group_collapsed_view(false)
        , gui_tooltip() {}


megamol::gui::InterfaceSlot::~InterfaceSlot() {

    // Remove all call slots from interface slot
    std::vector<ImGuiID> callslots_uids;
    for (auto& callslot_ptr : this->callslots) {
        callslots_uids.emplace_back(callslot_ptr->UID());
    }
    for (auto& callslot_uid : callslots_uids) {
        this->RemoveCallSlot(callslot_uid);
    }
    this->callslots.clear();
}


bool megamol::gui::InterfaceSlot::AddCallSlot(
    const CallSlotPtr_t& callslot_ptr, const InterfaceSlotPtr_t& parent_interfaceslot_ptr) {

    try {
        if (callslot_ptr == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        if (parent_interfaceslot_ptr == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Pointer to interface slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        if (this->is_callslot_compatible((*callslot_ptr))) {
            this->callslots.emplace_back(callslot_ptr);

            callslot_ptr->SetInterfaceSlotPtr(parent_interfaceslot_ptr);
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[GUI] Added call slot '%s' to interface slot of group.\n", callslot_ptr->Name().c_str());
#endif // GUI_VERBOSE
            return true;
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

    /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Call slot '%s' is incompatible to interface slot
    /// of group. [%s, %s, line %d]\n", callslot_ptr->name.c_str(), __FILE__, __FUNCTION__, __LINE__);
    return false;
}


bool megamol::gui::InterfaceSlot::RemoveCallSlot(ImGuiID callslot_uid) {

    try {
        for (auto iter = this->callslots.begin(); iter != this->callslots.end(); iter++) {
            if ((*iter)->UID() == callslot_uid) {

                (*iter)->SetInterfaceSlotPtr(nullptr);
#ifdef GUI_VERBOSE
                megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                    "[GUI] Removed call slot '%s' from interface slot of group.\n", (*iter)->Name().c_str());
#endif // GUI_VERBOSE
                (*iter).reset();
                this->callslots.erase(iter);

                return true;
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
    return false;
}


bool megamol::gui::InterfaceSlot::ContainsCallSlot(ImGuiID callslot_uid) {

    for (auto& callslot : this->callslots) {
        if (callslot_uid == callslot->UID()) {
            return true;
        }
    }
    return false;
}


bool megamol::gui::InterfaceSlot::IsConnectionValid(InterfaceSlot& interfaceslot) {

    if (auto callslot_ptr_1 = this->GetCompatibleCallSlot()) {
        if (auto callslot_ptr_2 = interfaceslot.GetCompatibleCallSlot()) {
            // Check for different group
            if (this->group_uid != interfaceslot.group_uid) {
                // Check for compatibility of call slots which are part of the interface slots
                return (callslot_ptr_1->IsConnectionValid((*callslot_ptr_2)));
            }
        }
    }
    return false;
}


bool megamol::gui::InterfaceSlot::IsConnectionValid(CallSlot& callslot) {

    if (this->is_callslot_compatible(callslot)) {
        return true;
    } else {
        // Call slot can only be added if parent module is not part of same group
        if (callslot.GetParentModule() == nullptr) {
            /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Call slots must have connceted parent
            /// module.
            /// [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        if (callslot.GetParentModule()->GroupUID() == this->group_uid) {
            /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Parent module of call slot should not be
            /// in same group as the interface. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        // Check for compatibility of call slots
        if (auto interface_callslot_ptr = this->GetCompatibleCallSlot()) {
            if (interface_callslot_ptr->IsConnectionValid(callslot)) {
                return true;
            }
        }
    }
    return false;
}


CallSlotPtr_t megamol::gui::InterfaceSlot::GetCompatibleCallSlot() {

    if (!this->callslots.empty()) {
        return this->callslots[0];
    }
    return nullptr;
}


CallSlotType megamol::gui::InterfaceSlot::GetCallSlotType() {

    CallSlotType ret_type = CallSlotType::CALLER;
    if (!this->callslots.empty()) {
        return this->callslots[0]->Type();
    }
    return ret_type;
}


bool megamol::gui::InterfaceSlot::IsEmpty() {
    return (this->callslots.empty());
}


bool megamol::gui::InterfaceSlot::is_callslot_compatible(CallSlot& callslot) {

    // Callee interface slots can only have one call slot
    if (!this->callslots.empty()) {
        if ((this->GetCallSlotType() == CallSlotType::CALLEE)) {
            /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Callee interface slots can only have one
            /// call slot connceted.
            /// [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
    }
    // Call slot can only be added if not already part of this interface
    if (this->ContainsCallSlot(callslot.UID())) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Call slots can only be added if not already
        /// part of this interface.
        /// [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Call slot can only be added if not already part of other interface
    if (callslot.InterfaceSlotPtr() != nullptr) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Call slots can only be added if not already
        /// part of other interface. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Call slot can only be added if parent module is part of same group
    if (callslot.GetParentModule() == nullptr) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Call slots must have connceted parent module.
        /// [%s, %s, line %d]\n",
        /// __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (callslot.GetParentModule()->GroupUID() != this->group_uid) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Parent module of call slot should be in same
        /// group as the interface. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Check for compatibility (with all available call slots...)
    size_t compatible_slot_count = 0;
    for (auto& interface_callslot_ptr : this->callslots) {
        // Check for same type and same compatible call indices
        if ((callslot.Type() == interface_callslot_ptr->Type()) &&
            (callslot.CompatibleCallIdxs() == interface_callslot_ptr->CompatibleCallIdxs())) {
            compatible_slot_count++;
        }
    }
    bool compatible = (compatible_slot_count == this->callslots.size());
    // Check for existing incompatible call slots
    if ((compatible_slot_count > 0) && (compatible_slot_count != this->callslots.size())) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Interface slot contains incompatible call
        /// slots.
        /// [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    return compatible;
}


void megamol::gui::InterfaceSlot::Draw(PresentPhase phase, megamol::gui::GraphItemsState_t& state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    ImGuiStyle& style = ImGui::GetStyle();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    try {
        ImVec2 actual_position = this->Position();
        float radius = GUI_SLOT_RADIUS * state.canvas.zooming;
        this->gui_label.clear();
        for (auto& callslot_ptr : this->CallSlots()) {
            std::string parent_module_name;
            if (callslot_ptr->GetParentModule() != nullptr) {
                parent_module_name = callslot_ptr->GetParentModule()->Name() + "::";
            }
            this->gui_label += (parent_module_name + callslot_ptr->Name() + " \n");
        }
        std::string button_label = "interfaceslot_" + std::to_string(this->uid);

        ImGui::PushID(static_cast<int>(this->uid));

        if (phase == megamol::gui::PresentPhase::INTERACTION) {

            // Button
            ImGui::SetCursorScreenPos(actual_position - ImVec2(radius, radius));
            ImGui::SetItemAllowOverlap();
            ImGui::InvisibleButton(
                button_label.c_str(), ImVec2(radius * 2.0f, radius * 2.0f), ImGuiButtonFlags_NoSetKeyOwner);
            ImGui::SetItemAllowOverlap();
            if (ImGui::IsItemActivated()) {
                state.interact.button_active_uid = this->uid;
            }
            if (ImGui::IsItemHovered()) {
                state.interact.button_hovered_uid = this->uid;
            }

            ImGui::PushFont(state.canvas.gui_font_ptr);

            // Context Menu
            if (ImGui::BeginPopupContextItem("invisible_button_context")) {
                state.interact.button_active_uid = this->uid;

                ImGui::TextDisabled("Interface Slot");
                ImGui::Separator();

                if (ImGui::MenuItem(
                        "Delete", state.hotkeys[HOTKEY_CONFIGURATOR_DELETE_GRAPH_ITEM].keycode.ToString().c_str())) {
                    state.interact.process_deletion = true;
                }

                ImGui::EndPopup();
            }

            // Drag & Drop
            if (ImGui::BeginDragDropTarget()) {
                if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(GUI_DND_CALLSLOT_UID_TYPE)) {
                    auto* dragged_slot_uid_ptr = (ImGuiID*)payload->Data;
                    state.interact.slot_drag_drop_uids.first = (*dragged_slot_uid_ptr);
                    state.interact.slot_drag_drop_uids.second = this->uid;
                }
                ImGui::EndDragDropTarget();
            }
            if (this->gui_selected) {
                auto dnd_flags =
                    ImGuiDragDropFlags_AcceptNoDrawDefaultRect; // | ImGuiDragDropFlags_SourceNoPreviewTooltip;
                if (ImGui::BeginDragDropSource(dnd_flags)) {
                    ImGui::SetDragDropPayload(GUI_DND_CALLSLOT_UID_TYPE, &this->uid, sizeof(ImGuiID));
                    std::string drag_str;
                    for (auto& callslot_ptr : this->CallSlots()) {
                        drag_str += (callslot_ptr->Name() + "\n");
                    }
                    ImGui::TextUnformatted(drag_str.c_str());
                    ImGui::EndDragDropSource();
                }
            }

            // Hover Tooltip
            if ((state.interact.interfaceslot_hovered_uid == this->uid) && !state.interact.callslot_show_label) {
                this->gui_tooltip.ToolTip(this->gui_label, ImGui::GetID(button_label.c_str()), 0.5f, 5.0f);
            } else {
                this->gui_tooltip.Reset();
            }

            ImGui::PopFont();

        } else if (phase == megamol::gui::PresentPhase::RENDERING) {

            bool active = (state.interact.button_active_uid == this->uid);
            bool hovered = (state.interact.button_hovered_uid == this->uid);
            bool mouse_clicked_anywhere =
                ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiPopupFlags_MouseButtonLeft);

            // Compatibility
            if (state.interact.callslot_compat_ptr != nullptr) {
                if (state.interact.callslot_compat_ptr->UID() != this->gui_last_compat_callslot_uid) {
                    this->gui_compatible = this->IsConnectionValid((*state.interact.callslot_compat_ptr));
                    this->gui_last_compat_callslot_uid = state.interact.callslot_compat_ptr->UID();
                }
            } else if (state.interact.interfaceslot_compat_ptr != nullptr) {
                if (state.interact.interfaceslot_compat_ptr->UID() != this->gui_last_compat_interface_uid) {
                    this->gui_compatible = this->IsConnectionValid((*state.interact.interfaceslot_compat_ptr));
                    this->gui_last_compat_interface_uid = state.interact.interfaceslot_compat_ptr->uid;
                }
            } else { /// (state.interact.callslot_compat_ptr == nullptr) && (state.interact.interfaceslot_compat_ptr ==
                     /// nullptr)
                this->gui_compatible = false;
                this->gui_last_compat_callslot_uid = GUI_INVALID_ID;
                this->gui_last_compat_interface_uid = GUI_INVALID_ID;
            }

            // Selection
            if (!this->gui_selected && active) {
                state.interact.interfaceslot_selected_uid = this->uid;
                this->gui_selected = true;
                state.interact.callslot_selected_uid = GUI_INVALID_ID;
                state.interact.call_selected_uid = GUI_INVALID_ID;
                state.interact.modules_selected_uids.clear();
                state.interact.group_selected_uid = GUI_INVALID_ID;
            }
            // Deselection
            else if (this->gui_selected && ((mouse_clicked_anywhere && !hovered) ||
                                               (state.interact.interfaceslot_selected_uid != this->uid))) {
                this->gui_selected = false;
                if (state.interact.interfaceslot_selected_uid == this->uid) {
                    state.interact.interfaceslot_selected_uid = GUI_INVALID_ID;
                }
            }

            // Hovering
            if (hovered) {
                state.interact.interfaceslot_hovered_uid = this->uid;
            }
            if (!hovered && (state.interact.interfaceslot_hovered_uid == this->uid)) {
                state.interact.interfaceslot_hovered_uid = GUI_INVALID_ID;
            }

            /// COLOR_INTERFACE_BACKGROUND
            ImVec4 tmpcol = style.Colors[ImGuiCol_FrameBg];
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_INTERFACE_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);
            /// COLOR_INTERFACE_BORDER
            tmpcol = style.Colors[ImGuiCol_ScrollbarGrabActive];
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_INTERFACE_BORDER = ImGui::ColorConvertFloat4ToU32(tmpcol);
            /// COLOR_INTERFACE_CURVE
            tmpcol = style.Colors[ImGuiCol_FrameBgHovered];
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_INTERFACE_CURVE = ImGui::ColorConvertFloat4ToU32(tmpcol);

            // Color modifications
            ImU32 slot_highlight_color = ImGui::ColorConvertFloat4ToU32(GUI_COLOR_SLOT_CALLER);
            if (this->GetCallSlotType() == CallSlotType::CALLEE) {
                slot_highlight_color = ImGui::ColorConvertFloat4ToU32(GUI_COLOR_SLOT_CALLEE);
            }
            ImU32 slot_color = COLOR_INTERFACE_BACKGROUND;
            if (this->gui_compatible) {
                slot_color = ImGui::ColorConvertFloat4ToU32(GUI_COLOR_SLOT_COMPATIBLE);
            }
            if (hovered || this->gui_selected) {
                slot_color = slot_highlight_color;
            }

            // Draw Slot
            draw_list->AddCircleFilled(actual_position, radius, slot_color);
            draw_list->AddCircle(actual_position, radius, COLOR_INTERFACE_BORDER);

            // Draw Curves
            if (!this->gui_group_collapsed_view) {
                for (auto& callslot_ptr : this->CallSlots()) {
                    draw_list->AddLine(actual_position, callslot_ptr->Position(), COLOR_INTERFACE_CURVE,
                        GUI_LINE_THICKNESS * state.canvas.zooming);
                }
            }

            // Text
            if (this->gui_group_collapsed_view) {
                auto type = this->GetCallSlotType();
                ImVec2 text_pos_left_upper = ImVec2(0.0f, 0.0f);
                auto text_size = ImGui::CalcTextSize(this->gui_label.c_str());
                text_pos_left_upper.y = actual_position.y - text_size.y / 2.0f;
                text_pos_left_upper.x = actual_position.x - text_size.x - (1.5f * radius);
                if (type == CallSlotType::CALLEE) {
                    text_pos_left_upper.x = actual_position.x + (1.5f * radius);
                }
                ImU32 slot_text_color = ImGui::ColorConvertFloat4ToU32(GUI_COLOR_SLOT_CALLER);
                if (type == CallSlotType::CALLEE) {
                    slot_text_color = ImGui::ColorConvertFloat4ToU32(GUI_COLOR_SLOT_CALLEE);
                }
                draw_list->AddText(text_pos_left_upper, slot_text_color, this->gui_label.c_str());
            }
        }

        ImGui::PopID();

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


ImVec2 megamol::gui::InterfaceSlot::Position(bool group_collapsed_view) {

    ImVec2 ret_position = this->gui_position;
    if ((!group_collapsed_view) && (!this->CallSlots().empty())) {
        auto only_callslot_ptr = this->CallSlots().front();
        ret_position.x = this->gui_position.x;
        ret_position.y = only_callslot_ptr->Position().y;
    }
    return ret_position;
}
