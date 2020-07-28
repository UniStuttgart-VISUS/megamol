/*
 * InterfaceSlot.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "InterfaceSlot.h"

#include "Call.h"
#include "CallSlot.h"
#include "Module.h"


using namespace megamol;
using namespace megamol::gui;


// GROUP INTERFACE SLOT PRESENTATION ###########################################

megamol::gui::InterfaceSlotPresentation::InterfaceSlotPresentation(void)
    : group()
    , label_visible(false)
    , position(ImVec2(FLT_MAX, FLT_MAX))
    , selected(false)
    , label()
    , last_compat_callslot_uid(GUI_INVALID_ID)
    , last_compat_interface_uid(GUI_INVALID_ID)
    , compatible(false)
    , tooltip() {

    this->group.uid = GUI_INVALID_ID;
    this->group.collapsed_view = false;
    this->group.collapsed_view = false;
}


megamol::gui::InterfaceSlotPresentation::~InterfaceSlotPresentation(void) {}


void megamol::gui::InterfaceSlotPresentation::Present(
    PresentPhase phase, megamol::gui::InterfaceSlot& inout_interfaceslot, megamol::gui::GraphItemsStateType& state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
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
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
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


// GROUP INTERFACE SLOT #######################################################

megamol::gui::InterfaceSlot::InterfaceSlot(ImGuiID uid, bool auto_create) : uid(uid), auto_created(auto_create) {}


megamol::gui::InterfaceSlot::~InterfaceSlot(void) {

    // Remove all call slots from interface slot
    std::vector<ImGuiID> callslots_uids;
    for (auto& callslot_ptr : this->callslots) {
        callslots_uids.emplace_back(callslot_ptr->uid);
    }
    for (auto& callslot_uid : callslots_uids) {
        this->RemoveCallSlot(callslot_uid);
    }
    this->callslots.clear();
}


bool megamol::gui::InterfaceSlot::AddCallSlot(
    const CallSlotPtrType& callslot_ptr, const InterfaceSlotPtrType& parent_interfaceslot_ptr) {

    try {
        if (callslot_ptr == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        if (parent_interfaceslot_ptr == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Pointer to interface slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        if (this->is_callslot_compatible((*callslot_ptr))) {
            this->callslots.emplace_back(callslot_ptr);

            callslot_ptr->present.group.interfaceslot_ptr = parent_interfaceslot_ptr;
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[Configurator] Added call slot '%s' to interface slot of group.\n", callslot_ptr->name.c_str());
#endif // GUI_VERBOSE
            return true;
        }
    } catch (std::exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    /// megamol::core::utility::log::Log::DefaultLog.WriteError("Call slot '%s' is incompatible to interface slot of
    /// group. [%s, %s, line %d]\n", callslot_ptr->name.c_str(), __FILE__, __FUNCTION__, __LINE__);
    return false;
}


bool megamol::gui::InterfaceSlot::RemoveCallSlot(ImGuiID callslot_uid) {

    try {
        for (auto iter = this->callslots.begin(); iter != this->callslots.end(); iter++) {
            if ((*iter)->uid == callslot_uid) {

                (*iter)->present.group.interfaceslot_ptr = nullptr;
#ifdef GUI_VERBOSE
                megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                    "[Configurator] Removed call slot '%s' from interface slot of group.\n", (*iter)->name.c_str());
#endif // GUI_VERBOSE
                (*iter).reset();
                this->callslots.erase(iter);

                return true;
            }
        }
    } catch (std::exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return false;
}


bool megamol::gui::InterfaceSlot::ContainsCallSlot(ImGuiID callslot_uid) {

    for (auto& callslot : this->callslots) {
        if (callslot_uid == callslot->uid) {
            return true;
        }
    }
    return false;
}


bool megamol::gui::InterfaceSlot::IsConnectionValid(InterfaceSlot& interfaceslot) {

    CallSlotPtrType callslot_ptr_1;
    CallSlotPtrType callslot_ptr_2;
    if (this->GetCompatibleCallSlot(callslot_ptr_1) && interfaceslot.GetCompatibleCallSlot(callslot_ptr_2)) {
        // Check for different group
        if (this->present.group.uid != interfaceslot.present.group.uid) {
            // Check for compatibility of call slots which are part of the interface slots
            return (callslot_ptr_1->IsConnectionValid((*callslot_ptr_2)));
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
            /// megamol::core::utility::log::Log::DefaultLog.WriteError("Call slots must have connceted parent module.
            /// [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        if (callslot.GetParentModule()->present.group.uid == this->present.group.uid) {
            /// megamol::core::utility::log::Log::DefaultLog.WriteError("Parent module of call slot should not be in
            /// same group as the interface. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        // Check for compatibility of call slots
        CallSlotPtrType interface_callslot_ptr;
        if (this->GetCompatibleCallSlot(interface_callslot_ptr)) {
            if (interface_callslot_ptr->IsConnectionValid(callslot)) {
                return true;
            }
        }
    }
    return false;
}


bool megamol::gui::InterfaceSlot::GetCompatibleCallSlot(CallSlotPtrType& out_callslot_ptr) {

    out_callslot_ptr.reset();
    if (!this->callslots.empty()) {
        out_callslot_ptr = this->callslots[0];
        return true;
    }
    return false;
}


bool megamol::gui::InterfaceSlot::IsConnected(void) {

    for (auto& callslot_ptr : this->callslots) {
        if (callslot_ptr->CallsConnected()) {
            return true;
        }
    }
    return false;
}


CallSlotType megamol::gui::InterfaceSlot::GetCallSlotType(void) {

    CallSlotType ret_type = CallSlotType::CALLER;
    if (!this->callslots.empty()) {
        return this->callslots[0]->type;
    }
    return ret_type;
}


bool megamol::gui::InterfaceSlot::IsEmpty(void) { return (this->callslots.empty()); }


bool megamol::gui::InterfaceSlot::is_callslot_compatible(CallSlot& callslot) {

    // Callee interface slots can only have one call slot
    if (this->callslots.size() > 0) {
        if ((this->GetCallSlotType() == CallSlotType::CALLEE)) {
            /// megamol::core::utility::log::Log::DefaultLog.WriteError("Callee interface slots can only have one call
            /// slot connceted.
            /// [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
    }
    // Call slot can only be added if not already part of this interface
    if (this->ContainsCallSlot(callslot.uid)) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("Call slots can only be added if not already part of
        /// this interface.
        /// [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Call slot can only be added if not already part of other interface
    if (callslot.present.group.interfaceslot_ptr != nullptr) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("Call slots can only be added if not already part of
        /// other interface. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Call slot can only be added if parent module is part of same group
    if (callslot.GetParentModule() == nullptr) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("Call slots must have connceted parent module. [%s,
        /// %s, line %d]\n",
        /// __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (callslot.GetParentModule()->present.group.uid != this->present.group.uid) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("Parent module of call slot should be in same group
        /// as the interface. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Check for compatibility (with all available call slots...)
    size_t compatible_slot_count = 0;
    for (auto& interface_callslot_ptr : this->callslots) {
        // Check for same type and same compatible call indices
        if ((callslot.type == interface_callslot_ptr->type) &&
            (callslot.compatible_call_idxs == interface_callslot_ptr->compatible_call_idxs)) {
            compatible_slot_count++;
        }
    }
    bool compatible = (compatible_slot_count == this->callslots.size());
    // Check for existing incompatible call slots
    if ((compatible_slot_count > 0) && (compatible_slot_count != this->callslots.size())) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("Interface slot contains incompatible call slots.
        /// [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    return compatible;
}
