/*
 * InterfaceSlot.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "InterfaceSlot.h"

#include "CallSlot.h"
#include "Module.h"


using namespace megamol;
using namespace megamol::gui;
using namespace megamol::gui::configurator;


megamol::gui::configurator::InterfaceSlot::InterfaceSlot(ImGuiID uid) : uid(uid) {}


megamol::gui::configurator::InterfaceSlot::~InterfaceSlot(void) {

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


bool megamol::gui::configurator::InterfaceSlot::AddCallSlot(
    const CallSlotPtrType& callslot_ptr, const InterfaceSlotPtrType& parent_interfaceslot_ptr) {

    try {
        if (callslot_ptr == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        if (parent_interfaceslot_ptr == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Pointer to interface slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        
        if (!(this->ContainsCallSlot(callslot_ptr->uid)) && (this->IsCallSlotCompatible((*callslot_ptr)))) {
            this->callslots.emplace_back(callslot_ptr);

            callslot_ptr->GUI_SetGroupInterface(parent_interfaceslot_ptr);

            vislib::sys::Log::DefaultLog.WriteInfo(
                "Added call slot '%s' to interface slot of group.\n", callslot_ptr->name.c_str());
            return true;
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


bool megamol::gui::configurator::InterfaceSlot::IsCallSlotCompatible(const CallSlot& callslot) {

    // Callee interface slots can only have one call slot
    if (this->callslots.size() > 0) {
        if ((this->GetCallSlotType() == CallSlotType::CALLEE)) {
            return false;
        }    
    }

    // Check for compatibility (with all available call slots...)
    size_t compatible = 0;
    for (auto& callslot_ptr : this->callslots) {
        if ((callslot.type == callslot_ptr->type) &&
            (callslot.compatible_call_idxs == callslot_ptr->compatible_call_idxs)) {
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


bool megamol::gui::configurator::InterfaceSlot::GetCompatibleCallSlot(CallSlotPtrType& out_callslot_ptr) {
    
    out_callslot_ptr.reset();
    if (!this->callslots.empty()) {
        out_callslot_ptr = this->callslots[0];
        return true;
    }   
    return false;
}


bool megamol::gui::configurator::InterfaceSlot::IsConnected(void) {
    
    for (auto& callslot_ptr : this->callslots) {
        if (callslot_ptr->CallsConnected()) {
            return true;
        }
    }
    return false;
}


CallSlotType megamol::gui::configurator::InterfaceSlot::GetCallSlotType(void) {
    
    CallSlotType ret_type = CallSlotType::CALLER;
    if (!this->callslots.empty()) {
        return this->callslots[0]->type;
    }   
    return ret_type;
}    
    

bool megamol::gui::configurator::InterfaceSlot::IsEmpty(void) { return (this->callslots.empty()); }


// GROUP INTERFACE SLOT PRESENTATION ###########################################

megamol::gui::configurator::InterfaceSlot::Presentation::Presentation(void)
    : group(), position(ImVec2(FLT_MAX, FLT_MAX)), utils(), selected(false) {
        
    this->group.uid = GUI_INVALID_ID;
    this->group.collapsed_view = false;
    this->group.collapsed_view = false;
}


megamol::gui::configurator::InterfaceSlot::Presentation::~Presentation(void) {}


void megamol::gui::configurator::InterfaceSlot::Presentation::Present(PresentPhase phase,
    megamol::gui::configurator::InterfaceSlot& inout_interfaceslot, megamol::gui::GraphItemsStateType& state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    ImGuiStyle& style = ImGui::GetStyle();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    try {
        ImVec2 actual_position = this->GetPosition(inout_interfaceslot);
        float radius = GUI_SLOT_RADIUS * state.canvas.zooming;
        bool compatible = false;
        if (state.interact.callslot_compat_ptr != nullptr) {
            CallSlotPtrType callslot_ptr;
            if (inout_interfaceslot.GetCompatibleCallSlot(callslot_ptr)) {
                compatible = (CallSlot::CheckCompatibleAvailableCallIndex(state.interact.callslot_compat_ptr, (*callslot_ptr)) != GUI_INVALID_ID);
            }
            compatible = compatible || inout_interfaceslot.IsCallSlotCompatible((*state.interact.callslot_compat_ptr));
        }
        std::string tooltip;
        if (!this->group.collapsed_view) {
            for (auto& callslot_ptr : inout_interfaceslot.GetCallSlots()) {
                tooltip += (callslot_ptr->name + "\n");
            }
        } else {
            for (auto& callslot_ptr : inout_interfaceslot.GetCallSlots()) {
                if (callslot_ptr->IsParentModuleConnected()) {
                    tooltip += (callslot_ptr->GetParentModule()->name + " > ");
                }
                tooltip += (callslot_ptr->name + "\n");
            }
        }

        std::string label = "interfaceslot_" + std::to_string(inout_interfaceslot.uid);

        ImGui::PushID(inout_interfaceslot.uid);

        if (phase == megamol::gui::PresentPhase::INTERACTION) {

            // Button
            ImGui::SetCursorScreenPos(actual_position - ImVec2(radius, radius));
            ImGui::SetItemAllowOverlap();
            ImGui::InvisibleButton(label.c_str(), ImVec2(radius * 2.0f, radius * 2.0f));
            ImGui::SetItemAllowOverlap();
            if (ImGui::IsItemActive()) {
                state.interact.button_active_uid = inout_interfaceslot.uid;
            }
            if (ImGui::IsItemHovered()) {
                state.interact.button_hovered_uid = inout_interfaceslot.uid;
            }

            // Context Menu
            if (ImGui::BeginPopupContextItem("invisible_button_context")) {
                state.interact.button_active_uid = inout_interfaceslot.uid;

                ImGui::TextUnformatted("Interface Slot");
                ImGui::Separator();
                if (ImGui::MenuItem("Delete",
                        std::get<0>(state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM]).ToString().c_str(), false, !inout_interfaceslot.IsConnected())) {
                    std::get<1>(state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM]) = true;
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
            if (state.interact.interfaceslot_hovered_uid == inout_interfaceslot.uid) {
                this->utils.HoverToolTip(tooltip, ImGui::GetID(label.c_str()), 0.5f, 5.0f);
            } else {
                this->utils.ResetHoverToolTip();
            }
        } else if (phase == megamol::gui::PresentPhase::RENDERING) {

            bool active = (state.interact.button_active_uid == inout_interfaceslot.uid);
            bool hovered = (state.interact.button_hovered_uid == inout_interfaceslot.uid);
            bool mouse_clicked_anywhere = ImGui::IsWindowHovered() && ImGui::GetIO().MouseClicked[0];

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

            tmpcol = style.Colors[ImGuiCol_ScrollbarGrab];
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_INTERFACE_LINE = ImGui::ColorConvertFloat4ToU32(tmpcol);

            // Color modification
            ImU32 slot_highlight_color = ImGui::ColorConvertFloat4ToU32(GUI_COLOR_SLOT_CALLER);;
            if (inout_interfaceslot.GetCallSlotType() == CallSlotType::CALLEE) {
                slot_highlight_color = ImGui::ColorConvertFloat4ToU32(GUI_COLOR_SLOT_CALLEE);
            }
            ImU32 slot_color = COLOR_INTERFACE_BACKGROUND;
            if (compatible) {
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
                    draw_list->AddLine(actual_position, callslot_ptr->GUI_GetPosition(), COLOR_INTERFACE_LINE,
                        GUI_LINE_THICKNESS * state.canvas.zooming);
                }
            }
        }

        ImGui::PopID();

    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


ImVec2 megamol::gui::configurator::InterfaceSlot::Presentation::GetPosition(InterfaceSlot& inout_interfaceslot) {

    auto only_callslot_ptr = inout_interfaceslot.GetCallSlots().front();
    ImVec2 ret_position = this->position;
    if (!this->group.collapsed_view) {
        ret_position.x = this->position.x;
        ret_position.y = only_callslot_ptr->GUI_GetPosition().y;
    }
    return ret_position;
}
