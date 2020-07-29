/*
 * CallSlot.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CallSlot.h"

#include "Call.h"
#include "InterfaceSlot.h"
#include "Module.h"


using namespace megamol;
using namespace megamol::gui;


// CALL SLOT PRESENTATION ####################################################

megamol::gui::CallSlotPresentation::CallSlotPresentation(void)
    : group()
    , label_visible(false)
    , position()
    , selected(false)
    , update_once(true)
    , show_modulestock(false)
    , last_compat_callslot_uid(GUI_INVALID_ID)
    , last_compat_interface_uid(GUI_INVALID_ID)
    , compatible(false)
    , tooltip() {

    this->group.interfaceslot_ptr.reset();
}


megamol::gui::CallSlotPresentation::~CallSlotPresentation(void) {}

void megamol::gui::CallSlotPresentation::Present(
    PresentPhase phase, megamol::gui::CallSlot& inout_callslot, megamol::gui::GraphItemsStateType& state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    ImGuiStyle& style = ImGui::GetStyle();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    try {
        // Apply update of position first
        if (this->update_once) {
            this->Update(inout_callslot, state.canvas);
            this->update_once = false;
        }

        // Get some information
        ImVec2 slot_position = this->position;
        float radius = GUI_SLOT_RADIUS * state.canvas.zooming;
        bool is_group_interface = (this->group.interfaceslot_ptr != nullptr);
        ImGuiID is_parent_module_group_uid = GUI_INVALID_ID;
        if (inout_callslot.IsParentModuleConnected()) {
            is_parent_module_group_uid = inout_callslot.GetParentModule()->present.group.uid;
        }
        ImVec2 text_pos_left_upper = ImVec2(0.0f, 0.0f);
        if (this->label_visible) {
            text_pos_left_upper.y = slot_position.y - ImGui::GetTextLineHeightWithSpacing() / 2.0f;
            if (inout_callslot.type == CallSlotType::CALLER) {
                text_pos_left_upper.x =
                    slot_position.x - ImGui::CalcTextSize(inout_callslot.name.c_str()).x - (1.5f * radius);
            } else if (inout_callslot.type == CallSlotType::CALLEE) {
                text_pos_left_upper.x = slot_position.x + (1.5f * radius);
            }
        }

        bool mouse_clicked_anywhere = ImGui::IsWindowHovered() && ImGui::GetIO().MouseClicked[0];

        // Clip call slots if lying ouside the canvas
        /// Is there a benefit since ImGui::PushClipRect is used?
        ImVec2 canvas_rect_min = state.canvas.position;
        ImVec2 canvas_rect_max = state.canvas.position + state.canvas.size;
        ImVec2 slot_rect_min = ImVec2(slot_position.x - radius, slot_position.y - radius);
        ImVec2 slot_rect_max = ImVec2(slot_position.x + radius, slot_position.y + radius);
        if (this->label_visible) {
            if (text_pos_left_upper.x < slot_rect_min.x) slot_rect_min.x = text_pos_left_upper.x;
            if (text_pos_left_upper.x > slot_rect_max.x) slot_rect_max.x = text_pos_left_upper.x;
            if (text_pos_left_upper.y < slot_rect_min.y) slot_rect_min.y = text_pos_left_upper.y;
            if (text_pos_left_upper.y > slot_rect_max.y) slot_rect_max.y = text_pos_left_upper.y;
        }
        if (!((canvas_rect_min.x < (slot_rect_max.x)) && (canvas_rect_max.x > (slot_rect_min.x)) &&
                (canvas_rect_min.y < (slot_rect_max.y)) && (canvas_rect_max.y > (slot_rect_min.y)))) {
            if (mouse_clicked_anywhere) {
                this->selected = false;
                if (state.interact.callslot_selected_uid == inout_callslot.uid) {
                    state.interact.callslot_selected_uid = GUI_INVALID_ID;
                }
            }
        } else {
            std::string button_label = "callslot_" + std::to_string(inout_callslot.uid);

            ImGui::PushID(inout_callslot.uid);

            if (phase == megamol::gui::PresentPhase::INTERACTION) {

                // Button
                ImGui::SetCursorScreenPos(slot_position - ImVec2(radius, radius));
                ImGui::SetItemAllowOverlap();
                ImGui::InvisibleButton(button_label.c_str(), ImVec2(radius * 2.0f, radius * 2.0f));
                ImGui::SetItemAllowOverlap();
                if (ImGui::IsItemActivated()) {
                    state.interact.button_active_uid = inout_callslot.uid;
                }
                if (ImGui::IsItemHovered()) {
                    state.interact.button_hovered_uid = inout_callslot.uid;
                }

                // Context Menu
                if (ImGui::BeginPopupContextItem("invisible_button_context")) {
                    state.interact.button_active_uid = inout_callslot.uid;

                    ImGui::TextDisabled("Call Slot");
                    ImGui::Separator();

                    bool enable_interface_creation =
                        (!is_group_interface && (is_parent_module_group_uid != GUI_INVALID_ID));
                    if (ImGui::MenuItem("Create new Interface Slot ", nullptr, false, enable_interface_creation)) {
                        state.interact.callslot_add_group_uid.first = inout_callslot.uid;
                        state.interact.callslot_add_group_uid.second = inout_callslot.GetParentModule()->uid;
                    }
                    if (ImGui::MenuItem("Remove from Interface Slot", nullptr, false, is_group_interface)) {
                        state.interact.callslot_remove_group_uid.first = inout_callslot.uid;
                        state.interact.callslot_remove_group_uid.second = inout_callslot.GetParentModule()->uid;
                    }
                    ImGui::Separator();

                    ImGui::TextDisabled("Description");
                    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 13.0f);
                    ImGui::TextUnformatted(inout_callslot.description.c_str());
                    ImGui::PopTextWrapPos();

                    ImGui::EndPopup();
                }

                // Drag & Drop
                if (ImGui::BeginDragDropTarget()) {
                    if (ImGui::AcceptDragDropPayload(GUI_DND_CALLSLOT_UID_TYPE) != nullptr) {
                        state.interact.slot_dropped_uid = inout_callslot.uid;
                    }
                    ImGui::EndDragDropTarget();
                }
                if (this->selected) {
                    auto dnd_flags =
                        ImGuiDragDropFlags_AcceptNoDrawDefaultRect; // | ImGuiDragDropFlags_SourceNoPreviewTooltip;
                    if (ImGui::BeginDragDropSource(dnd_flags)) {
                        ImGui::SetDragDropPayload(GUI_DND_CALLSLOT_UID_TYPE, &inout_callslot.uid, sizeof(ImGuiID));
                        ImGui::TextUnformatted(inout_callslot.name.c_str());
                        ImGui::EndDragDropSource();
                    }
                }

                // Hover Tooltip
                if ((state.interact.callslot_hovered_uid == inout_callslot.uid) && !this->label_visible) {
                    this->tooltip.ToolTip(inout_callslot.name, ImGui::GetID(button_label.c_str()), 0.5f, 5.0f);
                } else {
                    this->tooltip.Reset();
                }
            } else if (phase == megamol::gui::PresentPhase::RENDERING) {

                bool active = (state.interact.button_active_uid == inout_callslot.uid);
                bool hovered = (state.interact.button_hovered_uid == inout_callslot.uid);

                // Compatibility
                if (state.interact.callslot_compat_ptr != nullptr) {
                    if (state.interact.callslot_compat_ptr->uid != this->last_compat_callslot_uid) {
                        this->compatible = inout_callslot.IsConnectionValid((*state.interact.callslot_compat_ptr));
                        this->last_compat_callslot_uid = state.interact.callslot_compat_ptr->uid;
                    }
                } else if (state.interact.interfaceslot_compat_ptr != nullptr) {
                    if (state.interact.interfaceslot_compat_ptr->uid != this->last_compat_interface_uid) {
                        this->compatible = state.interact.interfaceslot_compat_ptr->IsConnectionValid(inout_callslot);
                        this->last_compat_interface_uid = state.interact.interfaceslot_compat_ptr->uid;
                    }
                } else { /// (state.interact.callslot_compat_ptr == nullptr) && (state.interact.interfaceslot_compat_ptr
                         /// == nullptr)
                    this->compatible = false;
                    this->last_compat_callslot_uid = GUI_INVALID_ID;
                    this->last_compat_interface_uid = GUI_INVALID_ID;
                }

                // Selection
                if (!this->selected && active) {
                    state.interact.callslot_selected_uid = inout_callslot.uid;
                    this->selected = true;
                    state.interact.call_selected_uid = GUI_INVALID_ID;
                    state.interact.modules_selected_uids.clear();
                    state.interact.group_selected_uid = GUI_INVALID_ID;
                    state.interact.interfaceslot_selected_uid = GUI_INVALID_ID;
                }
                // Deselection
                else if ((this->selected && ((mouse_clicked_anywhere && !hovered) ||
                                                (state.interact.callslot_selected_uid != inout_callslot.uid)))) {
                    this->selected = false;
                    if (state.interact.callslot_selected_uid == inout_callslot.uid) {
                        state.interact.callslot_selected_uid = GUI_INVALID_ID;
                    }
                }

                // Hovering
                if (hovered) {
                    state.interact.callslot_hovered_uid = inout_callslot.uid;
                }
                if (!hovered && (state.interact.callslot_hovered_uid == inout_callslot.uid)) {
                    state.interact.callslot_hovered_uid = GUI_INVALID_ID;
                }

                // Colors
                float brightness = (is_group_interface) ? (0.6f) : (1.0f);

                ImVec4 tmpcol = style.Colors[ImGuiCol_FrameBg];
                tmpcol.w *= brightness;
                tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
                const ImU32 COLOR_SLOT_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);

                tmpcol = style.Colors[ImGuiCol_ScrollbarGrabActive];
                tmpcol.w *= brightness;
                tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
                const ImU32 COLOR_SLOT_BORDER = ImGui::ColorConvertFloat4ToU32(tmpcol);

                // Draw Slot
                ImU32 slot_border_color = COLOR_SLOT_BORDER;
                ImU32 slot_background_color = COLOR_SLOT_BACKGROUND;
                if (this->compatible) {
                    tmpcol = GUI_COLOR_SLOT_COMPATIBLE;
                    tmpcol = ImVec4(tmpcol.x * brightness, tmpcol.y * brightness, tmpcol.z * brightness, tmpcol.w);
                    slot_background_color = ImGui::ColorConvertFloat4ToU32(tmpcol);
                }
                if (hovered || this->selected) {
                    tmpcol = GUI_COLOR_SLOT_CALLER;
                    if (inout_callslot.type == CallSlotType::CALLEE) {
                        tmpcol = GUI_COLOR_SLOT_CALLEE;
                    }
                    tmpcol = ImVec4(tmpcol.x * brightness, tmpcol.y * brightness, tmpcol.z * brightness, tmpcol.w);
                    slot_background_color = ImGui::ColorConvertFloat4ToU32(tmpcol);
                }
                const float segment_numer = 20.0f;
                draw_list->AddCircleFilled(slot_position, radius, slot_background_color, segment_numer);
                draw_list->AddCircle(slot_position, radius, slot_border_color, segment_numer);

                // Text
                if (this->label_visible) {
                    ImU32 slot_text_color = ImGui::ColorConvertFloat4ToU32(GUI_COLOR_SLOT_CALLER);
                    if (inout_callslot.type == CallSlotType::CALLEE) {
                        slot_text_color = ImGui::ColorConvertFloat4ToU32(GUI_COLOR_SLOT_CALLEE);
                    }
                    draw_list->AddText(text_pos_left_upper, slot_text_color, inout_callslot.name.c_str());
                }
            }
            ImGui::PopID();
        }

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


void megamol::gui::CallSlotPresentation::Update(
    megamol::gui::CallSlot& inout_callslot, const GraphCanvasType& in_canvas) {

    if (inout_callslot.IsParentModuleConnected()) {
        auto slot_count = inout_callslot.GetParentModule()->GetCallSlots(inout_callslot.type).size();
        size_t slot_idx = 0;
        for (size_t idx = 0; idx < slot_count; idx++) {
            if (inout_callslot.name == inout_callslot.GetParentModule()->GetCallSlots(inout_callslot.type)[idx]->name) {
                slot_idx = idx;
            }
        }

        float line_height = 0.0f;
        if (inout_callslot.GetParentModule()->present.label_visible) {
            line_height = ImGui::GetTextLineHeightWithSpacing() / in_canvas.zooming;
        }
        auto module_pos = inout_callslot.GetParentModule()->present.position;
        module_pos.y += line_height;
        ImVec2 pos = in_canvas.offset + module_pos * in_canvas.zooming;
        auto module_size = inout_callslot.GetParentModule()->present.GetSize();
        module_size.y -= line_height;
        ImVec2 size = module_size * in_canvas.zooming;
        this->position = ImVec2(pos.x + ((inout_callslot.type == CallSlotType::CALLER) ? (size.x) : (0.0f)),
            pos.y + size.y * ((float)slot_idx + 1) / ((float)slot_count + 1));
    }
}


// CALL SLOT ##################################################################

megamol::gui::CallSlot::CallSlot(ImGuiID uid)
    : uid(uid), name(), description(), compatible_call_idxs(), type(), parent_module(), connected_calls(), present() {}


megamol::gui::CallSlot::~CallSlot() {

    // Disconnects calls and parent module
    this->DisconnectCalls();
    this->DisconnectParentModule();
}


bool megamol::gui::CallSlot::CallsConnected(void) const {

    /// Check for unclean references
    for (auto& call_ptr : this->connected_calls) {
        if (call_ptr == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Pointer to one of the connected calls is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
        }
    }
    return (!this->connected_calls.empty());
}


bool megamol::gui::CallSlot::ConnectCall(const megamol::gui::CallPtrType& call_ptr) {

    if (call_ptr == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "Pointer to given call is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (this->type == CallSlotType::CALLER) {
        if (this->connected_calls.size() > 0) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "Caller slots can only be connected to one call. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
            return false;
        }
    }
    this->connected_calls.emplace_back(call_ptr);
    return true;
}


bool megamol::gui::CallSlot::DisconnectCall(ImGuiID call_uid) {

    try {
        for (auto call_iter = this->connected_calls.begin(); call_iter != this->connected_calls.end(); call_iter++) {
            if ((*call_iter) != nullptr) {
                if ((*call_iter)->uid == call_uid) {
                    (*call_iter)->DisconnectCallSlots(this->uid);
                    (*call_iter).reset();
                    if (call_iter != this->connected_calls.end()) {
                        this->connected_calls.erase(call_iter);
                    }
                    return true;
                }
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


bool megamol::gui::CallSlot::DisconnectCalls(void) {

    try {
        for (auto& call_ptr : this->connected_calls) {
            if (call_ptr != nullptr) {
                call_ptr->DisconnectCallSlots(this->uid);
            }
        }
        this->connected_calls.clear();

    } catch (std::exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


const std::vector<megamol::gui::CallPtrType>& megamol::gui::CallSlot::GetConnectedCalls(void) {

    /// Check for unclean references
    for (auto& call_ptr : this->connected_calls) {
        if (call_ptr == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Pointer to one of the connected calls is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
        }
    }
    return this->connected_calls;
}


bool megamol::gui::CallSlot::IsParentModuleConnected(void) const { return (this->parent_module != nullptr); }


bool megamol::gui::CallSlot::ConnectParentModule(megamol::gui::ModulePtrType parent_module) {

    if (parent_module == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "Pointer to given parent module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (this->parent_module != nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "Pointer to parent module is already set. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->parent_module = parent_module;
    return true;
}


bool megamol::gui::CallSlot::DisconnectParentModule(void) {

    if (parent_module == nullptr) {
#ifdef GUI_VERBOSE
/// megamol::core::utility::log::Log::DefaultLog.WriteWarn("Pointer to parent module is already nullptr. [%s, %s, line
/// %d]\n", __FILE__,
/// __FUNCTION__, __LINE__);
#endif // GUI_VERBOSE
        return false;
    }
    this->parent_module.reset();
    return true;
}


const megamol::gui::ModulePtrType& megamol::gui::CallSlot::GetParentModule(void) {

    if (this->parent_module == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "Returned pointer to parent module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    return this->parent_module;
}


ImGuiID megamol::gui::CallSlot::GetCompatibleCallIndex(
    const CallSlotPtrType& callslot_1, const CallSlotPtrType& callslot_2) {

    if ((callslot_1 != nullptr) && (callslot_2 != nullptr)) {
        if (callslot_1->GetParentModule() != callslot_2->GetParentModule() && (callslot_1->type != callslot_2->type)) {
            // Return first found compatible call index
            for (auto& comp_call_idx_1 : callslot_1->compatible_call_idxs) {
                for (auto& comp_call_idx_2 : callslot_2->compatible_call_idxs) {
                    if (comp_call_idx_1 == comp_call_idx_2) {
                        return static_cast<ImGuiID>(comp_call_idx_1);
                    }
                }
            }
        }
    }
    return GUI_INVALID_ID;
}


ImGuiID megamol::gui::CallSlot::GetCompatibleCallIndex(
    const CallSlotPtrType& callslot, const CallSlot::StockCallSlot& stock_callslot) {

    if (callslot != nullptr) {
        if (callslot->type != stock_callslot.type) {
            // Return first found compatible call index
            for (auto& comp_call_idx_1 : callslot->compatible_call_idxs) {
                for (auto& comp_call_idx_2 : stock_callslot.compatible_call_idxs) {
                    if (comp_call_idx_1 == comp_call_idx_2) {
                        return static_cast<ImGuiID>(comp_call_idx_1);
                    }
                }
            }
        }
    }
    return GUI_INVALID_ID;
}


bool megamol::gui::CallSlot::IsConnectionValid(CallSlot& callslot) {

    // Check for different type
    if (this->type == callslot.type) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("Call slots must have different types. [%s, %s, line
        /// %d]\n",
        /// __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Check for present parent module
    if ((callslot.GetParentModule() == nullptr) || (this->GetParentModule() == nullptr)) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("Call slots must have a connected parent module.
        /// [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Check for different parent module
    if ((this->GetParentModule()->uid == callslot.GetParentModule()->uid)) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("Call slots must have different parent modules. [%s,
        /// %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Check for at least one found compatible call index
    for (auto& selected_comp_callslot : callslot.compatible_call_idxs) {
        for (auto& current_comp_callslots : this->compatible_call_idxs) {
            if (selected_comp_callslot == current_comp_callslots) {
                return true;
            }
        }
    }
    return false;
}
