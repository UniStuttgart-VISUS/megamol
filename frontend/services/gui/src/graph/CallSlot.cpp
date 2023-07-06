/*
 * CallSlot.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "CallSlot.h"
#include "Call.h"
#include "InterfaceSlot.h"
#include "Module.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::CallSlot::CallSlot(ImGuiID uid, const std::string& name, const std::string& description,
    const std::vector<size_t>& compatible_call_idxs, CallSlotType type,
    megamol::core::AbstractCallSlotPresentation::Necessity necessity)
        : uid(uid)
        , name(name)
        , description(description)
        , compatible_call_idxs(compatible_call_idxs)
        , type(type)
        , necessity(necessity)
        , parent_module(nullptr)
        , connected_calls()
        , gui_interfaceslot_ptr(nullptr)
        , gui_selected(false)
        , gui_position(ImVec2(FLT_MAX, FLT_MAX))
        , gui_update_once(true)
        , gui_last_compat_callslot_uid(GUI_INVALID_ID)
        , gui_last_compat_interface_uid(GUI_INVALID_ID)
        , gui_compatible(false)
        , gui_tooltip() {}


megamol::gui::CallSlot::~CallSlot() {

    // Disconnects calls and parent module
    this->DisconnectCalls();
    this->DisconnectParentModule();
}


bool megamol::gui::CallSlot::CallsConnected() const {

    /// Check for unclean references
    for (auto& call_ptr : this->connected_calls) {
        if (call_ptr == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Pointer to one of the connected calls is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
        }
    }
    return (!this->connected_calls.empty());
}


bool megamol::gui::CallSlot::ConnectCall(const megamol::gui::CallPtr_t& call_ptr) {

    if (call_ptr == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Pointer to given call is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (this->type == CallSlotType::CALLER) {
        if (!this->connected_calls.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] Caller slots can only be connected to one call. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
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
                if ((*call_iter)->UID() == call_uid) {
                    (*call_iter)->DisconnectCallSlots(this->uid);
                    (*call_iter).reset();
                    if (call_iter != this->connected_calls.end()) {
                        this->connected_calls.erase(call_iter);
                    }
                    return true;
                }
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


bool megamol::gui::CallSlot::DisconnectCalls() {

    try {
        for (auto& call_ptr : this->connected_calls) {
            if (call_ptr != nullptr) {
                call_ptr->DisconnectCallSlots(this->uid);
            }
        }
        this->connected_calls.clear();

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


const std::vector<megamol::gui::CallPtr_t>& megamol::gui::CallSlot::GetConnectedCalls() {

    /// Check for unclean references
    for (auto& call_ptr : this->connected_calls) {
        if (call_ptr == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Pointer to one of the connected calls is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
        }
    }
    return this->connected_calls;
}


bool megamol::gui::CallSlot::IsParentModuleConnected() const {
    return (this->parent_module != nullptr);
}


bool megamol::gui::CallSlot::ConnectParentModule(megamol::gui::ModulePtr_t pm) {

    if (pm == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Pointer to given parent module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (this->parent_module != nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Pointer to parent module is already set. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->parent_module = pm;
    return true;
}


bool megamol::gui::CallSlot::DisconnectParentModule() {

    if (parent_module == nullptr) {
#ifdef GUI_VERBOSE
/// megamol::core::utility::log::Log::DefaultLog.WriteWarn("[GUI] Pointer to parent module is already nullptr. [%s, %s,
/// line %d]\n", __FILE__,
/// __FUNCTION__, __LINE__);
#endif // GUI_VERBOSE
        return false;
    }
    this->parent_module.reset();
    return true;
}


const megamol::gui::ModulePtr_t& megamol::gui::CallSlot::GetParentModule() {

    if (this->parent_module == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Returned pointer to parent module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
    }
    return this->parent_module;
}


ImGuiID megamol::gui::CallSlot::GetCompatibleCallIndex(
    const CallSlotPtr_t& callslot_1, const CallSlotPtr_t& callslot_2) {

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
    const CallSlotPtr_t& callslot, const CallSlot::StockCallSlot& stock_callslot) {

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
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Call slots must have different types. [%s,
        /// %s, line %d]\n",
        /// __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Check for present parent module
    if ((callslot.GetParentModule() == nullptr) || (this->GetParentModule() == nullptr)) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Call slots must have a connected parent
        /// module.
        /// [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Check for different parent module
    if ((this->GetParentModule()->UID() == callslot.GetParentModule()->UID())) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Call slots must have different parent
        /// modules. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
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


void megamol::gui::CallSlot::Draw(PresentPhase phase, megamol::gui::GraphItemsState_t& state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    ImGuiStyle& style = ImGui::GetStyle();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    try {
        // Apply update of position first
        if (this->gui_update_once) {
            this->Update(state);
            this->gui_update_once = false;
        }

        // Get some information
        ImVec2 slot_position = this->gui_position;
        float radius = GUI_SLOT_RADIUS * state.canvas.zooming;
        bool is_group_interface = (this->gui_interfaceslot_ptr != nullptr);
        ImGuiID is_parent_module_group_uid = GUI_INVALID_ID;
        if (this->IsParentModuleConnected()) {
            is_parent_module_group_uid = this->GetParentModule()->GroupUID();
        }
        ImVec2 text_pos_left_upper = ImVec2(0.0f, 0.0f);
        if (state.interact.callslot_show_label) {
            text_pos_left_upper.y = slot_position.y - ImGui::GetTextLineHeightWithSpacing() / 2.0f;
            if (this->type == CallSlotType::CALLER) {
                text_pos_left_upper.x = slot_position.x - ImGui::CalcTextSize(this->name.c_str()).x - (1.5f * radius);
            } else if (this->type == CallSlotType::CALLEE) {
                text_pos_left_upper.x = slot_position.x + (1.5f * radius);
            }
        }

        bool mouse_clicked_anywhere =
            ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiPopupFlags_MouseButtonLeft);

        std::string slot_label = this->name;
        std::string button_label = "callslot_" + std::to_string(this->uid);
        bool slot_required = ((this->necessity == megamol::core::AbstractCallSlotPresentation::SLOT_REQUIRED) &&
                              (!this->CallsConnected()));
        if (slot_required) {
            slot_label.append(" [REQUIRED]");
        }

        ImGui::PushID(static_cast<int>(this->uid));

        if (phase == megamol::gui::PresentPhase::INTERACTION) {

            // Button
            ImGui::SetCursorScreenPos(slot_position - ImVec2(radius, radius));
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

                ImGui::TextDisabled("Call Slot");
                ImGui::Separator();

                bool enable_interface_creation =
                    (!is_group_interface && (is_parent_module_group_uid != GUI_INVALID_ID));
                if (ImGui::MenuItem("Create new Interface Slot ", nullptr, false, enable_interface_creation)) {
                    state.interact.callslot_add_group_uid.first = this->uid;
                    state.interact.callslot_add_group_uid.second = this->GetParentModule()->UID();
                }
                if (ImGui::MenuItem("Remove from Interface Slot", nullptr, false, is_group_interface)) {
                    state.interact.callslot_remove_group_uid.first = this->uid;
                    state.interact.callslot_remove_group_uid.second = this->GetParentModule()->UID();
                }
                ImGui::Separator();

                ImGui::TextDisabled("Description");
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 13.0f);
                ImGui::TextUnformatted(this->description.c_str());
                ImGui::PopTextWrapPos();

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
                    ImGui::TextUnformatted(this->name.c_str());
                    ImGui::EndDragDropSource();
                }
            }

            // Hover Tooltip
            if ((state.interact.callslot_hovered_uid == this->uid) && !state.interact.callslot_show_label) {
                this->gui_tooltip.ToolTip(slot_label, ImGui::GetID(button_label.c_str()), 0.5f, 5.0f);
            } else {
                this->gui_tooltip.Reset();
            }

            ImGui::PopFont();

        } else if (phase == megamol::gui::PresentPhase::RENDERING) {

            bool active = (state.interact.button_active_uid == this->uid);
            bool hovered = (state.interact.button_hovered_uid == this->uid);

            // Compatibility
            if (state.interact.callslot_compat_ptr != nullptr) {
                if (state.interact.callslot_compat_ptr->uid != this->gui_last_compat_callslot_uid) {
                    this->gui_compatible = this->IsConnectionValid((*state.interact.callslot_compat_ptr));
                    this->gui_last_compat_callslot_uid = state.interact.callslot_compat_ptr->uid;
                }
            } else if (state.interact.interfaceslot_compat_ptr != nullptr) {
                if (state.interact.interfaceslot_compat_ptr->UID() != this->gui_last_compat_interface_uid) {
                    this->gui_compatible = state.interact.interfaceslot_compat_ptr->IsConnectionValid((*this));
                    this->gui_last_compat_interface_uid = state.interact.interfaceslot_compat_ptr->UID();
                }
            } else { /// (state.interact.callslot_compat_ptr == nullptr) && (state.interact.interfaceslot_compat_ptr
                     /// == nullptr)
                this->gui_compatible = false;
                this->gui_last_compat_callslot_uid = GUI_INVALID_ID;
                this->gui_last_compat_interface_uid = GUI_INVALID_ID;
            }

            // Selection
            if (!this->gui_selected && active) {
                state.interact.callslot_selected_uid = this->uid;
                this->gui_selected = true;
                state.interact.call_selected_uid = GUI_INVALID_ID;
                state.interact.modules_selected_uids.clear();
                state.interact.group_selected_uid = GUI_INVALID_ID;
                state.interact.interfaceslot_selected_uid = GUI_INVALID_ID;
            }
            // Deselection
            else if ((this->gui_selected && ((mouse_clicked_anywhere && !hovered) ||
                                                (state.interact.callslot_selected_uid != this->uid)))) {
                this->gui_selected = false;
                if (state.interact.callslot_selected_uid == this->uid) {
                    state.interact.callslot_selected_uid = GUI_INVALID_ID;
                }
            }

            // Hovering
            if (hovered) {
                state.interact.callslot_hovered_uid = this->uid;
            }
            if (!hovered && (state.interact.callslot_hovered_uid == this->uid)) {
                state.interact.callslot_hovered_uid = GUI_INVALID_ID;
            }

            float brightness = (is_group_interface) ? (0.6f) : (1.0f);
            /// COLOR_SLOT_BACKGROUND
            ImVec4 tmpcol = style.Colors[ImGuiCol_FrameBg];
            tmpcol.w *= brightness;
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_SLOT_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);
            /// COLOR_SLOT_BORDER
            tmpcol = style.Colors[ImGuiCol_ScrollbarGrabActive];
            tmpcol.w *= brightness;
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_SLOT_BORDER = ImGui::ColorConvertFloat4ToU32(tmpcol);

            // Draw Slot
            ImU32 slot_border_color = COLOR_SLOT_BORDER;
            ImU32 slot_background_color = COLOR_SLOT_BACKGROUND;
            if (slot_required) {
                slot_border_color = ImGui::ColorConvertFloat4ToU32(GUI_COLOR_SLOT_REQUIRED);
            }
            if (this->gui_compatible) {
                tmpcol = GUI_COLOR_SLOT_COMPATIBLE;
                tmpcol = ImVec4(tmpcol.x * brightness, tmpcol.y * brightness, tmpcol.z * brightness, tmpcol.w);
                slot_background_color = ImGui::ColorConvertFloat4ToU32(tmpcol);
            }
            if (hovered || this->gui_selected) {
                tmpcol = GUI_COLOR_SLOT_CALLER;
                if (this->type == CallSlotType::CALLEE) {
                    tmpcol = GUI_COLOR_SLOT_CALLEE;
                }
                tmpcol = ImVec4(tmpcol.x * brightness, tmpcol.y * brightness, tmpcol.z * brightness, tmpcol.w);
                slot_background_color = ImGui::ColorConvertFloat4ToU32(tmpcol);
            }

            float thickness = (1.0f * state.canvas.zooming);
            if (slot_required) {
                thickness = (2.0f * state.canvas.zooming);
            }
            draw_list->AddCircleFilled(slot_position, radius, slot_background_color);
            draw_list->AddCircle(slot_position, radius, slot_border_color, 0, thickness);

            // Text
            if (state.interact.callslot_show_label) {
                ImU32 slot_text_color = ImGui::ColorConvertFloat4ToU32(GUI_COLOR_SLOT_CALLER);
                if (this->type == CallSlotType::CALLEE) {
                    slot_text_color = ImGui::ColorConvertFloat4ToU32(GUI_COLOR_SLOT_CALLEE);
                }
                draw_list->AddText(text_pos_left_upper, slot_text_color, this->name.c_str());
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


void megamol::gui::CallSlot::Update(const GraphItemsState_t& state) {

    if (this->IsParentModuleConnected()) {
        auto slot_count = this->GetParentModule()->CallSlots(this->type).size();
        size_t slot_idx = 0;
        for (size_t idx = 0; idx < slot_count; idx++) {
            if (this->name == this->GetParentModule()->CallSlots(this->type)[idx]->name) {
                slot_idx = idx;
            }
        }

        float line_height = 0.0f;
        if (state.interact.module_show_label) {
            line_height = ImGui::GetTextLineHeightWithSpacing() / state.canvas.zooming;
        }
        auto module_pos = this->GetParentModule()->Position();
        module_pos.y += line_height;
        ImVec2 pos = state.canvas.offset + module_pos * state.canvas.zooming;
        auto module_size = this->GetParentModule()->Size();
        module_size.y -= line_height;
        ImVec2 size = module_size * state.canvas.zooming;
        this->gui_position = ImVec2(pos.x + ((this->type == CallSlotType::CALLER) ? (size.x) : (0.0f)),
            pos.y + size.y * ((float)slot_idx + 1) / ((float)slot_count + 1));
    }
}
