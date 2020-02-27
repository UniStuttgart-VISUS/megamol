/*
 * CallSlot.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "Call.h"
#include "CallSlot.h"
#include "Module.h"


using namespace megamol;
using namespace megamol::gui::configurator;


megamol::gui::configurator::CallSlot::CallSlot(int uid) : uid(uid), present() {
    this->parent_module.reset();
    connected_calls.clear();
}


megamol::gui::configurator::CallSlot::~CallSlot() {

    // Call separately and check for reference count
    this->DisConnectCalls();
    this->DisConnectParentModule();
}


bool megamol::gui::configurator::CallSlot::CallsConnected(void) const {

    /// TEMP Check for unclean references
    for (auto& call_ptr : this->connected_calls) {
        if (call_ptr == nullptr) {
            throw std::invalid_argument("Pointer to connected call is nullptr.");
        }
    }
    return (!this->connected_calls.empty());
}


bool megamol::gui::configurator::CallSlot::ConnectCall(megamol::gui::configurator::CallPtrType call) {

    if (call == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to given call is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (this->type == CallSlot::CallSlotType::CALLER) {
        if (this->connected_calls.size() > 0) {
            vislib::sys::Log::DefaultLog.WriteWarn(
                "Caller slots can only be connected to one call. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
            return false;
        }
    }
    this->connected_calls.emplace_back(call);
    return true;
}


bool megamol::gui::configurator::CallSlot::DisConnectCall(int call_uid, bool called_by_call) {

    try {
        for (auto call_iter = this->connected_calls.begin(); call_iter != this->connected_calls.end(); call_iter++) {
            if ((*call_iter) == nullptr) {
                vislib::sys::Log::DefaultLog.WriteWarn(
                    "Call is already disconnected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            } else {
                if ((*call_iter)->uid == call_uid) {
                    if (!called_by_call) {
                        (*call_iter)->DisConnectCallSlots();
                    }
                    (*call_iter).reset();
                    this->connected_calls.erase(call_iter);
                    return true;
                }
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


bool megamol::gui::configurator::CallSlot::DisConnectCalls(void) {

    try {
        // Since connected calls operate on this list for disconnecting slots
        // a local copy of the connected calls is required.
        auto connected_calls_copy = this->connected_calls;
        for (auto& call_ptr : connected_calls_copy) {
            if (call_ptr == nullptr) {
                vislib::sys::Log::DefaultLog.WriteWarn(
                    "Call is already disconnected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            } else {
                call_ptr->DisConnectCallSlots();
            }
        }
        this->connected_calls.clear();
        connected_calls_copy.clear();
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


const std::vector<megamol::gui::configurator::CallPtrType>& megamol::gui::configurator::CallSlot::GetConnectedCalls(
    void) {

    /// TEMP Check for unclean references
    for (auto& call_ptr : this->connected_calls) {
        if (call_ptr == nullptr) {
            throw std::invalid_argument("Pointer to connected call is nullptr.");
        }
    }

    return this->connected_calls;
}


bool megamol::gui::configurator::CallSlot::ParentModuleConnected(void) const {
    return (this->parent_module != nullptr);
}


bool megamol::gui::configurator::CallSlot::ConnectParentModule(
    megamol::gui::configurator::ModulePtrType parent_module) {

    if (parent_module == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to given parent module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    if (this->parent_module != nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to parent module is already set. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->parent_module = parent_module;
    return true;
}


bool megamol::gui::configurator::CallSlot::DisConnectParentModule(void) {

    if (parent_module == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to parent module is already nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->parent_module.reset();
    return true;
}


const megamol::gui::configurator::ModulePtrType megamol::gui::configurator::CallSlot::GetParentModule(void) {

    if (this->parent_module == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Returned pointer to parent module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    return this->parent_module;
}


// CALL SLOT PRESENTATION ####################################################

megamol::gui::configurator::CallSlot::Presentation::Presentation(void)
    : presentations(Presentation::DEFAULT), label_visible(true) {}

megamol::gui::configurator::CallSlot::Presentation::~Presentation(void) {}


bool megamol::gui::configurator::CallSlot::Presentation::Present(megamol::gui::configurator::CallSlot& call_slot) {

    try {

        if (ImGui::GetCurrentContext() == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        ImGui::PushID(call_slot.uid);

        /*
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        assert(draw_list != nullptr);
        draw_list->ChannelsSetCurrent(1); // Foreground

        const ImU32 COLOR_SLOT = IM_COL32(175, 175, 175, 255);
        const ImU32 COLOR_SLOT_BORDER = IM_COL32(225, 225, 225, 255);
        const ImU32 COLOR_SLOT_CALLER_LABEL = IM_COL32(0, 255, 255, 255);
        const ImU32 COLOR_SLOT_CALLER_HIGHTLIGHT = IM_COL32(0, 255, 255, 255);
        const ImU32 COLOR_SLOT_CALLEE_LABEL = IM_COL32(255, 92, 255, 255);
        const ImU32 COLOR_SLOT_CALLEE_HIGHTLIGHT = IM_COL32(255, 92, 255, 255);
        const ImU32 COLOR_SLOT_COMPATIBLE = IM_COL32(0, 255, 0, 255);

        ImU32 slot_color = COLOR_SLOT;
        ImU32 slot_highl_color;
        ImU32 slot_label_color;

        // Draw call slots for given module
        for (auto& slot_pair : mod->GetCallSlots()) {

            if (slot_pair.first == graph::CallSlot::CallSlotType::CALLER) {
                slot_highl_color = COLOR_SLOT_CALLER_HIGHTLIGHT;
                slot_label_color = COLOR_SLOT_CALLER_LABEL;
            } else if (slot_pair.first == graph::CallSlot::CallSlotType::CALLEE) {
                slot_highl_color = COLOR_SLOT_CALLEE_HIGHTLIGHT;
                slot_label_color = COLOR_SLOT_CALLEE_LABEL;
            }

            for (auto& slot : slot_pair.second) {
                ImGui::PushID(slot->uid);

                ImVec2 slot_position = slot->present.position;
                float radius = graph->gui.slot_radius * graph->gui.canvas_zooming;
                std::string slot_name = slot->name;
                slot_color = COLOR_SLOT;

                ImGui::SetCursorScreenPos(slot_position - ImVec2(radius, radius));
                std::string label = "slot_" + mod->full_name + slot_name + std::to_string(slot->uid);
                ImGui::InvisibleButton(label.c_str(), ImVec2(radius * 2.0f, radius * 2.0f));

                std::string tooltip = slot->description;
                if (!graph->gui.show_slot_names) {
                    tooltip = slot->name + " | " + tooltip;
                }
                this->utils.HoverToolTip(tooltip, ImGui::GetID(label.c_str()), 0.5f, 5.0f);
                auto hovered = ImGui::IsItemHovered();
                auto clicked = ImGui::IsItemClicked();
                int compat_call_idx = this->graph_manager.GetCompatibleCallIndex(graph->gui.selected_slot_ptr, slot);
                if (hovered) {
                    graph->gui.hovered_slot_uid = slot->uid;
                    // Check if selected call slot should be connected with current slot
                    if (graph->gui.process_selected_slot > 0) {
                        if (graph->AddCall(this->graph_manager.GetCallsStock(), compat_call_idx,
                                graph->gui.selected_slot_ptr, slot)) {
                            graph->gui.selected_slot_ptr = nullptr;
                        }
                    }
                }
                if (clicked) {
                    // Select / Unselect call slot
                    if (graph->gui.selected_slot_ptr != slot) {
                        graph->gui.selected_slot_ptr = slot;
                    } else {
                        graph->gui.selected_slot_ptr = nullptr;
                    }
                }
                if (hovered || (graph->gui.selected_slot_ptr == slot)) {
                    slot_color = slot_highl_color;
                }
                // Highlight if compatible to selected slot
                if (compat_call_idx > 0) {
                    slot_color = COLOR_SLOT_COMPATIBLE;
                }

                ImGui::SetCursorScreenPos(slot_position);
                draw_list->AddCircleFilled(slot_position, radius, slot_color);
                draw_list->AddCircle(slot_position, radius, COLOR_SLOT_BORDER);

                // Draw text
                if (graph->gui.show_slot_names) {
                    ImVec2 text_pos;
                    text_pos.y = slot_position.y - ImGui::GetFontSize() / 2.0f;
                    if (slot_pair.first == graph::CallSlot::CallSlotType::CALLER) {
                        text_pos.x = slot_position.x - this->utils.TextWidgetWidth(slot_name) - (2.0f * radius);
                    } else if (slot_pair.first == graph::CallSlot::CallSlotType::CALLEE) {
                        text_pos.x = slot_position.x + (2.0f * radius);
                    }
                    draw_list->AddText(text_pos, slot_label_color, slot_name.c_str());
                }

                ImGui::PopID();
            }
        }
        */

        ImGui::PopID();

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


void megamol::gui::configurator::CallSlot::Presentation::UpdatePosition() {}
