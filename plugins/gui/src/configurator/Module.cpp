/*
 * Module.cpp
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


megamol::gui::configurator::Module::Module(int uid) : uid(uid), present() {

    this->call_slots.clear();
    this->call_slots.emplace(
        megamol::gui::configurator::CallSlot::CallSlotType::CALLER, std::vector<CallSlotPtrType>());
    this->call_slots.emplace(
        megamol::gui::configurator::CallSlot::CallSlotType::CALLEE, std::vector<CallSlotPtrType>());
}


megamol::gui::configurator::Module::~Module() { this->RemoveAllCallSlots(); }


bool megamol::gui::configurator::Module::AddCallSlot(megamol::gui::configurator::CallSlotPtrType call_slot) {

    if (call_slot == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to given call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    auto type = call_slot->type;
    for (auto& call_slot_ptr : this->call_slots[type]) {
        if (call_slot_ptr == call_slot) {
            throw std::invalid_argument("Pointer to call slot already registered in modules call slot list.");
        }
    }
    this->call_slots[type].emplace_back(call_slot);
    return true;
}


bool megamol::gui::configurator::Module::RemoveAllCallSlots(void) {

    try {
        for (auto& call_slots_map : this->call_slots) {
            for (auto& call_slot_ptr : call_slots_map.second) {
                if (call_slot_ptr == nullptr) {
                    vislib::sys::Log::DefaultLog.WriteWarn(
                        "Call slot is already disconnected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                } else {
                    call_slot_ptr->DisConnectCalls();
                    call_slot_ptr->DisConnectParentModule();

                    vislib::sys::Log::DefaultLog.WriteWarn(
                        "Found %i references pointing to call slot. [%s, %s, line %d]\n", call_slot_ptr.use_count(),
                        __FILE__, __FUNCTION__, __LINE__);
                    assert(call_slot_ptr.use_count() == 1);

                    call_slot_ptr.reset();
                }
            }
            call_slots_map.second.clear();
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


const std::vector<megamol::gui::configurator::CallSlotPtrType>& megamol::gui::configurator::Module::GetCallSlots(
    megamol::gui::configurator::CallSlot::CallSlotType type) {

    // if (this->call_slots[type].empty()) {
    //    vislib::sys::Log::DefaultLog.WriteWarn(
    //        "Returned call slot list is empty. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    //}
    return this->call_slots[type];
}


const std::map<megamol::gui::configurator::CallSlot::CallSlotType,
    std::vector<megamol::gui::configurator::CallSlotPtrType>>&
megamol::gui::configurator::Module::GetCallSlots(void) {

    return this->call_slots;
}


// MODULE PRESENTATION ####################################################

megamol::gui::configurator::Module::Presentation::Presentation(void)
    : presentations(Module::Presentations::DEFAULT)
    , label_visible(true)
    , position(ImVec2(0.0f, 0.0f))
    , size(ImVec2(250.0f, 50.0f))
    , class_label()
    , name_label()
    , utils() {}


megamol::gui::configurator::Module::Presentation::~Presentation(void) {}


ImGuiID megamol::gui::configurator::Module::Presentation::Present(
    megamol::gui::configurator::Module& mod, ImVec2 canvas_offset, float canvas_zooming) {

    int retval_id = GUI_INVALID_ID;

    try {

        if (ImGui::GetCurrentContext() == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        ImGui::PushID(mod.uid);

        // Draw call slots ----------------------------------------------------
        /// Draw call slots prior to modules to catch mouse clicks on slot area lying over module box.

        int hovered_slot_uid = GUI_INVALID_ID;
        for (auto& slot_pair : mod.GetCallSlots()) {
            for (auto& slot : slot_pair.second) {
                auto id = slot->GUI_Present(canvas_offset, canvas_zooming);
                if (id != GUI_INVALID_ID) {
                    hovered_slot_uid = id;
                }
            }
        }

        // Draw module --------------------------------------------------------

        if ((this->position.x == 0.0f) && (this->position.y == 0.0f)) {
            this->position = canvas_offset;
        }

        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        assert(draw_list != nullptr);

        const ImU32 COLOR_MODULE_BACKGROUND = IM_COL32(64, 61, 64, 255);
        const ImU32 COLOR_MODULE_HIGHTLIGHT = IM_COL32(92, 92, 92, 255);
        const ImU32 COLOR_MODULE_BORDER = IM_COL32(128, 128, 128, 255);

        int hovered_module = GUI_INVALID_ID;

        ImVec2 module_size = this->size;
        ImVec2 module_rect_min = canvas_offset + this->position * canvas_zooming;
        ImVec2 module_rect_max = module_rect_min + module_size;
        ImVec2 module_center = module_rect_min + ImVec2(module_size.x / 2.0f, module_size.y / 2.0f);
        std::string label = this->class_label;

        // Draw text
        draw_list->ChannelsSetCurrent(1); // Foreground
        ImGui::BeginGroup();

        float line_offset = 0.0f;
        if (mod.is_view) {
            line_offset = -0.5f * ImGui::GetItemsLineHeightWithSpacing();
        }

        auto class_name_width = this->utils.TextWidgetWidth(label);
        ImGui::SetCursorScreenPos(
            module_center + ImVec2(-(class_name_width / 2.0f), line_offset - ImGui::GetItemsLineHeightWithSpacing()));
        ImGui::Text(label.c_str());

        label = this->name_label;
        auto name_width = this->utils.TextWidgetWidth(label);
        ImGui::SetCursorScreenPos(module_center + ImVec2(-(name_width / 2.0f), line_offset));
        ImGui::Text(label.c_str());

        if (mod.is_view) {
            if (false) {
                std::string view_label = "[View]";
                if (mod.is_view_instance) {
                    view_label = "[Main View]";
                }
                name_width = this->utils.TextWidgetWidth(view_label);
                ImGui::SetCursorScreenPos(module_center + ImVec2(-(name_width / 2.0f), -line_offset));
                ImGui::Text(view_label.c_str());
            } else {
                std::string view_label = "Main View";
                name_width = this->utils.TextWidgetWidth(view_label);
                ImGui::SetCursorScreenPos(module_center + ImVec2(-(name_width / 2.0f) - 20.0f, -line_offset));
                ImGui::Checkbox(view_label.c_str(), &mod.is_view_instance);
                ImGui::SameLine();
                this->utils.HelpMarkerToolTip(
                    "There should be only one main view.\nOtherwise first one found is used.");
                /// TODO ensure that there is always just one main view ...
            }
        }

        ImGui::EndGroup();

        // Draw box
        draw_list->ChannelsSetCurrent(0); // Background

        ImGui::SetCursorScreenPos(module_rect_min);
        label = "module_" + mod.full_name + std::to_string(mod.uid);
        ImGui::InvisibleButton(label.c_str(), module_size);
        // Gives slots which overlap modules priority for ToolTip and Context Menu.
        if (hovered_slot_uid == GUI_INVALID_ID) {
            this->utils.HoverToolTip(mod.description, ImGui::GetID(label.c_str()), 0.5f, 5.0f);
            // Context menu
            /// XXX
            /*
             if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem(
                        "Delete", std::get<0>(this->hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM]).ToString().c_str()))
                        {
                    std::get<1>(this->hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM]) = true;
                }
                if (ImGui::MenuItem("Rename")) {
                    this->gui.rename_popup_open = true;
                    this->gui.rename_popup_string = &mod->name;
                }

                ImGui::EndPopup();
            }
            */
        }
        bool module_active = ImGui::IsItemActive();
        if (module_active) {
            retval_id = mod.uid;
        }
        if (module_active && ImGui::IsMouseDragging(0)) {
            this->position = ((module_rect_min - canvas_offset) + ImGui::GetIO().MouseDelta) / canvas_zooming;
        }
        if (ImGui::IsItemHovered() && (hovered_module < 0)) {
            hovered_module = mod.uid;
        }
        ImU32 module_bg_color =
            (hovered_module == mod.uid || retval_id == mod.uid) ? COLOR_MODULE_HIGHTLIGHT : COLOR_MODULE_BACKGROUND;
        draw_list->AddRectFilled(module_rect_min, module_rect_max, module_bg_color, 4.0f);
        draw_list->AddRect(module_rect_min, module_rect_max, COLOR_MODULE_BORDER, 4.0f);

        ImGui::PopID();

    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    }

    return retval_id;
}
