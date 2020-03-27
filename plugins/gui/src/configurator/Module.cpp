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
using namespace megamol::gui;
using namespace megamol::gui::configurator;


megamol::gui::configurator::Module::Module(ImGuiID uid) : uid(uid), present() {

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
                    // vislib::sys::Log::DefaultLog.WriteWarn(
                    //     "Call slot is already disconnected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                } else {
                    call_slot_ptr->DisConnectCalls();
                    call_slot_ptr->DisConnectParentModule();

                    if (call_slot_ptr.use_count() > 1) {
                        vislib::sys::Log::DefaultLog.WriteError("Unclean deletion. Found %i references pointing to call slot. [%s, %s, line %d]\n",
                            call_slot_ptr.use_count(), __FILE__, __FUNCTION__, __LINE__);
                    }
           
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


const CallSlotPtrType megamol::gui::configurator::Module::GetCallSlot(ImGuiID call_slot_uid) {

    if (call_slot_uid != GUI_INVALID_ID) {
        for (auto& call_slot_map : this->GetCallSlots()) {
            for (auto& call_slot : call_slot_map.second) {
                if (call_slot->uid == call_slot_uid) {
                    return call_slot;
                }
            }
        }
    }

    return nullptr;
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
    , visible(true)
    , position(ImVec2(FLT_MAX, FLT_MAX))
    , size(ImVec2(0.0f, 0.0f))
    , class_label()
    , name_label()
    , utils()
    , selected(false)
    , update(true) {}


megamol::gui::configurator::Module::Presentation::~Presentation(void) {}


void megamol::gui::configurator::Module::Presentation::Present(megamol::gui::configurator::Module& inout_module, megamol::gui::GraphItemsStateType& state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    ImGuiStyle& style = ImGui::GetStyle();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    bool popup_rename = false;

    try {
        // Condition for initialization position (if position is not set yet via tag in project file)
        if ((this->position.x == FLT_MAX) && (this->position.y == FLT_MAX)) {
            this->position = ImVec2(10.0f, 10.0f) + (ImGui::GetWindowPos() - state.canvas.offset) / state.canvas.zooming;
        }
        // Update size if current values are invalid
        if (this->update || (this->size.x <= 0.0f) || (this->size.y <= 0.0f)) {
            this->UpdateSize(inout_module, state.canvas);
            this->update = false;
        }

        if (this->visible) {
            // Draw module --------------------------------------------------------
            ImVec2 module_size = this->size * state.canvas.zooming;
            ImVec2 module_rect_min = state.canvas.offset + this->position * state.canvas.zooming;
            ImVec2 module_rect_max = module_rect_min + module_size;
            ImVec2 module_center = module_rect_min + ImVec2(module_size.x / 2.0f, module_size.y / 2.0f);

            // Clip module if lying ouside the canvas 
            /// XXX Is there a benefit since ImGui::PushClipRect is used?
            /*
            ImVec2 canvas_rect_min = state.canvas.position;
            ImVec2 canvas_rect_max = state.canvas.position + state.canvas.size;
            if (!((canvas_rect_min.x < module_rect_max.x) && (canvas_rect_max.x > module_rect_min.x) &&
                    (canvas_rect_min.y < module_rect_max.y) && (canvas_rect_max.y > module_rect_min.y))) {
                if (mouse_clicked) {
                    this->selected = false;
                    if (state.interact.module_selected_uid == inout_module.uid) {
                        state.interact.module_selected_uid = GUI_INVALID_ID;
                    }                
                }
                if (this->selected) {
                    state.interact.module_selected_uid = inout_module.uid;
                }
                return;
            }
            else {
                ...
            }
            */
        
            ImGui::PushID(inout_module.uid);

            // Colors
            ImVec4 tmpcol = style.Colors[ImGuiCol_FrameBg]; // ImGuiCol_FrameBg ImGuiCol_Button
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_MODULE_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);

            tmpcol = style.Colors[ImGuiCol_FrameBgActive]; // ImGuiCol_FrameBgActive ImGuiCol_ButtonActive
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_MODULE_HIGHTLIGHT = ImGui::ColorConvertFloat4ToU32(tmpcol);

            tmpcol = style.Colors[ImGuiCol_ScrollbarGrabActive]; // ImGuiCol_Border ImGuiCol_ScrollbarGrabActive
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_MODULE_BORDER = ImGui::ColorConvertFloat4ToU32(tmpcol);

            // Draw box
            ImGui::SetCursorScreenPos(module_rect_min);
            std::string label = "module_" + inout_module.name;

            ImGui::SetItemAllowOverlap();
            ImGui::InvisibleButton(label.c_str(), module_size);
            ImGui::SetItemAllowOverlap();

            bool active = ImGui::IsItemActive();
            bool hovered = (ImGui::IsItemHovered() && (state.interact.callslot_hovered_uid == GUI_INVALID_ID) && 
                ((state.interact.module_hovered_uid == GUI_INVALID_ID) || (state.interact.module_hovered_uid == inout_module.uid)));
            bool mouse_clicked = ImGui::IsWindowHovered() && ImGui::GetIO().MouseClicked[0];

            if ((mouse_clicked && (!hovered || state.interact.callslot_hovered_uid != GUI_INVALID_ID)) || (state.interact.module_selected_uid != inout_module.uid)) {
                this->selected = false;
                if (state.interact.module_selected_uid == inout_module.uid) {
                    state.interact.module_selected_uid = GUI_INVALID_ID;
                }
            }     
            if (!hovered && (state.interact.module_hovered_uid == inout_module.uid)) {
                state.interact.module_hovered_uid = GUI_INVALID_ID;
            }        
            if (hovered) {
                state.interact.module_hovered_uid = inout_module.uid;              
            }              
            if (state.interact.callslot_hovered_uid == GUI_INVALID_ID) {
                // Context menu
                if (ImGui::BeginPopupContextItem("invisible_button_context")) {
                    ImGui::Text("Module");
                    ImGui::Separator();               
                    if (ImGui::MenuItem("Delete", std::get<0>(state.hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM]).ToString().c_str())) {
                        std::get<1>(state.hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM]) = true;
                        // Force selection
                        active = true;
                    }
                    if (ImGui::MenuItem("Rename")) {
                        popup_rename = true;
                    }
                    if (ImGui::BeginMenu("Add Group")) {
                        if (ImGui::MenuItem("New")) {
                            state.interact.module_add_group_uid.first =  inout_module.uid; 
                            state.interact.module_add_group_uid.second =  GUI_INVALID_ID; 
                        }
                        if (!state.groups.empty()) {
                            ImGui::Separator();
                        }
                        for (auto& group_pair : state.groups) {
                            if (ImGui::MenuItem(group_pair.second.c_str())) {
                                state.interact.module_add_group_uid.first = inout_module.uid;
                                state.interact.module_add_group_uid.second = group_pair.first;
                            } 
                        }
                        ImGui::EndMenu();
                    }
                    if (ImGui::MenuItem("Remove Group")) {
                        state.interact.module_remove_group_uid = inout_module.uid;
                    }                                
                    ImGui::EndPopup();
                }

                // Hover Tooltip
                std::string hover_text = inout_module.description;
                if (!this->label_visible) {
                    hover_text = "[" + inout_module.name + "] " + hover_text;
                }
                this->utils.HoverToolTip(hover_text.c_str(), ImGui::GetID(label.c_str()), 0.5f, 5.0f);     

                if (active) {
                    this->selected = true;
                    state.interact.module_selected_uid = inout_module.uid;
                    state.interact.callslot_selected_uid = GUI_INVALID_ID;
                    state.interact.call_selected_uid = GUI_INVALID_ID;
                    state.interact.group_selected_uid = GUI_INVALID_ID;                      
                }
                if (this->selected && ImGui::IsWindowHovered() && ImGui::IsMouseDragging(0)) {
                    this->position += (ImGui::GetIO().MouseDelta / state.canvas.zooming);
                    this->UpdateSize(inout_module, state.canvas);
                } 
            }
            
            ImU32 module_bg_color = (hovered || this->selected) ? COLOR_MODULE_HIGHTLIGHT : COLOR_MODULE_BACKGROUND;
            draw_list->AddRectFilled(module_rect_min, module_rect_max, module_bg_color, 5.0f, ImDrawCornerFlags_All);
            draw_list->AddRect(module_rect_min, module_rect_max, COLOR_MODULE_BORDER, 5.0f, ImDrawCornerFlags_All, 1.0f);

            // Draw text
            if (this->label_visible) {

                float line_offset = 0.0f;
                if (inout_module.is_view_instance) {
                    line_offset = -0.5f * ImGui::GetTextLineHeightWithSpacing();
                }

                label = this->name_label;
                float name_width = this->utils.TextWidgetWidth(label);
                ImVec2 text_pos_left_upper = (module_center + ImVec2(-(name_width / 2.0f), line_offset - ImGui::GetTextLineHeightWithSpacing()));
                draw_list->AddText(text_pos_left_upper, ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Text]), label.c_str());

                label = this->class_label;
                name_width = this->utils.TextWidgetWidth(label);
                text_pos_left_upper = (module_center + ImVec2(-(name_width / 2.0f), line_offset));
                draw_list->AddText(text_pos_left_upper, ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Text]), label.c_str());

                if (inout_module.is_view_instance) {
                    label = "[Main View]";
                    name_width = this->utils.TextWidgetWidth(label);
                    text_pos_left_upper = (module_center + ImVec2(-(name_width / 2.0f), line_offset + ImGui::GetTextLineHeightWithSpacing()));
                    draw_list->AddText(text_pos_left_upper, ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Text]), label.c_str());
                }
            }

            /// XXX Use ImGui::ArrowButton to show/hide parameters inside module box.

            // Draw call slots ----------------------------------------------------
            for (auto& slot_pair : inout_module.GetCallSlots()) {
                for (auto& slot : slot_pair.second) {
                    slot->GUI_Present(state);
                }
            }

            // Rename pop-up ------------------------------------------------------
            if (this->utils.RenamePopUp("Rename Project", popup_rename, inout_module.name)) {
                this->UpdateSize(inout_module, state.canvas);
            }

            ImGui::PopID();
        }

    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


void megamol::gui::configurator::Module::Presentation::UpdateSize(
    megamol::gui::configurator::Module& inout_mod, const GraphCanvasType& in_canvas) {

    float max_label_length = 0.0f;
    if (this->label_visible) {
        this->class_label = "Class: " + inout_mod.class_name;
        float class_name_length = this->utils.TextWidgetWidth(this->class_label);
        this->name_label = "Name: " + inout_mod.name; // inout_mod.FullName();
        float name_length = this->utils.TextWidgetWidth(inout_mod.present.name_label);
        max_label_length = std::max(class_name_length, name_length);
    }
    max_label_length /= in_canvas.zooming;

    float max_slot_name_length = 0.0f;
    for (auto& call_slot_type_list : inout_mod.GetCallSlots()) {
        for (auto& call_slot : call_slot_type_list.second) {
            if (call_slot->GUI_GetLabelVisibility()) {
                max_slot_name_length = std::max(this->utils.TextWidgetWidth(call_slot->name), max_slot_name_length);
            }
        }
    }
    if (max_slot_name_length != 0.0f) {
        max_slot_name_length = (2.0f * max_slot_name_length / in_canvas.zooming) + (1.0f * GUI_CALL_SLOT_RADIUS);
    }

    float module_width = (max_label_length + max_slot_name_length) + (3.0f * GUI_CALL_SLOT_RADIUS);

    auto max_slot_count = std::max(inout_mod.GetCallSlots(CallSlot::CallSlotType::CALLEE).size(),
        inout_mod.GetCallSlots(CallSlot::CallSlotType::CALLER).size());
    float module_slot_height =
        (static_cast<float>(max_slot_count) * (GUI_CALL_SLOT_RADIUS * 2.0f) * 1.25f) + GUI_CALL_SLOT_RADIUS;

    float module_height = std::max(module_slot_height,
        (1.0f / in_canvas.zooming) * (ImGui::GetTextLineHeightWithSpacing() * ((inout_mod.is_view_instance) ? (4.0f) : (3.0f))));

    // Clamp to minimum size
    this->size = ImVec2(std::max(module_width, 75.0f), std::max(module_height, 25.0f));

    // Update all call slots
    for (auto& slot_pair : inout_mod.GetCallSlots()) {
        for (auto& slot : slot_pair.second) {
            slot->GUI_Update(in_canvas);
        }
    }
}
