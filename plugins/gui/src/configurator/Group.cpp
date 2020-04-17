/*
 * Group.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "Group.h"


using namespace megamol;
using namespace megamol::gui;
using namespace megamol::gui::configurator;


megamol::gui::configurator::Group::Group(ImGuiID uid) : uid(uid), name(), modules(), interfaceslots(), present() {}


megamol::gui::configurator::Group::~Group() {

    // Remove all modules from group
    std::vector<ImGuiID> module_uids;
    for (auto& module_ptr : this->modules) {
        module_uids.emplace_back(module_ptr->uid);
    }
    for (auto& module_uid : module_uids) {
        this->RemoveModule(module_uid);
    }
    this->modules.clear();

    // Remove all interface slots from group (should already be empty)
    this->interfaceslots[CallSlotType::CALLER].clear();
    this->interfaceslots[CallSlotType::CALLEE].clear();
}


bool megamol::gui::configurator::Group::AddModule(const ModulePtrType& module_ptr) {

    if (module_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Check if module is already part of the group
    for (auto& mod : this->modules) {
        if (mod->uid == module_ptr->uid) {
            vislib::sys::Log::DefaultLog.WriteInfo(
                "Module '%s' is already part of group '%s'.\n", mod->name.c_str(), this->name.c_str());
            return false;
        }
    }

    this->modules.emplace_back(module_ptr);

    module_ptr->GUI_SetGroupMembership(this->uid);
    module_ptr->GUI_SetGroupVisibility(this->present.ModulesVisible());
    module_ptr->GUI_SetGroupName(this->name);
    this->present.ForceUpdate();

    vislib::sys::Log::DefaultLog.WriteInfo(
        "Added module '%s' to group '%s'.\n", module_ptr->name.c_str(), this->name.c_str());
    return true;
}


bool megamol::gui::configurator::Group::RemoveModule(ImGuiID module_uid) {

    try {
        for (auto mod_iter = this->modules.begin(); mod_iter != this->modules.end(); mod_iter++) {
            if ((*mod_iter)->uid == module_uid) {

                // Remove call slots belonging to this module which are part of interface slots of this group.
                for (auto& callslot_map : (*mod_iter)->GetCallSlots()) {
                    for (auto& callslot_ptr : callslot_map.second) {
                        this->InterfaceSlot_RemoveCallSlot(callslot_ptr->uid);
                    }
                }

                (*mod_iter)->GUI_SetGroupMembership(GUI_INVALID_ID);
                (*mod_iter)->GUI_SetGroupVisibility(false);
                (*mod_iter)->GUI_SetGroupName("");
                this->present.ForceUpdate();

                vislib::sys::Log::DefaultLog.WriteInfo(
                    "Removed module '%s' from group '%s'.\n", (*mod_iter)->name.c_str(), this->name.c_str());
                (*mod_iter).reset();
                this->modules.erase(mod_iter);

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

    vislib::sys::Log::DefaultLog.WriteWarn("Invalid module uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}


bool megamol::gui::configurator::Group::ContainsModule(ImGuiID module_uid) {

    for (auto& mod : this->modules) {
        if (mod->uid == module_uid) {
            return true;
        }
    }
    return false;
}


bool megamol::gui::configurator::Group::InterfaceSlot_AddCallSlot(
    const CallSlotPtrType& callslot_ptr, ImGuiID interfaceslot_uid) {

    bool successfully_added = false;

    if (callslot_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Check if call slot is already part of the group
    for (auto& interfaceslot_ptr : this->interfaceslots[callslot_ptr->type]) {
        if (interfaceslot_ptr->ContainsCallSlot(callslot_ptr->uid)) {
            vislib::sys::Log::DefaultLog.WriteInfo("Call Slot '%s' is already part of interface slot of group '%s'.\n",
                callslot_ptr->name.c_str(), this->name.c_str());
            return false;
        }
    }

    // Only add if parent module is already part of the group.
    bool parent_module_group_member = false;
    if (callslot_ptr->IsParentModuleConnected()) {
        ImGuiID parent_module_uid = callslot_ptr->GetParentModule()->uid;
        for (auto& module_ptr : this->modules) {
            if (parent_module_uid == module_ptr->uid) {
                parent_module_group_member = true;
            }
        }
    } else {
        vislib::sys::Log::DefaultLog.WriteError(
            "Call slot has no parent module connected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    if (parent_module_group_member) {
        
        this->interfaceslots[callslot_ptr->type].emplace_back(std::make_shared<InterfaceSlot>(interfaceslot_uid));

        vislib::sys::Log::DefaultLog.WriteInfo("Added interface slot to group '%s'.\n", this->name.c_str());
        
        InterfaceSlotPtrType interfaceslot_ptr = this->interfaceslots[callslot_ptr->type].back();
        if (interfaceslot_ptr != nullptr) {
            successfully_added = interfaceslot_ptr->AddCallSlot(callslot_ptr, interfaceslot_ptr);
            interfaceslot_ptr->GUI_SetGroupView(this->present.IsViewCollapsed());
        }

    } else {
        vislib::sys::Log::DefaultLog.WriteError(
            "Parent module of call slot which should be added to group interface is not part of any group. [%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return successfully_added;
}


bool megamol::gui::configurator::Group::InterfaceSlot_RemoveCallSlot(ImGuiID callslots_uid) {

    try {
        for (auto& interfaceslot_map : this->interfaceslots) {
            for (auto& interfaceslot_ptr : interfaceslot_map.second) {
                if (interfaceslot_ptr->ContainsCallSlot(callslots_uid)) {
                    interfaceslot_ptr->RemoveCallSlot(callslots_uid);
                    // Delete empty interface slots
                    if (interfaceslot_ptr->IsEmpty()) {
                       this->DeleteInterfaceSlot(interfaceslot_ptr->uid);
                    }
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


bool megamol::gui::configurator::Group::InterfaceSlot_ContainsCallSlot(ImGuiID callslot_uid) {

    for (auto& interfaceslots_map : this->interfaceslots) {
        for (auto& interfaceslot_ptr : interfaceslots_map.second) {
            if (interfaceslot_ptr->ContainsCallSlot(callslot_uid)) {
                return true;
            }
        }
    }
    return false;
}


bool megamol::gui::configurator::Group::GetInterfaceSlot(
    ImGuiID interfaceslot_uid, InterfaceSlotPtrType& interfaceslot_ptr) {

    if (interfaceslot_uid != GUI_INVALID_ID) {
        for (auto& interfaceslots_map : this->interfaceslots) {
            for (auto& interfaceslot : interfaceslots_map.second) {
                if (interfaceslot->uid == interfaceslot_uid) {
                    interfaceslot_ptr = interfaceslot;
                    return true;
                }
            }
        }
    }
    return false;
}


bool megamol::gui::configurator::Group::DeleteInterfaceSlot(ImGuiID interfaceslot_uid) {
    
    if (interfaceslot_uid != GUI_INVALID_ID) {
        for (auto& interfaceslot_map : this->interfaceslots) {
            for (auto iter = interfaceslot_map.second.begin(); iter != interfaceslot_map.second.end(); iter++) {
                if ((*iter)->uid == interfaceslot_uid) {

                    // Remove all call slots from interface slot
                    std::vector<ImGuiID> callslots_uids;
                    for (auto& callslot_ptr : (*iter)->GetCallSlots()) {
                        callslots_uids.emplace_back(callslot_ptr->uid);
                    }
                    for (auto& callslot_uid : callslots_uids) {
                        (*iter)->RemoveCallSlot(callslot_uid);
                    }
    
                    if ((*iter).use_count() > 1) {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "Unclean deletion. Found %i references pointing to interface slot. [%s, %s, line %d]\n",
                        (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                    }

                    (*iter).reset();
                    interfaceslot_map.second.erase(iter);
                    vislib::sys::Log::DefaultLog.WriteInfo(
                        "Removed interface slot from group '%s'.\n", this->name.c_str());
                    return true;
                }
            }
        }
    }
    return false;
}


bool megamol::gui::configurator::Group::ContainsInterfaceSlot(ImGuiID interfaceslot_uid) {
    
    if (interfaceslot_uid != GUI_INVALID_ID) {
        for (auto& interfaceslots_map : this->interfaceslots) {
            for (auto& interfaceslot : interfaceslots_map.second) {
                if (interfaceslot->uid == interfaceslot_uid) {
                    return true;
                }
            }
        }
    }
    return false;
}
    
    

// GROUP PRESENTATION ####################################################

megamol::gui::configurator::Group::Presentation::Presentation(void)
    : border(GUI_SLOT_RADIUS * 4.0f)
    , position(ImVec2(FLT_MAX, FLT_MAX))
    , size(ImVec2(0.0f, 0.0f))
    , utils()
    , name_label()
    , collapsed_view(false)
    , allow_context(false)
    , selected(false)
    , update(true) {}


megamol::gui::configurator::Group::Presentation::~Presentation(void) {}


void megamol::gui::configurator::Group::Presentation::Present(
    PresentPhase phase, megamol::gui::configurator::Group& inout_group, GraphItemsStateType& state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    ImGuiStyle& style = ImGui::GetStyle();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    try {
        // Update size and position if current values are invalid or in expanded view
        if (this->update || !this->collapsed_view || (this->size.x <= 0.0f) || (this->size.y <= 0.0f)) {
            this->UpdatePositionSize(inout_group, state.canvas);
            for (auto& mod : inout_group.GetModules()) {
                mod->GUI_SetGroupVisibility(this->ModulesVisible());
            }
            this->update = false;
        }

        // Draw group --------------------------------------------------------

        ImVec2 group_size = this->size * state.canvas.zooming;
        ImVec2 group_rect_min = state.canvas.offset + this->position * state.canvas.zooming;
        ImVec2 group_rect_max = group_rect_min + group_size;
        ImVec2 group_center = group_rect_min + ImVec2(group_size.x / 2.0f, group_size.y / 2.0f);
        ImVec2 header_size = ImVec2(group_size.x, ImGui::GetTextLineHeightWithSpacing());
        ImVec2 header_rect_max = group_rect_min + header_size;
            
        ImGui::PushID(inout_group.uid);

        if (phase == PresentPhase::INTERACTION) {   
            
            // Header Button
            std::string label = "group_" + std::to_string(inout_group.uid);
            ImGui::SetCursorScreenPos(group_rect_min);
            ImGui::SetItemAllowOverlap();
            ImGui::InvisibleButton(label.c_str(), group_size);
            ImGui::SetItemAllowOverlap();
            if (ImGui::IsItemActive()) {
                state.interact.button_active_uid = inout_group.uid;
            }
            if (ImGui::IsItemHovered()) {
                state.interact.button_hovered_uid = inout_group.uid;
            }
                   
            ImVec2 mouse = ImGui::GetMousePos();
            if ((state.interact.group_hovered_uid == inout_group.uid) && 
                    ((mouse.x >= group_rect_min.x) && (mouse.y >= group_rect_min.y) && (mouse.x <= header_rect_max.x) && (mouse.y <= header_rect_max.y))) {
                this->allow_context = true;
            }

            // Context menu
            bool popup_rename = false;        
            if (this->allow_context && ImGui::BeginPopupContextItem("invisible_button_context")) {
                
                state.interact.button_active_uid = inout_group.uid;

                ImGui::TextUnformatted("Group");
                ImGui::Separator();
                std::string view = "Collapsed View";
                if (this->collapsed_view) {
                    view = "Expanded View";
                }
                if (ImGui::MenuItem(view.c_str())) {
                    this->collapsed_view = !this->collapsed_view;
                    for (auto& module_ptr : inout_group.GetModules()) {
                        module_ptr->GUI_SetGroupVisibility(this->ModulesVisible());
                    }
                    for (auto& interfaceslots_map : inout_group.interfaceslots) {
                        for (auto& interfaceslot_ptr : interfaceslots_map.second) {
                            interfaceslot_ptr->GUI_SetGroupView(this->collapsed_view);
                        }
                    }
                    this->UpdatePositionSize(inout_group, state.canvas);
                }
                /// XXX Not yet implemented
                // if (ImGui::MenuItem("Save")) {
                //    state.interact.group_save = true;
                //}
                if (ImGui::MenuItem("Rename")) {
                    popup_rename = true;
                }
                if (ImGui::MenuItem("Delete",
                        std::get<0>(state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM]).ToString().c_str())) {
                    std::get<1>(state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM]) = true;
                }
                ImGui::EndPopup();
            }
            else {
                this->allow_context = false;
            }
                    
            // Automatically delete empty group
            if (inout_group.GetModules().empty()) {
                std::get<1>(state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM]) = true;
                state.interact.button_active_uid = inout_group.uid;
            } 
            
            // Rename pop-up
            if (this->utils.RenamePopUp("Rename Group", popup_rename, inout_group.name)) {
                for (auto& module_ptr : inout_group.GetModules()) {
                    module_ptr->GUI_SetGroupName(inout_group.name);
                    module_ptr->GUI_Update(state.canvas);
                }
                this->UpdatePositionSize(inout_group, state.canvas);
            }
        }
        else if (phase == PresentPhase::RENDERING) {
            
            bool active = (state.interact.button_active_uid == inout_group.uid);
            bool hovered = (state.interact.button_hovered_uid == inout_group.uid);
            bool mouse_clicked_anywhere = ImGui::IsWindowHovered() && ImGui::GetIO().MouseClicked[0];
                      
            // Hovering
            if (hovered) {
                state.interact.group_hovered_uid = inout_group.uid;
            }
            if (!hovered && (state.interact.group_hovered_uid == inout_group.uid)) {
                state.interact.group_hovered_uid = GUI_INVALID_ID;
            }    
            
            // Limit activation and hovering to header
            bool mouse_inside_header = false;
            ImVec2 mouse = ImGui::GetMousePos();
            if ((mouse.x >= group_rect_min.x) && (mouse.y >= group_rect_min.y) && (mouse.x <= header_rect_max.x) && (mouse.y <= header_rect_max.y)) {
                mouse_inside_header = true;
            }
            if (!mouse_inside_header) {
                active = false;
                hovered = false;
            }
            
            // Selection     
            if (!this->selected && active) {
                state.interact.group_selected_uid = inout_group.uid;
                this->selected = true;
                state.interact.callslot_selected_uid = GUI_INVALID_ID;
                state.interact.modules_selected_uids.clear();
                state.interact.call_selected_uid = GUI_INVALID_ID;
                state.interact.interfaceslot_selected_uid = GUI_INVALID_ID;            
            }
            // Deselection
            else if (this->selected && ((mouse_clicked_anywhere && !hovered) || (active && GUI_MULTISELECT_MODIFIER) || (state.interact.group_selected_uid != inout_group.uid))) {
                this->selected = false;
                if (state.interact.group_selected_uid == inout_group.uid) {
                    state.interact.group_selected_uid = GUI_INVALID_ID;
                }
            }
            
            // Dragging
            if (this->selected && ImGui::IsWindowHovered() && ImGui::IsMouseDragging(0)) {
                ImVec2 tmp_pos;
                for (auto& mod : inout_group.GetModules()) {
                    tmp_pos = mod->GUI_GetPosition();
                    tmp_pos += (ImGui::GetIO().MouseDelta / state.canvas.zooming);
                    mod->GUI_SetPosition(tmp_pos);
                    mod->GUI_Update(state.canvas);
                }
                this->UpdatePositionSize(inout_group, state.canvas);
            }

            // Colors
            ImVec4 tmpcol = style.Colors[ImGuiCol_ScrollbarBg];
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_GROUP_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);

            tmpcol = style.Colors[ImGuiCol_FrameBg];
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_GROUP_HIGHTLIGHT = ImGui::ColorConvertFloat4ToU32(tmpcol);

            tmpcol = style.Colors[ImGuiCol_ScrollbarGrabHovered];
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_GROUP_BORDER = ImGui::ColorConvertFloat4ToU32(tmpcol);

            const ImU32 COLOR_TEXT = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Text]);

            tmpcol = style.Colors[ImGuiCol_FrameBgHovered];
            tmpcol.y = 0.75f;
            const ImU32 COLOR_HEADER = ImGui::ColorConvertFloat4ToU32(tmpcol);

            tmpcol = style.Colors[ImGuiCol_ButtonActive];
            tmpcol.y = 0.75f;
            const ImU32 COLOR_HEADER_HIGHLIGHT = ImGui::ColorConvertFloat4ToU32(tmpcol);

            // Background
            ImU32 group_bg_color = (this->selected) ? (COLOR_GROUP_HIGHTLIGHT) : (COLOR_GROUP_BACKGROUND);
            draw_list->AddRectFilled(group_rect_min, group_rect_max, group_bg_color, 0.0f);
            draw_list->AddRect(group_rect_min, group_rect_max, COLOR_GROUP_BORDER, 0.0f);

            // Draw text
            float name_width = GUIUtils::TextWidgetWidth(this->name_label);
            ImVec2 text_pos_left_upper =
                ImVec2((group_center.x - (name_width / 2.0f)), (group_rect_min.y + (style.ItemSpacing.y / 2.0f)));
            if (!this->collapsed_view) {
                text_pos_left_upper =
                    ImVec2((group_rect_min.x + style.ItemSpacing.x), (group_rect_min.y + (style.ItemSpacing.y / 2.0f)));
            }
            auto header_color = (this->selected) ? (COLOR_HEADER_HIGHLIGHT) : (COLOR_HEADER);
            draw_list->AddRectFilled(group_rect_min, header_rect_max, header_color, GUI_RECT_CORNER_RADIUS,
                (ImDrawCornerFlags_TopLeft | ImDrawCornerFlags_TopRight));
            draw_list->AddText(text_pos_left_upper, COLOR_TEXT, this->name_label.c_str());
        }
        
        ImGui::PopID();
        
        // INTERFACE SLOTS -----------------------------------------------------
        for (auto& interfaceslots_map : inout_group.GetInterfaceSlots()) {
            for (auto& interfaceslot_ptr : interfaceslots_map.second) {
                interfaceslot_ptr->GUI_Present(phase, state);
            }
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


void megamol::gui::configurator::Group::Presentation::UpdatePositionSize(
    megamol::gui::configurator::Group& inout_group, const GraphCanvasType& in_canvas) {

    this->name_label = "[Group] " + inout_group.name;
    float line_height = ImGui::GetTextLineHeightWithSpacing() / in_canvas.zooming;

    // POSITION
    float pos_minX = FLT_MAX;
    float pos_minY = FLT_MAX;
    ImVec2 tmp_pos;
    if (inout_group.GetModules().size() > 0) {
        for (auto& mod : inout_group.GetModules()) {
            tmp_pos = mod->GUI_GetPosition();
            pos_minX = std::min(tmp_pos.x, pos_minX);
            pos_minY = std::min(tmp_pos.y, pos_minY);
        }
        pos_minX -= this->border;
        pos_minY -= (this->border + line_height);
        this->position = ImVec2(pos_minX, pos_minY);
    } else {
        this->position = megamol::gui::configurator::Module::GUI_GetInitModulePosition(in_canvas);
    }

    // SIZE
    float group_width = 0.0f;
    float group_height = 0.0f;
    size_t caller_count = inout_group.interfaceslots[CallSlotType::CALLER].size();
    size_t callee_count = inout_group.interfaceslots[CallSlotType::CALLEE].size();
    size_t max_slot_count = std::max(caller_count, callee_count);

    group_width = (1.5f * GUIUtils::TextWidgetWidth(this->name_label) / in_canvas.zooming) + (3.0f * GUI_SLOT_RADIUS);
    group_height = std::max((3.0f * line_height),
        (line_height + (static_cast<float>(max_slot_count) * (GUI_SLOT_RADIUS * 2.0f) * 1.5f) + GUI_SLOT_RADIUS));

    if (!this->collapsed_view) {
        float pos_maxX = -FLT_MAX;
        float pos_maxY = -FLT_MAX;
        ImVec2 tmp_pos;
        ImVec2 tmp_size;
        for (auto& mod : inout_group.GetModules()) {
            tmp_pos = mod->GUI_GetPosition();
            tmp_size = mod->GUI_GetSize();
            pos_maxX = std::max(tmp_pos.x + tmp_size.x, pos_maxX);
            pos_maxY = std::max(tmp_pos.y + tmp_size.y, pos_maxY);
        }
        group_width = std::max(group_width, (pos_maxX + this->border) - pos_minX);
        group_height = std::max(group_height, (pos_maxY + this->border) - pos_minY);
    }
    // Clamp to minimum size
    this->size = ImVec2(std::max(group_width, 75.0f), std::max(group_height, 25.0f));


    // Set group interface position of call slots --------------------------

    ImVec2 pos = in_canvas.offset + this->position * in_canvas.zooming;
    pos.y += (line_height * in_canvas.zooming);
    ImVec2 size = this->size * in_canvas.zooming;
    size.y -= (line_height * in_canvas.zooming);

    size_t caller_idx = 0;
    size_t callee_idx = 0;
    ImVec2 callslot_group_position;

    for (auto& interfaceslots_map : inout_group.interfaceslots) {
        for (auto& interfaceslot_ptr : interfaceslots_map.second) {
            if (interfaceslots_map.first == CallSlotType::CALLER) {
                callslot_group_position =
                    ImVec2((pos.x + size.x), (pos.y + size.y * ((float)caller_idx + 1) / ((float)caller_count + 1)));
                caller_idx++;
            } else if (interfaceslots_map.first == CallSlotType::CALLEE) {
                callslot_group_position =
                    ImVec2(pos.x, (pos.y + size.y * ((float)callee_idx + 1) / ((float)callee_count + 1)));
                callee_idx++;
            }
            interfaceslot_ptr->GUI_SetPosition(callslot_group_position);
        }
    }
}
