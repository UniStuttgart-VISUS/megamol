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


megamol::gui::configurator::Group::Group(ImGuiID uid) : uid(uid), present(), modules(), callslots() {
}


megamol::gui::configurator::Group::~Group() { 
    
    // Reset modules
    for (auto& mod : this->modules) {
        
        mod->GUI_SetVisibility(true);
        mod->name_space.clear(); 
        mod.reset();             
    }
    // Reset call slots
    for (auto& callslot_map : this->callslots) {
        for (auto& callslot : callslot_map.second) {
            callslot->GUI_SetInterfaceView(false);
            callslot.reset();            
        }
    }
}


bool megamol::gui::configurator::Group::AddModule(const ModulePtrType& module_ptr) {

    if (module_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("Pointer to module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;  
    }

    // Check if module is already part of the group
    for (auto& mod : this->modules) {
        if (mod->uid == module_ptr->uid) {
            vislib::sys::Log::DefaultLog.WriteInfo("Module '%s' is already part of group '%s'.\n", mod->name.c_str(), this->name.c_str());
            return false;
        }

    }

    this->modules.emplace_back(module_ptr);
    
    module_ptr->name_space = this->name;
    module_ptr->GUI_SetVisibility(this->present.ModuleVisible());
    this->present.ApplyUpdate();
    
    vislib::sys::Log::DefaultLog.WriteInfo("Added module '%s' to group '%s'.\n", module_ptr->name.c_str(), this->name.c_str() );                          
    return true;
}


bool megamol::gui::configurator::Group::RemoveModule(ImGuiID module_uid) {

    try {        
        for (auto mod_iter = this->modules.begin(); mod_iter != this->modules.end(); mod_iter++) {
            if ((*mod_iter)->uid == module_uid) {
 
                // Remove call slots belonging to module
                std::vector<ImGuiID> callslot_uids;
                for (auto& callslot_map : this->callslots) {
                    for (auto& callslot : callslot_map.second) {
                        if (callslot->ParentModuleConnected()) {
                            if (callslot->GetParentModule()->uid == module_uid) {
                                callslot_uids.emplace_back(callslot->uid);
                            }
                        }
                    }                  
                }
                for (auto& callslot_uid : callslot_uids) {
                    this->RemoveCallSlot(callslot_uid);
                }
                
                (*mod_iter)->GUI_SetVisibility(true);
                (*mod_iter)->name_space.clear(); 
                
                vislib::sys::Log::DefaultLog.WriteInfo("Removed module '%s' from group '%s'.\n", (*mod_iter)->name.c_str(), this->name.c_str());
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

    for (auto& mod :  this->modules) {
        if (mod->uid == module_uid) {
            return true; 
        }
    }
    return false;
}


bool megamol::gui::configurator::Group::AddCallSlot(const CallSlotPtrType& callslot_ptr) {

    if (callslot_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("Pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;  
    }

    // Check if call slot is already part of the group
    for (auto& callslot_map : this->callslots) {
        for (auto callslot_iter = callslot_map.second.begin(); callslot_iter != callslot_map.second.end(); callslot_iter++) {
            if ((*callslot_iter)->uid == callslot_ptr->uid) {
                vislib::sys::Log::DefaultLog.WriteInfo("Call Slot '%s' is already part of group '%s'.\n", callslot_ptr->name.c_str(), this->name.c_str());
                return false;
            }
        }
    }
    
    // Only add if parent module is already part of the group.
    bool add = false;
    if (callslot_ptr->ParentModuleConnected()) {
        for (auto mod_iter = this->modules.begin(); mod_iter != this->modules.end(); mod_iter++) {
            if (callslot_ptr->GetParentModule()->uid == (*mod_iter)->uid) {
                add = true;
            }
        }
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("Call slot has no parent module connected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false; 
    }
    
    if (add) {
        this->callslots[callslot_ptr->type].emplace_back(callslot_ptr);
        
        callslot_ptr->GUI_SetInterfaceView(true);
        vislib::sys::Log::DefaultLog.WriteInfo("Added call slot '%s' to group '%s'.\n", callslot_ptr->name.c_str(), this->name.c_str() );  
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("Parent module of call slot to add is not part of the group. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false; 
    }
                 
    return true;
}


bool megamol::gui::configurator::Group::RemoveCallSlot(ImGuiID callslots_uid) {

    try {
        for (auto& callslot_map : this->callslots) {
            for (auto callslot_iter = callslot_map.second.begin(); callslot_iter != callslot_map.second.end(); callslot_iter++) {
                if ((*callslot_iter)->uid == callslots_uid) {
                    
                    (*callslot_iter)->GUI_SetInterfaceView(false);
                    
                    vislib::sys::Log::DefaultLog.WriteInfo("Removed call slot '%s' from group interface '%s'.\n", (*callslot_iter)->name.c_str(), this->name.c_str());  
                    (*callslot_iter).reset();                        
                    callslot_map.second.erase(callslot_iter);
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


bool megamol::gui::configurator::Group::ContainsCallSlot(ImGuiID callslot_uid) {

    for (auto& callslot_map : this->callslots) {
        for (auto& callslot : callslot_map.second) { 
            if (callslot->uid == callslot_uid) {
                return true; 
            }
        }
    }
    return false;
}


// GROUP PRESENTATION ####################################################

megamol::gui::configurator::Group::Presentation::Presentation(void) 
    : BORDER(10.0f)
    , position(ImVec2(FLT_MAX, FLT_MAX))
    , size(ImVec2(0.0f, 0.0f))
    , utils()
    , name_label()
    , collapsed_view(false)
    , selected(false)
    , update(true) {

}


megamol::gui::configurator::Group::Presentation::~Presentation(void) {}


void megamol::gui::configurator::Group::Presentation::Present(megamol::gui::configurator::Group& inout_group, GraphItemsStateType& state) {

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
        // Update size and position if current values are invalid or in expanded view
        if (this->update || !this->collapsed_view || (this->size.x <= 0.0f) || (this->size.y <= 0.0f)) {
            this->UpdatePositionSize(inout_group, state.canvas);
            if (this->update) {
                for (auto& mod : inout_group.GetModules()) {
                    mod->GUI_SetVisibility(!this->collapsed_view);
                }
            }
            this->update = false;
        }

        // Draw group --------------------------------------------------------
        ImVec2 group_size = this->size * state.canvas.zooming;
        ImVec2 group_rect_min = state.canvas.offset + this->position * state.canvas.zooming;
        ImVec2 group_rect_max = group_rect_min + group_size;
        ImVec2 group_center = group_rect_min + ImVec2(group_size.x / 2.0f, group_size.y / 2.0f);
       
        ImGui::PushID(inout_group.uid);

        // Colors
        ImVec4 tmpcol = style.Colors[ImGuiCol_ScrollbarBg]; // ImGuiCol_ScrollbarGrab ImGuiCol_FrameBg ImGuiCol_Button
        tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
        const ImU32 COLOR_GROUP_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);

        tmpcol = style.Colors[ImGuiCol_FrameBg]; // ImGuiCol_ScrollbarGrabHovered ImGuiCol_FrameBgActive ImGuiCol_ButtonActive
        tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
        const ImU32 COLOR_GROUP_HIGHTLIGHT = ImGui::ColorConvertFloat4ToU32(tmpcol);

        tmpcol = style.Colors[ImGuiCol_ScrollbarGrabHovered]; // ImGuiCol_Border ImGuiCol_ScrollbarGrabActive
        tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
        const ImU32 COLOR_GROUP_BORDER = ImGui::ColorConvertFloat4ToU32(tmpcol);

       // Draw box
        ImGui::SetCursorScreenPos(group_rect_min);
        std::string label = "group_" + inout_group.name;

        ImGui::SetItemAllowOverlap();        
        ImGui::InvisibleButton(label.c_str(), group_size);
        ImGui::SetItemAllowOverlap();

        bool active = ImGui::IsItemActive();
        bool hovered = (ImGui::IsItemHovered() && (state.interact.callslot_hovered_uid == GUI_INVALID_ID) && (state.interact.module_hovered_uid == GUI_INVALID_ID));
        bool mouse_clicked = ImGui::IsWindowHovered() && ImGui::GetIO().MouseClicked[0];

        // Automatically delete empty group.
        if (inout_group.GetModules().empty()) {
            std::get<1>(state.hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM]) = true;
            // Force selection
            active = true; 
        }

        // Context menu
        if (ImGui::BeginPopupContextItem("invisible_button_context")) {
            ImGui::Text("Group");
            ImGui::Separator();
            std::string view = "Collapsed View";
            if (this->collapsed_view) {
                view = "Expanded View";
            }
            if (ImGui::MenuItem(view.c_str())) {
                this->collapsed_view = !this->collapsed_view;
                for (auto& mod : inout_group.GetModules()) {
                     mod->GUI_SetVisibility(this->ModuleVisible());
                }
                this->UpdatePositionSize(inout_group, state.canvas);
            }              
            if (ImGui::MenuItem("Save")) {
                state.interact.group_save = true;
                // Force selection
                active = true;                 
            }
            if (ImGui::MenuItem("Rename")) {
                popup_rename = true;
            }            
            if (ImGui::MenuItem("Delete", std::get<0>(state.hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM]).ToString().c_str())) {
                std::get<1>(state.hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM]) = true;
                // Force selection
                active = true; 
            }                            
            ImGui::EndPopup();
        }     

        if ((mouse_clicked && !hovered) || (state.interact.group_selected_uid != inout_group.uid)) {
            this->selected = false;
            if (state.interact.group_selected_uid == inout_group.uid) {
                state.interact.group_selected_uid = GUI_INVALID_ID;
            }
        }     
        if (active) {
            this->selected = true;
            state.interact.group_selected_uid = inout_group.uid;
            state.interact.callslot_selected_uid = GUI_INVALID_ID;
            state.interact.module_selected_uid = GUI_INVALID_ID;
            state.interact.call_selected_uid = GUI_INVALID_ID;            
        }
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
    
        ImU32 group_bg_color = this->selected ? COLOR_GROUP_HIGHTLIGHT : COLOR_GROUP_BACKGROUND;
        draw_list->AddRectFilled(group_rect_min, group_rect_max, group_bg_color, 0.0f);
        draw_list->AddRect(group_rect_min, group_rect_max, COLOR_GROUP_BORDER, 0.0f);

        // Draw text
        if (this->collapsed_view) {
            float name_width = this->utils.TextWidgetWidth(this->name_label);
            ImVec2 text_pos_left_upper = (group_center + ImVec2(-(name_width / 2.0f), -0.5f * ImGui::GetTextLineHeightWithSpacing()));
            draw_list->AddText(text_pos_left_upper, ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Text]), this->name_label.c_str());
        }
        else {
            ImVec2 text_pos_left_upper = group_rect_min + ImVec2(this->BORDER, this->BORDER) * state.canvas.zooming;
            draw_list->AddText(text_pos_left_upper, ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Text]), this->name_label.c_str());
        }
        // Rename pop-up ------------------------------------------------------
        if (this->utils.RenamePopUp("Rename Group", popup_rename, inout_group.name)) {
            for (auto& mod : inout_group.GetModules()) {
                mod->name_space = inout_group.name;
                mod->GUI_Update(state.canvas);
            }
            this->UpdatePositionSize(inout_group, state.canvas);
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


void megamol::gui::configurator::Group::Presentation::UpdatePositionSize(
    megamol::gui::configurator::Group& inout_group, const GraphCanvasType& in_canvas) {

    this->name_label = "Group: " + inout_group.name;

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
        pos_minX -= this->BORDER;
        pos_minY -= (this->BORDER + (1.5f * ImGui::GetTextLineHeightWithSpacing() / in_canvas.zooming));
        this->position = ImVec2(pos_minX, pos_minY);
    }
    else {
        this->position = ImVec2(10.0f, 10.0f) + (ImGui::GetWindowPos() - in_canvas.offset) / in_canvas.zooming;
    }

    // SIZE
    float group_width = 0.0f;
    float group_height = 0.0f;
    if (this->collapsed_view) {
        
        group_width = 1.5f * this->utils.TextWidgetWidth(this->name_label) / in_canvas.zooming;
        group_height = 3.0f * ImGui::GetTextLineHeightWithSpacing() / in_canvas.zooming;
    }
    else {
        float pos_maxX  = -FLT_MAX;
        float pos_maxY = -FLT_MAX;
        ImVec2 tmp_pos;
        ImVec2 tmp_size;
        for (auto& mod : inout_group.GetModules()) {
            tmp_pos = mod->GUI_GetPosition();
            tmp_size = mod->GUI_GetSize();
            pos_maxX = std::max(tmp_pos.x + tmp_size.x, pos_maxX);
            pos_maxY = std::max(tmp_pos.y + tmp_size.y, pos_maxY);
        }

        group_width = (pos_maxX + this->BORDER) - pos_minX;
        group_height = (pos_maxY + this->BORDER) - pos_minY;
    }
    // Clamp to minimum size
    this->size = ImVec2(std::max(group_width, 75.0f), std::max(group_height, 25.0f));  


}
