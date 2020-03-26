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


megamol::gui::configurator::Group::Group(ImGuiID uid) : uid(uid), present() {

}


megamol::gui::configurator::Group::~Group() {  
}


bool megamol::gui::configurator::Group::AddModule(const ModulePtrType& module_ptr) {

    // Check if module was alreday added to group
    for (auto& mod :  this->modules) {
        if (mod->uid == module_ptr->uid) {
            vislib::sys::Log::DefaultLog.WriteInfo("Module '%s' is already part of group '%s'.\n", mod->name.c_str(), this->name.c_str());
            return false; 
        }
    }

    this->modules.emplace_back(module_ptr);
    vislib::sys::Log::DefaultLog.WriteInfo("Added module '%s' to group '%s'.\n", module_ptr->name.c_str(), this->name.c_str() );                          
    return true;
}


bool megamol::gui::configurator::Group::DeleteModule(ImGuiID module_uid) {

    try {
        for (auto iter = this->modules.begin(); iter != this->modules.end(); iter++) {
            if ((*iter)->uid == module_uid) {
                vislib::sys::Log::DefaultLog.WriteInfo("Deleted module '%s' from group '%s'.\n", (*iter)->name.c_str(), this->name.c_str() );  
                (*iter).reset();                        
                this->modules.erase(iter);
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


// GROUP PRESENTATION ####################################################

megamol::gui::configurator::Group::Presentation::Presentation(void) 
    : position(ImVec2(FLT_MAX, FLT_MAX))
    , size(ImVec2(0.0f, 0.0f))
    , utils()
    , name_label()
    , minimized_view(false)
    , selected(false)
    , update_once(true) {

}


megamol::gui::configurator::Group::Presentation::~Presentation(void) {}


void megamol::gui::configurator::Group::Presentation::Present(megamol::gui::configurator::Group& inout_group, StateType& state) {

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
        if (this->update_once || (this->size.x <= 0.0f) || (this->size.y <= 0.0f)) {
            this->UpdateSize(inout_group, state.canvas);
            this->update_once = false;
        }

        // Draw group --------------------------------------------------------
        ImVec2 group_size = this->size * state.canvas.zooming;
        ImVec2 group_rect_min = state.canvas.offset + this->position * state.canvas.zooming;
        ImVec2 group_rect_max = group_rect_min + group_size;
        ImVec2 group_center = group_rect_min + ImVec2(group_size.x / 2.0f, group_size.y / 2.0f);
       
        ImGui::PushID(inout_group.uid);

        // Colors
        ImVec4 tmpcol = style.Colors[ImGuiCol_FrameBg]; // ImGuiCol_FrameBg ImGuiCol_Button
        tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
        const ImU32 COLOR_GROUP_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);
        tmpcol = style.Colors[ImGuiCol_FrameBgActive]; // ImGuiCol_FrameBgActive ImGuiCol_ButtonActive
        tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
        const ImU32 COLOR_GROUP_HIGHTLIGHT = ImGui::ColorConvertFloat4ToU32(tmpcol);
        tmpcol = style.Colors[ImGuiCol_ScrollbarGrabActive]; // ImGuiCol_Border ImGuiCol_ScrollbarGrabActive
        tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
        const ImU32 COLOR_GROUP_BORDER = ImGui::ColorConvertFloat4ToU32(tmpcol);

       // Draw box
        ImGui::SetCursorScreenPos(group_rect_min);
        std::string label = "group_" + inout_group.name;

        ImGui::SetItemAllowOverlap();        
        ImGui::InvisibleButton(label.c_str(), group_size);
        ImGui::SetItemAllowOverlap();

        bool active = ImGui::IsItemActive();
        bool hovered = (ImGui::IsItemHovered() && (state.interact.callslot_hovered_uid == GUI_INVALID_ID));
        bool mouse_clicked = ImGui::IsWindowHovered() && ImGui::GetIO().MouseClicked[0];

        // Context menu
        if (ImGui::BeginPopupContextItem("invisible_button_context")) {

            if (ImGui::MenuItem("Rename", nullptr, false, false)) {
                /// TODO
            }             
            if (ImGui::MenuItem("Collapse", nullptr, this->minimized_view)) {
                this->minimized_view = true;
                this->UpdateSize(inout_group, state.canvas);
            }              
            if (ImGui::MenuItem("Expand", nullptr, !this->minimized_view)) {
                this->minimized_view = false;
                this->UpdateSize(inout_group, state.canvas);
            }
            if (ImGui::MenuItem("Save Group", nullptr, false, false)) {
                /// TODO
                // --confGroupInterface={<call_slot_names>}
            }          
            if (ImGui::MenuItem("Delete Group", nullptr, false, false)) {
                /// TODO
            }                                        
            ImGui::EndPopup();
        }     
        
        if ((mouse_clicked && !hovered) || (state.interact.item_selected_uid != inout_group.uid)) {
            this->selected = false;
            if (state.interact.item_selected_uid == inout_group.uid) {
                state.interact.item_selected_uid = GUI_INVALID_ID;
            }
        }     
        if (active) {
            this->selected = true;
            state.interact.item_selected_uid = inout_group.uid;
            state.interact.item_selected_uid = GUI_INVALID_ID;
        }
        if (this->selected && ImGui::IsWindowHovered() && ImGui::IsMouseDragging(0)) {
            this->position =
                ((group_rect_min - state.canvas.offset) + ImGui::GetIO().MouseDelta) / state.canvas.zooming;
            this->UpdateSize(inout_group, state.canvas);
        }                    
    
        ImU32 group_bg_color = (hovered || this->selected) ? COLOR_GROUP_HIGHTLIGHT : COLOR_GROUP_BACKGROUND;
        draw_list->AddRectFilled(group_rect_min, group_rect_max, group_bg_color, 5.0f);
        draw_list->AddRect(group_rect_min, group_rect_max, COLOR_GROUP_BORDER, 5.0f);

        // Draw text
        if (this->minimized_view) {
            float name_width = this->utils.TextWidgetWidth(this->name_label);
            ImGui::SetCursorScreenPos(group_center + ImVec2(-(name_width / 2.0f), 0.0f));
            ImGui::Text(this->name_label.c_str());
        }
        else {
            ImGui::SetCursorScreenPos(group_rect_min);
            ImGui::Text(this->name_label.c_str());

        }
        // Rename pop-up ------------------------------------------------------
        if (this->utils.RenamePopUp("Rename Group", popup_rename, inout_group.name)) {
            this->UpdateSize(inout_group, state.canvas);
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


void megamol::gui::configurator::Group::Presentation::UpdateSize(
    megamol::gui::configurator::Group& inout_group, const CanvasType& in_canvas) {

    float group_width = 0.0f;
    float group_height = 0.0f;
    if (this->minimized_view) {
        this->name_label = "Group: " + inout_group.name ;
        group_width = this->utils.TextWidgetWidth(this->name_label);
        group_width /= in_canvas.zooming;
        group_height = (1.0f / in_canvas.zooming) * (ImGui::GetTextLineHeightWithSpacing() * 2.0f);
    }
    else {
        const float border = 25.0f;
        float minX = FLT_MAX;
        float maxX  = -FLT_MAX;
        float minY = FLT_MAX; 
        float maxY = -FLT_MAX;
        ImVec2 tmp_size;
        for (auto& mod : inout_group.GetGroupModules()) {
            tmp_size = mod->GUI_GetSize();
            minX = std::min(tmp_size.x, minX);
            maxX = std::max(tmp_size.x, maxX);
            minY = std::min(tmp_size.y, minY);
            maxY = std::max(tmp_size.y, maxY);
        }
        group_width = (maxX - minX) + (2.0f*border);
        group_height = (maxY - minY) + (2.0f*border);
    }
    // Clamp to minimum size
    this->size = ImVec2(std::max(group_width, 75.0f), std::max(group_height, 25.0f));    
}