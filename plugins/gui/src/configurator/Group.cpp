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

megamol::gui::configurator::Group::Presentation::Presentation(void) {

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

    try {
        ImGui::PushID(inout_group.uid);
        /*

        ImGui::InvisibleButton(label.c_str(), ImVec2(radius * 2.0f, radius * 2.0f));
        ImGui::SetItemAllowOverlap();

        bool active = ImGui::IsItemActive();
        bool hovered = ImGui::IsItemHovered() && (state.interact.callslot_hovered_uid == GUI_INVALID_ID) && (state.interact.module_hovered_uid == GUI_INVALID_ID)));
        bool mouse_clicked = ImGui::IsWindowHovered() && ImGui::GetIO().MouseClicked[0];

        // Context menu
        if (ImGui::BeginPopupContextItem("invisible_button_context")) {
            if (ImGui::MenuItem("Rename", nullptr, false, false)) {
                /// TODO
            }             
            if (ImGui::MenuItem("Collapse", nullptr, false, false)) {
                /// TODO
            }              
            if (ImGui::MenuItem("Expand", nullptr, false, false)) {
                /// TODO
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


        */
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

