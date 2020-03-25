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


megamol::gui::configurator::Group::~Group() {  }



// GROUP PRESENTATION ####################################################

megamol::gui::configurator::Group::Presentation::Presentation(void) {

}


megamol::gui::configurator::Group::Presentation::~Presentation(void) {}


void megamol::gui::configurator::Group::Presentation::Present(megamol::gui::configurator::Group& inout_group,
    const CanvasType& in_canvas, megamol::gui::HotKeyArrayType& inout_hotkeys,
    megamol::gui::InteractType& interact_state) {

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
        bool hovered = ImGui::IsItemHovered() && (interact_state.callslot_hovered_uid == GUI_INVALID_ID) && (interact_state.module_hovered_uid == GUI_INVALID_ID)));
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

