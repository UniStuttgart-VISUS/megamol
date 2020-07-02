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


megamol::gui::configurator::Group::Group(ImGuiID uid) : uid(uid), name(), modules(), callslots(), present() {}


megamol::gui::configurator::Group::~Group() {

    // Reset modules
    for (auto& module_ptr : this->modules) {
        module_ptr->GUI_SetGroupMembership(GUI_INVALID_ID);
        module_ptr->GUI_SetGroupName("");
        module_ptr.reset();
    }
    // Reset call slots
    for (auto& callslot_map : this->callslots) {
        for (auto& callslot_ptr : callslot_map.second) {
            callslot_ptr->GUI_SetGroupInterface(false);
            callslot_ptr.reset();
        }
    }
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
    module_ptr->GUI_SetGroupVisibility(this->present.ModuleVisible());
    module_ptr->GUI_SetGroupName(this->name);
    this->present.ForceUpdate();

    this->restore_callslot_interface_sate();

    vislib::sys::Log::DefaultLog.WriteInfo(
        "Added module '%s' to group '%s'.\n", module_ptr->name.c_str(), this->name.c_str());
    return true;
}


bool megamol::gui::configurator::Group::RemoveModule(ImGuiID module_uid) {

    try {
        for (auto mod_iter = this->modules.begin(); mod_iter != this->modules.end(); mod_iter++) {
            if ((*mod_iter)->uid == module_uid) {

                // Remove call slots belonging to module
                UIDVectorType callslot_uids;
                for (auto& callslot_map : this->callslots) {
                    for (auto& callslot_ptr : callslot_map.second) {
                        if (callslot_ptr->ParentModuleConnected()) {
                            if (callslot_ptr->GetParentModule()->uid == module_uid) {
                                callslot_uids.emplace_back(callslot_ptr->uid);
                            }
                        }
                    }
                }
                for (auto& callslot_uid : callslot_uids) {
                    this->RemoveCallSlot(callslot_uid);
                }

                (*mod_iter)->GUI_SetGroupMembership(GUI_INVALID_ID);
                (*mod_iter)->GUI_SetGroupVisibility(false);
                (*mod_iter)->GUI_SetGroupName("");
                this->present.ForceUpdate();

                vislib::sys::Log::DefaultLog.WriteInfo(
                    "Removed module '%s' from group '%s'.\n", (*mod_iter)->name.c_str(), this->name.c_str());
                (*mod_iter).reset();
                this->modules.erase(mod_iter);

                this->restore_callslot_interface_sate();

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


bool megamol::gui::configurator::Group::AddCallSlot(const CallSlotPtrType& callslot_ptr) {

    if (callslot_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Check if call slot is already part of the group
    for (auto& callslot_map : this->callslots) {
        for (auto callslot_iter = callslot_map.second.begin(); callslot_iter != callslot_map.second.end();
             callslot_iter++) {
            if ((*callslot_iter)->uid == callslot_ptr->uid) {
                vislib::sys::Log::DefaultLog.WriteInfo(
                    "Call Slot '%s' is already part of group '%s'.\n", callslot_ptr->name.c_str(), this->name.c_str());
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
    } else {
        vislib::sys::Log::DefaultLog.WriteError(
            "Call slot has no parent module connected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    if (add) {
        callslot_ptr->GUI_SetGroupInterface(true);

        this->callslots[callslot_ptr->type].emplace_back(callslot_ptr);

        vislib::sys::Log::DefaultLog.WriteInfo(
            "Added call slot '%s' to group '%s'.\n", callslot_ptr->name.c_str(), this->name.c_str());
    } else {
        vislib::sys::Log::DefaultLog.WriteError(
            "Parent module of call slot to add is not part of the group. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
        return false;
    }

    return true;
}


bool megamol::gui::configurator::Group::RemoveCallSlot(ImGuiID callslots_uid) {

    try {
        for (auto& callslot_map : this->callslots) {
            for (auto callslot_iter = callslot_map.second.begin(); callslot_iter != callslot_map.second.end();
                 callslot_iter++) {
                if ((*callslot_iter)->uid == callslots_uid) {

                    (*callslot_iter)->GUI_SetGroupInterface(false);

                    vislib::sys::Log::DefaultLog.WriteInfo("Removed call slot '%s' from group interface '%s'.\n",
                        (*callslot_iter)->name.c_str(), this->name.c_str());
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
        for (auto& callslot_ptr : callslot_map.second) {
            if (callslot_ptr->uid == callslot_uid) {
                return true;
            }
        }
    }
    return false;
}


void megamol::gui::configurator::Group::restore_callslot_interface_sate(void) {

    for (auto& module_ptr : this->modules) {

        // Add connected call slots to group interface if connected module is not part of same group
        /// Caller
        for (auto& callerslot_ptr : module_ptr->GetCallSlots(CallSlotType::CALLER)) {
            if (callerslot_ptr->CallsConnected()) {
                for (auto& call : callerslot_ptr->GetConnectedCalls()) {
                    auto calleeslot_ptr = call->GetCallSlot(CallSlotType::CALLEE);
                    if (calleeslot_ptr->ParentModuleConnected()) {
                        ImGuiID parent_module_group_uid = calleeslot_ptr->GetParentModule()->GUI_GetGroupMembership();
                        if (parent_module_group_uid != this->uid) {
                            this->AddCallSlot(callerslot_ptr);
                        }
                    }
                }
            }
        }
        /// Callee
        for (auto& calleeslot_ptr : module_ptr->GetCallSlots(CallSlotType::CALLEE)) {
            if (calleeslot_ptr->CallsConnected()) {
                for (auto& call : calleeslot_ptr->GetConnectedCalls()) {
                    auto callerslot_ptr = call->GetCallSlot(CallSlotType::CALLER);
                    if (callerslot_ptr->ParentModuleConnected()) {
                        ImGuiID parent_module_group_uid = callerslot_ptr->GetParentModule()->GUI_GetGroupMembership();
                        if (parent_module_group_uid != this->uid) {
                            this->AddCallSlot(calleeslot_ptr);
                        }
                    }
                }
            }
        }
        // Remove connected call slots of group interface if connected module is part of same group
        /// Caller
        for (auto& callerslot_ptr : module_ptr->GetCallSlots(CallSlotType::CALLER)) {
            if (callerslot_ptr->CallsConnected()) {
                for (auto& call : callerslot_ptr->GetConnectedCalls()) {
                    auto calleeslot_ptr = call->GetCallSlot(CallSlotType::CALLEE);
                    if (calleeslot_ptr->ParentModuleConnected()) {
                        ImGuiID parent_module_group_uid = calleeslot_ptr->GetParentModule()->GUI_GetGroupMembership();
                        if (parent_module_group_uid == this->uid) {
                            this->RemoveCallSlot(calleeslot_ptr->uid);
                        }
                    }
                }
            }
        }
        /// Callee
        for (auto& calleeslot_ptr : module_ptr->GetCallSlots(CallSlotType::CALLEE)) {
            if (calleeslot_ptr->CallsConnected()) {
                for (auto& call : calleeslot_ptr->GetConnectedCalls()) {
                    auto callerslot_ptr = call->GetCallSlot(CallSlotType::CALLER);
                    if (callerslot_ptr->ParentModuleConnected()) {
                        ImGuiID parent_module_group_uid = callerslot_ptr->GetParentModule()->GUI_GetGroupMembership();
                        if (parent_module_group_uid == this->uid) {
                            this->RemoveCallSlot(callerslot_ptr->uid);
                        }
                    }
                }
            }
        }
    }
}


// GROUP PRESENTATION ####################################################

megamol::gui::configurator::Group::Presentation::Presentation(void)
    : border(GUI_CALL_SLOT_RADIUS * 2.0f)
    , position(ImVec2(FLT_MAX, FLT_MAX))
    , size(ImVec2(0.0f, 0.0f))
    , utils()
    , name_label()
    , collapsed_view(false)
    , selected(false)
    , update(true) {}


megamol::gui::configurator::Group::Presentation::~Presentation(void) {}


void megamol::gui::configurator::Group::Presentation::Present(
    megamol::gui::configurator::Group& inout_group, GraphItemsStateType& state) {

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
            for (auto& mod : inout_group.GetModules()) {
                mod->GUI_SetGroupVisibility(this->ModuleVisible());
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

        // Colors
        ImVec4 tmpcol = style.Colors[ImGuiCol_ScrollbarBg]; // ImGuiCol_ScrollbarGrab ImGuiCol_FrameBg ImGuiCol_Button
        tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
        const ImU32 COLOR_GROUP_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);

        tmpcol =
            style
                .Colors[ImGuiCol_FrameBg]; // ImGuiCol_ScrollbarGrabHovered ImGuiCol_FrameBgActive ImGuiCol_ButtonActive
        tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
        const ImU32 COLOR_GROUP_HIGHTLIGHT = ImGui::ColorConvertFloat4ToU32(tmpcol);

        tmpcol = style.Colors[ImGuiCol_ScrollbarGrabHovered]; // ImGuiCol_Border ImGuiCol_ScrollbarGrabActive
        tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
        const ImU32 COLOR_GROUP_BORDER = ImGui::ColorConvertFloat4ToU32(tmpcol);

        const ImU32 COLOR_TEXT = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Text]);
        tmpcol = style.Colors[ImGuiCol_FrameBgHovered];
        tmpcol.y = 0.75f;
        const ImU32 COLOR_HEADER = ImGui::ColorConvertFloat4ToU32(tmpcol);
        tmpcol = style.Colors[ImGuiCol_ButtonActive];
        tmpcol.y = 0.75f;
        const ImU32 COLOR_HEADER_HIGHLIGHT = ImGui::ColorConvertFloat4ToU32(tmpcol);

        // Draw box
        ImGui::SetCursorScreenPos(group_rect_min);
        std::string label = "group_" + inout_group.name;

        ImGui::SetItemAllowOverlap();
        ImGui::InvisibleButton(label.c_str(), header_size);
        ImGui::SetItemAllowOverlap();

        bool active = ImGui::IsItemActive();
        bool hovered = (ImGui::IsItemHovered() && (state.interact.callslot_hovered_uid == GUI_INVALID_ID) &&
                        (state.interact.module_hovered_uid == GUI_INVALID_ID));
        bool mouse_clicked = ImGui::IsWindowHovered() && ImGui::GetIO().MouseClicked[0];

        // Automatically delete empty group.
        if (inout_group.GetModules().empty()) {
            std::get<1>(state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM]) = true;
            state.interact.group_selected_uid = inout_group.uid; // Force selection (must be set in same frame)
        }

        // Context menu
        if (ImGui::BeginPopupContextItem("invisible_button_context")) {
            active = true; // Force selection

            ImGui::TextUnformatted("Group");
            ImGui::Separator();
            std::string view = "Collapsed View";
            if (this->collapsed_view) {
                view = "Expanded View";
            }
            if (ImGui::MenuItem(view.c_str())) {
                this->collapsed_view = !this->collapsed_view;
                for (auto& mod : inout_group.GetModules()) {
                    mod->GUI_SetGroupVisibility(this->ModuleVisible());
                }
                this->UpdatePositionSize(inout_group, state.canvas);
            }
            /*
            if (ImGui::MenuItem("Save")) {
                state.interact.group_save = true;
                state.interact.group_selected_uid = inout_group.uid; // Force selection (must be set in same frame)
            }
            */
            if (ImGui::MenuItem("Rename")) {
                popup_rename = true;
            }
            if (ImGui::MenuItem("Delete",
                    std::get<0>(state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM]).ToString().c_str())) {
                std::get<1>(state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM]) = true;
            }
            ImGui::EndPopup();
        }

        // Selection
        if (state.interact.group_selected_uid == inout_group.uid) {
            /// Call before "active" if-statement for one frame delayed check for last valid candidate for selection
            this->selected = true;
            state.interact.callslot_selected_uid = GUI_INVALID_ID;
            state.interact.modules_selected_uids.clear();
            state.interact.call_selected_uid = GUI_INVALID_ID;
        }
        if (active) {
            state.interact.group_selected_uid = inout_group.uid;
        }
        if ((mouse_clicked && !hovered) || (state.interact.group_selected_uid != inout_group.uid)) {
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

        // Rename pop-up ------------------------------------------------------
        if (this->utils.RenamePopUp("Rename Group", popup_rename, inout_group.name)) {
            for (auto& module_ptr : inout_group.GetModules()) {
                module_ptr->GUI_SetGroupName(inout_group.name);
                module_ptr->GUI_Update(state.canvas);
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
    size_t caller_count = inout_group.callslots[CallSlotType::CALLER].size();
    size_t callee_count = inout_group.callslots[CallSlotType::CALLEE].size();
    size_t max_slot_count = std::max(caller_count, callee_count);

    group_width =
        (1.5f * GUIUtils::TextWidgetWidth(this->name_label) / in_canvas.zooming) + (3.0f * GUI_CALL_SLOT_RADIUS);
    group_height = std::max((3.0f * line_height),
        (line_height + (static_cast<float>(max_slot_count) * (GUI_CALL_SLOT_RADIUS * 2.0f) * 1.5f) +
            GUI_CALL_SLOT_RADIUS));

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
    pos.y += line_height;
    ImVec2 size = this->size * in_canvas.zooming;
    size.y -= line_height;

    size_t caller_idx = 0;
    size_t callee_idx = 0;
    ImVec2 callslot_group_position;

    for (auto& callslot_map : inout_group.callslots) {
        for (auto& callslot_ptr : callslot_map.second) {
            if (callslot_map.first == CallSlotType::CALLER) {
                callslot_group_position =
                    ImVec2((pos.x + size.x), (pos.y + size.y * ((float)caller_idx + 1) / ((float)caller_count + 1)));
                caller_idx++;
            } else if (callslot_map.first == CallSlotType::CALLEE) {
                callslot_group_position =
                    ImVec2(pos.x, (pos.y + size.y * ((float)callee_idx + 1) / ((float)callee_count + 1)));
                callee_idx++;
            }
            callslot_ptr->GUI_SetGroupPosition(callslot_group_position);
        }
    }
}
