/*
 * Group.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */


#include "Group.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::Group::Group(ImGuiID uid)
        : uid(uid)
        , name()
        , modules()
        , interfaceslots()
        , gui_selected(false)
        , gui_position(ImVec2(FLT_MAX, FLT_MAX))
        , gui_size(ImVec2(0.0f, 0.0f))
        , gui_collapsed_view(false)
        , gui_allow_selection(false)
        , gui_update(true)
        , gui_position_bottom_center(ImVec2())
        , gui_rename_popup() {

    this->interfaceslots.emplace(megamol::gui::CallSlotType::CALLER, InterfaceSlotPtrVector_t());
    this->interfaceslots.emplace(megamol::gui::CallSlotType::CALLEE, InterfaceSlotPtrVector_t());
}


megamol::gui::Group::~Group() {

    // Remove all modules from group
    std::vector<ImGuiID> module_uids;
    for (auto& module_ptr : this->modules) {
        module_uids.emplace_back(module_ptr->UID());
    }
    for (auto& module_uid : module_uids) {
        this->RemoveModule(module_uid, false);
    }
    this->modules.clear();

    // Remove all interface slots from group (should already be empty)
    this->interfaceslots[CallSlotType::CALLER].clear();
    this->interfaceslots[CallSlotType::CALLEE].clear();
}


bool megamol::gui::Group::AddModule(const ModulePtr_t& module_ptr) {

    if (module_ptr == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Pointer to module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Check if module is already part of the group
    for (auto& mod : this->modules) {
        if (mod->UID() == module_ptr->UID()) {
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[GUI] Module '%s' is already part of group '%s'.\n", mod->Name().c_str(), this->name.c_str());
#endif // GUI_VERBOSE
            return false;
        }
    }

    this->modules.emplace_back(module_ptr);

    module_ptr->SetGroupUID(this->uid);
    module_ptr->SetHidden(this->gui_collapsed_view);
    module_ptr->SetGroupName(this->name);

    this->gui_update = true;
    this->RestoreInterfaceslots();

#ifdef GUI_VERBOSE
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
        "[GUI] Added module '%s' to group '%s'.\n", module_ptr->Name().c_str(), this->name.c_str());
#endif // GUI_VERBOSE
    return true;
}


bool megamol::gui::Group::RemoveModule(ImGuiID module_uid, bool restore_interface) {

    try {
        for (auto mod_iter = this->modules.begin(); mod_iter != this->modules.end(); mod_iter++) {
            if ((*mod_iter)->UID() == module_uid) {

                // Remove call slots from group interface
                for (auto& callslot_map : (*mod_iter)->CallSlots()) {
                    for (auto& callslot_ptr : callslot_map.second) {
                        this->InterfaceSlot_RemoveCallSlot(callslot_ptr->UID(), true);
                    }
                }

                (*mod_iter)->SetGroupUID(GUI_INVALID_ID);
                (*mod_iter)->SetHidden(false);
                (*mod_iter)->SetGroupName("");

#ifdef GUI_VERBOSE
                megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                    "[GUI] Removed module '%s' from group '%s'.\n", (*mod_iter)->Name().c_str(), this->name.c_str());
#endif // GUI_VERBOSE
                (*mod_iter).reset();
                this->modules.erase(mod_iter);

                if (restore_interface) {
                    /// E.g. RestoreInterfaceslots() should only be triggered,
                    /// after connected calls of deleted module are deleted.
                    this->RestoreInterfaceslots();
                }
                this->gui_update = true;

                return true;
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

    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
        "[GUI] Invalid module uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}


bool megamol::gui::Group::ContainsModule(ImGuiID module_uid) {

    for (auto& mod : this->modules) {
        if (mod->UID() == module_uid) {
            return true;
        }
    }
    return false;
}


InterfaceSlotPtr_t megamol::gui::Group::AddInterfaceSlot(const CallSlotPtr_t& callslot_ptr, bool auto_add) {

    if (callslot_ptr == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return nullptr;
    }

    // Check if call slot is already part of the group
    for (auto& interfaceslot_ptr : this->interfaceslots[callslot_ptr->Type()]) {
        if (interfaceslot_ptr->ContainsCallSlot(callslot_ptr->UID())) {
            return interfaceslot_ptr;
        }
    }

    // Only add if parent module is already part of the group.
    bool parent_module_group_uid = false;
    if (callslot_ptr->IsParentModuleConnected()) {
        ImGuiID parent_module_uid = callslot_ptr->GetParentModule()->UID();
        for (auto& module_ptr : this->modules) {
            if (parent_module_uid == module_ptr->UID()) {
                parent_module_group_uid = true;
            }
        }
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Call slot has no parent module connected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return nullptr;
    }

    if (parent_module_group_uid) {
        InterfaceSlotPtr_t interfaceslot_ptr =
            std::make_shared<InterfaceSlot>(megamol::gui::GenerateUniqueID(), auto_add);
        if (interfaceslot_ptr != nullptr) {
            interfaceslot_ptr->SetGroupUID(this->uid);
            this->interfaceslots[callslot_ptr->Type()].emplace_back(interfaceslot_ptr);
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[GUI] Added interface slot (uid %i) to group '%s'.\n", interfaceslot_ptr->UID(), this->name.c_str());
#endif // GUI_VERBOSE

            if (interfaceslot_ptr->AddCallSlot(callslot_ptr, interfaceslot_ptr)) {

                interfaceslot_ptr->SetGroupViewCollapsed(this->gui_collapsed_view);
                this->gui_update = true;

                return interfaceslot_ptr;
            }
        }
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Parent module of call slot which should be added to group interface "
            "is not part of any group. [%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        return nullptr;
    }
    return nullptr;
}


bool megamol::gui::Group::InterfaceSlot_RemoveCallSlot(ImGuiID callslots_uid, bool force) {

    bool retval = false;
    try {
        std::vector<ImGuiID> empty_interfaceslots_uids;
        for (auto& interfaceslot_map : this->interfaceslots) {
            for (auto& interfaceslot_ptr : interfaceslot_map.second) {
                if ((interfaceslot_ptr->IsAutoCreated() || force) &&
                    interfaceslot_ptr->ContainsCallSlot(callslots_uid)) {
                    interfaceslot_ptr->RemoveCallSlot(callslots_uid);
                    retval = true;
                    if (interfaceslot_ptr->IsEmpty()) {
                        empty_interfaceslots_uids.emplace_back(interfaceslot_ptr->UID());
                    }
                }
            }
        }
        // Delete empty interface slots
        for (auto& interfaceslot_uid : empty_interfaceslots_uids) {
            this->DeleteInterfaceSlot(interfaceslot_uid);
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
    return retval;
}


bool megamol::gui::Group::InterfaceSlot_ContainsCallSlot(ImGuiID callslot_uid) {

    for (auto& interfaceslots_map : this->interfaceslots) {
        for (auto& interfaceslot_ptr : interfaceslots_map.second) {
            if (interfaceslot_ptr->ContainsCallSlot(callslot_uid)) {
                return true;
            }
        }
    }
    return false;
}


InterfaceSlotPtr_t megamol::gui::Group::InterfaceSlotPtr(ImGuiID interfaceslot_uid) {

    if (interfaceslot_uid != GUI_INVALID_ID) {
        for (auto& interfaceslots_map : this->interfaceslots) {
            for (auto& interfaceslot : interfaceslots_map.second) {
                if (interfaceslot->UID() == interfaceslot_uid) {
                    return interfaceslot;
                }
            }
        }
    }
    return nullptr;
}


bool megamol::gui::Group::DeleteInterfaceSlot(ImGuiID interfaceslot_uid) {

    if (interfaceslot_uid != GUI_INVALID_ID) {
        for (auto& interfaceslot_map : this->interfaceslots) {
            for (auto iter = interfaceslot_map.second.begin(); iter != interfaceslot_map.second.end(); iter++) {
                if ((*iter)->UID() == interfaceslot_uid) {

                    // Remove all call slots from interface slot
                    std::vector<ImGuiID> callslots_uids;
                    for (auto& callslot_ptr : (*iter)->CallSlots()) {
                        callslots_uids.emplace_back(callslot_ptr->UID());
                    }
                    for (auto& callslot_uid : callslots_uids) {
                        (*iter)->RemoveCallSlot(callslot_uid);
                    }

                    if ((*iter).use_count() > 1) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] Unclean deletion. Found %i references pointing to interface slot. [%s, %s, line "
                            "%d]\n",
                            (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                    }

#ifdef GUI_VERBOSE
                    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                        "[GUI] Deleted interface slot (uid %i) from group '%s'.\n", (*iter)->UID(), this->name.c_str());
#endif // GUI_VERBOSE

                    (*iter).reset();
                    interfaceslot_map.second.erase(iter);

                    this->gui_update = true;

                    return true;
                }
            }
        }
    }
    return false;
}


void megamol::gui::Group::RestoreInterfaceslots() {

    /// 1] REMOVE connected call slots of group interface if connected module is part of same group
    for (auto& module_ptr : this->modules) {
        // CALLER
        for (auto& callerslot_ptr : module_ptr->CallSlots(CallSlotType::CALLER)) {
            for (auto& call_ptr : callerslot_ptr->GetConnectedCalls()) {
                auto calleeslot_ptr = call_ptr->CallSlotPtr(CallSlotType::CALLEE);
                if (calleeslot_ptr->IsParentModuleConnected()) {
                    ImGuiID parent_module_group_uid = calleeslot_ptr->GetParentModule()->GroupUID();
                    if (parent_module_group_uid == this->uid) {
                        this->InterfaceSlot_RemoveCallSlot(calleeslot_ptr->UID());
                    }
                }
            }
        }
        // CALLEE
        for (auto& calleeslot_ptr : module_ptr->CallSlots(CallSlotType::CALLEE)) {
            for (auto& call_ptr : calleeslot_ptr->GetConnectedCalls()) {
                auto callerslot_ptr = call_ptr->CallSlotPtr(CallSlotType::CALLER);
                if (callerslot_ptr->IsParentModuleConnected()) {
                    ImGuiID parent_module_group_uid = callerslot_ptr->GetParentModule()->GroupUID();
                    if (parent_module_group_uid == this->uid) {
                        this->InterfaceSlot_RemoveCallSlot(callerslot_ptr->UID());
                    }
                }
            }
        }
    }

    /// 2] ADD connected call slots to group interface if connected module is not part of same group
    for (auto& module_ptr : this->modules) {
        // CALLER
        for (auto& callerslot_ptr : module_ptr->CallSlots(CallSlotType::CALLER)) {
            for (auto& call_ptr : callerslot_ptr->GetConnectedCalls()) {
                auto calleeslot_ptr = call_ptr->CallSlotPtr(CallSlotType::CALLEE);
                if (calleeslot_ptr->IsParentModuleConnected()) {
                    ImGuiID parent_module_group_uid = calleeslot_ptr->GetParentModule()->GroupUID();
                    if (parent_module_group_uid != this->uid) {
                        this->AddInterfaceSlot(callerslot_ptr);
                    }
                }
            }
        }
        // CALLEE
        for (auto& calleeslot_ptr : module_ptr->CallSlots(CallSlotType::CALLEE)) {
            for (auto& call_ptr : calleeslot_ptr->GetConnectedCalls()) {
                auto callerslot_ptr = call_ptr->CallSlotPtr(CallSlotType::CALLER);
                if (callerslot_ptr->IsParentModuleConnected()) {
                    ImGuiID parent_module_group_uid = callerslot_ptr->GetParentModule()->GroupUID();
                    if (parent_module_group_uid != this->uid) {
                        this->AddInterfaceSlot(calleeslot_ptr);
                    }
                }
            }
        }
    }
}


void megamol::gui::Group::Draw(megamol::gui::PresentPhase phase, GraphItemsState_t& state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    ImGuiStyle& style = ImGui::GetStyle();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    try {
        // Update size and position if current values are invalid or in expanded view
        /// XXX First condition calls Update() every frame if not collapsed
        if (!this->gui_collapsed_view || this->gui_update || (this->gui_size.x <= 0.0f) || (this->gui_size.y <= 0.0f)) {
            this->Update(state.canvas);
            for (auto& mod : this->modules) {
                mod->SetHidden(this->gui_collapsed_view);
            }
            this->gui_update = false;
        }

        // Draw group --------------------------------------------------------

        ImVec2 group_size = this->gui_size * state.canvas.zooming;
        ImVec2 group_rect_min = state.canvas.offset + this->gui_position * state.canvas.zooming;
        ImVec2 group_rect_max = group_rect_min + group_size;
        ImVec2 group_center = group_rect_min + ImVec2(group_size.x / 2.0f, group_size.y / 2.0f);
        this->gui_position_bottom_center = group_rect_min + ImVec2(group_size.x / 2.0f, group_size.y);
        ImVec2 header_size = ImVec2(group_size.x, ImGui::GetTextLineHeightWithSpacing());
        ImVec2 header_rect_max = group_rect_min + header_size;

        ImGui::PushID(static_cast<int>(this->uid));

        bool changed_view = false;

        if (phase == megamol::gui::PresentPhase::INTERACTION) {

            // Limit selection to header
            this->gui_allow_selection = false;
            ImVec2 mouse_pos = ImGui::GetMousePos();
            if ((mouse_pos.x >= group_rect_min.x) && (mouse_pos.y >= group_rect_min.y) &&
                (mouse_pos.x <= header_rect_max.x) && (mouse_pos.y <= header_rect_max.y)) {
                this->gui_allow_selection = true;
            }

            // Button
            std::string button_label = "group_" + std::to_string(this->uid);
            ImGui::SetCursorScreenPos(group_rect_min);
            ImGui::SetItemAllowOverlap();
            ImGui::InvisibleButton(button_label.c_str(), group_size, ImGuiButtonFlags_NoSetKeyOwner);
            ImGui::SetItemAllowOverlap();
            if (ImGui::IsItemActivated()) {
                state.interact.button_active_uid = this->uid;
            }
            if (ImGui::IsItemHovered()) {
                state.interact.button_hovered_uid = this->uid;
            }

            ImGui::PushFont(state.canvas.gui_font_ptr);

            // Context menu
            bool popup_rename = false;
            if (ImGui::BeginPopupContextItem("invisible_button_context")) { /// this->allow_context &&

                state.interact.button_active_uid = this->uid;

                ImGui::TextDisabled("Group");
                ImGui::Separator();

                std::string view("Collapse");
                if (this->gui_collapsed_view) {
                    view = "Expand";
                }
                if (ImGui::MenuItem(view.c_str(), "'Double-Click' Header")) {
                    this->gui_collapsed_view = !this->gui_collapsed_view;
                    changed_view = true;
                }
                if (ImGui::MenuItem("Layout Modules")) {
                    state.interact.group_layout = true;
                }
                if (ImGui::MenuItem("Rename")) {
                    popup_rename = true;
                }
                if (ImGui::MenuItem(
                        "Delete", state.hotkeys[HOTKEY_CONFIGURATOR_DELETE_GRAPH_ITEM].keycode.ToString().c_str())) {
                    state.interact.process_deletion = true;
                }
                ImGui::EndPopup();
            } /// else { this->allow_context = false; }

            // Rename pop-up
            if (this->gui_rename_popup.Rename("Rename Group", popup_rename, this->name)) {
                for (auto& module_ptr : this->modules) {
                    std::string last_module_name = module_ptr->FullName();
                    module_ptr->SetGroupName(this->name);
                    module_ptr->Update();
                    if (state.interact.graph_is_running) {
                        state.interact.module_rename.push_back(StrPair_t(last_module_name, module_ptr->FullName()));
                    }
                }
                this->Update(state.canvas);
            }

            ImGui::PopFont();

        } else if (phase == megamol::gui::PresentPhase::RENDERING) {

            bool active = (state.interact.button_active_uid == this->uid);
            bool hovered = (state.interact.button_hovered_uid == this->uid);
            bool mouse_clicked_anywhere =
                ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiPopupFlags_MouseButtonLeft);

            // Hovering
            if (hovered) {
                state.interact.group_hovered_uid = this->uid;
            }
            if (!hovered && (state.interact.group_hovered_uid == this->uid)) {
                state.interact.group_hovered_uid = GUI_INVALID_ID;
            }

            // Adjust state for selection
            active = active && this->gui_allow_selection;
            hovered = hovered && this->gui_allow_selection;
            this->gui_allow_selection = false;
            // Selection
            if (!this->gui_selected && active) {
                state.interact.group_selected_uid = this->uid;
                this->gui_selected = true;
                state.interact.callslot_selected_uid = GUI_INVALID_ID;
                state.interact.modules_selected_uids.clear();
                state.interact.call_selected_uid = GUI_INVALID_ID;
                state.interact.interfaceslot_selected_uid = GUI_INVALID_ID;
            }
            // Deselection
            else if (this->gui_selected &&
                     ((mouse_clicked_anywhere && !hovered) || (active && ImGui::IsKeyPressed(ImGuiMod_Shift)) ||
                         (state.interact.group_selected_uid != this->uid))) {
                this->gui_selected = false;
                if (state.interact.group_selected_uid == this->uid) {
                    state.interact.group_selected_uid = GUI_INVALID_ID;
                }
            }

            // Toggle View
            if (active && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                this->gui_collapsed_view = !this->gui_collapsed_view;
                changed_view = true;
            }

            // Dragging
            if (this->gui_selected && ImGui::IsWindowHovered() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                this->SetPosition(state, (this->gui_position + (ImGui::GetIO().MouseDelta / state.canvas.zooming)));
            }

            /// COLOR_GROUP_BACKGROUND
            ImVec4 tmpcol = style.Colors[ImGuiCol_ChildBg];
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_GROUP_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);
            /// COLOR_GROUP_BACKGROUND_HIGHTLIGHT
            tmpcol = style.Colors[ImGuiCol_FrameBgActive];
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_GROUP_BACKGROUND_HIGHTLIGHT = ImGui::ColorConvertFloat4ToU32(tmpcol);
            /// COLOR_GROUP_BORDER
            tmpcol = style.Colors[ImGuiCol_ScrollbarGrabHovered];
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_GROUP_BORDER = ImGui::ColorConvertFloat4ToU32(tmpcol);
            /// COLOR_TEXT
            const ImU32 COLOR_TEXT = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Text]);

            // Background
            ImU32 group_bg_color =
                (this->gui_selected) ? (COLOR_GROUP_BACKGROUND_HIGHTLIGHT) : (COLOR_GROUP_BACKGROUND);
            draw_list->AddRectFilled(group_rect_min, group_rect_max, group_bg_color, 0.0f);
            draw_list->AddRect(group_rect_min, group_rect_max, COLOR_GROUP_BORDER, 0.0f);

            // Draw text
            float name_width = ImGui::CalcTextSize(this->name.c_str()).x;
            ImVec2 text_pos_left_upper =
                ImVec2((group_center.x - (name_width / 2.0f)), (group_rect_min.y + (style.ItemSpacing.y / 2.0f)));
            if (!this->gui_collapsed_view) {
                text_pos_left_upper =
                    ImVec2((group_rect_min.x + style.ItemSpacing.x), (group_rect_min.y + (style.ItemSpacing.y / 2.0f)));
            }
            auto header_color = (this->gui_selected) ? (GUI_COLOR_GROUP_HEADER_HIGHLIGHT) : (GUI_COLOR_GROUP_HEADER);
            draw_list->AddRectFilled(group_rect_min, header_rect_max, ImGui::ColorConvertFloat4ToU32(header_color),
                GUI_RECT_CORNER_RADIUS, (ImDrawFlags_RoundCornersTopLeft | ImDrawFlags_RoundCornersTopRight));
            draw_list->AddText(text_pos_left_upper, COLOR_TEXT, this->name.c_str());
        }

        ImGui::PopID();

        if (changed_view) {
            for (auto& module_ptr : this->modules) {
                module_ptr->SetHidden(this->gui_collapsed_view);
            }
            for (auto& interfaceslots_map : this->InterfaceSlots()) {
                for (auto& interfaceslot_ptr : interfaceslots_map.second) {
                    interfaceslot_ptr->SetGroupViewCollapsed(this->gui_collapsed_view);
                }
            }
            this->Update(state.canvas);
        }

        // INTERFACE SLOTS -----------------------------------------------------
        for (auto& interfaceslots_map : this->InterfaceSlots()) {
            for (auto& interfaceslot_ptr : interfaceslots_map.second) {
                interfaceslot_ptr->Draw(phase, state);
            }
        }

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


void megamol::gui::Group::SetPosition(const GraphItemsState_t& state, ImVec2 pos) {

    ImVec2 pos_delta = (pos - this->gui_position);

    // Moving modules and then updating group position
    ImVec2 tmp_pos;
    for (auto& module_ptr : this->modules) {
        tmp_pos = module_ptr->Position();
        tmp_pos += pos_delta;
        module_ptr->SetPosition(tmp_pos);
        module_ptr->Update();
    }
    this->Update(state.canvas);
}


void megamol::gui::Group::Update(const GraphCanvas_t& in_canvas) {

    const float line_height = ImGui::GetTextLineHeightWithSpacing() / in_canvas.zooming;
    const float slot_height = GUI_SLOT_RADIUS * 3.0f;

    // POSITION
    float pos_minX = FLT_MAX;
    float pos_minY = FLT_MAX;
    ImVec2 tmp_pos;
    if (!this->modules.empty()) {
        for (auto& mod : this->modules) {
            tmp_pos = mod->Position();
            pos_minX = std::min(tmp_pos.x, pos_minX);
            pos_minY = std::min(tmp_pos.y, pos_minY);
        }
        pos_minX -= GUI_GRAPH_BORDER;
        pos_minY -= (GUI_GRAPH_BORDER + line_height);
        this->gui_position = ImVec2(pos_minX, pos_minY);
    } else {
        this->gui_position = megamol::gui::Module::GetDefaultModulePosition(in_canvas);
    }

    float group_width = (1.5f * ImGui::CalcTextSize(this->name.c_str()).x / in_canvas.zooming);
    float group_height = (3.0f * line_height);
    std::vector<float> caller_label_heights;
    std::vector<float> callee_label_heights;
    float caller_max_label_width = 0.0f;
    float callee_max_label_width = 0.0f;
    for (auto& interfaceslot_map : this->InterfaceSlots()) {
        for (auto& interfaceslot_ptr : interfaceslot_map.second) {
            auto text_size = ImGui::CalcTextSize(interfaceslot_ptr->Label().c_str());
            if (interfaceslot_map.first == CallSlotType::CALLER) {
                caller_max_label_width = std::max(text_size.x, caller_max_label_width);
                caller_label_heights.emplace_back(
                    std::max((text_size.y / in_canvas.zooming + GUI_SLOT_RADIUS * 1.0f), slot_height));
            } else if (interfaceslot_map.first == CallSlotType::CALLEE) {
                callee_max_label_width = std::max(text_size.x, callee_max_label_width);
                callee_label_heights.emplace_back(
                    std::max((text_size.y / in_canvas.zooming + GUI_SLOT_RADIUS * 1.0f), slot_height));
            }
        }
    }

    if (this->gui_collapsed_view) {

        float max_label_width =
            ((caller_max_label_width + callee_max_label_width) / in_canvas.zooming) + (1.0f * GUI_SLOT_RADIUS);
        group_width = std::max(group_width, max_label_width) + (3.0f * GUI_SLOT_RADIUS);

        float caller_max_label_height = 0.0f;
        for (auto& lh : caller_label_heights) {
            caller_max_label_height += lh;
        }
        float callee_max_label_height = 0.0f;
        for (auto& lh : callee_label_heights) {
            callee_max_label_height += lh;
        }
        group_height = std::max(group_height,
            (line_height + GUI_SLOT_RADIUS + (std::max(caller_max_label_height, callee_max_label_height))));
    } else {

        float pos_maxX = -FLT_MAX;
        float pos_maxY = -FLT_MAX;
        ImVec2 tmp_size;
        for (auto& mod : this->modules) {
            tmp_pos = mod->Position();
            tmp_size = mod->Size();
            pos_maxX = std::max(tmp_pos.x + tmp_size.x, pos_maxX);
            pos_maxY = std::max(tmp_pos.y + tmp_size.y, pos_maxY);
        }
        group_width = std::max(group_width, (pos_maxX + GUI_GRAPH_BORDER) - pos_minX);
        group_height = std::max(group_height, (pos_maxY + GUI_GRAPH_BORDER) - pos_minY);
    }

    // Clamp to minimum size
    this->gui_size = ImVec2(std::max(group_width, (100.0f * megamol::gui::gui_scaling.Get())),
        std::max(group_height, (50.0f * megamol::gui::gui_scaling.Get())));

    // Set group interface position of call slots --------------------------
    ImVec2 group_pos = in_canvas.offset + this->gui_position * in_canvas.zooming;
    group_pos.y += (line_height * in_canvas.zooming);
    ImVec2 group_size = this->gui_size * in_canvas.zooming;
    group_size.y -= (line_height * in_canvas.zooming);
    float caller_y = group_pos.y;
    float callee_y = group_pos.y;
    for (auto& interfaceslot_map : this->InterfaceSlots()) {
        auto slot_cnt = interfaceslot_map.second.size();
        for (size_t i = 0; i < slot_cnt; i++) {
            ImVec2 callslot_group_position;
            if (interfaceslot_map.first == CallSlotType::CALLER) {
                auto caller_label_height = caller_label_heights[i] * in_canvas.zooming;
                callslot_group_position = ImVec2((group_pos.x + group_size.x), (caller_y + caller_label_height / 2.0f));
                caller_y += caller_label_height;
            } else if (interfaceslot_map.first == CallSlotType::CALLEE) {
                auto callee_label_height = callee_label_heights[i] * in_canvas.zooming;
                callslot_group_position = ImVec2(group_pos.x, (callee_y + callee_label_height / 2.0f));
                callee_y += callee_label_height;
            }
            interfaceslot_map.second[i]->SetPosition(callslot_group_position);
        }
    }

    this->spacial_sort_interfaceslots();
}
