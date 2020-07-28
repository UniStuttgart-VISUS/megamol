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


// GROUP PRESENTATION ####################################################

megamol::gui::GroupPresentation::GroupPresentation(void)
    : position(ImVec2(FLT_MAX, FLT_MAX))
    , size(ImVec2(0.0f, 0.0f))
    , collapsed_view(false)
    , allow_selection(false)
    , allow_context(false)
    , selected(false)
    , update(true)
    , rename_popup() {}


megamol::gui::GroupPresentation::~GroupPresentation(void) {}


void megamol::gui::GroupPresentation::Present(
    megamol::gui::PresentPhase phase, megamol::gui::Group& inout_group, GraphItemsStateType& state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
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
                mod->present.group.visible = this->ModulesVisible();
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

        bool changed_view = false;

        if (phase == megamol::gui::PresentPhase::INTERACTION) {

            // Limit selection to header
            this->allow_selection = false;
            ImVec2 mouse_pos = ImGui::GetMousePos();
            if ((mouse_pos.x >= group_rect_min.x) && (mouse_pos.y >= group_rect_min.y) &&
                (mouse_pos.x <= header_rect_max.x) && (mouse_pos.y <= header_rect_max.y)) {
                this->allow_selection = true;
                if (state.interact.group_hovered_uid == inout_group.uid) {
                    this->allow_context = true;
                }
            }

            // Button
            std::string button_label = "group_" + std::to_string(inout_group.uid);
            ImGui::SetCursorScreenPos(group_rect_min);
            ImGui::SetItemAllowOverlap();
            ImGui::InvisibleButton(button_label.c_str(), group_size);
            ImGui::SetItemAllowOverlap();
            if (ImGui::IsItemActivated()) {
                state.interact.button_active_uid = inout_group.uid;
            }
            if (ImGui::IsItemHovered()) {
                state.interact.button_hovered_uid = inout_group.uid;
            }

            // Context menu
            bool popup_rename = false;
            if (ImGui::BeginPopupContextItem("invisible_button_context")) { /// this->allow_context &&

                state.interact.button_active_uid = inout_group.uid;

                ImGui::TextDisabled("Group");
                ImGui::Separator();

                std::string view = "Collapse";
                if (this->collapsed_view) {
                    view = "Expand";
                }
                if (ImGui::MenuItem(view.c_str(), "'Double-Click' Header")) {
                    this->collapsed_view = !this->collapsed_view;
                    changed_view = true;
                }
                if (ImGui::MenuItem("Layout")) {
                    state.interact.group_layout = true;
                }
                if (ImGui::MenuItem("Rename")) {
                    popup_rename = true;
                }
                if (ImGui::MenuItem("Delete",
                        std::get<0>(state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM]).ToString().c_str())) {
                    state.interact.process_deletion = true;
                }
                ImGui::EndPopup();
            } /// else { this->allow_context = false; }

            // Rename pop-up
            if (this->rename_popup.PopUp("Rename Group", popup_rename, inout_group.name)) {
                for (auto& module_ptr : inout_group.GetModules()) {
                    module_ptr->present.group.name = inout_group.name;
                    module_ptr->UpdateGUI(state.canvas);
                }
                this->UpdatePositionSize(inout_group, state.canvas);
            }
        } else if (phase == megamol::gui::PresentPhase::RENDERING) {

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

            // Adjust state for selection
            active = active && this->allow_selection;
            hovered = hovered && this->allow_selection;
            this->allow_selection = false;
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
            else if (this->selected && ((mouse_clicked_anywhere && !hovered) || (active && GUI_MULTISELECT_MODIFIER) ||
                                           (state.interact.group_selected_uid != inout_group.uid))) {
                this->selected = false;
                if (state.interact.group_selected_uid == inout_group.uid) {
                    state.interact.group_selected_uid = GUI_INVALID_ID;
                }
            }

            // Toggle View
            if (active && ImGui::IsMouseDoubleClicked(0)) {
                this->collapsed_view = !this->collapsed_view;
                changed_view = true;
            }

            // Dragging
            if (this->selected && ImGui::IsWindowHovered() && ImGui::IsMouseDragging(0)) {
                this->SetPosition(
                    inout_group, state.canvas, (this->position + (ImGui::GetIO().MouseDelta / state.canvas.zooming)));
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
            float name_width = ImGui::CalcTextSize(inout_group.name.c_str()).x;
            ImVec2 text_pos_left_upper =
                ImVec2((group_center.x - (name_width / 2.0f)), (group_rect_min.y + (style.ItemSpacing.y / 2.0f)));
            if (!this->collapsed_view) {
                text_pos_left_upper =
                    ImVec2((group_rect_min.x + style.ItemSpacing.x), (group_rect_min.y + (style.ItemSpacing.y / 2.0f)));
            }
            auto header_color = (this->selected) ? (COLOR_HEADER_HIGHLIGHT) : (COLOR_HEADER);
            draw_list->AddRectFilled(group_rect_min, header_rect_max, header_color, GUI_RECT_CORNER_RADIUS,
                (ImDrawCornerFlags_TopLeft | ImDrawCornerFlags_TopRight));
            draw_list->AddText(text_pos_left_upper, COLOR_TEXT, inout_group.name.c_str());
        }

        ImGui::PopID();

        if (changed_view) {
            for (auto& module_ptr : inout_group.GetModules()) {
                module_ptr->present.group.visible = this->ModulesVisible();
            }
            for (auto& interfaceslots_map : inout_group.GetInterfaceSlots()) {
                for (auto& interfaceslot_ptr : interfaceslots_map.second) {
                    interfaceslot_ptr->present.group.collapsed_view = this->collapsed_view;
                }
            }
            this->UpdatePositionSize(inout_group, state.canvas);
        }

        // INTERFACE SLOTS -----------------------------------------------------
        for (auto& interfaceslots_map : inout_group.GetInterfaceSlots()) {
            for (auto& interfaceslot_ptr : interfaceslots_map.second) {
                interfaceslot_ptr->PresentGUI(phase, state);
            }
        }

    } catch (std::exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


void megamol::gui::GroupPresentation::SetPosition(Group& inout_group, const GraphCanvasType& in_canvas, ImVec2 pos) {

    ImVec2 pos_delta = (pos - this->position);

    // Moving modules and then updating group position
    ImVec2 tmp_pos;
    for (auto& module_ptr : inout_group.GetModules()) {
        tmp_pos = module_ptr->present.position;
        tmp_pos += pos_delta;
        module_ptr->present.position = tmp_pos;
        module_ptr->UpdateGUI(in_canvas);
    }
    this->UpdatePositionSize(inout_group, in_canvas);
}


void megamol::gui::GroupPresentation::UpdatePositionSize(
    megamol::gui::Group& inout_group, const GraphCanvasType& in_canvas) {

    float line_height = ImGui::GetTextLineHeightWithSpacing() / in_canvas.zooming;

    // POSITION
    float pos_minX = FLT_MAX;
    float pos_minY = FLT_MAX;
    ImVec2 tmp_pos;
    if (inout_group.GetModules().size() > 0) {
        for (auto& mod : inout_group.GetModules()) {
            tmp_pos = mod->present.position;
            pos_minX = std::min(tmp_pos.x, pos_minX);
            pos_minY = std::min(tmp_pos.y, pos_minY);
        }
        pos_minX -= GUI_GRAPH_BORDER;
        pos_minY -= (GUI_GRAPH_BORDER + line_height);
        this->position = ImVec2(pos_minX, pos_minY);
    } else {
        this->position = megamol::gui::ModulePresentation::GetDefaultModulePosition(in_canvas);
    }

    // SIZE
    float group_width = 0.0f;
    float group_height = 0.0f;
    size_t caller_count = inout_group.GetInterfaceSlots().operator[](CallSlotType::CALLER).size();
    size_t callee_count = inout_group.GetInterfaceSlots().operator[](CallSlotType::CALLEE).size();
    size_t max_slot_count = std::max(caller_count, callee_count);

    // WIDTH
    float max_label_length = 0.0f;
    // Consider interface slot label width only in collapsed view
    if (this->collapsed_view) {
        for (auto& interfaceslot_map : inout_group.GetInterfaceSlots()) {
            for (auto& interfaceslot_ptr : interfaceslot_map.second) {
                max_label_length =
                    std::max(ImGui::CalcTextSize(interfaceslot_ptr->present.GetLabel().c_str()).x, max_label_length);
            }
        }
        if (max_label_length > 0.0f) {
            max_label_length = (2.0f * max_label_length / in_canvas.zooming) + (1.0f * GUI_SLOT_RADIUS);
        }
    }
    group_width =
        std::max((1.5f * ImGui::CalcTextSize(inout_group.name.c_str()).x / in_canvas.zooming), max_label_length) +
        (3.0f * GUI_SLOT_RADIUS);

    // HEIGHT
    group_height = std::max((3.0f * line_height),
        (line_height + (static_cast<float>(max_slot_count) * (GUI_SLOT_RADIUS * 2.0f) * 1.5f) + GUI_SLOT_RADIUS));

    if (!this->collapsed_view) {
        float pos_maxX = -FLT_MAX;
        float pos_maxY = -FLT_MAX;
        ImVec2 tmp_pos;
        ImVec2 tmp_size;
        for (auto& mod : inout_group.GetModules()) {
            tmp_pos = mod->present.position;
            tmp_size = mod->present.GetSize();
            pos_maxX = std::max(tmp_pos.x + tmp_size.x, pos_maxX);
            pos_maxY = std::max(tmp_pos.y + tmp_size.y, pos_maxY);
        }
        group_width = std::max(group_width, (pos_maxX + GUI_GRAPH_BORDER) - pos_minX);
        group_height = std::max(group_height, (pos_maxY + GUI_GRAPH_BORDER) - pos_minY);
    }
    // Clamp to minimum size
    this->size = ImVec2(std::max(group_width, 100.0f), std::max(group_height, 50.0f));


    // Set group interface position of call slots --------------------------

    ImVec2 group_pos = in_canvas.offset + this->position * in_canvas.zooming;
    group_pos.y += (line_height * in_canvas.zooming);
    ImVec2 group_size = this->size * in_canvas.zooming;
    group_size.y -= (line_height * in_canvas.zooming);

    size_t caller_idx = 0;
    size_t callee_idx = 0;
    ImVec2 callslot_group_position;

    for (auto& interfaceslots_map : inout_group.GetInterfaceSlots()) {
        for (auto& interfaceslot_ptr : interfaceslots_map.second) {
            if (interfaceslots_map.first == CallSlotType::CALLER) {
                callslot_group_position = ImVec2((group_pos.x + group_size.x),
                    (group_pos.y + group_size.y * ((float)caller_idx + 1) / ((float)caller_count + 1)));
                caller_idx++;
            } else if (interfaceslots_map.first == CallSlotType::CALLEE) {
                callslot_group_position = ImVec2(
                    group_pos.x, (group_pos.y + group_size.y * ((float)callee_idx + 1) / ((float)callee_count + 1)));
                callee_idx++;
            }
            interfaceslot_ptr->present.SetPosition(callslot_group_position);
        }
    }
}


// GROUP ######################################################################

megamol::gui::Group::Group(ImGuiID uid) : uid(uid), name(), modules(), interfaceslots(), present() {}


megamol::gui::Group::~Group() {

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


bool megamol::gui::Group::AddModule(const ModulePtrType& module_ptr) {

    if (module_ptr == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Pointer to module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Check if module is already part of the group
    for (auto& mod : this->modules) {
        if (mod->uid == module_ptr->uid) {
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[Configurator] Module '%s' is already part of group '%s'.\n", mod->name.c_str(), this->name.c_str());
#endif // GUI_VERBOSE
            return false;
        }
    }

    this->modules.emplace_back(module_ptr);

    module_ptr->present.group.uid = this->uid;
    module_ptr->present.group.visible = this->present.ModulesVisible();
    module_ptr->present.group.name = this->name;

    this->present.ForceUpdate();
    this->RestoreInterfaceslots();

#ifdef GUI_VERBOSE
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
        "[Configurator] Added module '%s' to group '%s'.\n", module_ptr->name.c_str(), this->name.c_str());
#endif // GUI_VERBOSE
    return true;
}


bool megamol::gui::Group::RemoveModule(ImGuiID module_uid) {

    try {
        for (auto mod_iter = this->modules.begin(); mod_iter != this->modules.end(); mod_iter++) {
            if ((*mod_iter)->uid == module_uid) {

                // Remove call slots from group interface
                for (auto& callslot_map : (*mod_iter)->GetCallSlots()) {
                    for (auto& callslot_ptr : callslot_map.second) {
                        this->InterfaceSlot_RemoveCallSlot(callslot_ptr->uid, true);
                    }
                }

                (*mod_iter)->present.group.uid = GUI_INVALID_ID;
                (*mod_iter)->present.group.visible = false;
                (*mod_iter)->present.group.name = "";

#ifdef GUI_VERBOSE
                megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                    "[Configurator] Removed module '%s' from group '%s'.\n", (*mod_iter)->name.c_str(),
                    this->name.c_str());
#endif // GUI_VERBOSE
                (*mod_iter).reset();
                this->modules.erase(mod_iter);

                this->present.ForceUpdate();

                return true;
            }
        }
    } catch (std::exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
        "Invalid module uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}


bool megamol::gui::Group::ContainsModule(ImGuiID module_uid) {

    for (auto& mod : this->modules) {
        if (mod->uid == module_uid) {
            return true;
        }
    }
    return false;
}


ImGuiID megamol::gui::Group::AddInterfaceSlot(const CallSlotPtrType& callslot_ptr, bool auto_add) {

    if (callslot_ptr == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    }

    // Check if call slot is already part of the group
    for (auto& interfaceslot_ptr : this->interfaceslots[callslot_ptr->type]) {
        if (interfaceslot_ptr->ContainsCallSlot(callslot_ptr->uid)) {
            return interfaceslot_ptr->uid;
        }
    }

    // Only add if parent module is already part of the group.
    bool parent_module_group_uid = false;
    if (callslot_ptr->IsParentModuleConnected()) {
        ImGuiID parent_module_uid = callslot_ptr->GetParentModule()->uid;
        for (auto& module_ptr : this->modules) {
            if (parent_module_uid == module_ptr->uid) {
                parent_module_group_uid = true;
            }
        }
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Call slot has no parent module connected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    }

    if (parent_module_group_uid) {
        InterfaceSlotPtrType interfaceslot_ptr =
            std::make_shared<InterfaceSlot>(megamol::gui::GenerateUniqueID(), auto_add);
        if (interfaceslot_ptr != nullptr) {
            interfaceslot_ptr->present.group.uid = this->uid;
            this->interfaceslots[callslot_ptr->type].emplace_back(interfaceslot_ptr);
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[Configurator] Added interface slot (uid %i) to group '%s'.\n", interfaceslot_ptr->uid,
                this->name.c_str());
#endif // GUI_VERBOSE

            if (interfaceslot_ptr->AddCallSlot(callslot_ptr, interfaceslot_ptr)) {
                interfaceslot_ptr->present.group.collapsed_view = this->present.IsViewCollapsed();
                return interfaceslot_ptr->uid;
            }
        }
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Parent module of call slot which should be added to group interface "
            "is not part of any group. [%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    }
    return GUI_INVALID_ID;
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
                        empty_interfaceslots_uids.emplace_back(interfaceslot_ptr->uid);
                    }
                }
            }
        }
        // Delete empty interface slots
        for (auto& interfaceslot_uid : empty_interfaceslots_uids) {
            this->DeleteInterfaceSlot(interfaceslot_uid);
        }
    } catch (std::exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
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


bool megamol::gui::Group::GetInterfaceSlot(ImGuiID interfaceslot_uid, InterfaceSlotPtrType& interfaceslot_ptr) {

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


bool megamol::gui::Group::DeleteInterfaceSlot(ImGuiID interfaceslot_uid) {

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
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "Unclean deletion. Found %i references pointing to interface slot. [%s, %s, line %d]\n",
                            (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                    }

#ifdef GUI_VERBOSE
                    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                        "[Configurator] Deleted interface slot (uid %i) from group '%s'.\n", (*iter)->uid,
                        this->name.c_str());
#endif // GUI_VERBOSE

                    (*iter).reset();
                    interfaceslot_map.second.erase(iter);

                    return true;
                }
            }
        }
    }
    return false;
}


bool megamol::gui::Group::ContainsInterfaceSlot(ImGuiID interfaceslot_uid) {

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


void megamol::gui::Group::RestoreInterfaceslots(void) {

    /// 1] REMOVE connected call slots of group interface if connected module is part of same group
    for (auto& module_ptr : this->modules) {
        // CALLER
        for (auto& callerslot_ptr : module_ptr->GetCallSlots(CallSlotType::CALLER)) {
            for (auto& call_ptr : callerslot_ptr->GetConnectedCalls()) {
                auto calleeslot_ptr = call_ptr->GetCallSlot(CallSlotType::CALLEE);
                if (calleeslot_ptr->IsParentModuleConnected()) {
                    ImGuiID parent_module_group_uid = calleeslot_ptr->GetParentModule()->present.group.uid;
                    if (parent_module_group_uid == this->uid) {
                        this->InterfaceSlot_RemoveCallSlot(calleeslot_ptr->uid);
                    }
                }
            }
        }
        // CALLEE
        for (auto& calleeslot_ptr : module_ptr->GetCallSlots(CallSlotType::CALLEE)) {
            for (auto& call_ptr : calleeslot_ptr->GetConnectedCalls()) {
                auto callerslot_ptr = call_ptr->GetCallSlot(CallSlotType::CALLER);
                if (callerslot_ptr->IsParentModuleConnected()) {
                    ImGuiID parent_module_group_uid = callerslot_ptr->GetParentModule()->present.group.uid;
                    if (parent_module_group_uid == this->uid) {
                        this->InterfaceSlot_RemoveCallSlot(callerslot_ptr->uid);
                    }
                }
            }
        }
    }

    /// 2] ADD connected call slots to group interface if connected module is not part of same group
    for (auto& module_ptr : this->modules) {
        // CALLER
        for (auto& callerslot_ptr : module_ptr->GetCallSlots(CallSlotType::CALLER)) {
            for (auto& call_ptr : callerslot_ptr->GetConnectedCalls()) {
                auto calleeslot_ptr = call_ptr->GetCallSlot(CallSlotType::CALLEE);
                if (calleeslot_ptr->IsParentModuleConnected()) {
                    ImGuiID parent_module_group_uid = calleeslot_ptr->GetParentModule()->present.group.uid;
                    if (parent_module_group_uid != this->uid) {
                        this->AddInterfaceSlot(callerslot_ptr);
                    }
                }
            }
        }
        // CALLEE
        for (auto& calleeslot_ptr : module_ptr->GetCallSlots(CallSlotType::CALLEE)) {
            for (auto& call_ptr : calleeslot_ptr->GetConnectedCalls()) {
                auto callerslot_ptr = call_ptr->GetCallSlot(CallSlotType::CALLER);
                if (callerslot_ptr->IsParentModuleConnected()) {
                    ImGuiID parent_module_group_uid = callerslot_ptr->GetParentModule()->present.group.uid;
                    if (parent_module_group_uid != this->uid) {
                        this->AddInterfaceSlot(calleeslot_ptr);
                    }
                }
            }
        }
    }
}
