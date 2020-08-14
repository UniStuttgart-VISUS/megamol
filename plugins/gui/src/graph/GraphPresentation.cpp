/*
 * Graph.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

/**
 * USED HOTKEYS:
 *
 * - Selection, Drag & Drop:    Left Mouse Button
 * - Context Menu:              Right Mouse Button
 * - Zooming:                   Mouse Wheel
 * - Scrolling:                 Ctrl  +  Left Mouse Button
 */

#include "stdafx.h"
#include "GraphPresentation.h"

#include "Graph.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::GraphPresentation::GraphPresentation(void)
    : params_visible(true)
    , params_readonly(false)
    , param_extended_mode(false)
    , update(true)
    , show_grid(false)
    , show_call_names(true)
    , show_slot_names(false)
    , show_module_names(true)
    , show_parameter_sidebar(true)
    , change_show_parameter_sidebar(false)
    , graph_layout(0)
    , parameter_sidebar_width(300.0f)
    , reset_zooming(true)
    , param_name_space()
    , multiselect_start_pos()
    , multiselect_end_pos()
    , multiselect_done(false)
    , canvas_hovered(false)
    , current_font_scaling(1.0f)
    , graph_state()
    , search_widget()
    , splitter_widget()
    , rename_popup()
    , tooltip() {

    this->graph_state.canvas.position = ImVec2(0.0f, 0.0f);
    this->graph_state.canvas.size = ImVec2(1.0f, 1.0f);
    this->graph_state.canvas.scrolling = ImVec2(0.0f, 0.0f);
    this->graph_state.canvas.zooming = 1.0f;
    this->graph_state.canvas.offset = ImVec2(0.0f, 0.0f);

    this->graph_state.interact.process_deletion = false;
    this->graph_state.interact.button_active_uid = GUI_INVALID_ID;
    this->graph_state.interact.button_hovered_uid = GUI_INVALID_ID;

    this->graph_state.interact.group_selected_uid = GUI_INVALID_ID;
    this->graph_state.interact.group_hovered_uid = GUI_INVALID_ID;
    this->graph_state.interact.group_layout = false;

    this->graph_state.interact.modules_selected_uids.clear();
    this->graph_state.interact.module_hovered_uid = GUI_INVALID_ID;
    this->graph_state.interact.module_mainview_uid = GUI_INVALID_ID;
    this->graph_state.interact.modules_add_group_uids.clear();
    this->graph_state.interact.modules_remove_group_uids.clear();
    this->graph_state.interact.modules_layout = false;

    this->graph_state.interact.call_selected_uid = GUI_INVALID_ID;
    this->graph_state.interact.call_hovered_uid = GUI_INVALID_ID;

    this->graph_state.interact.slot_dropped_uid = GUI_INVALID_ID;

    this->graph_state.interact.callslot_selected_uid = GUI_INVALID_ID;
    this->graph_state.interact.callslot_hovered_uid = GUI_INVALID_ID;
    this->graph_state.interact.callslot_add_group_uid = UIDPair_t(GUI_INVALID_ID, GUI_INVALID_ID);
    this->graph_state.interact.callslot_remove_group_uid = UIDPair_t(GUI_INVALID_ID, GUI_INVALID_ID);
    this->graph_state.interact.callslot_compat_ptr.reset();

    this->graph_state.interact.interfaceslot_selected_uid = GUI_INVALID_ID;
    this->graph_state.interact.interfaceslot_hovered_uid = GUI_INVALID_ID;
    this->graph_state.interact.interfaceslot_compat_ptr.reset();

    this->graph_state.groups.clear();
    // this->graph_state.hotkeys are already initialzed
}


megamol::gui::GraphPresentation::~GraphPresentation(void) {}


void megamol::gui::GraphPresentation::Present(megamol::gui::Graph& inout_graph, GraphState_t& state) {

    try {
        /// Hide running graph of core instance
        if (inout_graph.RunningState().IsFalse()) return;

        if (ImGui::GetCurrentContext() == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return;
        }
        ImGuiIO& io = ImGui::GetIO();

        ImGuiID graph_uid = inout_graph.uid;
        ImGui::PushID(graph_uid);

        // State Init ----------------------------
        if (this->change_show_parameter_sidebar) {
            state.show_parameter_sidebar = this->show_parameter_sidebar;
            this->change_show_parameter_sidebar = false;
        }
        this->show_parameter_sidebar = state.show_parameter_sidebar;

        this->graph_state.hotkeys = state.hotkeys;
        this->graph_state.groups.clear();
        for (auto& group : inout_graph.GetGroups()) {
            std::pair<ImGuiID, std::string> group_pair(group->uid, group->name);
            this->graph_state.groups.emplace_back(group_pair);
        }
        this->graph_state.interact.slot_dropped_uid = GUI_INVALID_ID;

        // Compatible slot pointers
        this->graph_state.interact.callslot_compat_ptr.reset();
        this->graph_state.interact.interfaceslot_compat_ptr.reset();
        //  Consider hovered slots only if there is no drag and drop
        bool slot_draganddrop_active = false;
        if (const ImGuiPayload* payload = ImGui::GetDragDropPayload()) {
            if (payload->IsDataType(GUI_DND_CALLSLOT_UID_TYPE)) {
                slot_draganddrop_active = true;
            }
        }
        ImGuiID slot_uid = GUI_INVALID_ID;
        if (this->graph_state.interact.callslot_selected_uid != GUI_INVALID_ID) {
            slot_uid = this->graph_state.interact.callslot_selected_uid;
        } else if ((this->graph_state.interact.interfaceslot_selected_uid == GUI_INVALID_ID) &&
                   (!slot_draganddrop_active)) {
            slot_uid = this->graph_state.interact.callslot_hovered_uid;
        }
        if (slot_uid != GUI_INVALID_ID) {
            for (auto& module_ptr : inout_graph.GetModules()) {
                CallSlotPtr_t callslot_ptr;
                if (module_ptr->GetCallSlot(slot_uid, callslot_ptr)) {
                    this->graph_state.interact.callslot_compat_ptr = callslot_ptr;
                }
            }
        }
        // Compatible call slot and/or interface slot ptr
        if (this->graph_state.interact.callslot_compat_ptr == nullptr) {
            slot_uid = GUI_INVALID_ID;
            if (this->graph_state.interact.interfaceslot_selected_uid != GUI_INVALID_ID) {
                slot_uid = this->graph_state.interact.interfaceslot_selected_uid;
            } else if (!slot_draganddrop_active) {
                slot_uid = this->graph_state.interact.interfaceslot_hovered_uid;
            }
            if (slot_uid != GUI_INVALID_ID) {
                for (auto& group_ptr : inout_graph.GetGroups()) {
                    InterfaceSlotPtr_t interfaceslot_ptr;
                    if (group_ptr->GetInterfaceSlot(slot_uid, interfaceslot_ptr)) {
                        this->graph_state.interact.interfaceslot_compat_ptr = interfaceslot_ptr;
                    }
                }
            }
        }

        // Tab showing this graph ---------------
        bool popup_rename = false;
        ImGuiTabItemFlags tab_flags = ImGuiTabItemFlags_None;
        if (inout_graph.IsDirty()) {
            tab_flags |= ImGuiTabItemFlags_UnsavedDocument;
        }
        std::string graph_label = "    " + inout_graph.name + "  ###graph" + std::to_string(graph_uid);
        if (inout_graph.RunningState().IsTrue()) {
            graph_label = "    [RUNNING]  " + graph_label;
        }
        // Checking for closed tab below
        bool open = true;
        if (ImGui::BeginTabItem(graph_label.c_str(), &open, tab_flags)) {
            // Context menu
            if (ImGui::BeginPopupContextItem()) {

                ImGui::TextDisabled("Project");
                ImGui::Separator();

                if (ImGui::MenuItem("Save")) {
                    state.graph_save = true;
                }

                if (ImGui::MenuItem("Rename")) {
                    popup_rename = true;
                }

                if (!inout_graph.GetFilename().empty()) {
                    ImGui::Separator();
                    ImGui::TextDisabled("Filename");
                    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 13.0f);
                    ImGui::TextUnformatted(inout_graph.GetFilename().c_str());
                    ImGui::PopTextWrapPos();
                }

                ImGui::EndPopup();
            }

            // Draw -----------------------------
            this->present_menu(inout_graph);

            float graph_width_auto = 0.0f;
            if (this->show_parameter_sidebar) {
                this->splitter_widget.Widget(
                    SplitterWidget::FixedSplitterSide::RIGHT, graph_width_auto, this->parameter_sidebar_width);
            }

            // Load font for canvas
            ImFont* font_ptr = nullptr;
            unsigned int scalings_count = static_cast<unsigned int>(state.font_scalings.size());
            if (scalings_count == 0) {
                throw std::invalid_argument("Array for graph fonts is empty.");
            } else if (scalings_count == 1) {
                font_ptr = io.Fonts->Fonts[0];
                this->current_font_scaling = state.font_scalings[0];
            } else {
                for (unsigned int i = 0; i < scalings_count; i++) {
                    bool apply = false;
                    if (i == 0) {
                        if (this->graph_state.canvas.zooming <= state.font_scalings[i]) {
                            apply = true;
                        }
                    } else if (i == (scalings_count - 1)) {
                        if (this->graph_state.canvas.zooming >= state.font_scalings[i]) {
                            apply = true;
                        }
                    } else {
                        if ((state.font_scalings[i - 1] < this->graph_state.canvas.zooming) &&
                            (this->graph_state.canvas.zooming < state.font_scalings[i + 1])) {
                            apply = true;
                        }
                    }
                    if (apply) {
                        font_ptr = io.Fonts->Fonts[i];
                        this->current_font_scaling = state.font_scalings[i];
                        break;
                    }
                }
            }
            if (font_ptr != nullptr) {
                ImGui::PushFont(font_ptr);
                this->present_canvas(inout_graph, graph_width_auto);
                ImGui::PopFont();
            } else {
                throw std::invalid_argument("Pointer to font is nullptr.");
            }
            if (this->show_parameter_sidebar) {
                ImGui::SameLine();
                this->present_parameters(inout_graph, this->parameter_sidebar_width);
            }

            state.graph_selected_uid = inout_graph.uid;
            ImGui::EndTabItem();
        }

        // State processing ---------------------
        this->ResetStatePointers();
        bool reset_state = false;
        // Add module to group
        if (!this->graph_state.interact.modules_add_group_uids.empty()) {
            ModulePtr_t module_ptr;
            ImGuiID new_group_uid = GUI_INVALID_ID;
            for (auto& uid_pair : this->graph_state.interact.modules_add_group_uids) {
                module_ptr.reset();
                for (auto& mod : inout_graph.GetModules()) {
                    if (mod->uid == uid_pair.first) {
                        module_ptr = mod;
                    }
                }
                if (module_ptr != nullptr) {

                    // Add module to new or already existing group
                    // Create new group for multiple selected modules only once!
                    ImGuiID group_uid = GUI_INVALID_ID;
                    if ((uid_pair.second == GUI_INVALID_ID) && (new_group_uid == GUI_INVALID_ID)) {
                        new_group_uid = inout_graph.AddGroup();
                    }
                    if (uid_pair.second == GUI_INVALID_ID) {
                        group_uid = new_group_uid;
                    } else {
                        group_uid = uid_pair.second;
                    }

                    GroupPtr_t add_group_ptr;
                    if (inout_graph.GetGroup(group_uid, add_group_ptr)) {
                        // Remove module from previous associated group
                        ImGuiID module_group_uid = module_ptr->present.group.uid;
                        GroupPtr_t remove_group_ptr;
                        bool restore_interfaceslots = false;
                        if (inout_graph.GetGroup(module_group_uid, remove_group_ptr)) {
                            if (remove_group_ptr->uid != add_group_ptr->uid) {
                                remove_group_ptr->RemoveModule(module_ptr->uid);
                                restore_interfaceslots = true;
                            }
                        }
                        // Add module to group
                        add_group_ptr->AddModule(module_ptr);
                        inout_graph.ForceSetDirty();
                        // Restore interface slots after adding module to new group
                        if (restore_interfaceslots) {
                            remove_group_ptr->RestoreInterfaceslots();
                        }
                    }
                }
            }
            reset_state = true;
        }
        // Remove module from group
        if (!this->graph_state.interact.modules_remove_group_uids.empty()) {
            for (auto& module_uid : this->graph_state.interact.modules_remove_group_uids) {
                for (auto& remove_group_ptr : inout_graph.GetGroups()) {
                    if (remove_group_ptr->ContainsModule(module_uid)) {
                        remove_group_ptr->RemoveModule(module_uid);
                        inout_graph.ForceSetDirty();
                    }
                }
            }
            reset_state = true;
        }
        // Create new interface slot for call slot
        ImGuiID callslot_uid = this->graph_state.interact.callslot_add_group_uid.first;
        if (callslot_uid != GUI_INVALID_ID) {
            CallSlotPtr_t callslot_ptr = nullptr;
            for (auto& mod : inout_graph.GetModules()) {
                for (auto& callslot_map : mod->GetCallSlots()) {
                    for (auto& callslot : callslot_map.second) {
                        if (callslot->uid == callslot_uid) {
                            callslot_ptr = callslot;
                        }
                    }
                }
            }
            if (callslot_ptr != nullptr) {
                ImGuiID module_uid = this->graph_state.interact.callslot_add_group_uid.second;
                if (module_uid != GUI_INVALID_ID) {
                    for (auto& group : inout_graph.GetGroups()) {
                        if (group->ContainsModule(module_uid)) {
                            group->AddInterfaceSlot(callslot_ptr, false);
                            inout_graph.ForceSetDirty();
                        }
                    }
                }
            }
            reset_state = true;
        }
        // Remove call slot from interface of group
        callslot_uid = this->graph_state.interact.callslot_remove_group_uid.first;
        if (callslot_uid != GUI_INVALID_ID) {
            CallSlotPtr_t callslot_ptr = nullptr;
            for (auto& mod : inout_graph.GetModules()) {
                for (auto& callslot_map : mod->GetCallSlots()) {
                    for (auto& callslot : callslot_map.second) {
                        if (callslot->uid == callslot_uid) {
                            callslot_ptr = callslot;
                        }
                    }
                }
            }
            ImGuiID module_uid = this->graph_state.interact.callslot_remove_group_uid.second;
            if (module_uid != GUI_INVALID_ID) {
                for (auto& group : inout_graph.GetGroups()) {
                    if (group->ContainsModule(module_uid)) {
                        if (group->InterfaceSlot_RemoveCallSlot(callslot_uid, true)) {
                            inout_graph.ForceSetDirty();
                            // Delete call which are connected outside the group
                            std::vector<ImGuiID> call_uids;
                            CallSlotType other_type = (callslot_ptr->type == CallSlotType::CALLEE)
                                                          ? (CallSlotType::CALLER)
                                                          : (CallSlotType::CALLEE);
                            for (auto& call_ptr : callslot_ptr->GetConnectedCalls()) {
                                CallSlotPtr_t other_callslot_ptr = call_ptr->GetCallSlot(other_type);
                                if (other_callslot_ptr->IsParentModuleConnected()) {
                                    if (other_callslot_ptr->GetParentModule()->present.group.uid != group->uid) {
                                        call_uids.emplace_back(call_ptr->uid);
                                    }
                                }
                            }
                            for (auto& call_uid : call_uids) {
                                inout_graph.DeleteCall(call_uid);
                            }
                        }
                    }
                }
            }
            reset_state = true;
        }
        // Process module/call/group deletion
        if ((this->graph_state.interact.process_deletion) ||
            (!io.WantTextInput &&
                std::get<1>(this->graph_state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM]))) {
            if (!this->graph_state.interact.modules_selected_uids.empty()) {
                for (auto& module_uid : this->graph_state.interact.modules_selected_uids) {
                    inout_graph.DeleteModule(module_uid);
                }
            }
            if (this->graph_state.interact.call_selected_uid != GUI_INVALID_ID) {
                inout_graph.DeleteCall(this->graph_state.interact.call_selected_uid);
            }
            if (this->graph_state.interact.group_selected_uid != GUI_INVALID_ID) {
                inout_graph.DeleteGroup(this->graph_state.interact.group_selected_uid);
            }
            if (this->graph_state.interact.interfaceslot_selected_uid != GUI_INVALID_ID) {
                for (auto& group_ptr : inout_graph.GetGroups()) {
                    InterfaceSlotPtr_t interfaceslot_ptr;
                    if (group_ptr->GetInterfaceSlot(
                            this->graph_state.interact.interfaceslot_selected_uid, interfaceslot_ptr)) {
                        // Delete all calls connected
                        std::vector<ImGuiID> call_uids;
                        for (auto& callslot_ptr : interfaceslot_ptr->GetCallSlots()) {
                            for (auto& call_ptr : callslot_ptr->GetConnectedCalls()) {
                                auto caller = call_ptr->GetCallSlot(CallSlotType::CALLER);
                                auto callee = call_ptr->GetCallSlot(CallSlotType::CALLEE);
                                if (caller->IsParentModuleConnected() && callee->IsParentModuleConnected()) {
                                    if (caller->GetParentModule()->present.group.uid !=
                                        callee->GetParentModule()->present.group.uid) {
                                        call_uids.emplace_back(call_ptr->uid);
                                    }
                                }
                            }
                        }
                        for (auto& call_uid : call_uids) {
                            inout_graph.DeleteCall(call_uid);
                        }
                        interfaceslot_ptr.reset();

                        group_ptr->DeleteInterfaceSlot(this->graph_state.interact.interfaceslot_selected_uid);
                        inout_graph.ForceSetDirty();
                    }
                }
            }
            reset_state = true;
        }
        // Delete empty group(s)
        std::vector<ImGuiID> delete_empty_groups_uids;
        for (auto& group_ptr : inout_graph.GetGroups()) {
            if (group_ptr->GetModules().empty()) {
                delete_empty_groups_uids.emplace_back(group_ptr->uid);
            }
        }
        for (auto& group_uid : delete_empty_groups_uids) {
            if (inout_graph.DeleteGroup(group_uid)) {
                reset_state = true;
            }
        }
        if (reset_state) {
            // Reset interact state for modules and call slots
            this->graph_state.interact.process_deletion = false;
            this->graph_state.interact.group_selected_uid = GUI_INVALID_ID;
            this->graph_state.interact.group_hovered_uid = GUI_INVALID_ID;
            this->graph_state.interact.interfaceslot_selected_uid = GUI_INVALID_ID;
            this->graph_state.interact.interfaceslot_hovered_uid = GUI_INVALID_ID;
            this->graph_state.interact.modules_selected_uids.clear();
            this->graph_state.interact.module_hovered_uid = GUI_INVALID_ID;
            this->graph_state.interact.module_mainview_uid = GUI_INVALID_ID;
            this->graph_state.interact.modules_add_group_uids.clear();
            this->graph_state.interact.modules_remove_group_uids.clear();
            this->graph_state.interact.call_selected_uid = GUI_INVALID_ID;
            this->graph_state.interact.call_hovered_uid = GUI_INVALID_ID;
            this->graph_state.interact.callslot_selected_uid = GUI_INVALID_ID;
            this->graph_state.interact.callslot_hovered_uid = GUI_INVALID_ID;
            this->graph_state.interact.callslot_add_group_uid = UIDPair_t(GUI_INVALID_ID, GUI_INVALID_ID);
            this->graph_state.interact.callslot_remove_group_uid = UIDPair_t(GUI_INVALID_ID, GUI_INVALID_ID);
            this->graph_state.interact.slot_dropped_uid = GUI_INVALID_ID;
        }

        // Layout graph
        /// One frame delay required for making sure canvas data is completely updated previously
        if (this->graph_layout > 0) {
            if (this->graph_layout > 1) {
                this->layout_graph(inout_graph);
                this->graph_layout = 0;
            } else {
                this->graph_layout++;
            }
        }
        // Layout modules of selected group
        if (this->graph_state.interact.group_layout) {
            for (auto& group_ptr : inout_graph.GetGroups()) {
                if (group_ptr->uid == this->graph_state.interact.group_selected_uid) {
                    ImVec2 init_position = ImVec2(FLT_MAX, FLT_MAX);
                    for (auto& module_ptr : group_ptr->GetModules()) {
                        init_position.x = std::min(module_ptr->present.position.x, init_position.x);
                        init_position.y = std::min(module_ptr->present.position.y, init_position.y);
                    }
                    this->layout(group_ptr->GetModules(), GroupPtrVector_t(), init_position);
                }
            }
            this->graph_state.interact.group_layout = false;
            this->update = true;
        }
        // Layout selelected modules
        if (this->graph_state.interact.modules_layout) {
            ImVec2 init_position = ImVec2(FLT_MAX, FLT_MAX);
            ModulePtrVector_t selected_modules;
            for (auto& module_ptr : inout_graph.GetModules()) {
                for (auto& selected_module_uid : this->graph_state.interact.modules_selected_uids) {
                    if (module_ptr->uid == selected_module_uid) {
                        init_position.x = std::min(module_ptr->present.position.x, init_position.x);
                        init_position.y = std::min(module_ptr->present.position.y, init_position.y);
                        selected_modules.emplace_back(module_ptr);
                    }
                }
            }
            this->layout(selected_modules, GroupPtrVector_t(), init_position);
            this->graph_state.interact.modules_layout = false;
        }
        // Set delete flag if tab was closed
        bool popup_prevent_close_permanent = false;
        if (!open) {
            if (inout_graph.RunningState().IsTrue()) {
                popup_prevent_close_permanent = true;
            } else {
                state.graph_delete = true;
                state.graph_selected_uid = inout_graph.uid;
            }
        }
        // Propoagate unhandeled hotkeys back to configurator state
        state.hotkeys = this->graph_state.hotkeys;

        // Prevent closing tab of running project pop-up
        bool tmp;
        MinimalPopUp::PopUp(
            "Close Project", popup_prevent_close_permanent, "Running Project can not be closed!", "OK", tmp, "", tmp);

        // Rename pop-up
        if (this->rename_popup.PopUp("Rename Project", popup_rename, inout_graph.name)) {
            inout_graph.ForceSetDirty();
        }

        ImGui::PopID();

    } catch (std::exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


bool megamol::gui::GraphPresentation::StateFromJsonString(Graph& inout_graph, const std::string& in_json_string) {

    try {
        if (in_json_string.empty()) {
            return false;
        }

        bool found = false;
        bool valid = true;

        nlohmann::json json;
        json = nlohmann::json::parse(in_json_string);

        if (!json.is_object()) {
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] State is no valid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
#endif // GUI_VERBOSE
            return false;
        }

        for (auto& header_item : json.items()) {
            if (header_item.key() == GUI_JSON_TAG_GRAPHS) {
                for (auto& content_item : header_item.value().items()) {
                    std::string json_graph_id = content_item.key();
                    if (json_graph_id == inout_graph.GetFilename()) { /// = graph filename
                        found = true;
                        auto config_state = content_item.value();

                        // show_parameter_sidebar
                        bool tmp_show_parameter_sidebar;
                        this->change_show_parameter_sidebar = false;
                        if (config_state.at("show_parameter_sidebar").is_boolean()) {
                            config_state.at("show_parameter_sidebar").get_to(tmp_show_parameter_sidebar);
                            this->change_show_parameter_sidebar = true;
                            this->show_parameter_sidebar = tmp_show_parameter_sidebar;
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read 'show_parameter_sidebar' as boolean. [%s, %s, line "
                                "%d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }
                        // parameter_sidebar_width
                        if (config_state.at("parameter_sidebar_width").is_number_float()) {
                            config_state.at("parameter_sidebar_width").get_to(this->parameter_sidebar_width);
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read first value of "
                                "'parameter_sidebar_width' as float. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }
                        // show_grid
                        if (config_state.at("show_grid").is_boolean()) {
                            config_state.at("show_grid").get_to(this->show_grid);
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read 'show_grid' as boolean. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }

                        // show_call_names
                        if (config_state.at("show_call_names").is_boolean()) {
                            config_state.at("show_call_names").get_to(this->show_call_names);
                            for (auto& call : inout_graph.GetCalls()) {
                                call->present.label_visible = this->show_call_names;
                            }
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read 'show_call_names' as boolean. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }
                        // show_slot_names
                        if (config_state.at("show_slot_names").is_boolean()) {
                            config_state.at("show_slot_names").get_to(this->show_slot_names);
                            for (auto& mod : inout_graph.GetModules()) {
                                for (auto& callslot_types : mod->GetCallSlots()) {
                                    for (auto& callslots : callslot_types.second) {
                                        callslots->present.label_visible = this->show_slot_names;
                                    }
                                }
                            }
                            for (auto& group_ptr : inout_graph.GetGroups()) {
                                for (auto& interfaceslots_map : group_ptr->GetInterfaceSlots()) {
                                    for (auto& interfaceslot_ptr : interfaceslots_map.second) {
                                        interfaceslot_ptr->present.label_visible = this->show_slot_names;
                                    }
                                }
                            }
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read 'show_slot_names' as boolean. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }
                        // show_module_names
                        if (config_state.at("show_module_names").is_boolean()) {
                            config_state.at("show_module_names").get_to(this->show_module_names);
                            for (auto& mod : inout_graph.GetModules()) {
                                mod->present.label_visible = this->show_module_names;
                            }
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read 'show_module_names' as boolean. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }
                        // params_visible
                        if (config_state.at("params_visible").is_boolean()) {
                            config_state.at("params_visible").get_to(this->params_visible);
                            /// Do not apply. Already refelcted in parameter gui state.
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read 'params_visible' as boolean. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }
                        // params_readonly
                        if (config_state.at("params_readonly").is_boolean()) {
                            config_state.at("params_readonly").get_to(this->params_readonly);
                            /// Do not apply. Already refelcted in parameter gui state.
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read 'params_readonly' as boolean. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }
                        // param_extended_mode
                        if (config_state.at("param_extended_mode").is_boolean()) {
                            config_state.at("param_extended_mode").get_to(this->param_extended_mode);
                            for (auto& module_ptr : inout_graph.GetModules()) {
                                for (auto& parameter : module_ptr->parameters) {
                                    parameter.present.extended = this->param_extended_mode;
                                }
                            }
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read 'param_extended_mode' as boolean. [%s, %s, line "
                                "%d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }
                        // canvas_scrolling
                        if (config_state.at("canvas_scrolling").is_array() &&
                            (config_state.at("canvas_scrolling").size() == 2)) {
                            if (config_state.at("canvas_scrolling")[0].is_number_float()) {
                                config_state.at("canvas_scrolling")[0].get_to(this->graph_state.canvas.scrolling.x);
                            } else {
                                megamol::core::utility::log::Log::DefaultLog.WriteError(
                                    "[GUI] JSON state: Failed to read first value of 'canvas_scrolling' as float. [%s, "
                                    "%s, "
                                    "line %d]\n",
                                    __FILE__, __FUNCTION__, __LINE__);
                            }
                            if (config_state.at("canvas_scrolling")[1].is_number_float()) {
                                config_state.at("canvas_scrolling")[1].get_to(this->graph_state.canvas.scrolling.y);
                            } else {
                                megamol::core::utility::log::Log::DefaultLog.WriteError(
                                    "[GUI] JSON state: Failed to read second value of 'canvas_scrolling' as float. "
                                    "[%s, %s, "
                                    "line %d]\n",
                                    __FILE__, __FUNCTION__, __LINE__);
                            }
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read 'canvas_scrolling' as "
                                "array of size two. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }
                        // canvas_zooming
                        if (config_state.at("canvas_zooming").is_number_float()) {
                            config_state.at("canvas_zooming").get_to(this->graph_state.canvas.zooming);
                            this->reset_zooming = false;
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read first value of "
                                "'canvas_zooming' as float. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }

                        // modules
                        for (auto& module_item : content_item.value().items()) {
                            if (module_item.key() == "modules") {
                                for (auto& module_state : module_item.value().items()) {
                                    std::string module_fullname = module_state.key();
                                    auto position_item = module_state.value();
                                    valid = true;

                                    // graph_position
                                    ImVec2 module_position;
                                    if (position_item.at("graph_position").is_array() &&
                                        (position_item.at("graph_position").size() == 2)) {
                                        if (position_item.at("graph_position")[0].is_number_float()) {
                                            position_item.at("graph_position")[0].get_to(module_position.x);
                                        } else {
                                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                                "[GUI] JSON state: Failed to read first value of 'graph_position' as "
                                                "float. "
                                                "[%s, %s, line %d]\n",
                                                __FILE__, __FUNCTION__, __LINE__);
                                            valid = false;
                                        }
                                        if (position_item.at("graph_position")[1].is_number_float()) {
                                            position_item.at("graph_position")[1].get_to(module_position.y);
                                        } else {
                                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                                "[GUI] JSON state: Failed to read second value of 'graph_position' as "
                                                "float. "
                                                "[%s, %s, line %d]\n",
                                                __FILE__, __FUNCTION__, __LINE__);
                                            valid = false;
                                        }
                                    } else {
                                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                                            "[GUI] JSON state: Failed to read 'graph_position' as array of size two. "
                                            "[%s, "
                                            "%s, line %d]\n",
                                            __FILE__, __FUNCTION__, __LINE__);
                                        valid = false;
                                    }

                                    // Apply graph position to module
                                    if (valid) {
                                        bool module_found = false;
                                        for (auto& module_ptr : inout_graph.GetModules()) {
                                            if (module_ptr->FullName() == module_fullname) {
                                                module_ptr->present.position = module_position;
                                                module_found = true;
                                            }
                                        }
                                        if (!module_found) {
                                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                                "[GUI] JSON state: Unable to find module '%s' to apply graph position "
                                                "in "
                                                "configurator. [%s, %s, line %d]\n",
                                                module_fullname.c_str(), __FILE__, __FUNCTION__, __LINE__);
                                        }
                                    }
                                }
                            }
                        }

                        // interfaces
                        for (auto& interfaces_item : content_item.value().items()) {
                            if (interfaces_item.key() == "interfaces") {
                                for (auto& interface_state : interfaces_item.value().items()) {
                                    std::string group_name = interface_state.key();
                                    auto interfaceslot_items = interface_state.value();

                                    // interfaces
                                    for (auto& interfaceslot_item : interfaceslot_items.items()) {
                                        valid = true;
                                        std::vector<std::string> calleslot_fullnames;
                                        for (auto& callslot_item : interfaceslot_item.value().items()) {
                                            if (callslot_item.value().is_string()) {
                                                calleslot_fullnames.emplace_back(
                                                    callslot_item.value().get<std::string>());
                                            } else {
                                                megamol::core::utility::log::Log::DefaultLog.WriteError(
                                                    "[GUI] JSON state: Failed to read value of call slot as string. "
                                                    "[%s, %s, "
                                                    "line %d]\n",
                                                    __FILE__, __FUNCTION__, __LINE__);
                                                valid = false;
                                            }
                                        }

                                        // Add interface slot containing found calls slots to group
                                        if (valid) {
                                            // Find pointers to call slots by name
                                            CallSlotPtrVector_t callslot_ptr_vector;
                                            for (auto& callsslot_fullname : calleslot_fullnames) {
                                                auto split_pos = callsslot_fullname.rfind("::");
                                                if (split_pos != std::string::npos) {
                                                    std::string callslot_name =
                                                        callsslot_fullname.substr(split_pos + 2);
                                                    std::string module_fullname =
                                                        callsslot_fullname.substr(0, (split_pos));
                                                    for (auto& module_ptr : inout_graph.GetModules()) {
                                                        if (module_ptr->FullName() == module_fullname) {
                                                            for (auto& callslot_map : module_ptr->GetCallSlots()) {
                                                                for (auto& callslot_ptr : callslot_map.second) {
                                                                    if (callslot_ptr->name == callslot_name) {
                                                                        callslot_ptr_vector.emplace_back(callslot_ptr);
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            if (!callslot_ptr_vector.empty()) {
                                                bool group_found = false;
                                                for (auto& group_ptr : inout_graph.GetGroups()) {
                                                    if (group_ptr->name == group_name) {
                                                        auto callslot_ptr = callslot_ptr_vector[0];
                                                        // First remove previously added interface slot which was
                                                        // automatically added during adding module to group
                                                        this->ResetStatePointers();
                                                        for (size_t i = 1; i < callslot_ptr_vector.size(); i++) {
                                                            if (group_ptr->InterfaceSlot_ContainsCallSlot(
                                                                    callslot_ptr_vector[i]->uid)) {
                                                                group_ptr->InterfaceSlot_RemoveCallSlot(
                                                                    callslot_ptr_vector[i]->uid, true);
                                                            }
                                                        }
                                                        ImGuiID interfaceslot_uid =
                                                            group_ptr->AddInterfaceSlot(callslot_ptr);
                                                        if (interfaceslot_uid != GUI_INVALID_ID) {
                                                            InterfaceSlotPtr_t interfaceslot_ptr;
                                                            if (group_ptr->GetInterfaceSlot(
                                                                    interfaceslot_uid, interfaceslot_ptr)) {
                                                                for (size_t i = 1; i < callslot_ptr_vector.size();
                                                                     i++) {
                                                                    interfaceslot_ptr->AddCallSlot(
                                                                        callslot_ptr_vector[i], interfaceslot_ptr);
                                                                }
                                                            }
                                                        }
                                                        group_found = true;
                                                    }
                                                }
                                                if (!group_found) {
                                                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                                                        "[GUI] JSON state: Unable to find group '%s' to add interface "
                                                        "slot. "
                                                        "[%s, %s, line %d]\n",
                                                        group_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (found) {
            this->update = true;
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[GUI] Read graph state for '%s' from JSON string.", inout_graph.name.c_str());
#endif // GUI_VERBOSE
        } else {
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] Could not find graph state in JSON. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
#endif // GUI_VERBOSE
            return false;
        }

    } catch (nlohmann::json::type_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::invalid_iterator& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::out_of_range& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::other_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error - Unable to parse JSON string. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


bool megamol::gui::GraphPresentation::StateToJSON(Graph& inout_graph, nlohmann::json& out_json) {

    try {
        /// Append to given json
        // out_json.clear();

        std::string json_graph_id = inout_graph.GetFilename(); /// = graph filename

        // ! State of graph is only stored if project was saved to file previously. Otherwise the project could not be
        // loaded again.
        if (!json_graph_id.empty()) {

            out_json[GUI_JSON_TAG_GRAPHS][json_graph_id]["show_parameter_sidebar"] = this->show_parameter_sidebar;
            out_json[GUI_JSON_TAG_GRAPHS][json_graph_id]["parameter_sidebar_width"] = this->parameter_sidebar_width;
            out_json[GUI_JSON_TAG_GRAPHS][json_graph_id]["show_grid"] = this->show_grid;
            out_json[GUI_JSON_TAG_GRAPHS][json_graph_id]["show_call_names"] = this->show_call_names;
            out_json[GUI_JSON_TAG_GRAPHS][json_graph_id]["show_slot_names"] = this->show_slot_names;
            out_json[GUI_JSON_TAG_GRAPHS][json_graph_id]["show_module_names"] = this->show_module_names;
            out_json[GUI_JSON_TAG_GRAPHS][json_graph_id]["params_visible"] = this->params_visible;
            out_json[GUI_JSON_TAG_GRAPHS][json_graph_id]["params_readonly"] = this->params_readonly;
            out_json[GUI_JSON_TAG_GRAPHS][json_graph_id]["param_extended_mode"] = this->param_extended_mode;
            out_json[GUI_JSON_TAG_GRAPHS][json_graph_id]["canvas_scrolling"] = {
                this->graph_state.canvas.scrolling.x, this->graph_state.canvas.scrolling.y};
            out_json[GUI_JSON_TAG_GRAPHS][json_graph_id]["canvas_zooming"] = this->graph_state.canvas.zooming;

            // Module positions
            for (auto& module_ptr : inout_graph.GetModules()) {
                out_json[GUI_JSON_TAG_GRAPHS][json_graph_id]["modules"][module_ptr->FullName()]["graph_position"] = {
                    module_ptr->present.position.x, module_ptr->present.position.y};
            }
            // Group interface slots
            size_t interface_number = 0;
            for (auto& group_ptr : inout_graph.GetGroups()) {
                for (auto& interfaceslots_map : group_ptr->GetInterfaceSlots()) {
                    for (auto& interface_ptr : interfaceslots_map.second) {
                        std::string interface_label = "interface_slot_" + std::to_string(interface_number);
                        for (auto& callslot_ptr : interface_ptr->GetCallSlots()) {
                            std::string callslot_fullname;
                            if (callslot_ptr->IsParentModuleConnected()) {
                                callslot_fullname =
                                    callslot_ptr->GetParentModule()->FullName() + "::" + callslot_ptr->name;
                            }

                            out_json[GUI_JSON_TAG_GRAPHS][json_graph_id]["interfaces"][group_ptr->name]
                                    [interface_label] += callslot_fullname;
                        }
                        interface_number++;
                    }
                }
            }
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Wrote graph state to JSON.");
#endif // GUI_VERBOSE
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] State of project '%s' is not being saved. Save project to file in "
                "order to get its state saved. [%s, %s, line %d]\n",
                inout_graph.name.c_str(), __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

    } catch (nlohmann::json::type_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::invalid_iterator& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::out_of_range& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::other_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error - Unable to write JSON of state. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
        return false;
    }

    return true;
}


void megamol::gui::GraphPresentation::present_menu(megamol::gui::Graph& inout_graph) {

    const std::string delimiter(" | ");

    const float child_height = ImGui::GetFrameHeightWithSpacing() * 1.0f;
    auto child_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NavFlattened;
    ImGui::BeginChild("graph_menu", ImVec2(0.0f, child_height), false, child_flags);

    // Main View Checkbox
    ModulePtr_t selected_mod_ptr;
    if (inout_graph.GetModule(this->graph_state.interact.module_mainview_uid, selected_mod_ptr)) {
        this->graph_state.interact.module_mainview_uid = GUI_INVALID_ID;
    } else if (this->graph_state.interact.modules_selected_uids.size() == 1) {
        for (auto& mod : inout_graph.GetModules()) {
            if ((this->graph_state.interact.modules_selected_uids[0] == mod->uid) && (mod->is_view)) {
                selected_mod_ptr = mod;
            }
        }
    }
    if (selected_mod_ptr == nullptr) {
        GUIUtils::ReadOnlyWigetStyle(true);
        bool checked = false;
        ImGui::Checkbox("Main View", &checked);
        GUIUtils::ReadOnlyWigetStyle(false);
    } else {
        ImGui::Checkbox("Main View", &selected_mod_ptr->is_view_instance);
        // Set all other (view) modules to non main views
        if (selected_mod_ptr->is_view_instance) {
            for (auto& mod : inout_graph.GetModules()) {
                if (selected_mod_ptr->uid != mod->uid) {
                    mod->is_view_instance = false;
                }
            }
        }
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(delimiter.c_str());
    ImGui::SameLine();

    auto button_size = ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight());

    const float scroll_fac = 10.0f;
    ImGui::Text("Scrolling: %.2f,%.2f", this->graph_state.canvas.scrolling.x, this->graph_state.canvas.scrolling.y);
    ImGui::SameLine();
    ImGui::TextUnformatted("H:");
    ImGui::SameLine();
    if (ImGui::Button("+###hor_incr_scrolling", button_size)) {
        this->graph_state.canvas.scrolling.x += scroll_fac;
    }
    ImGui::SameLine();
    if (ImGui::Button("-###hor_decr_scrolling", button_size)) {
        this->graph_state.canvas.scrolling.x -= scroll_fac;
    }
    ImGui::SameLine();
    ImGui::TextUnformatted("V:");
    ImGui::SameLine();
    if (ImGui::Button("+###vert_incr_scrolling", button_size)) {
        this->graph_state.canvas.scrolling.y += scroll_fac;
    }
    ImGui::SameLine();
    if (ImGui::Button("-###vert_decr_scrolling", button_size)) {
        this->graph_state.canvas.scrolling.y -= scroll_fac;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset###reset_scrolling")) {
        this->graph_state.canvas.scrolling = ImVec2(0.0f, 0.0f);
        this->update = true;
    }
    ImGui::SameLine();
    this->tooltip.Marker("Middle Mouse Button");
    ImGui::SameLine();
    ImGui::TextUnformatted(delimiter.c_str());
    ImGui::SameLine();

    const float zoom_fac = 1.1f; // =10%
    ImGui::Text("Zooming: %.2f", this->graph_state.canvas.zooming);
    ImGui::SameLine();
    if (ImGui::Button("+###incr_zooming", button_size)) {
        this->graph_state.canvas.zooming *= zoom_fac;
    }
    ImGui::SameLine();
    if (ImGui::Button("-###decr_zooming", button_size)) {
        this->graph_state.canvas.zooming /= zoom_fac;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset###reset_zooming")) {
        this->reset_zooming = true;
    }
    ImGui::SameLine();
    this->tooltip.Marker("Mouse Wheel");
    ImGui::SameLine();
    ImGui::TextUnformatted(delimiter.c_str());
    ImGui::SameLine();

    if (ImGui::Checkbox("Grid", &this->show_grid)) {
        inout_graph.ForceSetDirty();
    }

    ImGui::SameLine();

    if (ImGui::Checkbox("Call Names", &this->show_call_names)) {
        for (auto& call_ptr : inout_graph.GetCalls()) {
            call_ptr->present.label_visible = this->show_call_names;
        }
        this->update = true;
    }
    ImGui::SameLine();

    if (ImGui::Checkbox("Module Names", &this->show_module_names)) {
        for (auto& module_ptr : inout_graph.GetModules()) {
            module_ptr->present.label_visible = this->show_module_names;
        }
        this->update = true;
    }
    ImGui::SameLine();

    if (ImGui::Checkbox("Slot Names", &this->show_slot_names)) {
        for (auto& module_ptr : inout_graph.GetModules()) {
            for (auto& callslot_types : module_ptr->GetCallSlots()) {
                for (auto& callslots : callslot_types.second) {
                    callslots->present.label_visible = this->show_slot_names;
                }
            }
        }
        for (auto& group_ptr : inout_graph.GetGroups()) {
            for (auto& interfaceslots_map : group_ptr->GetInterfaceSlots()) {
                for (auto& interfaceslot_ptr : interfaceslots_map.second) {
                    interfaceslot_ptr->present.label_visible = this->show_slot_names;
                }
            }
        }
        this->update = true;
    }
    ImGui::SameLine();

    if (ImGui::Button("Layout Graph")) {
        this->graph_layout = 1;
    }

    ImGui::EndChild();
}


void megamol::gui::GraphPresentation::present_canvas(megamol::gui::Graph& inout_graph, float graph_width) {

    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    // Colors
    const ImU32 COLOR_CANVAS_BACKGROUND = ImGui::ColorConvertFloat4ToU32(
        style.Colors[ImGuiCol_ChildBg]); // ImGuiCol_ScrollbarBg ImGuiCol_ScrollbarGrab ImGuiCol_Border

    ImGui::PushStyleColor(ImGuiCol_ChildBg, COLOR_CANVAS_BACKGROUND);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(1, 1));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    auto child_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove;
    ImGui::BeginChild("region", ImVec2(graph_width, 0.0f), true, child_flags);

    this->canvas_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_None); // Ignores Pop-Ups like Context-Menu

    // UPDATE CANVAS -----------------------------------------------------------

    // Update canvas position
    ImVec2 new_position = ImGui::GetWindowPos();
    if ((this->graph_state.canvas.position.x != new_position.x) ||
        (this->graph_state.canvas.position.y != new_position.y)) {
        this->update = true;
    }
    this->graph_state.canvas.position = new_position;
    // Update canvas size
    ImVec2 new_size = ImGui::GetWindowSize();
    if ((this->graph_state.canvas.size.x != new_size.x) || (this->graph_state.canvas.size.y != new_size.y)) {
        this->update = true;
    }
    this->graph_state.canvas.size = new_size;
    // Update canvas offset
    ImVec2 new_offset =
        this->graph_state.canvas.position + (this->graph_state.canvas.scrolling * this->graph_state.canvas.zooming);
    if ((this->graph_state.canvas.offset.x != new_offset.x) || (this->graph_state.canvas.offset.y != new_offset.y)) {
        this->update = true;
    }
    this->graph_state.canvas.offset = new_offset;

    // Update position and size of modules (and  call slots) and groups.
    if (this->update) {
        inout_graph.ForceSetDirty();
        for (auto& mod : inout_graph.GetModules()) {
            mod->UpdateGUI(this->graph_state.canvas);
        }
        for (auto& group : inout_graph.GetGroups()) {
            group->UpdateGUI(this->graph_state.canvas);
        }
        this->update = false;
    }

    ImGui::PushClipRect(
        this->graph_state.canvas.position, this->graph_state.canvas.position + this->graph_state.canvas.size, true);

    // GRID --------------------------------------
    if (this->show_grid) {
        this->present_canvas_grid();
    }
    ImGui::PopStyleVar(2);

    // Render graph elements using collected button state
    this->graph_state.interact.button_active_uid = GUI_INVALID_ID;
    this->graph_state.interact.button_hovered_uid = GUI_INVALID_ID;
    for (size_t p = 0; p < 2; p++) {
        /// Phase 1: Interaction ---------------------------------------------------
        // Update button states of all graph elements
        /// Phase 2: Rendering -----------------------------------------------------
        PresentPhase phase = static_cast<PresentPhase>(p);

        // 1] GROUPS and INTERFACE SLOTS --------------
        for (auto& group_ptr : inout_graph.GetGroups()) {

            group_ptr->PresentGUI(phase, this->graph_state);

            // 2] MODULES and CALL SLOTS ----------------
            for (auto& module_ptr : inout_graph.GetModules()) {
                if (module_ptr->present.group.uid == group_ptr->uid) {

                    module_ptr->PresentGUI(phase, this->graph_state);

                    // 3] CALLS ---------------------------------;
                    /// Check only for calls of caller slots for considering each call only once
                    for (auto& callslots_ptr : module_ptr->GetCallSlots(CallSlotType::CALLER)) {
                        for (auto& call_ptr : callslots_ptr->GetConnectedCalls()) {

                            bool caller_group = false;
                            auto caller_ptr = call_ptr->GetCallSlot(CallSlotType::CALLER);
                            if (caller_ptr->IsParentModuleConnected()) {
                                if (caller_ptr->GetParentModule()->present.group.uid == group_ptr->uid) {
                                    caller_group = true;
                                }
                            }
                            bool callee_group = false;
                            auto callee_ptr = call_ptr->GetCallSlot(CallSlotType::CALLER);
                            if (callee_ptr->IsParentModuleConnected()) {
                                if (callee_ptr->GetParentModule()->present.group.uid == group_ptr->uid) {
                                    callee_group = true;
                                }
                            }
                            if (caller_group || callee_group) {

                                call_ptr->PresentGUI(phase, this->graph_state);
                            }
                        }
                    }
                }
            }
        }
        // MODULES (non group members)
        for (auto& module_ptr : inout_graph.GetModules()) {
            if (module_ptr->present.group.uid == GUI_INVALID_ID) {
                module_ptr->PresentGUI(phase, this->graph_state);
            }
        }
        // CALLS (connected to call slots which are not part of module which is group member)
        for (auto& call_ptr : inout_graph.GetCalls()) {
            bool caller_group = false;
            auto caller_ptr = call_ptr->GetCallSlot(CallSlotType::CALLER);
            if (caller_ptr->IsParentModuleConnected()) {
                if (caller_ptr->GetParentModule()->present.group.uid != GUI_INVALID_ID) {
                    caller_group = true;
                }
            }
            bool callee_group = false;
            auto callee_ptr = call_ptr->GetCallSlot(CallSlotType::CALLER);
            if (callee_ptr->IsParentModuleConnected()) {
                if (callee_ptr->GetParentModule()->present.group.uid != GUI_INVALID_ID) {
                    callee_group = true;
                }
            }
            if ((!caller_group) && (!callee_group)) {
                call_ptr->PresentGUI(phase, this->graph_state);
            }
        }
    }

    // Multiselection ----------------------------
    this->present_canvas_multiselection(inout_graph);

    // Dragged CALL ------------------------------
    this->present_canvas_dragged_call(inout_graph);

    ImGui::PopClipRect();

    // Zooming and Scaling ----------------------
    // Must be checked inside this canvas child window!
    // Check at the end of drawing for being applied in next frame when font scaling matches zooming.
    if ((ImGui::IsWindowHovered() && !ImGui::IsAnyItemActive()) || this->reset_zooming) {

        // Scrolling (2 = Middle Mouse Button)
        if (ImGui::IsMouseDragging(2)) { // io.KeyCtrl && ImGui::IsMouseDragging(0)) {
            this->graph_state.canvas.scrolling =
                this->graph_state.canvas.scrolling + ImGui::GetIO().MouseDelta / this->graph_state.canvas.zooming;
            this->update = true;
        }

        // Zooming (Mouse Wheel) + Reset
        if ((io.MouseWheel != 0) || this->reset_zooming) {
            float last_zooming = this->graph_state.canvas.zooming;
            ImVec2 current_mouse_pos;
            if (this->reset_zooming) {
                this->graph_state.canvas.zooming = 1.0f;
                current_mouse_pos = this->graph_state.canvas.offset -
                                    (this->graph_state.canvas.position + this->graph_state.canvas.size * 0.5f);
                this->reset_zooming = false;
            } else {
                const float factor = this->graph_state.canvas.zooming / 10.0f;
                this->graph_state.canvas.zooming = this->graph_state.canvas.zooming + (io.MouseWheel * factor);
                current_mouse_pos = this->graph_state.canvas.offset - ImGui::GetMousePos();
            }
            // Limit zooming
            this->graph_state.canvas.zooming =
                (this->graph_state.canvas.zooming <= 0.0f) ? 0.000001f : (this->graph_state.canvas.zooming);
            // Compensate zooming shift of origin
            ImVec2 scrolling_diff = (this->graph_state.canvas.scrolling * last_zooming) -
                                    (this->graph_state.canvas.scrolling * this->graph_state.canvas.zooming);
            this->graph_state.canvas.scrolling += (scrolling_diff / this->graph_state.canvas.zooming);
            // Move origin away from mouse position
            ImVec2 new_mouse_position = (current_mouse_pos / last_zooming) * this->graph_state.canvas.zooming;
            this->graph_state.canvas.scrolling +=
                ((new_mouse_position - current_mouse_pos) / this->graph_state.canvas.zooming);

            this->update = true;
        }
    }

    ImGui::EndChild();
    ImGui::PopStyleColor();

    // FONT scaling
    float font_scaling = this->graph_state.canvas.zooming / this->current_font_scaling;
    // Update when scaling of font has changed due to project tab switching
    if (ImGui::GetFont()->Scale != font_scaling) {
        this->update = true;
    }
    // Font scaling is applied next frame after ImGui::Begin()
    // Font for graph should not be the currently used font of the gui.
    ImGui::GetFont()->Scale = font_scaling;
}


void megamol::gui::GraphPresentation::present_parameters(megamol::gui::Graph& inout_graph, float graph_width) {

    ImGui::BeginGroup();

    float search_child_height = ImGui::GetFrameHeightWithSpacing() * 3.5f;
    auto child_flags =
        ImGuiWindowFlags_AlwaysUseWindowPadding | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NavFlattened;
    ImGui::BeginChild("parameter_search_child", ImVec2(graph_width, search_child_height), false, child_flags);

    ImGui::TextUnformatted("Parameters");
    ImGui::Separator();

    // Mode
    if (megamol::gui::ParameterPresentation::ParameterExtendedModeButton(this->param_extended_mode)) {
        inout_graph.ForceSetDirty();
        for (auto& module_ptr : inout_graph.GetModules()) {
            for (auto& parameter : module_ptr->parameters) {
                parameter.present.extended = this->param_extended_mode;
            }
        }
    }

    if (this->param_extended_mode) {
        ImGui::SameLine();

        // Visibility
        if (ImGui::Checkbox("Visibility", &this->params_visible)) {
            inout_graph.ForceSetDirty();
            for (auto& module_ptr : inout_graph.GetModules()) {
                for (auto& parameter : module_ptr->parameters) {
                    parameter.present.SetGUIVisible(this->params_visible);
                }
            }
        }
        ImGui::SameLine();

        // Read-only option
        if (ImGui::Checkbox("Read-Only", &this->params_readonly)) {
            inout_graph.ForceSetDirty();
            for (auto& module_ptr : inout_graph.GetModules()) {
                for (auto& parameter : module_ptr->parameters) {
                    parameter.present.SetGUIReadOnly(this->params_readonly);
                }
            }
        }
    }

    // Parameter Search
    if (std::get<1>(this->graph_state.hotkeys[megamol::gui::HotkeyIndex::PARAMETER_SEARCH])) {
        this->search_widget.SetSearchFocus(true);
    }
    std::string help_text =
        "[" + std::get<0>(this->graph_state.hotkeys[megamol::gui::HotkeyIndex::PARAMETER_SEARCH]).ToString() +
        "] Set keyboard focus to search input field.\n"
        "Case insensitive substring search in parameter names.";
    this->search_widget.Widget("graph_parameter_search", help_text);
    auto search_string = this->search_widget.GetSearchString();

    ImGui::Separator();

    ImGui::EndChild();

    child_flags = ImGuiWindowFlags_AlwaysVerticalScrollbar | ImGuiWindowFlags_NavFlattened |
                  ImGuiWindowFlags_AlwaysUseWindowPadding;
    ImGui::BeginChild("parameter_list_frame_child", ImVec2(graph_width, 0.0f), false, child_flags);

    if (!this->graph_state.interact.modules_selected_uids.empty()) {
        // Loop over all selected modules
        for (auto& module_uid : this->graph_state.interact.modules_selected_uids) {
            ModulePtr_t module_ptr;
            // Get pointer to currently selected module(s)
            if (inout_graph.GetModule(module_uid, module_ptr)) {
                ImGui::PushID(module_ptr->uid);

                // Set default state of header
                auto headerId = ImGui::GetID(module_ptr->name.c_str());
                auto headerState = ImGui::GetStateStorage()->GetInt(headerId, 1); // 0=close 1=open
                ImGui::GetStateStorage()->SetInt(headerId, headerState);

                if (ImGui::CollapsingHeader(module_ptr->name.c_str(), nullptr, ImGuiTreeNodeFlags_None)) {

                    // Draw parameters
                    module_ptr->present.param_groups.PresentGUI(module_ptr->parameters, module_ptr->FullName(),
                        search_string, vislib::math::Ternary(this->param_extended_mode), true,
                        ParameterPresentation::WidgetScope::LOCAL, nullptr, nullptr);
                }
                this->tooltip.ToolTip(module_ptr->description, ImGui::GetID(module_ptr->name.c_str()), 0.75f, 5.0f);

                ImGui::PopID();
            }
        }
    }
    ImGui::EndChild();

    ImGui::EndGroup();
}


void megamol::gui::GraphPresentation::present_canvas_grid(void) {

    ImGuiStyle& style = ImGui::GetStyle();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    // Color
    const ImU32 COLOR_GRID = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Border]);

    const float GRID_SIZE = 64.0f * this->graph_state.canvas.zooming;
    ImVec2 relative_offset = this->graph_state.canvas.offset - this->graph_state.canvas.position;

    for (float x = fmodf(relative_offset.x, GRID_SIZE); x < this->graph_state.canvas.size.x; x += GRID_SIZE) {
        draw_list->AddLine(ImVec2(x, 0.0f) + this->graph_state.canvas.position,
            ImVec2(x, this->graph_state.canvas.size.y) + this->graph_state.canvas.position, COLOR_GRID);
    }

    for (float y = fmodf(relative_offset.y, GRID_SIZE); y < this->graph_state.canvas.size.y; y += GRID_SIZE) {
        draw_list->AddLine(ImVec2(0.0f, y) + this->graph_state.canvas.position,
            ImVec2(this->graph_state.canvas.size.x, y) + this->graph_state.canvas.position, COLOR_GRID);
    }
}


void megamol::gui::GraphPresentation::present_canvas_dragged_call(megamol::gui::Graph& inout_graph) {

    if (const ImGuiPayload* payload = ImGui::GetDragDropPayload()) {
        if (payload->IsDataType(GUI_DND_CALLSLOT_UID_TYPE)) {
            ImGuiID* selected_slot_uid_ptr = (ImGuiID*)payload->Data;
            if (selected_slot_uid_ptr == nullptr) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Pointer to drag and drop payload data is nullptr. [%s, %s, line %d]\n", __FILE__,
                    __FUNCTION__, __LINE__);
                return;
            }

            ImGuiStyle& style = ImGui::GetStyle();
            ImDrawList* draw_list = ImGui::GetWindowDrawList();

            // Color
            const auto COLOR_CALL_CURVE = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Button]);

            ImVec2 current_pos = ImGui::GetMousePos();
            bool mouse_inside_canvas = false;
            if ((current_pos.x >= this->graph_state.canvas.position.x) &&
                (current_pos.x <= (this->graph_state.canvas.position.x + this->graph_state.canvas.size.x)) &&
                (current_pos.y >= this->graph_state.canvas.position.y) &&
                (current_pos.y <= (this->graph_state.canvas.position.y + this->graph_state.canvas.size.y))) {
                mouse_inside_canvas = true;
            }
            if (mouse_inside_canvas) {

                bool found_valid_slot = false;
                ImVec2 p1;

                CallSlotPtr_t selected_callslot_ptr;
                for (auto& module_ptr : inout_graph.GetModules()) {
                    CallSlotPtr_t callslot_ptr;
                    if (module_ptr->GetCallSlot((*selected_slot_uid_ptr), callslot_ptr)) {
                        selected_callslot_ptr = callslot_ptr;
                    }
                }
                if (selected_callslot_ptr != nullptr) {
                    p1 = selected_callslot_ptr->present.GetPosition();
                    found_valid_slot = true;
                }
                if (!found_valid_slot) {
                    InterfaceSlotPtr_t selected_interfaceslot_ptr;
                    for (auto& group_ptr : inout_graph.GetGroups()) {
                        InterfaceSlotPtr_t interfaceslot_ptr;
                        if (group_ptr->GetInterfaceSlot((*selected_slot_uid_ptr), interfaceslot_ptr)) {
                            selected_interfaceslot_ptr = interfaceslot_ptr;
                        }
                    }
                    if (selected_interfaceslot_ptr != nullptr) {
                        p1 = selected_interfaceslot_ptr->present.GetPosition((*selected_interfaceslot_ptr));
                        found_valid_slot = true;
                    }
                }

                if (found_valid_slot) {
                    ImVec2 p2 = ImGui::GetMousePos();
                    if (p2.x < p1.x) {
                        ImVec2 tmp = p1;
                        p1 = p2;
                        p2 = tmp;
                    }
                    if (glm::length(glm::vec2(p1.x, p1.y) - glm::vec2(p2.x, p2.y)) > GUI_SLOT_RADIUS) {
                        draw_list->AddBezierCurve(p1, p1 + ImVec2(+50, 0), p2 + ImVec2(-50, 0), p2, COLOR_CALL_CURVE,
                            GUI_LINE_THICKNESS * this->graph_state.canvas.zooming);
                    }
                }
            }
        }
    }
}


void megamol::gui::GraphPresentation::present_canvas_multiselection(Graph& inout_graph) {

    bool no_graph_item_selected = ((this->graph_state.interact.callslot_selected_uid == GUI_INVALID_ID) &&
                                   (this->graph_state.interact.call_selected_uid == GUI_INVALID_ID) &&
                                   (this->graph_state.interact.modules_selected_uids.empty()) &&
                                   (this->graph_state.interact.interfaceslot_selected_uid == GUI_INVALID_ID) &&
                                   (this->graph_state.interact.group_selected_uid == GUI_INVALID_ID));

    if (no_graph_item_selected && ImGui::IsWindowHovered() && ImGui::IsMouseDragging(0)) {

        this->multiselect_end_pos = ImGui::GetMousePos();
        this->multiselect_done = true;

        ImGuiStyle& style = ImGui::GetStyle();
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        assert(draw_list != nullptr);

        ImVec4 tmpcol = style.Colors[ImGuiCol_FrameBg];
        tmpcol.w = 0.2f; // alpha
        const ImU32 COLOR_MULTISELECT_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);
        const ImU32 COLOR_MULTISELECT_BORDER = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Border]);

        draw_list->AddRectFilled(multiselect_start_pos, multiselect_end_pos, COLOR_MULTISELECT_BACKGROUND,
            GUI_RECT_CORNER_RADIUS, ImDrawCornerFlags_All);

        float border = 1.0f;
        draw_list->AddRect(multiselect_start_pos, multiselect_end_pos, COLOR_MULTISELECT_BORDER, GUI_RECT_CORNER_RADIUS,
            ImDrawCornerFlags_All, border);
    } else if (this->multiselect_done && ImGui::IsWindowHovered() && ImGui::IsMouseReleased(0)) {
        ImVec2 outer_rect_min = ImVec2(std::min(this->multiselect_start_pos.x, this->multiselect_end_pos.x),
            std::min(this->multiselect_start_pos.y, this->multiselect_end_pos.y));
        ImVec2 outer_rect_max = ImVec2(std::max(this->multiselect_start_pos.x, this->multiselect_end_pos.x),
            std::max(this->multiselect_start_pos.y, this->multiselect_end_pos.y));
        ImVec2 inner_rect_min, inner_rect_max;
        ImVec2 module_size;
        this->graph_state.interact.modules_selected_uids.clear();
        for (auto& module_ptr : inout_graph.GetModules()) {
            bool group_member = (module_ptr->present.group.uid != GUI_INVALID_ID);
            if (!group_member || (group_member && module_ptr->present.group.visible)) {
                module_size = module_ptr->present.GetSize() * this->graph_state.canvas.zooming;
                inner_rect_min =
                    this->graph_state.canvas.offset + module_ptr->present.position * this->graph_state.canvas.zooming;
                inner_rect_max = inner_rect_min + module_size;
                if (((outer_rect_min.x < inner_rect_max.x) && (outer_rect_max.x > inner_rect_min.x) &&
                        (outer_rect_min.y < inner_rect_max.y) && (outer_rect_max.y > inner_rect_min.y))) {
                    this->graph_state.interact.modules_selected_uids.emplace_back(module_ptr->uid);
                }
            }
        }
        this->multiselect_done = false;
    } else {
        this->multiselect_start_pos = ImGui::GetMousePos();
    }
}


void megamol::gui::GraphPresentation::layout_graph(megamol::gui::Graph& inout_graph) {

    ImVec2 init_position = megamol::gui::ModulePresentation::GetDefaultModulePosition(this->graph_state.canvas);

    /// 1] Layout all grouped modules
    for (auto& group_ptr : inout_graph.GetGroups()) {
        this->layout(group_ptr->GetModules(), GroupPtrVector_t(), init_position);
        group_ptr->UpdateGUI(this->graph_state.canvas);
    }

    /// 2] Layout ungrouped modules and groups
    ModulePtrVector_t ungrouped_modules;
    for (auto& module_ptr : inout_graph.GetModules()) {
        if (module_ptr->present.group.uid == GUI_INVALID_ID) {
            ungrouped_modules.emplace_back(module_ptr);
        }
    }
    this->layout(ungrouped_modules, inout_graph.GetGroups(), init_position);

    this->update = true;
}


void megamol::gui::GraphPresentation::layout(
    const ModulePtrVector_t& modules, const GroupPtrVector_t& groups, ImVec2 init_position) {

    struct LayoutItem {
        ModulePtr_t module_ptr;
        GroupPtr_t group_ptr;
        bool considered;

        LayoutItem() : module_ptr(nullptr), group_ptr(nullptr), considered(false) {}
        ~LayoutItem() {}
    };
    std::vector<std::vector<LayoutItem>> layers;
    layers.clear();

    // Fill first layer with graph elements having no connected callee slots
    layers.emplace_back();
    for (auto& group_ptr : groups) {
        bool any_connected_callee = false;
        for (auto& interfaceslot_ptr : group_ptr->GetInterfaceSlots(CallSlotType::CALLEE)) {
            if (this->connected_interfaceslot(modules, groups, interfaceslot_ptr)) {
                any_connected_callee = true;
            }
        }
        if (!any_connected_callee) {
            LayoutItem layout_item;
            layout_item.module_ptr.reset();
            layout_item.group_ptr = group_ptr;
            layers.back().emplace_back(layout_item);
        }
    }
    for (auto& module_ptr : modules) {
        bool any_connected_callee = false;
        for (auto& calleeslot_ptr : module_ptr->GetCallSlots(CallSlotType::CALLEE)) {
            if (this->connected_callslot(modules, groups, calleeslot_ptr)) {
                any_connected_callee = true;
            }
        }
        if (!any_connected_callee) {
            LayoutItem layout_item;
            layout_item.module_ptr = module_ptr;
            layout_item.group_ptr.reset();
            layers.back().emplace_back(layout_item);
        }
    }

    // Loop while graph elements are added to new layer
    bool added_item = true;
    while (added_item) {
        added_item = false;
        layers.emplace_back();

        // Loop through graph elements of last filled layer
        for (auto& layer_item : layers[layers.size() - 2]) {
            CallSlotPtrVector_t callerslots;
            if (layer_item.module_ptr != nullptr) {
                for (auto& callerslot_ptr : layer_item.module_ptr->GetCallSlots(CallSlotType::CALLER)) {
                    if (this->connected_callslot(modules, groups, callerslot_ptr)) {
                        callerslots.emplace_back(callerslot_ptr);
                    }
                }
            } else if (layer_item.group_ptr != nullptr) {
                for (auto& interfaceslot_slot : layer_item.group_ptr->GetInterfaceSlots(CallSlotType::CALLER)) {
                    for (auto& callerslot_ptr : interfaceslot_slot->GetCallSlots()) {
                        if (this->connected_callslot(modules, groups, callerslot_ptr)) {
                            callerslots.emplace_back(callerslot_ptr);
                        }
                    }
                }
            }
            for (auto& callerslot_ptr : callerslots) {
                if (callerslot_ptr->CallsConnected()) {
                    for (auto& call_ptr : callerslot_ptr->GetConnectedCalls()) {
                        if (call_ptr->GetCallSlot(CallSlotType::CALLEE)->IsParentModuleConnected()) {

                            auto add_module_ptr = call_ptr->GetCallSlot(CallSlotType::CALLEE)->GetParentModule();
                            if (this->contains_module(modules, add_module_ptr->uid)) {
                                // Add module only if not already present. Prevents cyclic dependency
                                bool module_already_added = false;
                                for (auto& previous_layer : layers) {
                                    for (auto& previous_layer_item : previous_layer) {
                                        if (previous_layer_item.module_ptr != nullptr) {
                                            if (previous_layer_item.module_ptr->uid == add_module_ptr->uid) {
                                                module_already_added = true;
                                            }
                                        }
                                    }
                                }
                                if (!module_already_added) {
                                    LayoutItem layout_item;
                                    layout_item.module_ptr = add_module_ptr;
                                    layout_item.group_ptr.reset();
                                    layers.back().emplace_back(layout_item);
                                    added_item = true;
                                }
                            } else if (add_module_ptr->present.group.uid != GUI_INVALID_ID) {
                                ImGuiID group_uid = add_module_ptr->present.group.uid; // != GUI_INVALID_ID
                                if (this->contains_group(groups, group_uid)) {
                                    GroupPtr_t add_group_ptr;
                                    for (auto& group_ptr : groups) {
                                        if (group_ptr->uid == group_uid) {
                                            add_group_ptr = group_ptr;
                                        }
                                    }
                                    if (add_group_ptr != nullptr) {
                                        // Add group only if not already present. Prevents cyclic dependency
                                        bool group_already_added = false;
                                        for (auto& previous_layer : layers) {
                                            for (auto& previous_layer_item : previous_layer) {
                                                if (previous_layer_item.group_ptr != nullptr) {
                                                    if (previous_layer_item.group_ptr->uid == add_group_ptr->uid) {
                                                        group_already_added = true;
                                                    }
                                                }
                                            }
                                        }
                                        if (!group_already_added) {
                                            LayoutItem layout_item;
                                            layout_item.module_ptr.reset();
                                            layout_item.group_ptr = add_group_ptr;
                                            layers.back().emplace_back(layout_item);
                                            added_item = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Move modules back to right layer
    for (size_t i = 0; i < layers.size(); i++) {
        for (size_t j = 0; j < layers[i].size(); j++) {

            // Collect caller slots of current graph element
            CallSlotPtrVector_t callerslots;
            if (layers[i][j].module_ptr != nullptr) {
                for (auto& callerslot_ptr : layers[i][j].module_ptr->GetCallSlots(CallSlotType::CALLER)) {
                    callerslots.emplace_back(callerslot_ptr);
                }
            } else if (layers[i][j].group_ptr != nullptr) {
                for (auto& interfaceslot_slot : layers[i][j].group_ptr->GetInterfaceSlots(CallSlotType::CALLER)) {
                    for (auto& callerslot_ptr : interfaceslot_slot->GetCallSlots()) {
                        callerslots.emplace_back(callerslot_ptr);
                    }
                }
            }
            // Collect all connected callee slots
            CallSlotPtrVector_t current_calleeslots;
            for (auto& callerslot_ptr : callerslots) {
                for (auto& call_ptr : callerslot_ptr->GetConnectedCalls()) {
                    auto calleeslot_ptr = call_ptr->GetCallSlot(CallSlotType::CALLEE);
                    current_calleeslots.emplace_back(calleeslot_ptr);
                }
            }

            // Search for connected graph elements lying in same or lower layer and move graph element
            for (size_t k = 0; k <= i; k++) {
                for (size_t m = 0; m < layers[k].size(); m++) {
                    if (!layers[k][m].considered) {
                        CallSlotPtrVector_t other_calleeslots;
                        if (layers[k][m].module_ptr != nullptr) {
                            for (auto& calleeslot_ptr : layers[k][m].module_ptr->GetCallSlots(CallSlotType::CALLEE)) {
                                other_calleeslots.emplace_back(calleeslot_ptr);
                            }
                        } else if (layers[k][m].group_ptr != nullptr) {
                            for (auto& interfaceslot_slot :
                                layers[k][m].group_ptr->GetInterfaceSlots(CallSlotType::CALLEE)) {
                                for (auto& calleeslot_ptr : interfaceslot_slot->GetCallSlots()) {
                                    other_calleeslots.emplace_back(calleeslot_ptr);
                                }
                            }
                        }
                        for (auto& current_calleeslot_ptr : current_calleeslots) {
                            for (auto& other_calleeslot_ptr : other_calleeslots) {
                                if (current_calleeslot_ptr->uid == other_calleeslot_ptr->uid) {
                                    if ((i + 1) == layers.size()) {
                                        layers.emplace_back();
                                    }
                                    layers[i + 1].emplace_back(layers[k][m]);
                                    layers[k][m].module_ptr.reset();
                                    layers[k][m].group_ptr.reset();

                                    layers[i][j].considered = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// DEBUG
    /*
    for (size_t i = 0; i < layers.size(); i++) {
        std::cout << ">>> --- LAYER --- " <<  i <<  std::endl;
        size_t layer_items_count = layers[i].size();
        for (size_t j = 0; j < layer_items_count; j++)  {
            auto layer_item = layers[i][j];
            std::cout << ">>> ITEM " << j << " - " <<  ((layer_item.module_ptr != nullptr)? ("Module: " +
    layer_item.module_ptr->FullName()) : ("")) <<
            ((layer_item.group_ptr != nullptr) ? ("GROUP: " + layer_item.group_ptr->name) : ("")) << std::endl;
        }
    }
    std::cout << std::endl;
    */

    // Calculate new position of graph elements
    ImVec2 pos = init_position;
    float max_call_width = 25.0f;
    float max_graph_element_width = 0.0f;

    size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++) {

        if (this->show_call_names) {
            max_call_width = 0.0f;
        }
        max_graph_element_width = 0.0f;
        pos.y = init_position.y;
        bool found_layer_item = false;

        size_t layer_items_count = layers[i].size();
        for (size_t j = 0; j < layer_items_count; j++) {
            auto layer_item = layers[i][j];

            if (layer_item.module_ptr != nullptr) {
                if (this->show_call_names) {
                    for (auto& callerslot_ptr : layer_item.module_ptr->GetCallSlots(CallSlotType::CALLER)) {
                        if (callerslot_ptr->CallsConnected() &&
                            this->connected_callslot(modules, groups, callerslot_ptr)) {
                            for (auto& call_ptr : callerslot_ptr->GetConnectedCalls()) {
                                auto call_name_length = ImGui::CalcTextSize(call_ptr->class_name.c_str()).x;
                                max_call_width = std::max(call_name_length, max_call_width);
                            }
                        }
                    }
                }
                layer_item.module_ptr->present.position = pos;
                auto module_size = layer_item.module_ptr->present.GetSize();
                pos.y += (module_size.y + GUI_GRAPH_BORDER);
                max_graph_element_width = std::max(module_size.x, max_graph_element_width);
                found_layer_item = true;

            } else if (layer_item.group_ptr != nullptr) {
                if (this->show_call_names) {
                    for (auto& interfaceslot_slot : layer_item.group_ptr->GetInterfaceSlots(CallSlotType::CALLER)) {
                        for (auto& callerslot_ptr : interfaceslot_slot->GetCallSlots()) {
                            if (callerslot_ptr->CallsConnected() &&
                                this->connected_callslot(modules, groups, callerslot_ptr)) {
                                for (auto& call_ptr : callerslot_ptr->GetConnectedCalls()) {
                                    auto call_name_length = ImGui::CalcTextSize(call_ptr->class_name.c_str()).x;
                                    max_call_width = std::max(call_name_length, max_call_width);
                                }
                            }
                        }
                    }
                }
                layer_item.group_ptr->SetGUIPosition(this->graph_state.canvas, pos);
                auto group_size = layer_item.group_ptr->present.GetSize();
                pos.y += (group_size.y + GUI_GRAPH_BORDER);
                max_graph_element_width = std::max(group_size.x, max_graph_element_width);
                found_layer_item = true;
            }
        }
        if (found_layer_item) {
            pos.x += (max_graph_element_width + max_call_width + (2.0f * GUI_GRAPH_BORDER));
        }
    }
}


bool megamol::gui::GraphPresentation::connected_callslot(
    const ModulePtrVector_t& modules, const GroupPtrVector_t& groups, const CallSlotPtr_t& callslot_ptr) {

    bool retval = false;
    for (auto& call_ptr : callslot_ptr->GetConnectedCalls()) {
        CallSlotType type =
            (callslot_ptr->type == CallSlotType::CALLER) ? (CallSlotType::CALLEE) : (CallSlotType::CALLER);
        auto connected_callslot_ptr = call_ptr->GetCallSlot(type);
        if (connected_callslot_ptr != nullptr) {
            if (this->contains_callslot(modules, connected_callslot_ptr->uid)) {
                retval = true;
                break;
            }
            if (connected_callslot_ptr->present.group.interfaceslot_ptr != nullptr) {
                if (this->contains_interfaceslot(
                        groups, connected_callslot_ptr->present.group.interfaceslot_ptr->uid)) {
                    retval = true;
                    break;
                }
            }
        }
    }
    return retval;
}


bool megamol::gui::GraphPresentation::connected_interfaceslot(
    const ModulePtrVector_t& modules, const GroupPtrVector_t& groups, const InterfaceSlotPtr_t& interfaceslot_ptr) {

    bool retval = false;
    for (auto& callslot_ptr : interfaceslot_ptr->GetCallSlots()) {
        for (auto& call_ptr : callslot_ptr->GetConnectedCalls()) {
            CallSlotType type =
                (callslot_ptr->type == CallSlotType::CALLER) ? (CallSlotType::CALLEE) : (CallSlotType::CALLER);
            auto connected_callslot_ptr = call_ptr->GetCallSlot(type);
            if (connected_callslot_ptr != nullptr) {
                if (this->contains_callslot(modules, connected_callslot_ptr->uid)) {
                    retval = true;
                    break;
                }
                if (connected_callslot_ptr->present.group.interfaceslot_ptr != nullptr) {
                    if (this->contains_interfaceslot(
                            groups, connected_callslot_ptr->present.group.interfaceslot_ptr->uid)) {
                        retval = true;
                        break;
                    }
                }
            }
        }
    }
    return retval;
}


bool megamol::gui::GraphPresentation::contains_callslot(const ModulePtrVector_t& modules, ImGuiID callslot_uid) {

    for (auto& module_ptr : modules) {
        for (auto& callslots_map : module_ptr->GetCallSlots()) {
            for (auto& callslot_ptr : callslots_map.second) {
                if (callslot_ptr->uid == callslot_uid) {
                    return true;
                }
            }
        }
    }
    return false;
}


bool megamol::gui::GraphPresentation::contains_interfaceslot(
    const GroupPtrVector_t& groups, ImGuiID interfaceslot_uid) {

    for (auto& group_ptr : groups) {
        for (auto& interfaceslots_map : group_ptr->GetInterfaceSlots()) {
            for (auto& interfaceslot_ptr : interfaceslots_map.second) {
                if (interfaceslot_ptr->uid == interfaceslot_uid) {
                    return true;
                }
            }
        }
    }
    return false;
}


bool megamol::gui::GraphPresentation::contains_module(const ModulePtrVector_t& modules, ImGuiID module_uid) {

    for (auto& module_ptr : modules) {
        if (module_ptr->uid == module_uid) {
            return true;
        }
    }
    return false;
}


bool megamol::gui::GraphPresentation::contains_group(const GroupPtrVector_t& groups, ImGuiID group_uid) {

    for (auto& group_ptr : groups) {
        if (group_ptr->uid == group_uid) {
            return true;
        }
    }
    return false;
}
