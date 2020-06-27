/*
 * Graph.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Graph.h"


using namespace megamol;
using namespace megamol::gui;
using namespace megamol::gui::configurator;


ImGuiID megamol::gui::configurator::Graph::generated_uid = 0; /// must be greater than or equal to zero


megamol::gui::configurator::Graph::Graph(const std::string& graph_name)
    : uid(this->generate_unique_id())
    , name(graph_name)
    , group_name_uid(0)
    , modules()
    , calls()
    , groups()
    , dirty_flag(true)
    , present() {}


megamol::gui::configurator::Graph::~Graph(void) {}


ImGuiID megamol::gui::configurator::Graph::AddModule(
    const ModuleStockVectorType& stock_modules, const std::string& module_class_name) {

    try {
        for (auto& mod : stock_modules) {
            if (module_class_name == mod.class_name) {
                ImGuiID mod_uid = this->generate_unique_id();
                auto mod_ptr = std::make_shared<Module>(mod_uid);
                mod_ptr->class_name = mod.class_name;
                mod_ptr->description = mod.description;
                mod_ptr->plugin_name = mod.plugin_name;
                mod_ptr->is_view = mod.is_view;
                mod_ptr->name = this->generate_unique_module_name(mod.class_name);
                mod_ptr->is_view_instance = false;
                mod_ptr->GUI_SetLabelVisibility(this->present.GetModuleLabelVisibility());

                for (auto& p : mod.parameters) {
                    Parameter param_slot(this->generate_unique_id(), p.type, p.storage, p.minval, p.maxval);
                    param_slot.full_name = p.full_name;
                    param_slot.description = p.description;
                    param_slot.SetValueString(p.default_value, true);
                    param_slot.GUI_SetLabelVisibility(this->present.params_visible);
                    param_slot.GUI_SetReadOnly(this->present.params_readonly);
                    param_slot.GUI_SetExpert(this->present.params_expert);

                    mod_ptr->parameters.emplace_back(param_slot);
                }

                for (auto& call_slots_type : mod.call_slots) {
                    for (auto& c : call_slots_type.second) {
                        CallSlot call_slot(this->generate_unique_id());
                        call_slot.name = c.name;
                        call_slot.description = c.description;
                        call_slot.compatible_call_idxs = c.compatible_call_idxs;
                        call_slot.type = c.type;
                        call_slot.GUI_SetLabelVisibility(this->present.GetCallSlotLabelVisibility());

                        mod_ptr->AddCallSlot(std::make_shared<CallSlot>(call_slot));
                    }
                }

                for (auto& call_slot_type_list : mod_ptr->GetCallSlots()) {
                    for (auto& call_slot : call_slot_type_list.second) {
                        call_slot->ConnectParentModule(mod_ptr);
                    }
                }

                this->modules.emplace_back(mod_ptr);
                vislib::sys::Log::DefaultLog.WriteInfo(
                    "Added module '%s' to project '%s'.\n", mod_ptr->class_name.c_str(), this->name.c_str());

                this->dirty_flag = true;

                return mod_uid;
            }
        }
    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    }

    vislib::sys::Log::DefaultLog.WriteError("Unable to find module in stock: %s [%s, %s, line %d]\n",
        module_class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
    return GUI_INVALID_ID;
}


bool megamol::gui::configurator::Graph::DeleteModule(ImGuiID module_uid) {

    try {
        for (auto iter = this->modules.begin(); iter != this->modules.end(); iter++) {
            if ((*iter)->uid == module_uid) {

                // First reset module and call slot pointers in groups
                for (auto& group : this->groups) {
                    if (group->ContainsModule(module_uid)) {
                        group->RemoveModule(module_uid);
                    }
                    if (group->EmptyModules()) {
                        this->DeleteGroup(group->uid);
                    }
                }

                // Second remove call slots
                (*iter)->RemoveAllCallSlots();

                if ((*iter).use_count() > 1) {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "Unclean deletion. Found %i references pointing to module. [%s, %s, line %d]\n",
                        (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                }

                vislib::sys::Log::DefaultLog.WriteInfo(
                    "Deleted module '%s' from  project '%s'.\n", (*iter)->class_name.c_str(), this->name.c_str());
                (*iter).reset();
                this->modules.erase(iter);

                this->delete_disconnected_calls();
                this->dirty_flag = true;
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


bool megamol::gui::configurator::Graph::GetModule(
    ImGuiID module_uid, megamol::gui::configurator::ModulePtrType& out_module_ptr) {

    if (module_uid != GUI_INVALID_ID) {
        for (auto& module_ptr : this->modules) {
            if (module_ptr->uid == module_uid) {
                out_module_ptr = module_ptr;
                return true;
            }
        }
    }
    return false;
}


bool megamol::gui::configurator::Graph::AddCall(
    const CallStockVectorType& stock_calls, CallSlotPtrType call_slot_1, CallSlotPtrType call_slot_2) {

    try {

        auto compat_idx = CallSlot::GetCompatibleCallIndex(call_slot_1, call_slot_2);
        if (compat_idx == GUI_INVALID_ID) {
            vislib::sys::Log::DefaultLog.WriteWarn(
                "Unable to find compatible call. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        Call::StockCall call_stock_data = stock_calls[compat_idx];

        auto call_ptr = std::make_shared<Call>(this->generate_unique_id());
        call_ptr->class_name = call_stock_data.class_name;
        call_ptr->description = call_stock_data.description;
        call_ptr->plugin_name = call_stock_data.plugin_name;
        call_ptr->functions = call_stock_data.functions;
        call_ptr->GUI_SetLabelVisibility(this->present.GetCallLabelVisibility());

        if ((call_slot_1->type == CallSlotType::CALLER) && (call_slot_1->CallsConnected())) {
            call_slot_1->DisConnectCalls();
            this->delete_disconnected_calls();
        }
        if ((call_slot_2->type == CallSlotType::CALLER) && (call_slot_2->CallsConnected())) {
            call_slot_2->DisConnectCalls();
            this->delete_disconnected_calls();
        }

        if (call_ptr->ConnectCallSlots(call_slot_1, call_slot_2) && call_slot_1->ConnectCall(call_ptr) &&
            call_slot_2->ConnectCall(call_ptr)) {

            this->calls.emplace_back(call_ptr);
            vislib::sys::Log::DefaultLog.WriteInfo(
                "Added call '%s' to project '%s'.\n", call_ptr->class_name.c_str(), this->name.c_str());

            // Add connected call slots to interface of group of the parent module
            if (call_slot_1->ParentModuleConnected() && call_slot_2->ParentModuleConnected()) {
                ImGuiID slot_1_parent_group_uid = call_slot_1->GetParentModule()->GUI_GetGroupMembership();
                ImGuiID slot_2_parent_group_uid = call_slot_2->GetParentModule()->GUI_GetGroupMembership();
                if (slot_1_parent_group_uid != slot_2_parent_group_uid) {
                    for (auto& group : this->groups) {
                        if (group->uid == slot_1_parent_group_uid) {
                            group->AddCallSlot(call_slot_1);
                        }
                    }
                    for (auto& group : this->groups) {
                        if (group->uid == slot_2_parent_group_uid) {
                            group->AddCallSlot(call_slot_2);
                        }
                    }
                }
            }

            this->dirty_flag = true;
        } else {
            this->DeleteCall(call_ptr->uid);
            vislib::sys::Log::DefaultLog.WriteWarn("Unable to connect call: %s [%s, %s, line %d]\n",
                call_ptr->class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
            return false;
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


bool megamol::gui::configurator::Graph::DeleteCall(ImGuiID call_uid) {

    try {
        for (auto iter = this->calls.begin(); iter != this->calls.end(); iter++) {
            if ((*iter)->uid == call_uid) {
                (*iter)->DisConnectCallSlots();

                if ((*iter).use_count() > 1) {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "Unclean deletion. Found %i references pointing to call. [%s, %s, line %d]\n",
                        (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                }

                vislib::sys::Log::DefaultLog.WriteInfo("Deleted call '%s' from  project '%s'.\n",
                    (*iter)->class_name.c_str(), this->name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                (*iter).reset();
                this->calls.erase(iter);

                this->dirty_flag = true;
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

    vislib::sys::Log::DefaultLog.WriteWarn("Invalid call uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}


ImGuiID megamol::gui::configurator::Graph::AddGroup(const std::string& group_name) {

    try {
        ImGuiID group_id = this->generate_unique_id();
        auto group_ptr = std::make_shared<Group>(group_id);
        group_ptr->name = (group_name.empty()) ? (this->generate_unique_group_name()) : (group_name);
        this->groups.emplace_back(group_ptr);

        vislib::sys::Log::DefaultLog.WriteInfo(
            "Added group '%s' to project '%s'.\n", group_ptr->name.c_str(), this->name.c_str());
        return group_id;

    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    }

    return GUI_INVALID_ID;
}


bool megamol::gui::configurator::Graph::DeleteGroup(ImGuiID group_uid) {

    try {
        for (auto iter = this->groups.begin(); iter != this->groups.end(); iter++) {
            if ((*iter)->uid == group_uid) {

                if ((*iter).use_count() > 1) {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "Unclean deletion. Found %i references pointing to group. [%s, %s, line %d]\n",
                        (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                }

                vislib::sys::Log::DefaultLog.WriteInfo("Deleted group '%s' from  project '%s'.\n",
                    (*iter)->name.c_str(), this->name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                (*iter).reset();
                this->groups.erase(iter);

                this->present.ForceUpdate();
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

    vislib::sys::Log::DefaultLog.WriteWarn("Invalid group uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}


ImGuiID megamol::gui::configurator::Graph::AddGroupModule(
    const std::string& group_name, const ModulePtrType& module_ptr) {

    try {
        // Only create new group if given name is not empty
        if (!group_name.empty()) {
            // Check if group with given name already exists
            ImGuiID existing_group_uid = GUI_INVALID_ID;
            for (auto& group : this->groups) {
                if (group->name == group_name) {
                    existing_group_uid = group->uid;
                }
            }
            // Create new group if there is no one with given name
            if (existing_group_uid == GUI_INVALID_ID) {
                existing_group_uid = this->AddGroup(group_name);
            }
            // Add module to group
            for (auto& group : this->groups) {
                if (group->uid == existing_group_uid) {
                    if (group->AddModule(module_ptr)) {
                        return existing_group_uid;
                    }
                }
            }
        }
    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    }

    return GUI_INVALID_ID;
}


bool megamol::gui::configurator::Graph::GetGroup(
    ImGuiID group_uid, megamol::gui::configurator::GroupPtrType& out_group_ptr) {

    if (group_uid != GUI_INVALID_ID) {
        for (auto& group_ptr : this->groups) {
            if (group_ptr->uid == group_uid) {
                out_group_ptr = group_ptr;
                return true;
            }
        }
    }
    return false;
}


bool megamol::gui::configurator::Graph::UniqueModuleRename(const std::string& module_name) {

    for (auto& mod : this->modules) {
        if (module_name == mod->name) {
            mod->name = this->generate_unique_module_name(module_name);
            this->present.ForceUpdate();
            return true;
        }
    }
    return false;
}


bool megamol::gui::configurator::Graph::IsMainViewSet(void) {

    for (auto& mod : this->modules) {
        if (mod->is_view_instance) {
            return true;
        }
    }
    return false;
}


bool megamol::gui::configurator::Graph::delete_disconnected_calls(void) {

    try {
        // Create separate uid list to avoid iterator conflict when operating on calls list while deleting.
        UIDVectorType call_uids;
        for (auto& call : this->calls) {
            if (!call->IsConnected()) {
                call_uids.emplace_back(call->uid);
            }
        }
        for (auto& id : call_uids) {
            this->DeleteCall(id);
            this->dirty_flag = true;
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


const std::string megamol::gui::configurator::Graph::generate_unique_group_name(void) {

    int new_name_id = 0;
    std::string new_name_prefix = "Group_";
    for (auto& group : this->groups) {
        if (group->name.find(new_name_prefix) == 0) {
            std::string int_postfix = group->name.substr(new_name_prefix.length());
            try {
                int last_id = std::stoi(int_postfix);
                new_name_id = std::max(new_name_id, last_id);
            } catch (...) {
            }
        }
    }
    return std::string(new_name_prefix + std::to_string(new_name_id + 1));
}


const std::string megamol::gui::configurator::Graph::generate_unique_module_name(const std::string& module_name) {

    int new_name_id = 0;
    std::string new_name_prefix = module_name + "_";
    for (auto& mod : this->modules) {
        if (mod->name.find(new_name_prefix) == 0) {
            std::string int_postfix = mod->name.substr(new_name_prefix.length());
            try {
                int last_id = std::stoi(int_postfix);
                new_name_id = std::max(new_name_id, last_id);
            } catch (...) {
            }
        }
    }
    return std::string(new_name_prefix + std::to_string(new_name_id + 1));
}


// GRAPH PRESENTATION ####################################################

megamol::gui::configurator::Graph::Presentation::Presentation(void)
    : params_visible(true)
    , params_readonly(false)
    , params_expert(false)
    , utils()
    , update(true)
    , show_grid(false)
    , show_call_names(true)
    , show_slot_names(false)
    , show_module_names(true)
    , layout_current_graph(false)
    , child_split_width(300.0f)
    , reset_zooming(true)
    , param_name_space()
    , multiselect_start_pos()
    , multiselect_end_pos()
    , multiselect_done(false)
    , canvas_hovered(false)
    , current_font_scaling(1.0f)
    , graph_state() {

    this->graph_state.canvas.position = ImVec2(0.0f, 0.0f);
    this->graph_state.canvas.size = ImVec2(1.0f, 1.0f);
    this->graph_state.canvas.scrolling = ImVec2(0.0f, 0.0f);
    this->graph_state.canvas.zooming = 1.0f;
    this->graph_state.canvas.offset = ImVec2(0.0f, 0.0f);

    this->graph_state.interact.group_selected_uid = GUI_INVALID_ID;
    this->graph_state.interact.group_save = false;

    this->graph_state.interact.modules_selected_uids.clear();
    this->graph_state.interact.module_hovered_uid = GUI_INVALID_ID;
    this->graph_state.interact.module_mainview_uid = GUI_INVALID_ID;
    this->graph_state.interact.modules_add_group_uids.clear();
    this->graph_state.interact.modules_remove_group_uids.clear();

    this->graph_state.interact.call_selected_uid = GUI_INVALID_ID;

    this->graph_state.interact.callslot_selected_uid = GUI_INVALID_ID;
    this->graph_state.interact.callslot_hovered_uid = GUI_INVALID_ID;
    this->graph_state.interact.callslot_dropped_uid = GUI_INVALID_ID;
    this->graph_state.interact.callslot_add_group_uid = UIDPairType(GUI_INVALID_ID, GUI_INVALID_ID);
    this->graph_state.interact.callslot_remove_group_uid = GUI_INVALID_ID;
    this->graph_state.interact.callslot_compat_ptr = nullptr;

    this->graph_state.groups.clear();

    // this->graph_state.hotkeys are already initialzed
}


megamol::gui::configurator::Graph::Presentation::~Presentation(void) {}


void megamol::gui::configurator::Graph::Presentation::Present(
    megamol::gui::configurator::Graph& inout_graph, GraphStateType& state) {

    try {
        if (ImGui::GetCurrentContext() == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return;
        }
        ImGuiIO& io = ImGui::GetIO();

        ImGuiID graph_uid = inout_graph.uid;
        ImGui::PushID(graph_uid);

        // State Init/Reset ----------------------
        this->graph_state.hotkeys = state.hotkeys;
        this->graph_state.groups.clear();
        for (auto& group : inout_graph.GetGroups()) {
            std::pair<ImGuiID, std::string> group_pair(group->uid, group->name);
            this->graph_state.groups.emplace_back(group_pair);
        }
        this->graph_state.interact.callslot_compat_ptr.reset();
        if (this->graph_state.interact.callslot_selected_uid != GUI_INVALID_ID) {
            for (auto& mods : inout_graph.GetModules()) {
                CallSlotPtrType call_slot_ptr;
                if (mods->GetCallSlot(this->graph_state.interact.callslot_selected_uid, call_slot_ptr)) {
                    this->graph_state.interact.callslot_compat_ptr = call_slot_ptr;
                }
            }
        }
        if (this->graph_state.interact.callslot_hovered_uid != GUI_INVALID_ID) {
            for (auto& mods : inout_graph.GetModules()) {
                CallSlotPtrType call_slot_ptr;
                if (mods->GetCallSlot(this->graph_state.interact.callslot_hovered_uid, call_slot_ptr)) {
                    this->graph_state.interact.callslot_compat_ptr = call_slot_ptr;
                }
            }
        }
        this->graph_state.interact.callslot_dropped_uid = GUI_INVALID_ID;

        // Tab showing this graph ---------------
        bool popup_rename = false;
        ImGuiTabItemFlags tab_flags = ImGuiTabItemFlags_None;
        if (inout_graph.IsDirty()) {
            tab_flags |= ImGuiTabItemFlags_UnsavedDocument;
        }
        std::string graph_label = "    " + inout_graph.name + "  ###graph" + std::to_string(graph_uid);
        bool open = true;
        if (ImGui::BeginTabItem(graph_label.c_str(), &open, tab_flags)) {
            // Context menu
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Rename")) {
                    popup_rename = true;
                }
                ImGui::EndPopup();
            }

            // Draw -----------------------------
            this->present_menu(inout_graph);

            float child_width_auto = 0.0f;
            if (state.show_parameter_sidebar) {
                this->utils.VerticalSplitter(
                    GUIUtils::FixedSplitterSide::RIGHT, child_width_auto, this->child_split_width);
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
                this->present_canvas(inout_graph, child_width_auto);
                ImGui::PopFont();
            } else {
                throw std::invalid_argument("Pointer to font is nullptr.");
            }

            if (state.show_parameter_sidebar) {
                ImGui::SameLine();
                this->present_parameters(inout_graph, this->child_split_width);
            }

            // Apply graph layout
            if (this->layout_current_graph) {
                this->layout_graph(inout_graph);
                this->layout_current_graph = false;
                this->update = true;
            }

            state.graph_selected_uid = inout_graph.uid;
            ImGui::EndTabItem();
        }

        // State processing ---------------------
        // Add module to group
        if (!this->graph_state.interact.modules_add_group_uids.empty()) {
            ModulePtrType module_ptr;
            ImGuiID new_group_uid = GUI_INVALID_ID;
            for (auto& uid_pair : this->graph_state.interact.modules_add_group_uids) {
                module_ptr.reset();
                for (auto& mod : inout_graph.GetModules()) {
                    if (mod->uid == uid_pair.first) {
                        module_ptr = mod;
                    }
                }
                if (module_ptr != nullptr) {

                    // Add module to new or alredy existing group
                    /// Create new group for multiple selected modules only once
                    ImGuiID group_uid = GUI_INVALID_ID;
                    if ((uid_pair.second == GUI_INVALID_ID) && (new_group_uid == GUI_INVALID_ID)) {
                        new_group_uid = inout_graph.AddGroup();
                    }
                    if (uid_pair.second == GUI_INVALID_ID) {
                        group_uid = new_group_uid;
                    } else {
                        group_uid = uid_pair.second;
                    }

                    GroupPtrType add_group_ptr;
                    if (inout_graph.GetGroup(group_uid, add_group_ptr)) {
                        // Remove module from previous associated group
                        ImGuiID module_group_member_uid = module_ptr->GUI_GetGroupMembership();
                        GroupPtrType remove_group_ptr;
                        if (inout_graph.GetGroup(module_group_member_uid, remove_group_ptr)) {
                            if (remove_group_ptr->uid != add_group_ptr->uid) {
                                remove_group_ptr->RemoveModule(module_ptr->uid);
                            }
                        }
                        // Add module to group
                        add_group_ptr->AddModule(module_ptr);
                    }
                }
            }
            this->graph_state.interact.modules_add_group_uids.clear();
        }
        // Remove module from group
        if (!this->graph_state.interact.modules_remove_group_uids.empty()) {
            for (auto& module_uid : this->graph_state.interact.modules_remove_group_uids) {
                for (auto& remove_group_ptr : inout_graph.GetGroups()) {
                    if (remove_group_ptr->ContainsModule(module_uid)) {
                        remove_group_ptr->RemoveModule(module_uid);
                    }
                }
            }
            this->graph_state.interact.modules_remove_group_uids.clear();
        }
        // Add call slot to group interface
        ImGuiID callslot_uid = this->graph_state.interact.callslot_add_group_uid.first;
        if (callslot_uid != GUI_INVALID_ID) {
            CallSlotPtrType callslot_ptr = nullptr;
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
                            group->AddCallSlot(callslot_ptr);
                        }
                    }
                }
            }
            this->graph_state.interact.callslot_add_group_uid.first = GUI_INVALID_ID;
            this->graph_state.interact.callslot_add_group_uid.second = GUI_INVALID_ID;
        }
        // Remove call slot from group interface
        callslot_uid = this->graph_state.interact.callslot_remove_group_uid;
        if (callslot_uid != GUI_INVALID_ID) {
            for (auto& group : inout_graph.GetGroups()) {
                if (group->ContainsCallSlot(callslot_uid)) {
                    group->RemoveCallSlot(callslot_uid);
                }
            }
            this->graph_state.interact.callslot_remove_group_uid = GUI_INVALID_ID;
        }
        // Process module/call/group deletion
        if (std::get<1>(this->graph_state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM])) {
            if (!this->graph_state.interact.modules_selected_uids.empty()) {
                for (auto& module_uid : this->graph_state.interact.modules_selected_uids) {
                    inout_graph.DeleteModule(module_uid);
                }
                // Reset interact state for modules and call slots
                this->graph_state.interact.modules_selected_uids.clear();
                this->graph_state.interact.module_hovered_uid = GUI_INVALID_ID;
                this->graph_state.interact.module_mainview_uid = GUI_INVALID_ID;
                this->graph_state.interact.modules_add_group_uids.clear();
                this->graph_state.interact.modules_remove_group_uids.clear();
                this->graph_state.interact.callslot_selected_uid = GUI_INVALID_ID;
                this->graph_state.interact.callslot_hovered_uid = GUI_INVALID_ID;
                this->graph_state.interact.callslot_dropped_uid = GUI_INVALID_ID;
                this->graph_state.interact.callslot_add_group_uid = UIDPairType(GUI_INVALID_ID, GUI_INVALID_ID);
                this->graph_state.interact.callslot_remove_group_uid = GUI_INVALID_ID;
                this->graph_state.interact.callslot_compat_ptr = nullptr;
            }
            if (this->graph_state.interact.call_selected_uid != GUI_INVALID_ID) {
                inout_graph.DeleteCall(this->graph_state.interact.call_selected_uid);
                // Reset interact state for calls
                this->graph_state.interact.call_selected_uid = GUI_INVALID_ID;
            }
            if (this->graph_state.interact.group_selected_uid != GUI_INVALID_ID) {
                inout_graph.DeleteGroup(this->graph_state.interact.group_selected_uid);
                // Reset interact state for groups
                this->graph_state.interact.group_selected_uid = GUI_INVALID_ID;
                this->graph_state.interact.group_save = false;
            }
        }
        // Set delete flag if tab was closed
        if (!open) {
            state.graph_delete = true;
        }
        // Propoagate unhandeled hotkeys back to configurator state
        state.hotkeys = this->graph_state.hotkeys;

        // Rename pop-up
        this->utils.RenamePopUp("Rename Project", popup_rename, inout_graph.name);

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


void megamol::gui::configurator::Graph::Presentation::present_menu(megamol::gui::configurator::Graph& inout_graph) {

    const float child_height = ImGui::GetFrameHeightWithSpacing() * 1.0f;
    auto child_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NavFlattened;
    ImGui::BeginChild("graph_menu", ImVec2(0.0f, child_height), false, child_flags);

    // Main View Checkbox
    ModulePtrType selected_mod_ptr;
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

    ImGui::Text("Scrolling: %.2f,%.2f", this->graph_state.canvas.scrolling.x, this->graph_state.canvas.scrolling.y);
    ImGui::SameLine();
    if (ImGui::Button("Reset###reset_scrolling")) {
        this->graph_state.canvas.scrolling = ImVec2(0.0f, 0.0f);
        this->update = true;
    }
    ImGui::SameLine();

    ImGui::Text("Zooming: %.2f", this->graph_state.canvas.zooming);
    ImGui::SameLine();
    if (ImGui::Button("Reset###reset_zooming")) {
        this->reset_zooming = true;
    }
    ImGui::SameLine();

    ImGui::Checkbox("Grid", &this->show_grid);

    ImGui::SameLine();

    if (ImGui::Checkbox("Call Names", &this->show_call_names)) {
        for (auto& call : inout_graph.get_calls()) {
            call->GUI_SetLabelVisibility(this->show_call_names);
        }
        this->update = true;
    }
    ImGui::SameLine();

    if (ImGui::Checkbox("Module Names", &this->show_module_names)) {
        for (auto& mod : inout_graph.GetModules()) {
            mod->GUI_SetLabelVisibility(this->show_module_names);
        }
        this->update = true;
    }
    ImGui::SameLine();

    if (ImGui::Checkbox("Slot Names", &this->show_slot_names)) {
        for (auto& mod : inout_graph.GetModules()) {
            for (auto& call_slot_types : mod->GetCallSlots()) {
                for (auto& call_slots : call_slot_types.second) {
                    call_slots->GUI_SetLabelVisibility(this->show_slot_names);
                }
            }
        }
        this->update = true;
    }
    ImGui::SameLine();

    if (ImGui::Button("Layout Graph")) {
        this->layout_current_graph = true;
    }

    ImGui::EndChild();
}


void megamol::gui::configurator::Graph::Presentation::present_canvas(
    megamol::gui::configurator::Graph& inout_graph, float child_width) {

    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    // Colors
    const ImU32 COLOR_CANVAS_BACKGROUND = ImGui::ColorConvertFloat4ToU32(
        style.Colors[ImGuiCol_ChildBg]); // ImGuiCol_ScrollbarBg ImGuiCol_ScrollbarGrab ImGuiCol_Border

    ImGui::PushStyleColor(ImGuiCol_ChildBg, COLOR_CANVAS_BACKGROUND);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(1, 1));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    auto child_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove;
    ImGui::BeginChild("region", ImVec2(child_width, 0.0f), true, child_flags);

    this->canvas_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows);

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
        for (auto& mod : inout_graph.GetModules()) {
            mod->GUI_Update(this->graph_state.canvas);
        }
        for (auto& group : inout_graph.GetGroups()) {
            group->GUI_Update(this->graph_state.canvas);
        }
        this->update = false;
    }

    ImGui::PushClipRect(
        this->graph_state.canvas.position, this->graph_state.canvas.position + this->graph_state.canvas.size, true);

    // 1] GRID ----------------------------------
    if (this->show_grid) {
        this->present_canvas_grid();
    }
    ImGui::PopStyleVar(2);

    // 2] GROUPS --------------------------------
    for (auto& group : inout_graph.GetGroups()) {
        group->GUI_Present(this->graph_state);
    }

    // 3] MODULES and CALL SLOTS ----------------
    for (auto& mod : inout_graph.GetModules()) {
        mod->GUI_Present(this->graph_state);
    }

    // 4] CALLS ---------------------------------;
    for (auto& call : inout_graph.get_calls()) {
        call->GUI_Present(this->graph_state);
    }

    // 5] Multiselection
    this->present_canvas_multiselection(inout_graph);

    // 5] Dragged CALL --------------------------
    this->present_canvas_dragged_call(inout_graph);

    ImGui::PopClipRect();

    // Zooming and Scaling ----------------------
    /// Must be checked inside canvas child window.
    /// Check at the end for being applied in next frame when font scaling matches zooming.
    if ((ImGui::IsWindowHovered() && !ImGui::IsAnyItemActive()) || this->reset_zooming) {

        // Scrolling (2 = Middle Mouse Button)
        if (ImGui::IsMouseDragging(2, 0.0f)) {
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


void megamol::gui::configurator::Graph::Presentation::present_parameters(
    megamol::gui::configurator::Graph& inout_graph, float child_width) {

    ImGui::BeginGroup();

    float search_child_height = ImGui::GetFrameHeightWithSpacing() * 3.5f;
    auto child_flags =
        ImGuiWindowFlags_AlwaysUseWindowPadding | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NavFlattened;
    ImGui::BeginChild("parameter_search_child", ImVec2(child_width, search_child_height), false, child_flags);

    ImGui::TextUnformatted("Parameters");
    ImGui::Separator();

    if (std::get<1>(this->graph_state.hotkeys[megamol::gui::HotkeyIndex::PARAMETER_SEARCH])) {
        this->utils.SetSearchFocus(true);
    }
    std::string help_text =
        "[" + std::get<0>(this->graph_state.hotkeys[megamol::gui::HotkeyIndex::PARAMETER_SEARCH]).ToString() +
        "] Set keyboard focus to search input field.\n"
        "Case insensitive substring search in parameter names.";
    this->utils.StringSearch("graph_parameter_search", help_text);
    auto search_string = this->utils.GetSearchString();

    // Mode
    ImGui::BeginGroup();
    this->utils.PointCircleButton("Mode");
    if (ImGui::BeginPopupContextItem("param_mode_button_context", 0)) { // 0 = left mouse button
        bool changed = false;
        if (ImGui::MenuItem("Basic", nullptr, (this->params_expert == false))) {
            this->params_expert = false;
            changed = true;
        }
        if (ImGui::MenuItem("Expert", nullptr, (this->params_expert == true))) {
            this->params_expert = true;
            changed = true;
        }
        if (changed) {
            for (auto& module_ptr : inout_graph.GetModules()) {
                for (auto& param : module_ptr->parameters) {
                    param.GUI_SetExpert(this->params_expert);
                }
            }
        }
        ImGui::EndPopup();
    }
    ImGui::EndGroup();

    if (this->params_expert) {
        ImGui::SameLine();

        // Visibility
        if (ImGui::Checkbox("Visibility", &this->params_visible)) {
            for (auto& module_ptr : inout_graph.GetModules()) {
                for (auto& param : module_ptr->parameters) {
                    param.GUI_SetLabelVisibility(this->params_visible);
                }
            }
        }
        ImGui::SameLine();

        // Read-only option
        if (ImGui::Checkbox("Read-Only", &this->params_readonly)) {
            for (auto& module_ptr : inout_graph.GetModules()) {
                for (auto& param : module_ptr->parameters) {
                    param.GUI_SetReadOnly(this->params_readonly);
                }
            }
        }
    }
    ImGui::Separator();

    ImGui::EndChild();

    child_flags = ImGuiWindowFlags_AlwaysVerticalScrollbar | ImGuiWindowFlags_NavFlattened |
                  ImGuiWindowFlags_AlwaysUseWindowPadding;
    ImGui::BeginChild("parameter_list_frame_child", ImVec2(child_width, 0.0f), false, child_flags);

    if (!this->graph_state.interact.modules_selected_uids.empty()) {
        // Loop over all selected modules
        for (auto& module_uid : this->graph_state.interact.modules_selected_uids) {
            ModulePtrType module_ptr;
            // Get pointer to currently selected module(s)
            if (inout_graph.GetModule(module_uid, module_ptr)) {
                if (module_ptr->parameters.size() > 0) {

                    ImGui::PushID(module_ptr->uid);

                    // Set default state of header
                    auto headerId = ImGui::GetID(module_ptr->name.c_str());
                    auto headerState = ImGui::GetStateStorage()->GetInt(headerId, 1); // 0=close 1=open
                    ImGui::GetStateStorage()->SetInt(headerId, headerState);

                    if (ImGui::CollapsingHeader(module_ptr->name.c_str(), nullptr, ImGuiTreeNodeFlags_None)) {
                        this->utils.HoverToolTip(
                            module_ptr->description, ImGui::GetID(module_ptr->name.c_str()), 0.75f, 5.0f);

                        bool param_name_space_open = true;
                        unsigned int param_indent_stack = 0;
                        for (auto& param : module_ptr->parameters) {
                            // Filter module by given search string
                            bool search_filter = true;
                            if (!search_string.empty()) {
                                search_filter =
                                    this->utils.FindCaseInsensitiveSubstring(param.full_name, search_string);
                            }

                            // Add Collapsing header depending on parameter namespace
                            std::string current_param_namespace = param.GetNameSpace();
                            if (current_param_namespace != this->param_name_space) {
                                this->param_name_space = current_param_namespace;
                                while (param_indent_stack > 0) {
                                    param_indent_stack--;
                                    ImGui::Unindent();
                                }

                                if (!this->param_name_space.empty()) {
                                    ImGui::Indent();
                                    std::string label = this->param_name_space + "###" + param.full_name;
                                    // Open all namespace headers when parameter search is active
                                    if (!search_string.empty()) {
                                        auto headerId = ImGui::GetID(label.c_str());
                                        ImGui::GetStateStorage()->SetInt(headerId, 1);
                                    }
                                    param_name_space_open =
                                        ImGui::CollapsingHeader(label.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
                                    param_indent_stack++;
                                } else {
                                    param_name_space_open = true;
                                }
                            }

                            // Draw parameter
                            if (search_filter && param_name_space_open) {
                                param.GUI_Present();
                            }
                        }

                        // Vertical spacing using dummy
                        ImGui::Dummy(ImVec2(1.0f, ImGui::GetFrameHeightWithSpacing()));
                    }
                    ImGui::PopID();
                }
            }
        }
    }
    ImGui::EndChild();

    ImGui::EndGroup();
}


void megamol::gui::configurator::Graph::Presentation::present_canvas_grid(void) {

    ImGuiStyle& style = ImGui::GetStyle();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

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


void megamol::gui::configurator::Graph::Presentation::present_canvas_dragged_call(
    megamol::gui::configurator::Graph& inout_graph) {

    if (const ImGuiPayload* payload = ImGui::GetDragDropPayload()) {
        if (payload->IsDataType(GUI_DND_CALL_UID_TYPE)) {
            ImGuiID* selected_call_slot_uid_ptr = (ImGuiID*)payload->Data;

            ImGuiStyle& style = ImGui::GetStyle();
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            assert(draw_list != nullptr);

            const auto COLOR_CALL_CURVE = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Button]);
            const float CURVE_THICKNESS = 3.0f;

            ImVec2 current_pos = ImGui::GetMousePos();
            bool mouse_inside_canvas = false;
            if ((current_pos.x >= this->graph_state.canvas.position.x) &&
                (current_pos.x <= (this->graph_state.canvas.position.x + this->graph_state.canvas.size.x)) &&
                (current_pos.y >= this->graph_state.canvas.position.y) &&
                (current_pos.y <= (this->graph_state.canvas.position.y + this->graph_state.canvas.size.y))) {
                mouse_inside_canvas = true;
            }
            if (mouse_inside_canvas) {

                CallSlotPtrType selected_call_slot_ptr;
                for (auto& mods : inout_graph.GetModules()) {
                    CallSlotPtrType call_slot_ptr;
                    if (mods->GetCallSlot(*selected_call_slot_uid_ptr, call_slot_ptr)) {
                        selected_call_slot_ptr = call_slot_ptr;
                    }
                }

                if (selected_call_slot_ptr != nullptr) {
                    ImVec2 p1 = selected_call_slot_ptr->GUI_GetPosition();
                    ImVec2 p2 = ImGui::GetMousePos();
                    if (glm::length(glm::vec2(p1.x, p1.y) - glm::vec2(p2.x, p2.y)) > GUI_CALL_SLOT_RADIUS) {
                        if (selected_call_slot_ptr->type == CallSlotType::CALLEE) {
                            ImVec2 tmp = p1;
                            p1 = p2;
                            p2 = tmp;
                        }
                        draw_list->AddBezierCurve(p1, p1 + ImVec2(+50, 0), p2 + ImVec2(-50, 0), p2, COLOR_CALL_CURVE,
                            CURVE_THICKNESS * this->graph_state.canvas.zooming);
                    }
                }
            }
        }
    }
}


void megamol::gui::configurator::Graph::Presentation::present_canvas_multiselection(Graph& inout_graph) {

    bool no_graph_item_selected = ((this->graph_state.interact.callslot_selected_uid == GUI_INVALID_ID) &&
                                   (this->graph_state.interact.call_selected_uid == GUI_INVALID_ID) &&
                                   (this->graph_state.interact.modules_selected_uids.empty()) &&
                                   (this->graph_state.interact.group_selected_uid == GUI_INVALID_ID));

    if (no_graph_item_selected && ImGui::IsMouseDragging(0)) {

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
        for (auto& module_ptr : inout_graph.modules) {
            bool group_member = (module_ptr->GUI_GetGroupMembership() != GUI_INVALID_ID);
            if (!group_member || (group_member && module_ptr->GUI_GetGroupVisibility())) {
                module_size = module_ptr->GUI_GetSize() * this->graph_state.canvas.zooming;
                inner_rect_min =
                    this->graph_state.canvas.offset + module_ptr->GUI_GetPosition() * this->graph_state.canvas.zooming;
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


bool megamol::gui::configurator::Graph::Presentation::layout_graph(megamol::gui::configurator::Graph& inout_graph) {

    /// Really simple layouting, sorting modules into differnet layers 'from left to right' following the calls.

    std::vector<std::vector<ModulePtrType>> layers;
    layers.clear();

    // Fill first layer with modules having no connected callee
    layers.emplace_back();
    for (auto& mod : inout_graph.GetModules()) {
        bool any_connected_callee = false;
        for (auto& callee_slot : mod->GetCallSlots(CallSlotType::CALLEE)) {
            if (callee_slot->CallsConnected()) {
                any_connected_callee = true;
            }
        }
        if (!any_connected_callee) {
            layers.back().emplace_back(mod);
        }
    }

    // Loop while modules are added to new layer.
    bool added_module = true;
    while (added_module) {
        added_module = false;
        // Add new layer
        layers.emplace_back();
        // Loop through last filled layer
        for (auto& layer_mod : layers[layers.size() - 2]) {
            for (auto& caller_slot : layer_mod->GetCallSlots(CallSlotType::CALLER)) {
                if (caller_slot->CallsConnected()) {
                    for (auto& call : caller_slot->GetConnectedCalls()) {
                        auto add_mod = call->GetCallSlot(CallSlotType::CALLEE)->GetParentModule();

                        // Add module only if not already present in current layer
                        bool module_already_added = false;
                        for (auto& last_layer_mod : layers.back()) {
                            if (last_layer_mod == add_mod) {
                                module_already_added = true;
                            }
                        }
                        if (!module_already_added) {
                            layers.back().emplace_back(add_mod);
                            added_module = true;
                        }
                    }
                }
            }
        }
    }

    // Deleting duplicate modules from back to front
    int layer_size = static_cast<int>(layers.size());
    for (int i = (layer_size - 1); i >= 0; i--) {
        for (auto& layer_module : layers[i]) {
            for (int j = (i - 1); j >= 0; j--) {
                for (auto module_iter = layers[j].begin(); module_iter != layers[j].end(); module_iter++) {
                    if ((*module_iter) == layer_module) {
                        layers[j].erase(module_iter);
                        break;
                    }
                }
            }
        }
    }

    // Calculate new positions of modules
    ImVec2 init_position = megamol::gui::configurator::Module::GUI_GetInitModulePosition(this->graph_state.canvas);
    ImVec2 pos = init_position;
    float max_call_width = 25.0f;
    float max_module_width = 0.0f;
    size_t layer_mod_cnt = 0;
    for (auto& layer : layers) {
        if (this->show_call_names) {
            max_call_width = 0.0f;
        }
        max_module_width = 0.0f;
        layer_mod_cnt = layer.size();
        for (size_t i = 0; i < layer_mod_cnt; i++) {
            auto mod = layer[i];
            if (this->show_call_names) {
                for (auto& caller_slot : mod->GetCallSlots(CallSlotType::CALLER)) {
                    if (caller_slot->CallsConnected()) {
                        for (auto& call : caller_slot->GetConnectedCalls()) {
                            auto call_name_length = GUIUtils::TextWidgetWidth(call->class_name) * 1.5f;
                            max_call_width =
                                (call_name_length > max_call_width) ? (call_name_length) : (max_call_width);
                        }
                    }
                }
            }
            mod->GUI_SetPosition(pos);
            auto mod_size = mod->GUI_GetSize();
            pos.y += mod_size.y + GUI_GRAPH_BORDER;
            max_module_width = (mod_size.x > max_module_width) ? (mod_size.x) : (max_module_width);
        }
        pos.x += (max_module_width + max_call_width + GUI_GRAPH_BORDER);

        pos.x += GUI_GRAPH_BORDER;
        pos.y = init_position.y + GUI_GRAPH_BORDER;
    }

    return true;
}
