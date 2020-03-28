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
    : modules(), calls(), groups(), uid(this->generate_unique_id()), name(graph_name), dirty_flag(true), group_name_uid(0), present() {}


megamol::gui::configurator::Graph::~Graph(void) {}


bool megamol::gui::configurator::Graph::AddModule(
    const ModuleStockVectorType& stock_modules, const std::string& module_class_name) {

    try {
        for (auto& mod : stock_modules) {
            if (module_class_name == mod.class_name) {
                auto mod_ptr = std::make_shared<Module>(this->generate_unique_id());
                mod_ptr->class_name = mod.class_name;
                mod_ptr->description = mod.description;
                mod_ptr->plugin_name = mod.plugin_name;
                mod_ptr->is_view = mod.is_view;
                mod_ptr->name = this->generate_unique_module_name(mod.class_name);
                mod_ptr->name_space = "";
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
                vislib::sys::Log::DefaultLog.WriteInfo("Added module '%s'. [%s, %s, line %d]\n",
                    mod_ptr->class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);

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

    vislib::sys::Log::DefaultLog.WriteError("Unable to find module in stock: %s [%s, %s, line %d]\n",
        module_class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
    return false;
}


bool megamol::gui::configurator::Graph::DeleteModule(ImGuiID module_uid) {

    try {
        for (auto iter = this->modules.begin(); iter != this->modules.end(); iter++) {
            if ((*iter)->uid == module_uid) {
                (*iter)->RemoveAllCallSlots();

                for (auto& group : this->groups) {
                    if (group.ContainsModule(module_uid)) {
                        group.RemoveModule(module_uid);
                    }
                    if (group.Empty()) {
                        this->DeleteGroup(group.uid);
                    }
                }   

                if ((*iter).use_count() > 1) {
                    vislib::sys::Log::DefaultLog.WriteError("Unclean deletion. Found %i references pointing to module. [%s, %s, line %d]\n",
                        (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                }

                vislib::sys::Log::DefaultLog.WriteInfo("Deleted module: %s [%s, %s, line %d]\n",
                    (*iter)->class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
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

    //vislib::sys::Log::DefaultLog.WriteWarn("Invalid module uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
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

        if ((call_slot_1->type == CallSlot::CallSlotType::CALLER) && (call_slot_1->CallsConnected())) {
            call_slot_1->DisConnectCalls();
            this->delete_disconnected_calls();
        }
        if ((call_slot_2->type == CallSlot::CallSlotType::CALLER) && (call_slot_2->CallsConnected())) {
            call_slot_2->DisConnectCalls();
            this->delete_disconnected_calls();
        }

        if (call_ptr->ConnectCallSlots(call_slot_1, call_slot_2) && call_slot_1->ConnectCall(call_ptr) &&
            call_slot_2->ConnectCall(call_ptr)) {

            this->calls.emplace_back(call_ptr);
            vislib::sys::Log::DefaultLog.WriteInfo(
                "Added call '%s'. [%s, %s, line %d]\n", call_ptr->class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);

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

bool megamol::gui::configurator::Graph::delete_disconnected_calls(void) {

    try {
        // Create separate uid list to avoid iterator conflict when operating on calls list while deleting.
        std::vector<ImGuiID> call_uids;
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


bool megamol::gui::configurator::Graph::DeleteCall(ImGuiID call_uid) {

    try {
        for (auto iter = this->calls.begin(); iter != this->calls.end(); iter++) {
            if ((*iter)->uid == call_uid) {
                (*iter)->DisConnectCallSlots();

                if ((*iter).use_count() > 1) {
                    vislib::sys::Log::DefaultLog.WriteError("Unclean deletion. Found %i references pointing to call. [%s, %s, line %d]\n",
                        (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                }

                vislib::sys::Log::DefaultLog.WriteInfo("Deleted call: %s [%s, %s, line %d]\n",
                    (*iter)->class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
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

    //vislib::sys::Log::DefaultLog.WriteWarn("Invalid call uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}


bool megamol::gui::configurator::Graph::UniqueModuleRename(const std::string& module_name) {

    for (auto& mod : this->modules) {
        if (module_name == mod->name) {
            mod->name = this->generate_unique_module_name(module_name);
            this->present.ApplyUpdate();
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


ImGuiID megamol::gui::configurator::Graph::AddGroup(void) {

   return this->add_group(this->generate_unique_group_name());
}


bool megamol::gui::configurator::Graph::AddGroupModule(const ModulePtrType& module_ptr) {

    // Check module namespace for group membership
    std::string name_space = module_ptr->name_space;

    ImGuiID group_uid = this->get_group_uid(name_space);
    if (group_uid == GUI_INVALID_ID) {
        group_uid = this->add_group(name_space);
    }
    if (group_uid != GUI_INVALID_ID) {
        for (auto& group : this->groups) {
            if (group.uid == group_uid) {
                return group.AddModule(module_ptr);
            }
        }
    }
    return false;
}


bool megamol::gui::configurator::Graph::DeleteGroup(ImGuiID group_uid) {

    bool found_group = false;
    try {
        auto tmp_groups = this->groups;
        this->groups.clear();

        for (auto& group : tmp_groups) {
            if (group.uid != group_uid) {
                this->groups.emplace_back(group);
            }
            else {
                for (auto& mod : group.GetGroupModules()) {
                    mod->name_space.clear(); 
                }
                this->present.ApplyUpdate();
                vislib::sys::Log::DefaultLog.WriteInfo("Deleted group '%s'.\n",group.name.c_str());  
                found_group = true;
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

    if (!found_group) {
        vislib::sys::Log::DefaultLog.WriteWarn("Invalid group uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }           
    return true;
}


ImGuiID megamol::gui::configurator::Graph::add_group(const std::string& group_name) {

    try {
        // Only create new group if group_name is not equal to empty global namespace
        ///  XXX or if group name equals namespace of graph <= ???
        if (!group_name.empty() && (group_name != this->name)) {

            ImGuiID group_id = this->generate_unique_id();
            Group group(group_id);
            group.name = group_name;
            this->groups.emplace_back(group);

            vislib::sys::Log::DefaultLog.WriteInfo("Added group '%s'. [%s, %s, line %d]\n",
                group.name.c_str(), __FILE__, __FUNCTION__, __LINE__);  

            return group_id;
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


ImGuiID megamol::gui::configurator::Graph::get_group_uid(const std::string& group_name) {

    for (auto& group : this->groups) {
        if (group.name == group_name) {
            return group.uid;
        }
    }
    return GUI_INVALID_ID;
}


std::string megamol::gui::configurator::Graph::generate_unique_module_name(const std::string& module_name) {


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
    : utils()
    , update(true)
    , show_grid(false)
    , show_call_names(true)
    , show_slot_names(false)
    , show_module_names(true)
    , layout_current_graph(false)
    , child_split_width(300.0f)
    , reset_zooming(true)
    , params_visible(true)
    , params_readonly(false)
    , params_expert(false)
    , param_name_space()
    , graphstate() {

    this->graphstate.canvas.position = ImVec2(0.0f, 0.0f);
    this->graphstate.canvas.size = ImVec2(1.0f, 1.0f);
    this->graphstate.canvas.scrolling = ImVec2(0.0f, 0.0f);
    this->graphstate.canvas.zooming = 1.0f;
    this->graphstate.canvas.offset = ImVec2(0.0f, 0.0f);

    this->graphstate.interact.group_selected_uid = GUI_INVALID_ID;
    this->graphstate.interact.group_save = false;
    this->graphstate.interact.module_selected_uid = GUI_INVALID_ID;
    this->graphstate.interact.module_hovered_uid = GUI_INVALID_ID;
    this->graphstate.interact.module_add_group_uid = UIDPairType(GUI_INVALID_ID, GUI_INVALID_ID);
    this->graphstate.interact.module_remove_group_uid = GUI_INVALID_ID;
    this->graphstate.interact.call_selected_uid = GUI_INVALID_ID;
    this->graphstate.interact.callslot_selected_uid = GUI_INVALID_ID;
    this->graphstate.interact.callslot_hovered_uid = GUI_INVALID_ID;
    this->graphstate.interact.callslot_dropped_uid = GUI_INVALID_ID;
    this->graphstate.interact.callslot_add_group_uid = UIDPairType(GUI_INVALID_ID, GUI_INVALID_ID);
    this->graphstate.interact.callslot_remove_group_uid = GUI_INVALID_ID;
    this->graphstate.interact.callslot_compat_ptr = nullptr;

    this->graphstate.groups.clear();

    //this->graphstate.hotkeys are already initialzed
}


megamol::gui::configurator::Graph::Presentation::~Presentation(void) {}


void megamol::gui::configurator::Graph::Presentation::Present(megamol::gui::configurator::Graph& inout_graph, GraphStateType& state) {

    try {
        if (ImGui::GetCurrentContext() == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return;
        }

        ImGuiIO& io = ImGui::GetIO();
        ImGuiID graph_uid = inout_graph.GetUID();

        bool popup_rename = false;

        ImGui::PushID(graph_uid);

        // State Reset --------------------------
        this->graphstate.hotkeys = state.hotkeys;
        this->graphstate.groups.clear();
        for (auto& group : inout_graph.get_graph_groups()) {
            std::pair<ImGuiID, std::string> group_pair(group.uid, group.name);
            this->graphstate.groups.emplace_back(group_pair);
        }
        this->graphstate.interact.callslot_compat_ptr.reset();
        if (this->graphstate.interact.callslot_selected_uid != GUI_INVALID_ID) {
            for (auto& mods : inout_graph.GetGraphModules()) {
                CallSlotPtrType call_slot_ptr = mods->GetCallSlot(this->graphstate.interact.callslot_selected_uid);
                if (call_slot_ptr != nullptr) {
                    this->graphstate.interact.callslot_compat_ptr = call_slot_ptr;
                }
            }
        }
        if (this->graphstate.interact.callslot_hovered_uid != GUI_INVALID_ID) {
            for (auto& mods : inout_graph.GetGraphModules()) {
                CallSlotPtrType call_slot_ptr = mods->GetCallSlot(this->graphstate.interact.callslot_hovered_uid);
                if (call_slot_ptr != nullptr) {
                    this->graphstate.interact.callslot_compat_ptr = call_slot_ptr;
                }
            }
        }
        this->graphstate.interact.callslot_dropped_uid = GUI_INVALID_ID;        

        // Tab showing this graph ---------------
        ImGuiTabItemFlags tab_flags = ImGuiTabItemFlags_None;
        if (inout_graph.IsDirty()) {
            tab_flags |= ImGuiTabItemFlags_UnsavedDocument;
        }
        std::string graph_label = "    " + inout_graph.GetName() + "  ###graph" + std::to_string(graph_uid);
        bool open = true;
        if (ImGui::BeginTabItem(graph_label.c_str(), &open, tab_flags)) {
            // Context menu
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Rename")) {
                    popup_rename = true;
                }
                ImGui::EndPopup();
            }

            // Apply graph layout
            if (this->layout_current_graph) {
                this->layout_graph(inout_graph);
                this->layout_current_graph = false;
                this->update = true;                
            }

            // Draw -----------------------------
            this->present_menu(inout_graph);

            float child_width_auto = 0.0f;
            if (state.show_parameter_sidebar) { 
                this->utils.VerticalSplitter(
                    GUIUtils::FixedSplitterSide::RIGHT, child_width_auto, this->child_split_width);
            }

            if (state.font == nullptr) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "Found no font for configurator. Provide font via GuiView::SetGraphFont(). [%s, %s, line %d]\n", __FILE__,
                    __FUNCTION__, __LINE__);
                return;
            }
            ImGui::PushFont(state.font);
            this->present_canvas(inout_graph, child_width_auto);
            ImGui::PopFont();  

            if (state.show_parameter_sidebar) { 
                ImGui::SameLine();
                this->present_parameters(inout_graph, this->child_split_width);
            }

            state.graph_selected_uid = inout_graph.GetUID();  
            ImGui::EndTabItem();
        }

        // State processing ---------------------
        // Add module to group
        ImGuiID module_uid = this->graphstate.interact.module_add_group_uid.first;
        if (module_uid != GUI_INVALID_ID) {
            ModulePtrType module_ptr = nullptr;
            for (auto& mod : inout_graph.GetGraphModules()) {
                if (mod->uid == module_uid) {
                    module_ptr = mod;
                }
            }
            if (module_ptr != nullptr) {
                ImGuiID group_uid = this->graphstate.interact.module_add_group_uid.second;
                if (group_uid == GUI_INVALID_ID) {
                    group_uid = inout_graph.AddGroup();
                }
                if (group_uid != GUI_INVALID_ID) {
                    for (auto& group : inout_graph.get_graph_groups()) {
                        if (group.uid == group_uid) {
                            if (module_ptr != nullptr) {
                                group.AddModule(module_ptr);
                            }
                        }
                    }    
                }
            }
            this->graphstate.interact.module_add_group_uid.first = GUI_INVALID_ID;
            this->graphstate.interact.module_add_group_uid.second = GUI_INVALID_ID;
        }
        // Remove module from group
        module_uid = this->graphstate.interact.module_remove_group_uid;
        if (module_uid != GUI_INVALID_ID) {
            for (auto& group : inout_graph.get_graph_groups()) {
                if (group.ContainsModule(module_uid)) {
                    group.RemoveModule(module_uid);
                }
            }
            this->graphstate.interact.module_remove_group_uid = GUI_INVALID_ID;
        }
        // Add call slot to group interface
        ImGuiID callslot_uid = this->graphstate.interact.callslot_add_group_uid.first;
        if (callslot_uid != GUI_INVALID_ID) {
            ImGuiID group_id = this->graphstate.interact.module_add_group_uid.second;
            /// TODO


            this->graphstate.interact.callslot_add_group_uid.first = GUI_INVALID_ID;
            this->graphstate.interact.callslot_add_group_uid.second = GUI_INVALID_ID;
        }         
        // Remove call slot from group interface  
        callslot_uid  = this->graphstate.interact.callslot_remove_group_uid;
        if (callslot_uid != GUI_INVALID_ID) {
            for (auto& group : inout_graph.get_graph_groups()) {
                if (group.ContainsCallSlot(callslot_uid)) {
                    group.RemoveCallSlot(callslot_uid);
                }
            }
            this->graphstate.interact.callslot_remove_group_uid = GUI_INVALID_ID;
        }        
        // Process module/call/group deletion 
        if (std::get<1>(this->graphstate.hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM])) {
            if (this->graphstate.interact.module_selected_uid != GUI_INVALID_ID) {
                inout_graph.DeleteModule(this->graphstate.interact.module_selected_uid);
            }
            if (this->graphstate.interact.call_selected_uid != GUI_INVALID_ID) {
                inout_graph.DeleteCall(this->graphstate.interact.call_selected_uid);
            }
            if (this->graphstate.interact.group_selected_uid != GUI_INVALID_ID) {
                inout_graph.DeleteGroup(this->graphstate.interact.group_selected_uid);
            }
        }
        // Set delete flag if tab was closed
        if (!open) {
            state.graph_delete = true;
        }

        // Rename pop-up
        this->utils.RenamePopUp("Rename Project", popup_rename, inout_graph.GetName());

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
    ModulePtrType selected_mod_ptr = nullptr;
    if (this->graphstate.interact.module_selected_uid != GUI_INVALID_ID) {
        for (auto& mod : inout_graph.GetGraphModules()) {
            if ((this->graphstate.interact.module_selected_uid == mod->uid) && (mod->is_view)) {
                selected_mod_ptr = mod;
            }
        }
    }
    if (selected_mod_ptr == nullptr) {
        this->utils.ReadOnlyWigetStyle(true);
        bool checked = false;
        ImGui::Checkbox("Main View", &checked);
        this->utils.ReadOnlyWigetStyle(false);
    } else {
        if (ImGui::Checkbox("Main View", &selected_mod_ptr->is_view_instance)) {
            this->update = true;
            if (selected_mod_ptr->is_view_instance) {
                // Set all other modules to non main views
                for (auto& mod : inout_graph.GetGraphModules()) {
                    if (this->graphstate.interact.module_selected_uid != mod->uid) {
                        mod->is_view_instance = false;
                    }
                }
            }
        }
    }
    ImGui::SameLine();

    ImGui::Text("Scrolling: %.2f,%.2f", this->graphstate.canvas.scrolling.x, this->graphstate.canvas.scrolling.y);
    ImGui::SameLine();
    if (ImGui::Button("Reset###reset_scrolling")) {
        this->graphstate.canvas.scrolling = ImVec2(0.0f, 0.0f);
        this->update = true;
    }
    ImGui::SameLine();

    ImGui::Text("Zooming: %.2f", this->graphstate.canvas.zooming);
    ImGui::SameLine();
    if (ImGui::Button("Reset###reset_zooming")) {
        this->reset_zooming = true;
    }
    ImGui::SameLine();

    ImGui::Checkbox("Grid", &this->show_grid);

    ImGui::SameLine();

    if (ImGui::Checkbox("Call Names", &this->show_call_names)) {
        for (auto& call : inout_graph.get_graph_calls()) {
            call->GUI_SetLabelVisibility(this->show_call_names);
        }
        this->update = true;
    }
    ImGui::SameLine();

    if (ImGui::Checkbox("Module Names", &this->show_module_names)) {
        for (auto& mod : inout_graph.GetGraphModules()) {
            mod->GUI_SetLabelVisibility(this->show_module_names);
        }
        this->update = true;
    }
    ImGui::SameLine();

    if (ImGui::Checkbox("Slot Names", &this->show_slot_names)) {
        for (auto& mod : inout_graph.GetGraphModules()) {
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
    const ImU32 COLOR_CANVAS_BACKGROUND = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ChildBg]); // ImGuiCol_ScrollbarBg ImGuiCol_ScrollbarGrab ImGuiCol_Border

    ImGui::PushStyleColor(ImGuiCol_ChildBg, COLOR_CANVAS_BACKGROUND);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(1, 1));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    auto child_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NavFlattened;
    ImGui::BeginChild("region", ImVec2(child_width, 0.0f), true, child_flags);

    // Update canvas position
    ImVec2 new_position = ImGui::GetWindowPos();
    if ((this->graphstate.canvas.position.x != new_position.x) || (this->graphstate.canvas.position.y != new_position.y)) {
        this->update = true;
    }
    this->graphstate.canvas.position = new_position;
    // Update canvas size
    ImVec2 new_size = ImGui::GetWindowSize();
    if ((this->graphstate.canvas.size.x != new_size.x) || (this->graphstate.canvas.size.y != new_size.y)) {
        this->update = true;
    }
    this->graphstate.canvas.size = new_size;
    // Update canvas offset
    ImVec2 new_offset = this->graphstate.canvas.position + (this->graphstate.canvas.scrolling * this->graphstate.canvas.zooming);
    if ((this->graphstate.canvas.offset.x != new_offset.x) || (this->graphstate.canvas.offset.y != new_offset.y)) {
        this->update = true;
    }
    this->graphstate.canvas.offset = new_offset;  

    // Update module size and call slot positions
    if (this->update) {
        for (auto& mod : inout_graph.GetGraphModules()) {
            mod->GUI_Update(this->graphstate.canvas);
        }
        for (auto& group : inout_graph.get_graph_groups()) {
            group.GUI_Update(this->graphstate.canvas);
        }        
        this->update = false;
    }

    ImGui::PushClipRect(this->graphstate.canvas.position, this->graphstate.canvas.position + this->graphstate.canvas.size, true);
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    // 1] GRID ----------------------------------
    if (this->show_grid) {
        this->present_canvas_grid();
    }
    ImGui::PopStyleVar(2);

    // 2] GROUPS --------------------------------
    for (auto& group : inout_graph.get_graph_groups()) {
        group.GUI_Present(this->graphstate);

    }

    // 3] MODULES and CALL SLOTS ----------------
    for (auto& mod : inout_graph.GetGraphModules()) {
        mod->GUI_Present(this->graphstate);
    }

    // 4] CALLS ---------------------------------;
    for (auto& call : inout_graph.get_graph_calls()) {
        call->GUI_Present(this->graphstate);
    }

    // 5] Dragged CALL --------------------------
    this->present_canvas_dragged_call(inout_graph);     

    ImGui::PopClipRect();

    // Zooming and Scaling ----------------------
    /// Must be checked inside canvas child window.
    /// Check at the end for being applied in next frame when font scaling matches zooming.
    if ((ImGui::IsWindowHovered() && !ImGui::IsAnyItemActive()) || this->reset_zooming) {

        // Scrolling (2 = Middle Mouse Button)
        if (ImGui::IsMouseDragging(2, 0.0f)) {
            this->graphstate.canvas.scrolling = this->graphstate.canvas.scrolling + ImGui::GetIO().MouseDelta / this->graphstate.canvas.zooming;
            this->update = true;
        }

        // Zooming (Mouse Wheel) + Reset
        if ((io.MouseWheel != 0) || this->reset_zooming) {
            float last_zooming = this->graphstate.canvas.zooming;
            ImVec2 current_mouse_pos;
            if (this->reset_zooming) {
                this->graphstate.canvas.zooming = 1.0f;
                current_mouse_pos = this->graphstate.canvas.offset - (this->graphstate.canvas.position + this->graphstate.canvas.size * 0.5f);
                this->reset_zooming= false;
            }
            else {
                const float factor = this->graphstate.canvas.zooming / 10.0f;
                this->graphstate.canvas.zooming = this->graphstate.canvas.zooming + (io.MouseWheel * factor);
                current_mouse_pos = this->graphstate.canvas.offset - ImGui::GetMousePos();
            }
            // Limit zooming
            this->graphstate.canvas.zooming = (this->graphstate.canvas.zooming <= 0.0f) ? 0.000001f : (this->graphstate.canvas.zooming);
            // Compensate zooming shift of origin
            ImVec2 scrolling_diff =
                (this->graphstate.canvas.scrolling * last_zooming) - (this->graphstate.canvas.scrolling * this->graphstate.canvas.zooming);
            this->graphstate.canvas.scrolling += (scrolling_diff / this->graphstate.canvas.zooming);
            // Move origin away from mouse position
            ImVec2 new_mouse_position = (current_mouse_pos / last_zooming) * this->graphstate.canvas.zooming;
            this->graphstate.canvas.scrolling += ((new_mouse_position - current_mouse_pos) / this->graphstate.canvas.zooming);

            this->update = true;
        }
    }

    ImGui::EndChild();
    ImGui::PopStyleColor();

    // Update when scaling of font has changed due to project tab switching
    if (ImGui::GetFont()->Scale != this->graphstate.canvas.zooming) {
        this->update = true;
    }
    // Font scaling is applied next frame after ImGui::Begin()
    // Font for graph should not be the currently used font of the gui.    
    ImGui::GetFont()->Scale = this->graphstate.canvas.zooming;
}


void megamol::gui::configurator::Graph::Presentation::present_parameters(megamol::gui::configurator::Graph& inout_graph, float child_width) {

    ImGui::BeginGroup();

    float param_child_height = ImGui::GetFrameHeightWithSpacing() * 3.5f;
    auto child_flags = ImGuiWindowFlags_AlwaysUseWindowPadding | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NavFlattened;
    ImGui::BeginChild("parameter_search_child", ImVec2(child_width, param_child_height), false, child_flags);

    ImGui::Text("Parameters");
    ImGui::Separator();

    if (std::get<1>(this->graphstate.hotkeys[HotkeyIndex::PARAMETER_SEARCH])) {
        this->utils.SetSearchFocus(true);
    }
    std::string help_text = "[" + std::get<0>(this->graphstate.hotkeys[HotkeyIndex::PARAMETER_SEARCH]).ToString() +
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
            for (auto& modptr : inout_graph.GetGraphModules()) {
                for (auto& param : modptr->parameters) {
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
            for (auto& modptr : inout_graph.GetGraphModules()) {
                for (auto& param : modptr->parameters) {
                    param.GUI_SetLabelVisibility(this->params_visible);
                }
            }
        }
        ImGui::SameLine();

        // Read-only option
        if (ImGui::Checkbox("Read-Only", &this->params_readonly)) {
            for (auto& modptr : inout_graph.GetGraphModules()) {
                for (auto& param : modptr->parameters) {
                    param.GUI_SetReadOnly(this->params_readonly);
                }
            }
        }
    }
    ImGui::Separator();

    ImGui::EndChild();

    // Get pointer to currently selected module
    ModulePtrType modptr;
    if (this->graphstate.interact.module_selected_uid != GUI_INVALID_ID) {
        for (auto& mod : inout_graph.GetGraphModules()) {
            if (mod->uid == this->graphstate.interact.module_selected_uid) {
                modptr = mod;
            }
        }
    }
    if (modptr != nullptr) {
        float param_child_height = ImGui::GetFrameHeightWithSpacing() * 1.0f;
        ImGui::BeginChild("parameter_info_child", ImVec2(child_width, param_child_height), false, child_flags);

        ImGui::Text("Selected Module:");
        ImGui::SameLine();
        ImGui::TextColored(ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive), modptr->name.c_str());

        ImGui::EndChild();
        auto child_flags = ImGuiWindowFlags_AlwaysVerticalScrollbar | ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NavFlattened;
        ImGui::BeginChild("parameter_list_child", ImVec2(child_width, 0.0f), true, child_flags);

        bool param_name_space_open = true;
        unsigned int param_indent_stack = 0;

        for (auto& param : modptr->parameters) {
            // Filter module by given search string
            bool search_filter = true;
            if (!search_string.empty()) {
                search_filter = this->utils.FindCaseInsensitiveSubstring(param.full_name, search_string);
            }

            // Add Collapsing header depending on parameter namespace
            std::string current_param_namespace = param.GetNameSpace();
            if (current_param_namespace != this->param_name_space) {
                this->param_name_space = current_param_namespace;
                while (param_indent_stack > 0) {
                    param_indent_stack--;
                    ImGui::Unindent();
                }

                ImGui::Separator();
                if (!this->param_name_space.empty()) {
                    ImGui::Indent();
                    std::string label = this->param_name_space + "###" + param.full_name;
                    // Open all namespace headers when parameter search is active
                    if (!search_string.empty()) {
                        auto headerId = ImGui::GetID(label.c_str());
                        ImGui::GetStateStorage()->SetInt(headerId, 1);
                    }
                    param_name_space_open = ImGui::CollapsingHeader(label.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
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

        ImGui::EndChild();
    }

    ImGui::EndGroup();
}


void megamol::gui::configurator::Graph::Presentation::present_canvas_grid(void) {

    ImGuiStyle& style = ImGui::GetStyle();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    const ImU32 COLOR_GRID = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Border]);
    const float GRID_SIZE = 64.0f * this->graphstate.canvas.zooming;

    ImVec2 relative_offset = this->graphstate.canvas.offset - this->graphstate.canvas.position;

    for (float x = fmodf(relative_offset.x, GRID_SIZE); x < this->graphstate.canvas.size.x; x += GRID_SIZE) {
        draw_list->AddLine(ImVec2(x, 0.0f) + this->graphstate.canvas.position,
            ImVec2(x, this->graphstate.canvas.size.y) + this->graphstate.canvas.position, COLOR_GRID);
    }

    for (float y = fmodf(relative_offset.y, GRID_SIZE); y < this->graphstate.canvas.size.y; y += GRID_SIZE) {
        draw_list->AddLine(ImVec2(0.0f, y) + this->graphstate.canvas.position,
            ImVec2(this->graphstate.canvas.size.x, y) + this->graphstate.canvas.position, COLOR_GRID);
    }

    // DEBUG Cross in origin
    // draw_list->AddLine(ImVec2(this->graphstate.canvas.offset.x - GRID_SIZE, this->graphstate.canvas.offset.y),ImVec2(this->graphstate.canvas.offset.x + GRID_SIZE, this->graphstate.canvas.offset.y), IM_COL32(128, 0, 0, 255), 1.0f);
    // draw_list->AddLine(ImVec2(this->graphstate.canvas.offset.x, this->graphstate.canvas.offset.y - GRID_SIZE), ImVec2(this->graphstate.canvas.offset.x, this->graphstate.canvas.offset.y + GRID_SIZE), IM_COL32(0, 128, 0, 255), 1.0f);
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
            if ((current_pos.x >= this->graphstate.canvas.position.x) &&
                (current_pos.x <= (this->graphstate.canvas.position.x + this->graphstate.canvas.size.x)) &&
                (current_pos.y >= this->graphstate.canvas.position.y) &&
                (current_pos.y <= (this->graphstate.canvas.position.y + this->graphstate.canvas.size.y))) {
                mouse_inside_canvas = true;
            }
            if (mouse_inside_canvas) {

                CallSlotPtrType selected_call_slot_ptr;
                for (auto& mods : inout_graph.GetGraphModules()) {
                    CallSlotPtrType call_slot_ptr = mods->GetCallSlot(*selected_call_slot_uid_ptr);
                    if (call_slot_ptr != nullptr) {
                        selected_call_slot_ptr = call_slot_ptr;
                    }
                }

                if (selected_call_slot_ptr != nullptr) {
                    ImVec2 p1 = selected_call_slot_ptr->GUI_GetPosition();
                    ImVec2 p2 = ImGui::GetMousePos();
                    if (glm::length(glm::vec2(p1.x, p1.y) - glm::vec2(p2.x, p2.y)) > GUI_CALL_SLOT_RADIUS) {
                        if (selected_call_slot_ptr->type == CallSlot::CallSlotType::CALLEE) {
                            ImVec2 tmp = p1;
                            p1 = p2;
                            p2 = tmp;
                        }
                        draw_list->AddBezierCurve(p1, p1 + ImVec2(+50, 0), p2 + ImVec2(-50, 0), p2, COLOR_CALL_CURVE,
                            CURVE_THICKNESS * this->graphstate.canvas.zooming);
                    }
                }
            }
        }
    }
}


bool megamol::gui::configurator::Graph::Presentation::layout_graph(megamol::gui::configurator::Graph& inout_graph) {

    // Really simple layouting sorting modules into differnet layers

    ImGuiStyle& style = ImGui::GetStyle();
    std::vector<std::vector<ModulePtrType>> layers;
    layers.clear();

    // Fill first layer with modules having no connected callee
    layers.emplace_back();
    for (auto& mod : inout_graph.GetGraphModules()) {
        bool any_connected_callee = false;
        for (auto& callee_slot : mod->GetCallSlots(CallSlot::CallSlotType::CALLEE)) {
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
            for (auto& caller_slot : layer_mod->GetCallSlots(CallSlot::CallSlotType::CALLER)) {
                if (caller_slot->CallsConnected()) {
                    for (auto& call : caller_slot->GetConnectedCalls()) {
                        auto add_mod = call->GetCallSlot(CallSlot::CallSlotType::CALLEE)->GetParentModule();

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
    const float border_offset = GUI_CALL_SLOT_RADIUS * 4.0f;
    ImVec2 init_position = ImVec2(-1.0f * this->graphstate.canvas.scrolling.x, -1.0f * this->graphstate.canvas.scrolling.y);
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
        pos.x += border_offset;
        pos.y = init_position.y + border_offset;
        for (size_t i = 0; i < layer_mod_cnt; i++) {
            auto mod = layer[i];
            if (this->show_call_names) {
                for (auto& caller_slot : mod->GetCallSlots(CallSlot::CallSlotType::CALLER)) {
                    if (caller_slot->CallsConnected()) {
                        for (auto& call : caller_slot->GetConnectedCalls()) {
                            auto call_name_length = this->utils.TextWidgetWidth(call->class_name) * 1.5f;
                            max_call_width =
                                (call_name_length > max_call_width) ? (call_name_length) : (max_call_width);
                        }
                    }
                }
            }
            mod->GUI_SetPosition(pos);
            auto mod_size = mod->GUI_GetSize();
            pos.y += mod_size.y + border_offset;
            max_module_width = (mod_size.x > max_module_width) ? (mod_size.x) : (max_module_width);
        }
        pos.x += (max_module_width + max_call_width + border_offset);
    }

    return true;
}
