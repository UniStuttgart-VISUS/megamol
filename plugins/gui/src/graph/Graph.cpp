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


megamol::gui::Graph::Graph(const std::string& graph_name, GraphCoreInterface core_interface)
        : uid(megamol::gui::GenerateUniqueID())
        , name(graph_name)
        , modules()
        , calls()
        , groups()
        , dirty_flag(true)
        , filename()
        , sync_queue()
        , graph_core_interface(core_interface)
        , gui_graph_state() 
        , gui_show_grid()
        , gui_show_call_label()
        , gui_show_call_slots_label()
        , gui_show_slot_label()
        , gui_show_module_label()
        , gui_show_parameter_sidebar()
        , gui_params_visible()
        , gui_params_readonly()
        , gui_param_extended_mode()
        , gui_change_show_parameter_sidebar()
        , gui_graph_layout()
        , gui_parameter_sidebar_width()
        , gui_reset_zooming()
        , gui_increment_zooming()
        , gui_decrement_zooming()
        , gui_param_name_space()
        , gui_current_graph_entry_name()
        , gui_multiselect_start_pos()
        , gui_multiselect_end_pos()
        , gui_multiselect_done()
        , gui_canvas_hovered()
        , gui_current_font_scaling()
        , gui_search_widget()
        , gui_splitter_widget()
        , gui_rename_popup()
        , gui_tooltip() {

    this->gui_graph_state.canvas.position = ImVec2(0.0f, 0.0f);
    this->gui_graph_state.canvas.size = ImVec2(1.0f, 1.0f);
    this->gui_graph_state.canvas.scrolling = ImVec2(0.0f, 0.0f);
    this->gui_graph_state.canvas.zooming = 1.0f;
    this->gui_graph_state.canvas.offset = ImVec2(0.0f, 0.0f);

    this->gui_graph_state.interact.process_deletion = false;
    this->gui_graph_state.interact.button_active_uid = GUI_INVALID_ID;
    this->gui_graph_state.interact.button_hovered_uid = GUI_INVALID_ID;

    this->gui_graph_state.interact.group_selected_uid = GUI_INVALID_ID;
    this->gui_graph_state.interact.group_hovered_uid = GUI_INVALID_ID;
    this->gui_graph_state.interact.group_layout = false;

    this->gui_graph_state.interact.modules_selected_uids.clear();
    this->gui_graph_state.interact.module_hovered_uid = GUI_INVALID_ID;
    this->gui_graph_state.interact.modules_add_group_uids.clear();
    this->gui_graph_state.interact.modules_remove_group_uids.clear();
    this->gui_graph_state.interact.modules_layout = false;
    this->gui_graph_state.interact.module_rename.clear();
    this->gui_graph_state.interact.module_graphentry_changed = vislib::math::Ternary::TRI_UNKNOWN;
    this->gui_graph_state.interact.module_param_child_position = ImVec2(-1.0f, -1.0f);

    this->gui_graph_state.interact.call_selected_uid = GUI_INVALID_ID;
    this->gui_graph_state.interact.call_hovered_uid = GUI_INVALID_ID;

    this->gui_graph_state.interact.slot_dropped_uid = GUI_INVALID_ID;

    this->gui_graph_state.interact.callslot_selected_uid = GUI_INVALID_ID;
    this->gui_graph_state.interact.callslot_hovered_uid = GUI_INVALID_ID;
    this->gui_graph_state.interact.callslot_add_group_uid = UIDPair_t(GUI_INVALID_ID, GUI_INVALID_ID);
    this->gui_graph_state.interact.callslot_remove_group_uid = UIDPair_t(GUI_INVALID_ID, GUI_INVALID_ID);
    this->gui_graph_state.interact.callslot_compat_ptr.reset();

    this->gui_graph_state.interact.interfaceslot_selected_uid = GUI_INVALID_ID;
    this->gui_graph_state.interact.interfaceslot_hovered_uid = GUI_INVALID_ID;
    this->gui_graph_state.interact.interfaceslot_compat_ptr.reset();

    this->gui_graph_state.interact.graph_core_interface = GraphCoreInterface::NO_INTERFACE;

    this->gui_graph_state.groups.clear();
    // this->gui_graph_state.hotkeys are already initialzed

}

megamol::gui::Graph::~Graph(void) {

    this->Clear();
}


ImGuiID megamol::gui::Graph::AddEmptyModule(void) {

    try {
        ImGuiID mod_uid = megamol::gui::GenerateUniqueID();
        auto mod_ptr = std::make_shared<Module>(mod_uid);
        this->modules.emplace_back(mod_ptr);
        this->ForceSetDirty();

#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Added empty module to project.\n");
#endif // GUI_VERBOSE

        return mod_uid;
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    }

    megamol::core::utility::log::Log::DefaultLog.WriteError(
        "[GUI] Unable to add empty module. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return GUI_INVALID_ID;
}


ImGuiID megamol::gui::Graph::AddModule(const ModuleStockVector_t& stock_modules, const std::string& module_class_name) {

    try {
        for (auto& mod : stock_modules) {
            if (module_class_name == mod.class_name) {
                ImGuiID mod_uid = megamol::gui::GenerateUniqueID();
                auto mod_ptr = std::make_shared<Module>(mod_uid, mod.class_name, mod.description, mod.plugin_name);
                mod_ptr->is_view = mod.is_view;
                mod_ptr->name = this->generate_unique_module_name(mod.class_name);
                mod_ptr->graph_entry_name.clear();
                mod_ptr->label_visible = this->GetModuleLabelVisibility();

                for (auto& p : mod.parameters) {
                    Parameter param_slot(megamol::gui::GenerateUniqueID(), p.type, p.storage, p.minval, p.maxval,
                        p.full_name, p.description);
                    param_slot.SetValueString(p.default_value, true, true);
                    param_slot.SetGUIVisible(p.gui_visibility);
                    param_slot.SetGUIReadOnly(p.gui_read_only);
                    param_slot.SetGUIPresentation(p.gui_presentation);
                    // Apply current global configurator parameter gui settings
                    // Do not apply global read-only and visibility.
                    param_slot.extended = this->param_extended_mode;

                    mod_ptr->parameters.emplace_back(param_slot);
                }

                for (auto& callslots_type : mod.callslots) {
                    for (auto& c : callslots_type.second) {
                        auto callslot_ptr = std::make_shared<CallSlot>(megamol::gui::GenerateUniqueID());
                        callslot_ptr->name = c.name;
                        callslot_ptr->description = c.description;
                        callslot_ptr->compatible_call_idxs = c.compatible_call_idxs;
                        callslot_ptr->type = c.type;
                        callslot_ptr->label_visible = this->GetSlotLabelVisibility();
                        callslot_ptr->ConnectParentModule(mod_ptr);

                        mod_ptr->AddCallSlot(callslot_ptr);
                    }
                }

                QueueData queue_data;
                queue_data.class_name = mod_ptr->class_name;
                queue_data.name_id = mod_ptr->FullName();
                this->PushSyncQueue(QueueAction::ADD_MODULE, queue_data);

                this->modules.emplace_back(mod_ptr);
                this->ForceSetDirty();

#ifdef GUI_VERBOSE
                megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                    "[GUI] Added module '%s' (uid %i) to project '%s'.\n", mod_ptr->class_name.c_str(), mod_ptr->uid,
                    this->name.c_str());
#endif // GUI_VERBOSE

                return mod_uid;
            }
        }
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    }

    megamol::core::utility::log::Log::DefaultLog.WriteError(
        "[GUI] Unable to find module in stock: '%s'. [%s, %s, line %d]\n", module_class_name.c_str(), __FILE__,
        __FUNCTION__, __LINE__);
    return GUI_INVALID_ID;
}


bool megamol::gui::Graph::DeleteModule(ImGuiID module_uid, bool force) {

    try {
        for (auto iter = this->modules.begin(); iter != this->modules.end(); iter++) {
            if ((*iter)->uid == module_uid) {

                if (!force && (*iter)->IsGraphEntry()) {
                    if (this->GetCoreInterface() == GraphCoreInterface::CORE_INSTANCE_GRAPH) {
                        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                            "[GUI] The action [Delete graph entry/ view instance] is not yet supported for the graph "
                            "using the 'Core Instance Graph' interface. Open project from file to make desired "
                            "changes. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        return false;
                    }
                }

                this->ResetStatePointers();
                QueueData queue_data;

                // 1) Reset module and call slot pointers in groups
                auto current_full_name = (*iter)->FullName();
                GroupPtr_t module_group_ptr = nullptr;
                for (auto& group_ptr : this->groups) {
                    if (group_ptr->ContainsModule(module_uid)) {
                        group_ptr->RemoveModule(module_uid);
                        module_group_ptr = group_ptr;

                        queue_data.name_id = current_full_name;
                        queue_data.rename_id = (*iter)->FullName();
                        this->PushSyncQueue(QueueAction::RENAME_MODULE, queue_data);
                    }
                }

                // 2)  Delete calls
                for (auto& callslot_map : (*iter)->GetCallSlots()) {
                    for (auto& callslot_ptr : callslot_map.second) {
                        std::vector<ImGuiID> delete_call_uids;
                        for (auto& call_ptr : callslot_ptr->GetConnectedCalls()) {
                            if (call_ptr != nullptr) {
                                delete_call_uids.push_back(call_ptr->uid);
                            }
                        }
                        for (auto& call_uid : delete_call_uids) {
                            this->DeleteCall(call_uid);
                        }
                    }
                }

                // 3)  Remove call slots
                (*iter)->DeleteCallSlots();

                // 4) Automatically restore interfaceslots
                if (module_group_ptr != nullptr) {
                    module_group_ptr->RestoreInterfaceslots();
                }

#ifdef GUI_VERBOSE
                megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                    "[GUI] Deleted module '%s' (uid %i) from  project '%s'.\n", (*iter)->FullName().c_str(),
                    (*iter)->uid, this->name.c_str());
#endif // GUI_VERBOSE

                queue_data.name_id = (*iter)->FullName();
                this->PushSyncQueue(QueueAction::DELETE_MODULE, queue_data);

                // 5) Delete module
                if ((*iter).use_count() > 1) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Unclean deletion. Found %i references pointing to module. [%s, %s, line %d]\n",
                        (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                }
                (*iter).reset();
                this->modules.erase(iter);

                this->ForceSetDirty();
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


bool megamol::gui::Graph::GetModule(ImGuiID module_uid, megamol::gui::ModulePtr_t& out_module_ptr) {

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


bool megamol::gui::Graph::ModuleExists(const std::string& module_fullname) {

    return (std::find_if(this->modules.begin(), this->modules.end(), [&](megamol::gui::ModulePtr_t& module_ptr) {
        return (module_ptr->FullName() == module_fullname);
    }) != this->modules.end());
}


bool megamol::gui::Graph::AddCall(const CallStockVector_t& stock_calls, ImGuiID slot_1_uid, ImGuiID slot_2_uid) {

    try {
        if ((slot_1_uid == GUI_INVALID_ID) || (slot_2_uid == GUI_INVALID_ID)) {
#ifdef GUI_VERBOSE
            /// megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Invalid slot uid given. [%s, %s, line
            /// %d]\n",
            /// __FILE__,
            /// __FUNCTION__,
            /// __LINE__);
#endif // GUI_VERBOSE
            return false;
        }

        CallSlotPtr_t drag_callslot_ptr;
        CallSlotPtr_t drop_callslot_ptr;
        for (auto& module_ptr : this->modules) {
            CallSlotPtr_t callslot_ptr;
            if (module_ptr->GetCallSlot(slot_1_uid, callslot_ptr)) {
                drag_callslot_ptr = callslot_ptr;
            }
            if (module_ptr->GetCallSlot(slot_2_uid, callslot_ptr)) {

                drop_callslot_ptr = callslot_ptr;
            }
        }

        InterfaceSlotPtr_t drag_interfaceslot_ptr;
        InterfaceSlotPtr_t drop_interfaceslot_ptr;
        for (auto& group_ptr : this->groups) {
            InterfaceSlotPtr_t interfaceslot_ptr;
            if (group_ptr->GetInterfaceSlot(slot_1_uid, interfaceslot_ptr)) {
                drag_interfaceslot_ptr = interfaceslot_ptr;
            }
            if (group_ptr->GetInterfaceSlot(slot_2_uid, interfaceslot_ptr)) {
                drop_interfaceslot_ptr = interfaceslot_ptr;
            }
        }

        // CallSlot <-> CallSlot
        if ((drag_callslot_ptr != nullptr) && (drop_callslot_ptr != nullptr)) {
            this->AddCall(stock_calls, drag_callslot_ptr, drop_callslot_ptr);
        }
        // InterfaceSlot <-> CallSlot
        else if (((drag_interfaceslot_ptr != nullptr) && (drop_callslot_ptr != nullptr)) ||
                 ((drag_callslot_ptr != nullptr) && (drop_interfaceslot_ptr != nullptr))) {

            InterfaceSlotPtr_t interface_ptr =
                (drag_interfaceslot_ptr != nullptr) ? (drag_interfaceslot_ptr) : (drop_interfaceslot_ptr);
            CallSlotPtr_t callslot_ptr = (drop_callslot_ptr != nullptr) ? (drop_callslot_ptr) : (drag_callslot_ptr);

            ImGuiID interfaceslot_group_uid = interface_ptr->group.uid;
            ImGuiID callslot_group_uid = GUI_INVALID_ID;
            if (callslot_ptr->IsParentModuleConnected()) {
                callslot_group_uid = callslot_ptr->GetParentModule()->group.uid;
            }

            if (interfaceslot_group_uid == callslot_group_uid) {
                if (interface_ptr->AddCallSlot(callslot_ptr, interface_ptr)) {
                    CallSlotType compatible_callslot_type = (interface_ptr->GetCallSlotType() == CallSlotType::CALLEE)
                                                                ? (CallSlotType::CALLER)
                                                                : (CallSlotType::CALLEE);
                    // Get call slot the interface slot is connected to and add call for new added call slot
                    CallSlotPtr_t connect_callslot_ptr;
                    for (auto& interface_callslots_ptr : interface_ptr->GetCallSlots()) {
                        if (interface_callslots_ptr->uid != callslot_ptr->uid) {
                            for (auto& call_ptr : interface_callslots_ptr->GetConnectedCalls()) {
                                connect_callslot_ptr = call_ptr->GetCallSlot(compatible_callslot_type);
                            }
                        }
                    }
                    if (connect_callslot_ptr != nullptr) {
                        if (!this->AddCall(stock_calls, callslot_ptr, connect_callslot_ptr)) {
                            interface_ptr->RemoveCallSlot(callslot_ptr->uid);
                        }
                    }
                }
            } else if (interfaceslot_group_uid != callslot_group_uid) {
                // Add calls to all call slots the call slots of the interface are connected to.
                for (auto& interface_callslots_ptr : interface_ptr->GetCallSlots()) {
                    this->AddCall(stock_calls, callslot_ptr, interface_callslots_ptr);
                }
            }
        }
        // InterfaceSlot <-> InterfaceSlot
        else if ((drag_interfaceslot_ptr != nullptr) && (drop_interfaceslot_ptr != nullptr)) {
            if (drag_interfaceslot_ptr->IsConnectionValid((*drop_interfaceslot_ptr))) {
                for (auto& drag_interface_callslots_ptr : drag_interfaceslot_ptr->GetCallSlots()) {
                    for (auto& drop_interface_callslots_ptr : drop_interfaceslot_ptr->GetCallSlots()) {
                        this->AddCall(stock_calls, drag_interface_callslots_ptr, drop_interface_callslots_ptr);
                    }
                }
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

    return true;
}


bool megamol::gui::Graph::AddCall(
    const CallStockVector_t& stock_calls, CallSlotPtr_t callslot_1, CallSlotPtr_t callslot_2) {

    try {
        if ((callslot_1 == nullptr) || (callslot_2 == nullptr)) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] Pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        if (!callslot_1->IsConnectionValid((*callslot_2))) {
            return false;
        }
        auto compat_idx = CallSlot::GetCompatibleCallIndex(callslot_1, callslot_2);
        if (compat_idx == GUI_INVALID_ID) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] Unable to find index of compatible call. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        Call::StockCall call_stock_data = stock_calls[compat_idx];

        auto call_ptr = std::make_shared<Call>(megamol::gui::GenerateUniqueID());
        call_ptr->class_name = call_stock_data.class_name;
        call_ptr->description = call_stock_data.description;
        call_ptr->plugin_name = call_stock_data.plugin_name;
        call_ptr->functions = call_stock_data.functions;
        call_ptr->gui_label_visible = this->GetCallLabelVisibility();
        call_ptr->gui_slots_visible = this->GetCallSlotLabelVisibility();

        return this->AddCall(call_ptr, callslot_1, callslot_2);

    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
}


bool megamol::gui::Graph::AddCall(CallPtr_t& call_ptr, CallSlotPtr_t callslot_1, CallSlotPtr_t callslot_2) {

    if (call_ptr == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Pointer to call is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if ((callslot_1 == nullptr) || (callslot_2 == nullptr)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Delete calls when CALLERs should be connected to new call slot
    if ((callslot_1->type == CallSlotType::CALLER) && (callslot_1->CallsConnected())) {
        std::vector<ImGuiID> calls_uids;
        for (auto& call : callslot_1->GetConnectedCalls()) {
            calls_uids.emplace_back(call->uid);
        }
        for (auto& call_uid : calls_uids) {
            this->DeleteCall(call_uid);
        }
    }
    if ((callslot_2->type == CallSlotType::CALLER) && (callslot_2->CallsConnected())) {
        std::vector<ImGuiID> calls_uids;
        for (auto& call : callslot_2->GetConnectedCalls()) {
            calls_uids.emplace_back(call->uid);
        }
        for (auto& call_uid : calls_uids) {
            this->DeleteCall(call_uid);
        }
    }

    if (call_ptr->ConnectCallSlots(callslot_1, callslot_2) && callslot_1->ConnectCall(call_ptr) &&
        callslot_2->ConnectCall(call_ptr)) {

        QueueData queue_data;
        queue_data.class_name = call_ptr->class_name;
        bool valid_ptr = false;
        auto caller_ptr = call_ptr->GetCallSlot(megamol::gui::CallSlotType::CALLER);
        if (caller_ptr != nullptr) {
            if (caller_ptr->GetParentModule() != nullptr) {
                queue_data.caller = caller_ptr->GetParentModule()->FullName() + "::" + caller_ptr->name;
                valid_ptr = true;
            }
        }
        if (!valid_ptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Pointer to caller slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        }
        valid_ptr = false;
        auto callee_ptr = call_ptr->GetCallSlot(megamol::gui::CallSlotType::CALLEE);
        if (callee_ptr != nullptr) {
            if (callee_ptr->GetParentModule() != nullptr) {
                queue_data.callee = callee_ptr->GetParentModule()->FullName() + "::" + callee_ptr->name;
                valid_ptr = true;
            }
        }
        if (!valid_ptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Pointer to callee slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        }
        this->PushSyncQueue(QueueAction::ADD_CALL, queue_data);

        this->calls.emplace_back(call_ptr);
        this->ForceSetDirty();

#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Added call '%s' (uid %i) to project '%s'.\n",
            call_ptr->class_name.c_str(), call_ptr->uid, this->name.c_str());
#endif // GUI_VERBOSE

        // Add connected call slots to interface of group of the parent module
        if (callslot_1->IsParentModuleConnected() && callslot_2->IsParentModuleConnected()) {
            ImGuiID slot_1_parent_group_uid = callslot_1->GetParentModule()->group.uid;
            ImGuiID slot_2_parent_group_uid = callslot_2->GetParentModule()->group.uid;
            if (slot_1_parent_group_uid != slot_2_parent_group_uid) {
                if ((slot_1_parent_group_uid != GUI_INVALID_ID) && (callslot_1->gui_interfaceslot_ptr == nullptr)) {
                    for (auto& group_ptr : this->groups) {
                        if (group_ptr->uid == slot_1_parent_group_uid) {
                            group_ptr->AddInterfaceSlot(callslot_1);
                        }
                    }
                }
                if ((slot_2_parent_group_uid != GUI_INVALID_ID) && (callslot_2->gui_interfaceslot_ptr == nullptr)) {
                    for (auto& group_ptr : this->groups) {
                        if (group_ptr->uid == slot_2_parent_group_uid) {
                            group_ptr->AddInterfaceSlot(callslot_2);
                        }
                    }
                }
            }
        }

        return true;
    } else {
        this->DeleteCall(call_ptr->uid);
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Unable to create call. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
}


bool megamol::gui::Graph::DeleteCall(ImGuiID call_uid) {

    try {
        std::vector<ImGuiID> delete_calls_uids;
        delete_calls_uids.emplace_back(call_uid);

        // Also delete other calls, which are connceted to same interface slot and call slot
        ImGuiID caller_uid = GUI_INVALID_ID;
        ImGuiID callee_uid = GUI_INVALID_ID;
        for (auto& call_ptr : this->calls) {
            if (call_ptr->uid == call_uid) {
                auto caller = call_ptr->GetCallSlot(CallSlotType::CALLER);
                auto callee = call_ptr->GetCallSlot(CallSlotType::CALLEE);
                if (caller != nullptr) {
                    caller_uid = caller->uid;
                    if (caller->gui_interfaceslot_ptr != nullptr) {
                        caller_uid = caller->gui_interfaceslot_ptr->uid;
                    }
                }
                if (callee != nullptr) {
                    callee_uid = callee->uid;
                    if (callee->gui_interfaceslot_ptr != nullptr) {
                        callee_uid = callee->gui_interfaceslot_ptr->uid;
                    }
                }
            }
        }
        for (auto& call_ptr : this->calls) {
            if (call_ptr->uid != call_uid) {
                bool caller_fits = false;
                bool callee_fits = false;
                auto caller = call_ptr->GetCallSlot(CallSlotType::CALLER);
                auto callee = call_ptr->GetCallSlot(CallSlotType::CALLEE);
                if (caller != nullptr) {
                    if (caller->gui_interfaceslot_ptr != nullptr) {
                        caller_fits = (caller_uid == caller->gui_interfaceslot_ptr->uid);
                    } else {
                        caller_fits = (caller_uid == caller->uid);
                    }
                }
                if (callee != nullptr) {
                    if (callee->gui_interfaceslot_ptr != nullptr) {
                        callee_fits = (callee_uid == callee->gui_interfaceslot_ptr->uid);
                    } else {
                        callee_fits = (callee_uid == callee->uid);
                    }
                }
                if (caller_fits && callee_fits) {
                    delete_calls_uids.emplace_back(call_ptr->uid);
                }
            }
        }

        // Actual deletion of calls
        for (auto& delete_call_uid : delete_calls_uids) {
            for (auto iter = this->calls.begin(); iter != this->calls.end(); iter++) {
                if ((*iter)->uid == delete_call_uid) {

                    QueueData queue_data;
                    bool valid_caller_ptr = false;
                    auto caller_ptr = (*iter)->GetCallSlot(megamol::gui::CallSlotType::CALLER);
                    if (caller_ptr != nullptr) {
                        if (caller_ptr->GetParentModule() != nullptr) {
                            queue_data.caller = caller_ptr->GetParentModule()->FullName() + "::" + caller_ptr->name;
                            valid_caller_ptr = true;
                        }
                    }
                    if (!valid_caller_ptr) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] Pointer to caller slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                            __LINE__);
                    }

                    bool valid_callee_ptr = false;
                    auto callee_ptr = (*iter)->GetCallSlot(megamol::gui::CallSlotType::CALLEE);
                    if (callee_ptr != nullptr) {
                        if (callee_ptr->GetParentModule() != nullptr) {
                            queue_data.callee = callee_ptr->GetParentModule()->FullName() + "::" + callee_ptr->name;
                            valid_callee_ptr = true;
                        }
                    }
                    if (!valid_callee_ptr) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] Pointer to callee slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                            __LINE__);
                    }

                    if (!valid_caller_ptr || !valid_callee_ptr) {
                        return false;
                    }

                    this->PushSyncQueue(QueueAction::DELETE_CALL, queue_data);

                    this->ResetStatePointers();

                    (*iter)->DisconnectCallSlots();

                    if ((*iter).use_count() > 1) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] Unclean deletion. Found %i references pointing to call. [%s, %s, line %d]\n",
                            (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                    }
#ifdef GUI_VERBOSE
                    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                        "[GUI] Deleted call '%s' (uid %i) from  project '%s'.\n", (*iter)->class_name.c_str(),
                        (*iter)->uid, this->name.c_str());
#endif // GUI_VERBOSE

                    (*iter).reset();
                    this->calls.erase(iter);

                    this->ForceSetDirty();
                    break;
                }
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
    return true;
}


ImGuiID megamol::gui::Graph::AddGroup(const std::string& group_name) {

    try {
        ImGuiID group_id = megamol::gui::GenerateUniqueID();
        auto group_ptr = std::make_shared<Group>(group_id);
        group_ptr->name = (group_name.empty()) ? (this->generate_unique_group_name()) : (group_name);
        this->groups.emplace_back(group_ptr);
        this->ForceSetDirty();

#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Added group '%s' (uid %i) to project '%s'.\n",
            group_ptr->name.c_str(), group_ptr->uid, this->name.c_str());
#endif // GUI_VERBOSE
        return group_id;

    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    }

    return GUI_INVALID_ID;
}


bool megamol::gui::Graph::GetGroup(ImGuiID group_uid, megamol::gui::GroupPtr_t& out_group_ptr) {

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


bool megamol::gui::Graph::DeleteGroup(ImGuiID group_uid) {

    // ! No syncronisation of module renaming considered
    try {
        for (auto iter = this->groups.begin(); iter != this->groups.end(); iter++) {
            if ((*iter)->uid == group_uid) {

                this->ResetStatePointers();

                if ((*iter).use_count() > 1) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Unclean deletion. Found %i references pointing to group. [%s, %s, line %d]\n",
                        (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                }
#ifdef GUI_VERBOSE
                megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                    "[GUI] Deleted group '%s' (uid %i) from  project '%s'.\n", (*iter)->name.c_str(), (*iter)->uid,
                    this->name.c_str());
#endif // GUI_VERBOSE
                (*iter).reset();
                this->groups.erase(iter);

                this->ForceSetDirty();

                this->ForceUpdate();
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
        "[GUI] Invalid group uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}


ImGuiID megamol::gui::Graph::AddGroupModule(const std::string& group_name, const ModulePtr_t& module_ptr) {

    try {
        // Only create new group if given name is not empty
        if (!group_name.empty()) {
            // Check if group with given name already exists
            ImGuiID existing_group_uid = GUI_INVALID_ID;
            for (auto& group_ptr : this->groups) {
                if (group_ptr->name == group_name) {
                    existing_group_uid = group_ptr->uid;
                }
            }
            // Create new group if there is no one with given name
            if (existing_group_uid == GUI_INVALID_ID) {
                existing_group_uid = this->AddGroup(group_name);
            }
            // Add module to group
            for (auto& group_ptr : this->groups) {
                if (group_ptr->uid == existing_group_uid) {
                    Graph::QueueData queue_data;
                    queue_data.name_id = module_ptr->FullName();
                    if (group_ptr->AddModule(module_ptr)) {
                        queue_data.rename_id = module_ptr->FullName();
                        this->PushSyncQueue(Graph::QueueAction::RENAME_MODULE, queue_data);
                        this->ForceSetDirty();
                        return existing_group_uid;
                    }
                }
            }
        }
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    }

    return GUI_INVALID_ID;
}


void megamol::gui::Graph::Clear(void) {

    this->ResetStatePointers();
    // Groups are implicitly deleted when last module of group is deleted.

    // 1) Delete all modules
    std::vector<ImGuiID> module_uids;
    for (auto& module_ptr : this->modules) {
        module_uids.emplace_back(module_ptr->uid);
    }
    for (auto& module_uid : module_uids) {
        this->DeleteModule(module_uid, true);
    }

    // 2) Delete all calls
    std::vector<ImGuiID> call_uids;
    for (auto& call_ptr : this->calls) {
        call_uids.emplace_back(call_ptr->GetID());
    }
    for (auto& call_uid : call_uids) {
        this->DeleteCall(call_uid);
    }
}


bool megamol::gui::Graph::UniqueModuleRename(const std::string& module_full_name) {

    for (auto& mod : this->modules) {
        if (module_full_name == mod->FullName()) {
            mod->name = this->generate_unique_module_name(mod->name);

            QueueData queue_data;
            queue_data.name_id = module_full_name;
            queue_data.rename_id = mod->FullName();
            this->PushSyncQueue(QueueAction::RENAME_MODULE, queue_data);

            this->ForceUpdate();

            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] Renamed existing module '%s' while adding module with same name. "
                "This is required for successful unambiguous parameter addressing which uses the module "
                "name. [%s, "
                "%s, line %d]\n",
                module_full_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
            return true;
        }
    }
    return false;
}


bool megamol::gui::Graph::PushSyncQueue(QueueAction action, const QueueData& in_data) {

    // Use sync queue only when interface to core graph is available
    if (!this->HasCoreInterface())
        return false;

    // Validate and process given data
    megamol::gui::Graph::QueueData queue_data = in_data;
    switch (action) {
    case (QueueAction::ADD_MODULE): {
        if (queue_data.name_id.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Graph sync queue action ADD_MODULE is missing data for 'name_id'. [%s, %s, line %d]\n", __FILE__,
                __FUNCTION__, __LINE__);
            return false;
        }
        if (queue_data.class_name.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Graph sync queue action ADD_MODULE is missing data for 'class_name'. [%s, %s, line %d]\n",
                __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
    } break;
    case (QueueAction::DELETE_MODULE): {
        if (queue_data.name_id.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Graph sync queue action ADD_MODULE is missing data for 'name_id'. [%s, %s, line %d]\n", __FILE__,
                __FUNCTION__, __LINE__);
            return false;
        }
    } break;
    case (QueueAction::RENAME_MODULE): {
        if (queue_data.name_id.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Graph sync queue action ADD_MODULE is missing data for 'name_id'. [%s, %s, line %d]\n", __FILE__,
                __FUNCTION__, __LINE__);
            return false;
        }
        if (queue_data.rename_id.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Graph sync queue action RENAME_MODULE is missing data for 'rename_id'. [%s, %s, line %d]\n",
                __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
    } break;
    case (QueueAction::ADD_CALL): {
        if (queue_data.class_name.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Graph sync queue action ADD_CALL is missing data for 'class_name'. [%s, %s, line %d]\n",
                __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        if (queue_data.caller.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Graph sync queue action ADD_CALL is missing data for 'caller'. [%s, %s, line %d]\n", __FILE__,
                __FUNCTION__, __LINE__);
            return false;
        }
        if (queue_data.callee.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Graph sync queue action ADD_CALL is missing data for 'callee'. [%s, %s, line %d]\n", __FILE__,
                __FUNCTION__, __LINE__);
            return false;
        }
    } break;
    case (QueueAction::DELETE_CALL): {
        if (queue_data.caller.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Graph sync queue action DELETE_CALL is missing data for 'caller'. [%s, %s, line %d]\n", __FILE__,
                __FUNCTION__, __LINE__);
            return false;
        }
        if (queue_data.callee.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Graph sync queue action DELETE_CALL is missing data for 'callee'. [%s, %s, line %d]\n", __FILE__,
                __FUNCTION__, __LINE__);
            return false;
        }
    } break;
    case (QueueAction::CREATE_GRAPH_ENTRY): {
    } break;
    case (QueueAction::REMOVE_GRAPH_ENTRY): {
    } break;
    default: {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown graph sync queue action. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    } break;
    }

    this->sync_queue.push(SyncQueueData_t(action, queue_data));
    return true;
}


bool megamol::gui::Graph::PopSyncQueue(QueueAction& out_action, QueueData& out_data) {

    if (!this->sync_queue.empty()) {
        out_action = std::get<0>(this->sync_queue.front());
        out_data = std::get<1>(this->sync_queue.front());
        this->sync_queue.pop();
        return true;
    }
    return false;
}


bool megamol::gui::Graph::StateFromJSON(const nlohmann::json& in_json) {

    try {
        if (!in_json.is_object()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Invalid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        for (auto& header_item : in_json.items()) {
            if (header_item.key() == GUI_JSON_TAG_GRAPHS) {
                for (auto& content_item : header_item.value().items()) {
                    std::string json_graph_id = content_item.key();
                    GUIUtils::Utf8Decode(json_graph_id);
                    if (json_graph_id == GUI_JSON_TAG_PROJECT) {
                        auto graph_state = content_item.value();

                        /// std::string filename;
                        /// megamol::core::utility::get_json_value<std::string>(graph_state, {"project_file"},
                        /// &filename); this->SetFilename(filename);

                        megamol::core::utility::get_json_value<std::string>(graph_state, {"project_name"}, &this->name);

                        bool tmp_show_parameter_sidebar = false;
                        this->change_show_parameter_sidebar = false;
                        if (megamol::core::utility::get_json_value<bool>(
                                graph_state, {"show_parameter_sidebar"}, &tmp_show_parameter_sidebar)) {
                            this->change_show_parameter_sidebar = true;
                            this->show_parameter_sidebar = tmp_show_parameter_sidebar;
                        }

                        megamol::core::utility::get_json_value<float>(
                            graph_state, {"parameter_sidebar_width"}, &this->parameter_sidebar_width);

                        megamol::core::utility::get_json_value<bool>(graph_state, {"show_grid"}, &this->show_grid);

                        if (megamol::core::utility::get_json_value<bool>(
                                graph_state, {"show_call_label"}, &this->show_call_label)) {
                            for (auto& call : this->GetCalls()) {
                                call->gui_label_visible = this->show_call_label;
                            }
                        }
                        if (megamol::core::utility::get_json_value<bool>(
                                graph_state, {"show_call_slots_label"}, &this->show_call_slots_label)) {
                            for (auto& call : this->GetCalls()) {
                                call->gui_slots_visible = this->show_call_slots_label;
                            }
                        }
                        if (megamol::core::utility::get_json_value<bool>(
                                graph_state, {"show_module_label"}, &this->show_module_label)) {
                            for (auto& mod : this->GetModules()) {
                                mod->label_visible = this->show_module_label;
                            }
                        }

                        if (megamol::core::utility::get_json_value<bool>(
                                graph_state, {"show_slot_label"}, &this->show_slot_label)) {
                            for (auto& mod : this->GetModules()) {
                                for (auto& callslot_types : mod->GetCallSlots()) {
                                    for (auto& callslots : callslot_types.second) {
                                        callslots->label_visible = this->show_slot_label;
                                    }
                                }
                            }
                            for (auto& group_ptr : this->GetGroups()) {
                                for (auto& interfaceslots_map : group_ptr->GetInterfaceSlots()) {
                                    for (auto& interfaceslot_ptr : interfaceslots_map.second) {
                                        interfaceslot_ptr->label_visible = this->show_slot_label;
                                    }
                                }
                            }
                        }

                        megamol::core::utility::get_json_value<bool>(
                            graph_state, {"params_visible"}, &this->params_visible);

                        megamol::core::utility::get_json_value<bool>(
                            graph_state, {"params_readonly"}, &this->params_readonly);

                        if (megamol::core::utility::get_json_value<bool>(
                                graph_state, {"param_extended_mode"}, &this->param_extended_mode)) {
                            for (auto& module_ptr : this->GetModules()) {
                                for (auto& parameter : module_ptr->parameters) {
                                    parameter.extended = this->param_extended_mode;
                                }
                            }
                        }

                        std::array<float, 2> canvas_scrolling;
                        megamol::core::utility::get_json_value<float>(
                            graph_state, {"canvas_scrolling"}, canvas_scrolling.data(), canvas_scrolling.size());
                        this->gui_graph_state.canvas.scrolling = ImVec2(canvas_scrolling[0], canvas_scrolling[1]);

                        if (megamol::core::utility::get_json_value<float>(
                                graph_state, {"canvas_zooming"}, &this->gui_graph_state.canvas.zooming)) {
                            this->reset_zooming = false;
                        }

                        // modules
                        for (auto& module_item : graph_state.items()) {
                            if (module_item.key() == GUI_JSON_TAG_MODULES) {
                                for (auto& module_state : module_item.value().items()) {
                                    std::string module_fullname = module_state.key();
                                    auto position_item = module_state.value();
                                    std::array<float, 2> graph_position;
                                    megamol::core::utility::get_json_value<float>(module_state.value(),
                                        {"graph_position"}, graph_position.data(), graph_position.size());
                                    auto module_position = ImVec2(graph_position[0], graph_position[1]);

                                    // Apply graph position to module
                                    bool module_found = false;
                                    for (auto& module_ptr : this->GetModules()) {
                                        if (module_ptr->FullName() == module_fullname) {
                                            module_ptr->position = module_position;
                                            module_found = true;
                                        }
                                    }
                                    if (!module_found) {
                                        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                                            "[GUI] JSON state: Unable to find module '%s' to apply graph position "
                                            "in configurator. [%s, %s, line %d]\n",
                                            module_fullname.c_str(), __FILE__, __FUNCTION__, __LINE__);
                                    }
                                }
                            }
                        }

                        // interfaces
                        for (auto& interfaces_item : graph_state.items()) {
                            if (interfaces_item.key() == GUI_JSON_TAG_INTERFACES) {
                                for (auto& interface_state : interfaces_item.value().items()) {
                                    std::string group_name = interface_state.key();
                                    auto interfaceslot_items = interface_state.value();

                                    // interfaces
                                    for (auto& interfaceslot_item : interfaceslot_items.items()) {
                                        std::vector<std::string> calleslot_fullnames;
                                        for (auto& callslot_item : interfaceslot_item.value().items()) {
                                            std::string callslot_name;
                                            megamol::core::utility::get_json_value<std::string>(
                                                callslot_item.value(), {}, &callslot_name);
                                            calleslot_fullnames.emplace_back(callslot_name);
                                        }

                                        // Add interface slot containing found calls slots to group
                                        // Find pointers to call slots by name
                                        CallSlotPtrVector_t callslot_ptr_vector;
                                        for (auto& callsslot_fullname : calleslot_fullnames) {
                                            auto split_pos = callsslot_fullname.rfind("::");
                                            if (split_pos != std::string::npos) {
                                                std::string callslot_name = callsslot_fullname.substr(split_pos + 2);
                                                std::string module_fullname = callsslot_fullname.substr(0, (split_pos));
                                                for (auto& module_ptr : this->GetModules()) {
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
                                            for (auto& group_ptr : this->GetGroups()) {
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
                                                            for (size_t i = 1; i < callslot_ptr_vector.size(); i++) {
                                                                interfaceslot_ptr->AddCallSlot(
                                                                    callslot_ptr_vector[i], interfaceslot_ptr);
                                                            }
                                                        }
                                                    }
                                                    group_found = true;
                                                }
                                            }
                                            if (!group_found) {
                                                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
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

        this->update = true;
#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Read graph state from JSON.", this->name.c_str());
#endif // GUI_VERBOSE

    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON Error - Unable to read state from JSON. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


bool megamol::gui::Graph::StateToJSON(nlohmann::json& inout_json) {

    try {
        // Write graph state
        /// std::string filename = this->GetFilename();
        /// GUIUtils::Utf8Encode(filename);
        /// inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["project_file"] = filename;
        GUIUtils::Utf8Encode(this->name);
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["project_name"] = this->name;
        GUIUtils::Utf8Decode(this->name);
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["show_parameter_sidebar"] = this->show_parameter_sidebar;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["parameter_sidebar_width"] =
            this->parameter_sidebar_width;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["show_grid"] = this->show_grid;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["show_call_label"] = this->show_call_label;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["show_call_slots_label"] = this->show_call_slots_label;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["show_slot_label"] = this->show_slot_label;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["show_module_label"] = this->show_module_label;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["params_visible"] = this->params_visible;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["params_readonly"] = this->params_readonly;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["param_extended_mode"] = this->param_extended_mode;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["canvas_scrolling"] = {
            this->gui_graph_state.canvas.scrolling.x, this->gui_graph_state.canvas.scrolling.y};
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["canvas_zooming"] = this->gui_graph_state.canvas.zooming;

        // Write module positions
        for (auto& module_ptr : this->GetModules()) {
            inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT][GUI_JSON_TAG_MODULES][module_ptr->FullName()]
                      ["graph_position"] = {module_ptr->position.x, module_ptr->position.y};
        }

        // Write group interface slots
        size_t interface_number = 0;
        for (auto& group_ptr : this->GetGroups()) {
            for (auto& interfaceslots_map : group_ptr->GetInterfaceSlots()) {
                for (auto& interface_ptr : interfaceslots_map.second) {
                    std::string interface_label = "interface_slot_" + std::to_string(interface_number);
                    for (auto& callslot_ptr : interface_ptr->GetCallSlots()) {
                        std::string callslot_fullname;
                        if (callslot_ptr->IsParentModuleConnected()) {
                            callslot_fullname = callslot_ptr->GetParentModule()->FullName() + "::" + callslot_ptr->name;
                        }
                        GUIUtils::Utf8Encode(callslot_fullname);
                        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT][GUI_JSON_TAG_INTERFACES][group_ptr->name]
                                  [interface_label] += callslot_fullname;
                    }
                    interface_number++;
                }
            }
        }

#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Wrote graph state to JSON.");
#endif // GUI_VERBOSE

    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error - Unable to write JSON of state. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
        return false;
    }

    return true;
}


void megamol::gui::Graph::Present(megamol::gui::Graph& inout_graph, GraphState_t& state) {

    try {
        if (ImGui::GetCurrentContext() == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return;
        }
        ImGuiIO& io = ImGui::GetIO();

        ImGuiID graph_uid = inout_graph.uid;
        ImGui::PushID(graph_uid);

        // Tab showing this graph ---------------
        bool popup_rename = false;
        ImGuiTabItemFlags tab_flags = ImGuiTabItemFlags_None;
        if (inout_graph.IsDirty()) {
            tab_flags |= ImGuiTabItemFlags_UnsavedDocument;
        }
        std::string graph_label = "    " + inout_graph.name + "  ###graph" + std::to_string(graph_uid);
        if (inout_graph.HasCoreInterface()) {
            graph_label = "    [RUNNING]  " + graph_label;
        }

        // Checking for closed tab below
        bool open = true;
        if (ImGui::BeginTabItem(graph_label.c_str(), &open, tab_flags)) {

            // State Init ----------------------------
            if (this->change_show_parameter_sidebar) {
                state.show_parameter_sidebar = this->show_parameter_sidebar;
                this->change_show_parameter_sidebar = false;
            }
            this->show_parameter_sidebar = state.show_parameter_sidebar;

            this->gui_graph_state.hotkeys = state.hotkeys;
            this->gui_graph_state.groups.clear();
            for (auto& group : inout_graph.GetGroups()) {
                std::pair<ImGuiID, std::string> group_pair(group->uid, group->name);
                this->gui_graph_state.groups.emplace_back(group_pair);
            }
            this->gui_graph_state.interact.slot_dropped_uid = GUI_INVALID_ID;
            this->gui_graph_state.interact.graph_core_interface = inout_graph.GetCoreInterface();

            // Compatible slot pointers
            this->gui_graph_state.interact.callslot_compat_ptr.reset();
            this->gui_graph_state.interact.interfaceslot_compat_ptr.reset();
            //  Consider hovered slots only if there is no drag and drop
            bool slot_draganddrop_active = false;
            if (const ImGuiPayload* payload = ImGui::GetDragDropPayload()) {
                if (payload->IsDataType(GUI_DND_CALLSLOT_UID_TYPE)) {
                    slot_draganddrop_active = true;
                }
            }
            ImGuiID slot_uid = GUI_INVALID_ID;
            if (this->gui_graph_state.interact.callslot_selected_uid != GUI_INVALID_ID) {
                slot_uid = this->gui_graph_state.interact.callslot_selected_uid;
            } else if ((this->gui_graph_state.interact.interfaceslot_selected_uid == GUI_INVALID_ID) &&
                       (!slot_draganddrop_active)) {
                slot_uid = this->gui_graph_state.interact.callslot_hovered_uid;
            }
            if (slot_uid != GUI_INVALID_ID) {
                for (auto& module_ptr : inout_graph.GetModules()) {
                    CallSlotPtr_t callslot_ptr;
                    if (module_ptr->GetCallSlot(slot_uid, callslot_ptr)) {
                        this->gui_graph_state.interact.callslot_compat_ptr = callslot_ptr;
                    }
                }
            }
            // Compatible call slot and/or interface slot ptr
            if (this->gui_graph_state.interact.callslot_compat_ptr == nullptr) {
                slot_uid = GUI_INVALID_ID;
                if (this->gui_graph_state.interact.interfaceslot_selected_uid != GUI_INVALID_ID) {
                    slot_uid = this->gui_graph_state.interact.interfaceslot_selected_uid;
                } else if (!slot_draganddrop_active) {
                    slot_uid = this->gui_graph_state.interact.interfaceslot_hovered_uid;
                }
                if (slot_uid != GUI_INVALID_ID) {
                    for (auto& group_ptr : inout_graph.GetGroups()) {
                        InterfaceSlotPtr_t interfaceslot_ptr;
                        if (group_ptr->GetInterfaceSlot(slot_uid, interfaceslot_ptr)) {
                            this->gui_graph_state.interact.interfaceslot_compat_ptr = interfaceslot_ptr;
                        }
                    }
                }
            }

            // Context menu ---------------------
            if (ImGui::BeginPopupContextItem()) {

                ImGui::TextDisabled("Project");
                ImGui::Separator();

                if (ImGui::MenuItem("Save")) {
                    if (inout_graph.HasCoreInterface()) {
                        state.global_graph_save = true;
                    } else {
                        state.configurator_graph_save = true;
                    }
                }

                if (ImGui::MenuItem("Rename")) {
                    popup_rename = true;
                }

                if (!inout_graph.GetFilename().empty()) {
                    ImGui::Separator();
                    ImGui::TextDisabled("Filename");
                    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 13.0f);
                    std::string filename = inout_graph.GetFilename();
                    GUIUtils::Utf8Encode(filename);
                    ImGui::TextUnformatted(filename.c_str());
                    ImGui::PopTextWrapPos();
                }

                ImGui::EndPopup();
            }

            // Draw -----------------------------
            this->draw_menu(inout_graph);

            if (megamol::gui::gui_scaling.PendingChange()) {
                this->parameter_sidebar_width *= megamol::gui::gui_scaling.TransitonFactor();
            }
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
                        if (this->gui_graph_state.canvas.zooming <= state.font_scalings[i]) {
                            apply = true;
                        }
                    } else if (i == (scalings_count - 1)) {
                        if (this->gui_graph_state.canvas.zooming >= state.font_scalings[i]) {
                            apply = true;
                        }
                    } else {
                        if ((state.font_scalings[i - 1] < this->gui_graph_state.canvas.zooming) &&
                            (this->gui_graph_state.canvas.zooming < state.font_scalings[i + 1])) {
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
                this->draw_canvas(inout_graph, graph_width_auto);
                ImGui::PopFont();
            } else {
                throw std::invalid_argument("Pointer to font is nullptr.");
            }
            if (this->show_parameter_sidebar) {
                ImGui::SameLine();
                this->draw_parameters(inout_graph, this->parameter_sidebar_width);
            }

            state.graph_selected_uid = inout_graph.uid;

            // State processing ---------------------
            this->ResetStatePointers();
            bool reset_state = false;
            // Add module renaming event to graph synchronization queue -----------
            if (!this->gui_graph_state.interact.module_rename.empty()) {
                Graph::QueueData queue_data;
                for (auto& str_pair : this->gui_graph_state.interact.module_rename) {
                    queue_data.name_id = str_pair.first;
                    queue_data.rename_id = str_pair.second;
                    inout_graph.PushSyncQueue(Graph::QueueAction::RENAME_MODULE, queue_data);
                }
                this->gui_graph_state.interact.module_rename.clear();
            }
            // Add module graph entry event to graph synchronization queue ----------
            if (this->gui_graph_state.interact.module_graphentry_changed != vislib::math::Ternary::TRI_UNKNOWN) {
                // Choose single selected view module
                ModulePtr_t selected_mod_ptr;
                if (this->gui_graph_state.interact.modules_selected_uids.size() == 1) {
                    for (auto& mod : inout_graph.GetModules()) {
                        if ((this->gui_graph_state.interact.modules_selected_uids.front() == mod->uid) && (mod->is_view)) {
                            selected_mod_ptr = mod;
                        }
                    }
                }
                if (selected_mod_ptr != nullptr) {
                    Graph::QueueData queue_data;
                    if (this->gui_graph_state.interact.module_graphentry_changed == vislib::math::Ternary::TRI_TRUE) {
                        // Remove all graph entries
                        for (auto module_ptr : inout_graph.GetModules()) {
                            if (module_ptr->is_view && module_ptr->IsGraphEntry()) {
                                module_ptr->graph_entry_name.clear();
                                queue_data.name_id = module_ptr->FullName();
                                inout_graph.PushSyncQueue(Graph::QueueAction::REMOVE_GRAPH_ENTRY, queue_data);
                            }
                        }
                        // Add new graph entry
                        queue_data.name_id = selected_mod_ptr->FullName();
                        selected_mod_ptr->graph_entry_name = inout_graph.GenerateUniqueGraphEntryName();
                        inout_graph.PushSyncQueue(Graph::QueueAction::CREATE_GRAPH_ENTRY, queue_data);
                    } else {
                        queue_data.name_id = selected_mod_ptr->FullName();
                        selected_mod_ptr->graph_entry_name.clear();
                        inout_graph.PushSyncQueue(Graph::QueueAction::REMOVE_GRAPH_ENTRY, queue_data);
                    }
                }
                this->gui_graph_state.interact.module_graphentry_changed = vislib::math::Ternary::TRI_UNKNOWN;
            }
            // Add module to group ------------------------------------------------
            if (!this->gui_graph_state.interact.modules_add_group_uids.empty()) {
                if (inout_graph.GetCoreInterface() == GraphCoreInterface::CORE_INSTANCE_GRAPH) {
                    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                        "[GUI] The action [Add Module to Group] is not yet supported for the graph "
                        "using the 'Core Instance Graph' interface. Open project from file to make desired "
                        "changes. [%s, %s, line %d]\n",
                        __FILE__, __FUNCTION__, __LINE__);
                } else {
                    ModulePtr_t module_ptr;
                    ImGuiID new_group_uid = GUI_INVALID_ID;
                    for (auto& uid_pair : this->gui_graph_state.interact.modules_add_group_uids) {
                        module_ptr.reset();
                        for (auto& mod : inout_graph.GetModules()) {
                            if (mod->uid == uid_pair.first) {
                                module_ptr = mod;
                            }
                        }
                        if (module_ptr != nullptr) {
                            std::string current_module_fullname = module_ptr->FullName();

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
                                Graph::QueueData queue_data;
                                queue_data.name_id = module_ptr->FullName();

                                // Remove module from previous associated group
                                ImGuiID module_group_uid = module_ptr->group.uid;
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
                                queue_data.rename_id = module_ptr->FullName();
                                inout_graph.PushSyncQueue(Graph::QueueAction::RENAME_MODULE, queue_data);
                                inout_graph.ForceSetDirty();
                                // Restore interface slots after adding module to new group
                                if (restore_interfaceslots) {
                                    remove_group_ptr->RestoreInterfaceslots();
                                }
                            }
                        }
                    }
                }
                reset_state = true;
            }
            // Remove module from group -------------------------------------------
            if (!this->gui_graph_state.interact.modules_remove_group_uids.empty()) {
                if (inout_graph.GetCoreInterface() == GraphCoreInterface::CORE_INSTANCE_GRAPH) {
                    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                        "[GUI] The action [Remove Module from Group] is not yet supported for the graph "
                        "using the 'Core Instance Graph' interface. Open project from file to make desired "
                        "changes. [%s, %s, line %d]\n",
                        __FILE__, __FUNCTION__, __LINE__);
                } else {
                    for (auto& module_uid : this->gui_graph_state.interact.modules_remove_group_uids) {
                        ModulePtr_t module_ptr;
                        for (auto& mod : inout_graph.GetModules()) {
                            if (mod->uid == module_uid) {
                                module_ptr = mod;
                            }
                        }
                        for (auto& remove_group_ptr : inout_graph.GetGroups()) {
                            if (remove_group_ptr->ContainsModule(module_uid)) {
                                Graph::QueueData queue_data;
                                queue_data.name_id = module_ptr->FullName();
                                remove_group_ptr->RemoveModule(module_ptr->uid);
                                queue_data.rename_id = module_ptr->FullName();
                                inout_graph.PushSyncQueue(Graph::QueueAction::RENAME_MODULE, queue_data);
                                inout_graph.ForceSetDirty();
                            }
                        }
                    }
                }
                reset_state = true;
            }
            // Create new interface slot for call slot ----------------------------
            ImGuiID callslot_uid = this->gui_graph_state.interact.callslot_add_group_uid.first;
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
                    ImGuiID module_uid = this->gui_graph_state.interact.callslot_add_group_uid.second;
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
            // Remove call slot from interface of group ---------------------------
            callslot_uid = this->gui_graph_state.interact.callslot_remove_group_uid.first;
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
                ImGuiID module_uid = this->gui_graph_state.interact.callslot_remove_group_uid.second;
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
                                        if (other_callslot_ptr->GetParentModule()->group.uid != group->uid) {
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
            // Process module/call/group deletion ---------------------------------
            if ((this->gui_graph_state.interact.process_deletion) ||
                (!io.WantTextInput &&
                    this->gui_graph_state.hotkeys[megamol::gui::HotkeyIndex::DELETE_GRAPH_ITEM].is_pressed)) {
                if (!this->gui_graph_state.interact.modules_selected_uids.empty()) {
                    for (auto& module_uid : this->gui_graph_state.interact.modules_selected_uids) {
                        inout_graph.DeleteModule(module_uid);
                    }
                }
                if (this->gui_graph_state.interact.call_selected_uid != GUI_INVALID_ID) {
                    inout_graph.DeleteCall(this->gui_graph_state.interact.call_selected_uid);
                }
                if (this->gui_graph_state.interact.group_selected_uid != GUI_INVALID_ID) {
                    if (inout_graph.GetCoreInterface() == GraphCoreInterface::CORE_INSTANCE_GRAPH) {
                        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                            "[GUI] The action [Delete Group] is not yet supported for the graph "
                            "using the 'Core Instance Graph' interface. Open project from file to make desired "
                            "changes. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                    } else {
                        // Save old name of modules
                        std::vector<std::pair<ImGuiID, std::string>> module_uid_name_pair;
                        GroupPtr_t group_ptr;
                        if (inout_graph.GetGroup(this->gui_graph_state.interact.group_selected_uid, group_ptr)) {
                            for (auto& module_ptr : group_ptr->GetModules()) {
                                module_uid_name_pair.push_back({module_ptr->uid, module_ptr->FullName()});
                            }
                        }
                        group_ptr.reset();
                        // Delete group
                        inout_graph.DeleteGroup(this->gui_graph_state.interact.group_selected_uid);
                        // Push module renaming to sync queue
                        for (auto& module_ptr : inout_graph.GetModules()) {
                            for (auto& module_pair : module_uid_name_pair) {
                                if (module_ptr->uid == module_pair.first) {
                                    Graph::QueueData queue_data;
                                    queue_data.name_id = module_pair.second;
                                    queue_data.rename_id = module_ptr->FullName();
                                    inout_graph.PushSyncQueue(Graph::QueueAction::RENAME_MODULE, queue_data);
                                }
                            }
                        }
                    }
                }
                if (this->gui_graph_state.interact.interfaceslot_selected_uid != GUI_INVALID_ID) {
                    for (auto& group_ptr : inout_graph.GetGroups()) {
                        InterfaceSlotPtr_t interfaceslot_ptr;
                        if (group_ptr->GetInterfaceSlot(
                                this->gui_graph_state.interact.interfaceslot_selected_uid, interfaceslot_ptr)) {
                            // Delete all calls connected
                            std::vector<ImGuiID> call_uids;
                            for (auto& callslot_ptr : interfaceslot_ptr->GetCallSlots()) {
                                for (auto& call_ptr : callslot_ptr->GetConnectedCalls()) {
                                    auto caller = call_ptr->GetCallSlot(CallSlotType::CALLER);
                                    auto callee = call_ptr->GetCallSlot(CallSlotType::CALLEE);
                                    if (caller->IsParentModuleConnected() && callee->IsParentModuleConnected()) {
                                        if (caller->GetParentModule()->group.uid !=
                                            callee->GetParentModule()->group.uid) {
                                            call_uids.emplace_back(call_ptr->uid);
                                        }
                                    }
                                }
                            }
                            for (auto& call_uid : call_uids) {
                                inout_graph.DeleteCall(call_uid);
                            }
                            interfaceslot_ptr.reset();

                            group_ptr->DeleteInterfaceSlot(this->gui_graph_state.interact.interfaceslot_selected_uid);
                            inout_graph.ForceSetDirty();
                        }
                    }
                }

                reset_state = true;
            }
            // Delete empty group(s) ----------------------------------------------
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

            // Reset interact state for modules and call slots --------------------
            if (reset_state) {
                this->gui_graph_state.interact.process_deletion = false;
                this->gui_graph_state.interact.group_selected_uid = GUI_INVALID_ID;
                this->gui_graph_state.interact.group_hovered_uid = GUI_INVALID_ID;
                this->gui_graph_state.interact.interfaceslot_selected_uid = GUI_INVALID_ID;
                this->gui_graph_state.interact.interfaceslot_hovered_uid = GUI_INVALID_ID;
                this->gui_graph_state.interact.modules_selected_uids.clear();
                this->gui_graph_state.interact.module_hovered_uid = GUI_INVALID_ID;
                this->gui_graph_state.interact.modules_add_group_uids.clear();
                this->gui_graph_state.interact.modules_remove_group_uids.clear();
                this->gui_graph_state.interact.call_selected_uid = GUI_INVALID_ID;
                this->gui_graph_state.interact.call_hovered_uid = GUI_INVALID_ID;
                this->gui_graph_state.interact.callslot_selected_uid = GUI_INVALID_ID;
                this->gui_graph_state.interact.callslot_hovered_uid = GUI_INVALID_ID;
                this->gui_graph_state.interact.callslot_add_group_uid = UIDPair_t(GUI_INVALID_ID, GUI_INVALID_ID);
                this->gui_graph_state.interact.callslot_remove_group_uid = UIDPair_t(GUI_INVALID_ID, GUI_INVALID_ID);
                this->gui_graph_state.interact.slot_dropped_uid = GUI_INVALID_ID;
            }

            // Layout graph -------------------------------------------------------
            /// One frame delay required for making sure canvas data is completely updated previously
            if (this->graph_layout > 0) {
                if (this->graph_layout > 1) {
                    this->layout_graph(inout_graph);
                    this->graph_layout = 0;
                } else {
                    this->graph_layout++;
                }
            }
            // Layout modules of selected group -----------------------------------
            if (this->gui_graph_state.interact.group_layout) {
                for (auto& group_ptr : inout_graph.GetGroups()) {
                    if (group_ptr->uid == this->gui_graph_state.interact.group_selected_uid) {
                        ImVec2 init_position = ImVec2(FLT_MAX, FLT_MAX);
                        for (auto& module_ptr : group_ptr->GetModules()) {
                            init_position.x = std::min(module_ptr->position.x, init_position.x);
                            init_position.y = std::min(module_ptr->position.y, init_position.y);
                        }
                        this->layout(group_ptr->GetModules(), GroupPtrVector_t(), init_position);
                    }
                }
                this->gui_graph_state.interact.group_layout = false;
                this->update = true;
            }
            // Layout selelected modules ------------------------------------------
            if (this->gui_graph_state.interact.modules_layout) {
                ImVec2 init_position = ImVec2(FLT_MAX, FLT_MAX);
                ModulePtrVector_t selected_modules;
                for (auto& module_ptr : inout_graph.GetModules()) {
                    for (auto& selected_module_uid : this->gui_graph_state.interact.modules_selected_uids) {
                        if (module_ptr->uid == selected_module_uid) {
                            init_position.x = std::min(module_ptr->position.x, init_position.x);
                            init_position.y = std::min(module_ptr->position.y, init_position.y);
                            selected_modules.emplace_back(module_ptr);
                        }
                    }
                }
                this->layout(selected_modules, GroupPtrVector_t(), init_position);
                this->gui_graph_state.interact.modules_layout = false;
            }
            // Set delete flag if tab was closed ----------------------------------
            bool popup_prevent_close_permanent = false;
            if (!open) {
                if (inout_graph.HasCoreInterface()) {
                    popup_prevent_close_permanent = true;
                } else {
                    state.graph_delete = true;
                    state.graph_selected_uid = inout_graph.uid;
                }
            }

            // Propoagate unhandeled hotkeys back to configurator state -----------
            state.hotkeys = this->gui_graph_state.hotkeys;

            // Prevent closing tab of running project pop-up ----------------------
            bool tmp;
            MinimalPopUp::PopUp("Close Project", popup_prevent_close_permanent,
                "Running project can not be closed in configurator.", "OK", tmp, "", tmp);

            // Rename pop-up ------------------------------------------------------
            if (this->rename_popup.PopUp("Rename Project", popup_rename, inout_graph.name)) {
                inout_graph.ForceSetDirty();
            }

            // Module Parameter Child -------------------------------------------------
            // Choose single selected view module
            ModulePtr_t selected_mod_ptr;
            if (this->gui_graph_state.interact.modules_selected_uids.size() == 1) {
                for (auto& mod : inout_graph.GetModules()) {
                    if ((this->gui_graph_state.interact.modules_selected_uids.front() == mod->uid)) {
                        selected_mod_ptr = mod;
                    }
                }
            }
            if (selected_mod_ptr == nullptr) {
                this->gui_graph_state.interact.module_param_child_position = ImVec2(-1.0f, -1.0f);
            } else {
                if ((this->gui_graph_state.interact.module_param_child_position.x > 0.0f) &&
                    (this->gui_graph_state.interact.module_param_child_position.y > 0.0f)) {
                    std::string pop_up_id = "module_param_child";

                    if (!ImGui::IsPopupOpen(pop_up_id.c_str())) {
                        ImGui::OpenPopup(pop_up_id.c_str(), ImGuiPopupFlags_None);
                        this->gui_graph_state.interact.module_param_child_position.x += ImGui::GetFrameHeight();
                        ImGui::SetNextWindowPos(this->gui_graph_state.interact.module_param_child_position);
                        ImGui::SetNextWindowSize(ImVec2(10.0f, 10.0f));
                    }
                    auto popup_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar |
                                       ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar |
                                       ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove;
                    if (ImGui::BeginPopup(pop_up_id.c_str(), popup_flags)) {
                        // Draw parameters
                        selected_mod_ptr->param_groups.PresentGUI(selected_mod_ptr->parameters,
                            selected_mod_ptr->FullName(), "", vislib::math::Ternary::TRI_UNKNOWN, false,
                            Parameter::WidgetScope::LOCAL, nullptr, nullptr, GUI_INVALID_ID, nullptr);

                        ImVec2 popup_size = ImGui::GetWindowSize();
                        bool param_popup_open = ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId);
                        bool module_parm_child_popup_hovered = false;
                        if ((ImGui::GetMousePos().x >= this->gui_graph_state.interact.module_param_child_position.x) &&
                            (ImGui::GetMousePos().x <=
                                (this->gui_graph_state.interact.module_param_child_position.x + popup_size.x)) &&
                            (ImGui::GetMousePos().y >= this->gui_graph_state.interact.module_param_child_position.y) &&
                            (ImGui::GetMousePos().y <=
                                (this->gui_graph_state.interact.module_param_child_position.y + popup_size.y))) {
                            module_parm_child_popup_hovered = true;
                        }
                        if (!param_popup_open && ((ImGui::IsMouseClicked(0) && !module_parm_child_popup_hovered) ||
                                                     ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape)))) {

                            this->gui_graph_state.interact.module_param_child_position = ImVec2(-1.0f, -1.0f);
                            // Reset module selection to prevent irrgular dragging
                            this->gui_graph_state.interact.modules_selected_uids.clear();
                            ImGui::CloseCurrentPopup();
                        }

                        // Save actual position since pop-up might be moved away from right edge
                        this->gui_graph_state.interact.module_param_child_position = ImGui::GetWindowPos();
                        ImGui::EndPopup();
                    }
                }
            }

            ImGui::EndTabItem();
        }

        ImGui::PopID();

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


void megamol::gui::Graph::draw_menu(megamol::gui::Graph& inout_graph) {

    ImGuiStyle& style = ImGui::GetStyle();
    auto button_size = ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight());

    float child_height = ImGui::GetFrameHeightWithSpacing();
    auto child_flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NavFlattened | ImGuiWindowFlags_MenuBar;
    ImGui::BeginChild("graph_menu", ImVec2(0.0f, child_height), false, child_flags);

    ImGui::BeginMenuBar();

    // GRAPH LAYOUT
    if (ImGui::Button("Layout Graph")) {
        this->graph_layout = 1;
    }
    ImGui::Separator();

    // MODULES
    if (ImGui::BeginMenu("Modules")) {
        if (ImGui::MenuItem("Label", nullptr, &this->show_module_label)) {
            for (auto& module_ptr : inout_graph.GetModules()) {
                module_ptr->label_visible = this->show_module_label;
            }
            this->update = true;
        }
        ImGui::EndMenu();
    }
    ImGui::Separator();

    // CALLS
    if (ImGui::BeginMenu("Calls")) {
        if (ImGui::MenuItem("Label", nullptr, &this->show_call_label)) {
            for (auto& call_ptr : inout_graph.GetCalls()) {
                call_ptr->gui_label_visible = this->show_call_label;
            }
            this->update = true;
        }
        if (ImGui::MenuItem("Slots", nullptr, &this->show_call_slots_label)) {
            for (auto& call_ptr : inout_graph.GetCalls()) {
                call_ptr->gui_slots_visible = this->show_call_slots_label;
            }
            this->update = true;
        }
        ImGui::EndMenu();
    }
    ImGui::Separator();

    // SLOTS
    if (ImGui::BeginMenu("Slots")) {
        if (ImGui::MenuItem("Label", nullptr, &this->show_slot_label)) {
            for (auto& module_ptr : inout_graph.GetModules()) {
                for (auto& callslot_types : module_ptr->GetCallSlots()) {
                    for (auto& callslots : callslot_types.second) {
                        callslots->label_visible = this->show_slot_label;
                    }
                }
            }
            for (auto& group_ptr : inout_graph.GetGroups()) {
                for (auto& interfaceslots_map : group_ptr->GetInterfaceSlots()) {
                    for (auto& interfaceslot_ptr : interfaceslots_map.second) {
                        interfaceslot_ptr->label_visible = this->show_slot_label;
                    }
                }
            }
            this->update = true;
        }
        ImGui::EndMenu();
    }
    ImGui::Separator();

    // GRID
    if (ImGui::BeginMenu("Grid")) {
        ImGui::MenuItem("Show", nullptr, &this->show_grid);
        ImGui::EndMenu();
    }
    ImGui::Separator();

    // Choose single selected view module
    ModulePtr_t selected_mod_ptr;
    if (this->gui_graph_state.interact.modules_selected_uids.size() == 1) {
        for (auto& mod : inout_graph.GetModules()) {
            if ((this->gui_graph_state.interact.modules_selected_uids.front() == mod->uid) && (mod->is_view)) {
                selected_mod_ptr = mod;
            }
        }
    }
    // Graph Entry Checkbox
    const float min_text_width = 3.0f * ImGui::GetFrameHeightWithSpacing();
    if (selected_mod_ptr == nullptr) {
        GUIUtils::ReadOnlyWigetStyle(true);
        bool is_graph_entry = false;
        this->current_graph_entry_name.clear();
        ImGui::Checkbox("Graph Entry", &is_graph_entry);
        // ImGui::SameLine(0.0f, min_text_width + 2.0f * style.ItemSpacing.x);
        GUIUtils::ReadOnlyWigetStyle(false);
    } else {
        bool is_graph_entry = selected_mod_ptr->IsGraphEntry();
        if (ImGui::Checkbox("Graph Entry", &is_graph_entry)) {
            if (inout_graph.GetCoreInterface() == GraphCoreInterface::CORE_INSTANCE_GRAPH) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "[GUI] The action [Change Graph Entry] is not yet supported for the graph "
                    "using the 'Core Instance Graph' interface. Open project from file to make desired "
                    "changes. [%s, %s, line %d]\n",
                    __FILE__, __FUNCTION__, __LINE__);
            } else {
                Graph::QueueData queue_data;
                if (is_graph_entry) {
                    // Remove all graph entries
                    for (auto module_ptr : inout_graph.GetModules()) {
                        if (module_ptr->is_view && module_ptr->IsGraphEntry()) {
                            module_ptr->graph_entry_name.clear();
                            queue_data.name_id = module_ptr->FullName();
                            inout_graph.PushSyncQueue(Graph::QueueAction::REMOVE_GRAPH_ENTRY, queue_data);
                        }
                    }
                    // Add new graph entry
                    selected_mod_ptr->graph_entry_name = inout_graph.GenerateUniqueGraphEntryName();
                    queue_data.name_id = selected_mod_ptr->FullName();
                    inout_graph.PushSyncQueue(Graph::QueueAction::CREATE_GRAPH_ENTRY, queue_data);
                } else {
                    selected_mod_ptr->graph_entry_name.clear();
                    queue_data.name_id = selected_mod_ptr->FullName();
                    inout_graph.PushSyncQueue(Graph::QueueAction::REMOVE_GRAPH_ENTRY, queue_data);
                }
            }
        }

        this->current_graph_entry_name = selected_mod_ptr->graph_entry_name;
        float input_text_width = std::max(min_text_width,
            (ImGui::CalcTextSize(this->current_graph_entry_name.c_str()).x + 2.0f * style.ItemSpacing.x));
        ImGui::PushItemWidth(input_text_width);
        GUIUtils::Utf8Encode(this->current_graph_entry_name);
        ImGui::InputText("###current_graph_entry_name", &this->current_graph_entry_name, ImGuiInputTextFlags_None);
        GUIUtils::Utf8Decode(this->current_graph_entry_name);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            if (inout_graph.GetCoreInterface() == GraphCoreInterface::CORE_INSTANCE_GRAPH) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "[GUI] The action [Change Graph Entry] is not yet supported for the graph "
                    "using the 'Core Instance Graph' interface. Open project from file to make desired "
                    "changes. [%s, %s, line %d]\n",
                    __FILE__, __FUNCTION__, __LINE__);
            } else {
                selected_mod_ptr->graph_entry_name = this->current_graph_entry_name;
            }
        } else {
            this->current_graph_entry_name = selected_mod_ptr->graph_entry_name;
        }
        ImGui::PopItemWidth();
    }
    ImGui::Separator();

    // SCROLLING
    const float scroll_fac = 10.0f;
    ImGui::Text("Scrolling: %.2f, %.2f", this->gui_graph_state.canvas.scrolling.x, this->gui_graph_state.canvas.scrolling.y);
    ImGui::TextUnformatted("H:");
    if (ImGui::Button("+###hor_incr_scrolling", button_size)) {
        this->gui_graph_state.canvas.scrolling.x += scroll_fac;
        this->update = true;
    }
    if (ImGui::Button("-###hor_decr_scrolling", button_size)) {
        this->gui_graph_state.canvas.scrolling.x -= scroll_fac;
        this->update = true;
    }
    ImGui::TextUnformatted("V:");
    if (ImGui::Button("+###vert_incr_scrolling", button_size)) {
        this->gui_graph_state.canvas.scrolling.y += scroll_fac;
        this->update = true;
    }
    if (ImGui::Button("-###vert_decr_scrolling", button_size)) {
        this->gui_graph_state.canvas.scrolling.y -= scroll_fac;
        this->update = true;
    }
    if (ImGui::Button("Reset###reset_scrolling")) {
        this->gui_graph_state.canvas.scrolling = ImVec2(0.0f, 0.0f);
        this->update = true;
    }
    this->tooltip.Marker("Middle Mouse Button");
    ImGui::Separator();

    // ZOOMING
    ImGui::Text("Zooming: %.2f", this->gui_graph_state.canvas.zooming);
    if (ImGui::Button("+###incr_zooming", button_size)) {
        this->increment_zooming = true;
    }
    if (ImGui::Button("-###decr_zooming", button_size)) {
        this->decrement_zooming = true;
    }
    if (ImGui::Button("Reset###reset_zooming")) {
        this->reset_zooming = true;
    }
    this->tooltip.Marker("Mouse Wheel");
    ImGui::Separator();

    ImGui::EndMenuBar();

    ImGui::EndChild();
}


void megamol::gui::Graph::draw_canvas(megamol::gui::Graph& inout_graph, float graph_width) {

    ImGuiIO& io = ImGui::GetIO();

    // Colors
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(1, 1));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    auto child_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove;
    ImGui::BeginChild("region", ImVec2(graph_width, 0.0f), true, child_flags);

    this->canvas_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_None); // Ignores Pop-Ups like Context-Menu

    // UPDATE CANVAS -----------------------------------------------------------

    // Update canvas position
    ImVec2 new_position = ImGui::GetWindowPos();
    if (this->gui_graph_state.canvas.position != new_position) {
        this->update = true;
    }
    this->gui_graph_state.canvas.position = new_position;
    // Update canvas size
    ImVec2 new_size = ImGui::GetWindowSize();
    if (this->gui_graph_state.canvas.size != new_size) {
        this->update = true;
    }
    this->gui_graph_state.canvas.size = new_size;
    // Update canvas offset
    ImVec2 new_offset =
        this->gui_graph_state.canvas.position + (this->gui_graph_state.canvas.scrolling * this->gui_graph_state.canvas.zooming);
    if (this->gui_graph_state.canvas.offset != new_offset) {
        this->update = true;
    }
    this->gui_graph_state.canvas.offset = new_offset;

    // Update position and size of modules (and  call slots) and groups.
    if (this->update) {
        for (auto& mod : inout_graph.GetModules()) {
            mod->UpdateGUI(this->gui_graph_state.canvas);
        }
        for (auto& group : inout_graph.GetGroups()) {
            group->UpdateGUI(this->gui_graph_state.canvas);
        }
        this->update = false;
    }

    ImGui::PushClipRect(
        this->gui_graph_state.canvas.position, this->gui_graph_state.canvas.position + this->gui_graph_state.canvas.size, true);

    // GRID --------------------------------------
    if (this->show_grid) {
        this->draw_canvas_grid();
    }
    ImGui::PopStyleVar(2);

    // Render graph elements using collected button state
    this->gui_graph_state.interact.button_active_uid = GUI_INVALID_ID;
    this->gui_graph_state.interact.button_hovered_uid = GUI_INVALID_ID;
    for (size_t p = 0; p < 2; p++) {
        /// Phase 1: Interaction ---------------------------------------------------
        //  - Update button states of all graph elements
        /// Phase 2: Rendering -----------------------------------------------------
        //  - Draw all graph elements
        PresentPhase phase = static_cast<PresentPhase>(p);

        // 1] GROUPS and INTERFACE SLOTS --------------------------------------
        for (auto& group_ptr : inout_graph.GetGroups()) {
            group_ptr->PresentGUI(phase, this->gui_graph_state);

            // 3] MODULES and CALL SLOTS (of group) ---------------------------
            for (auto& module_ptr : inout_graph.GetModules()) {
                if (module_ptr->group.uid == group_ptr->uid) {
                    module_ptr->PresentGUI(phase, this->gui_graph_state);
                }
            }

            // 2] CALLS (of group) --------------------------------------------
            for (auto& module_ptr : inout_graph.GetModules()) {
                if (module_ptr->group.uid == group_ptr->uid) {
                    /// Check only for calls of caller slots for considering each call only once
                    for (auto& callslots_ptr : module_ptr->GetCallSlots(CallSlotType::CALLER)) {
                        for (auto& call_ptr : callslots_ptr->GetConnectedCalls()) {

                            bool caller_group = false;
                            auto caller_ptr = call_ptr->GetCallSlot(CallSlotType::CALLER);
                            if (caller_ptr->IsParentModuleConnected()) {
                                if (caller_ptr->GetParentModule()->group.uid == group_ptr->uid) {
                                    caller_group = true;
                                }
                            }
                            bool callee_group = false;
                            auto callee_ptr = call_ptr->GetCallSlot(CallSlotType::CALLER);
                            if (callee_ptr->IsParentModuleConnected()) {
                                if (callee_ptr->GetParentModule()->group.uid == group_ptr->uid) {
                                    callee_group = true;
                                }
                            }
                            if (caller_group || callee_group) {

                                call_ptr->Draw(phase, this->gui_graph_state);
                            }
                        }
                    }
                }
            }
        }

        // 4] MODULES and CALL SLOTS (non group members) ----------------------
        for (auto& module_ptr : inout_graph.GetModules()) {
            if (module_ptr->group.uid == GUI_INVALID_ID) {
                module_ptr->PresentGUI(phase, this->gui_graph_state);
            }
        }

        // 5] CALLS  (non group members) --------------------------------------
        /// (connected to call slots which are not part of module which is group member)
        for (auto& call_ptr : inout_graph.GetCalls()) {
            bool caller_group = false;
            auto caller_ptr = call_ptr->GetCallSlot(CallSlotType::CALLER);
            if (caller_ptr->IsParentModuleConnected()) {
                if (caller_ptr->GetParentModule()->group.uid != GUI_INVALID_ID) {
                    caller_group = true;
                }
            }
            bool callee_group = false;
            auto callee_ptr = call_ptr->GetCallSlot(CallSlotType::CALLER);
            if (callee_ptr->IsParentModuleConnected()) {
                if (callee_ptr->GetParentModule()->group.uid != GUI_INVALID_ID) {
                    callee_group = true;
                }
            }
            if ((!caller_group) && (!callee_group)) {
                call_ptr->Draw(phase, this->gui_graph_state);
            }
        }
    }

    // Multiselection ----------------------------
    this->draw_canvas_multiselection(inout_graph);

    // Dragged CALL ------------------------------
    this->draw_canvas_dragged_call(inout_graph);

    ImGui::PopClipRect();

    // Zooming and Scaling ----------------------
    // Must be checked inside this canvas child window!
    // Check at the end of drawing for being applied in next frame when font scaling matches zooming.
    if ((ImGui::IsWindowHovered() && !ImGui::IsAnyItemActive()) || this->reset_zooming || this->increment_zooming ||
        this->decrement_zooming) {

        // Scrolling (2 = Middle Mouse Button)
        if (ImGui::IsMouseDragging(2)) { // io.KeyCtrl && ImGui::IsMouseDragging(0)) {
            this->gui_graph_state.canvas.scrolling =
                this->gui_graph_state.canvas.scrolling + ImGui::GetIO().MouseDelta / this->gui_graph_state.canvas.zooming;
            this->update = true;
        }

        // Zooming (Mouse Wheel) + Reset
        if ((io.MouseWheel != 0) || this->reset_zooming || this->increment_zooming || this->decrement_zooming) {
            float last_zooming = this->gui_graph_state.canvas.zooming;
            // Center mouse position as init value
            ImVec2 current_mouse_pos = this->gui_graph_state.canvas.offset -
                                       (this->gui_graph_state.canvas.position + this->gui_graph_state.canvas.size * 0.5f);
            const float zoom_fac = 1.1f; // = 10%
            if (this->reset_zooming) {
                this->gui_graph_state.canvas.zooming = 1.0f;
                this->reset_zooming = false;
            } else {
                if (io.MouseWheel != 0) {
                    const float factor = this->gui_graph_state.canvas.zooming / 10.0f;
                    this->gui_graph_state.canvas.zooming += (io.MouseWheel * factor);
                    current_mouse_pos = this->gui_graph_state.canvas.offset - ImGui::GetMousePos();
                } else if (this->increment_zooming) {
                    this->gui_graph_state.canvas.zooming *= zoom_fac;
                    this->increment_zooming = false;
                } else if (this->decrement_zooming) {
                    this->gui_graph_state.canvas.zooming /= zoom_fac;
                    this->decrement_zooming = false;
                }
            }
            // Limit zooming
            this->gui_graph_state.canvas.zooming =
                (this->gui_graph_state.canvas.zooming <= 0.0f) ? 0.000001f : (this->gui_graph_state.canvas.zooming);
            // Compensate zooming shift of origin
            ImVec2 scrolling_diff = (this->gui_graph_state.canvas.scrolling * last_zooming) -
                                    (this->gui_graph_state.canvas.scrolling * this->gui_graph_state.canvas.zooming);
            this->gui_graph_state.canvas.scrolling += (scrolling_diff / this->gui_graph_state.canvas.zooming);
            // Move origin away from mouse position
            ImVec2 new_mouse_position = (current_mouse_pos / last_zooming) * this->gui_graph_state.canvas.zooming;
            this->gui_graph_state.canvas.scrolling +=
                ((new_mouse_position - current_mouse_pos) / this->gui_graph_state.canvas.zooming);

            this->update = true;
        }
    }

    ImGui::EndChild();

    // FONT scaling
    float font_scaling = this->gui_graph_state.canvas.zooming / this->current_font_scaling;
    // Update when scaling of font has changed due to project tab switching
    if (ImGui::GetFont()->Scale != font_scaling) {
        this->update = true;
    }
    // Font scaling is applied next frame after ImGui::Begin()
    // Font for graph should not be the currently used font of the gui.
    ImGui::GetFont()->Scale = font_scaling;
}


void megamol::gui::Graph::draw_parameters(megamol::gui::Graph& inout_graph, float graph_width) {

    ImGui::BeginGroup();

    float search_child_height = ImGui::GetFrameHeightWithSpacing() * 3.5f;
    auto child_flags =
        ImGuiWindowFlags_AlwaysUseWindowPadding | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NavFlattened;
    ImGui::BeginChild("parameter_search_child", ImVec2(graph_width, search_child_height), false, child_flags);

    ImGui::TextUnformatted("Parameters");
    ImGui::Separator();

    // Mode
    if (megamol::gui::Parameter::ParameterExtendedModeButton("parameter_search_child", this->param_extended_mode)) {
        for (auto& module_ptr : inout_graph.GetModules()) {
            for (auto& parameter : module_ptr->parameters) {
                parameter.extended = this->param_extended_mode;
            }
        }
    }

    if (this->param_extended_mode) {
        ImGui::SameLine();

        // Visibility
        if (ImGui::Checkbox("Visibility", &this->params_visible)) {
            for (auto& module_ptr : inout_graph.GetModules()) {
                for (auto& parameter : module_ptr->parameters) {
                    parameter.SetGUIVisible(this->params_visible);
                }
            }
        }
        ImGui::SameLine();

        // Read-only option
        if (ImGui::Checkbox("Read-Only", &this->params_readonly)) {
            for (auto& module_ptr : inout_graph.GetModules()) {
                for (auto& parameter : module_ptr->parameters) {
                    parameter.SetGUIReadOnly(this->params_readonly);
                }
            }
        }
    }

    // Parameter Search
    if (this->gui_graph_state.hotkeys[megamol::gui::HotkeyIndex::PARAMETER_SEARCH].is_pressed) {
        this->search_widget.SetSearchFocus(true);
    }
    std::string help_text = "[" +
                            this->gui_graph_state.hotkeys[megamol::gui::HotkeyIndex::PARAMETER_SEARCH].keycode.ToString() +
                            "] Set keyboard focus to search input field.\n"
                            "Case insensitive substring search in module and parameter names.";
    this->search_widget.Widget("graph_parameter_search", help_text);


    ImGui::EndChild();

    // ------------------------------------------------------------------------

    child_flags = ImGuiWindowFlags_AlwaysVerticalScrollbar | ImGuiWindowFlags_NavFlattened |
                  ImGuiWindowFlags_AlwaysUseWindowPadding;
    ImGui::BeginChild("parameter_param_frame_child", ImVec2(graph_width, 0.0f), false, child_flags);

    if (!this->gui_graph_state.interact.modules_selected_uids.empty()) {
        // Get module groups
        std::map<std::string, std::vector<ModulePtr_t>> group_map;
        for (auto& module_uid : this->gui_graph_state.interact.modules_selected_uids) {
            ModulePtr_t module_ptr;
            // Get pointer to currently selected module(s)
            if (inout_graph.GetModule(module_uid, module_ptr)) {
                auto group_name = module_ptr->group.name;
                if (!group_name.empty()) {
                    group_map["::" + group_name].emplace_back(module_ptr);
                } else {
                    group_map[""].emplace_back(module_ptr);
                }
            }
        }
        for (auto& group : group_map) {
            std::string search_string = this->search_widget.GetSearchString();
            bool indent = false;
            bool group_header_open = group.first.empty();
            if (!group_header_open) {
                group_header_open =
                    GUIUtils::GroupHeader(megamol::gui::HeaderType::MODULE_GROUP, group.first, search_string);
                indent = true;
                ImGui::Indent();
            }
            if (group_header_open) {
                for (auto& module_ptr : group.second) {
                    ImGui::PushID(module_ptr->uid);
                    std::string module_label = module_ptr->FullName();

                    // Draw module header
                    bool module_header_open =
                        GUIUtils::GroupHeader(megamol::gui::HeaderType::MODULE, module_label, search_string);
                    // Module description as hover tooltip
                    this->tooltip.ToolTip(module_ptr->description, ImGui::GetID(module_label.c_str()), 0.5f, 5.0f);

                    // Draw parameters
                    if (module_header_open) {
                        module_ptr->param_groups.PresentGUI(module_ptr->parameters, module_ptr->FullName(),
                            search_string, vislib::math::Ternary(this->param_extended_mode), true,
                            Parameter::WidgetScope::LOCAL, nullptr, nullptr, GUI_INVALID_ID, nullptr);
                    }

                    ImGui::PopID();
                }
            }
            if (indent) {
                ImGui::Unindent();
            }
        }
    }
    ImGui::EndChild();

    ImGui::EndGroup();
}


void megamol::gui::Graph::draw_canvas_grid(void) {

    ImGuiStyle& style = ImGui::GetStyle();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    // Color
    const ImU32 COLOR_GRID = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Border]);

    const float GRID_SIZE = (64.0f * megamol::gui::gui_scaling.Get()) * this->gui_graph_state.canvas.zooming;
    ImVec2 relative_offset = this->gui_graph_state.canvas.offset - this->gui_graph_state.canvas.position;

    for (float x = fmodf(relative_offset.x, GRID_SIZE); x < this->gui_graph_state.canvas.size.x; x += GRID_SIZE) {
        draw_list->AddLine(ImVec2(x, 0.0f) + this->gui_graph_state.canvas.position,
            ImVec2(x, this->gui_graph_state.canvas.size.y) + this->gui_graph_state.canvas.position, COLOR_GRID);
    }

    for (float y = fmodf(relative_offset.y, GRID_SIZE); y < this->gui_graph_state.canvas.size.y; y += GRID_SIZE) {
        draw_list->AddLine(ImVec2(0.0f, y) + this->gui_graph_state.canvas.position,
            ImVec2(this->gui_graph_state.canvas.size.x, y) + this->gui_graph_state.canvas.position, COLOR_GRID);
    }
}


void megamol::gui::Graph::draw_canvas_dragged_call(megamol::gui::Graph& inout_graph) {

    if (const ImGuiPayload* payload = ImGui::GetDragDropPayload()) {
        if (payload->IsDataType(GUI_DND_CALLSLOT_UID_TYPE)) {
            ImGuiID* selected_slot_uid_ptr = (ImGuiID*) payload->Data;
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
            if ((current_pos.x >= this->gui_graph_state.canvas.position.x) &&
                (current_pos.x <= (this->gui_graph_state.canvas.position.x + this->gui_graph_state.canvas.size.x)) &&
                (current_pos.y >= this->gui_graph_state.canvas.position.y) &&
                (current_pos.y <= (this->gui_graph_state.canvas.position.y + this->gui_graph_state.canvas.size.y))) {
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
                    p1 = selected_callslot_ptr->GetPosition();
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
                        p1 = selected_interfaceslot_ptr->GetPosition((*selected_interfaceslot_ptr));
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
                            GUI_LINE_THICKNESS * this->gui_graph_state.canvas.zooming);
                    }
                }
            }
        }
    }
}


void megamol::gui::Graph::draw_canvas_multiselection(Graph& inout_graph) {

    bool no_graph_item_selected = ((this->gui_graph_state.interact.callslot_selected_uid == GUI_INVALID_ID) &&
                                   (this->gui_graph_state.interact.call_selected_uid == GUI_INVALID_ID) &&
                                   (this->gui_graph_state.interact.modules_selected_uids.empty()) &&
                                   (this->gui_graph_state.interact.interfaceslot_selected_uid == GUI_INVALID_ID) &&
                                   (this->gui_graph_state.interact.group_selected_uid == GUI_INVALID_ID));

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

        float border = (1.0f * megamol::gui::gui_scaling.Get());
        draw_list->AddRect(multiselect_start_pos, multiselect_end_pos, COLOR_MULTISELECT_BORDER, GUI_RECT_CORNER_RADIUS,
            ImDrawCornerFlags_All, border);
    } else if (this->multiselect_done && ImGui::IsWindowHovered() && ImGui::IsMouseReleased(0)) {
        ImVec2 outer_rect_min = ImVec2(std::min(this->multiselect_start_pos.x, this->multiselect_end_pos.x),
            std::min(this->multiselect_start_pos.y, this->multiselect_end_pos.y));
        ImVec2 outer_rect_max = ImVec2(std::max(this->multiselect_start_pos.x, this->multiselect_end_pos.x),
            std::max(this->multiselect_start_pos.y, this->multiselect_end_pos.y));
        ImVec2 inner_rect_min, inner_rect_max;
        ImVec2 module_size;
        this->gui_graph_state.interact.modules_selected_uids.clear();
        for (auto& module_ptr : inout_graph.GetModules()) {
            bool group_member = (module_ptr->group.uid != GUI_INVALID_ID);
            if (!group_member || (group_member && module_ptr->group.visible)) {
                module_size = module_ptr->GetSize() * this->gui_graph_state.canvas.zooming;
                inner_rect_min =
                    this->gui_graph_state.canvas.offset + module_ptr->position * this->gui_graph_state.canvas.zooming;
                inner_rect_max = inner_rect_min + module_size;
                if (((outer_rect_min.x < inner_rect_max.x) && (outer_rect_max.x > inner_rect_min.x) &&
                        (outer_rect_min.y < inner_rect_max.y) && (outer_rect_max.y > inner_rect_min.y))) {
                    this->gui_graph_state.interact.modules_selected_uids.emplace_back(module_ptr->uid);
                }
            }
        }
        this->multiselect_done = false;
    } else {
        this->multiselect_start_pos = ImGui::GetMousePos();
    }
}


void megamol::gui::Graph::layout_graph(megamol::gui::Graph& inout_graph) {

    ImVec2 init_position = megamol::gui::ModulePresentation::GetDefaultModulePosition(this->gui_graph_state.canvas);

    /// 1] Layout all grouped modules
    for (auto& group_ptr : inout_graph.GetGroups()) {
        this->layout(group_ptr->GetModules(), GroupPtrVector_t(), init_position);
        group_ptr->UpdateGUI(this->gui_graph_state.canvas);
    }

    /// 2] Layout ungrouped modules and groups
    ModulePtrVector_t ungrouped_modules;
    for (auto& module_ptr : inout_graph.GetModules()) {
        if (module_ptr->group.uid == GUI_INVALID_ID) {
            ungrouped_modules.emplace_back(module_ptr);
        }
    }
    this->layout(ungrouped_modules, inout_graph.GetGroups(), init_position);

    this->update = true;
}


void megamol::gui::Graph::layout(
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
                                // Add module only if not already  Prevents cyclic dependency
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
                            } else if (add_module_ptr->group.uid != GUI_INVALID_ID) {
                                ImGuiID group_uid = add_module_ptr->group.uid; // != GUI_INVALID_ID
                                if (this->contains_group(groups, group_uid)) {
                                    GroupPtr_t add_group_ptr;
                                    for (auto& group_ptr : groups) {
                                        if (group_ptr->uid == group_uid) {
                                            add_group_ptr = group_ptr;
                                        }
                                    }
                                    if (add_group_ptr != nullptr) {
                                        // Add group only if not already  Prevents cyclic dependency
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
    float max_call_width = (25.0f * megamol::gui::gui_scaling.Get());
    float max_graph_element_width = 0.0f;

    size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++) {

        if (this->show_call_label || this->show_call_slots_label) {
            max_call_width = 0.0f;
        }
        max_graph_element_width = 0.0f;
        pos.y = init_position.y;
        bool found_layer_item = false;

        size_t layer_items_count = layers[i].size();
        for (size_t j = 0; j < layer_items_count; j++) {
            auto layer_item = layers[i][j];

            if (layer_item.module_ptr != nullptr) {
                if (this->show_call_label) {
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
                if (this->show_call_slots_label) {
                    for (auto& callerslot_ptr : layer_item.module_ptr->GetCallSlots(CallSlotType::CALLER)) {
                        if (callerslot_ptr->CallsConnected() &&
                            this->connected_callslot(modules, groups, callerslot_ptr)) {
                            for (auto& call_ptr : callerslot_ptr->GetConnectedCalls()) {
                                auto call_name_length = ImGui::CalcTextSize(call_ptr->GetSlotsLabel().c_str()).x;
                                max_call_width = std::max(call_name_length, max_call_width);
                            }
                        }
                    }
                }
                layer_item.module_ptr->position = pos;
                auto module_size = layer_item.module_ptr->GetSize();
                pos.y += (module_size.y + GUI_GRAPH_BORDER);
                max_graph_element_width = std::max(module_size.x, max_graph_element_width);
                found_layer_item = true;

            } else if (layer_item.group_ptr != nullptr) {
                if (this->show_call_label) {
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
                if (this->show_call_slots_label) {
                    for (auto& interfaceslot_slot : layer_item.group_ptr->GetInterfaceSlots(CallSlotType::CALLER)) {
                        for (auto& callerslot_ptr : interfaceslot_slot->GetCallSlots()) {
                            if (callerslot_ptr->CallsConnected() &&
                                this->connected_callslot(modules, groups, callerslot_ptr)) {
                                for (auto& call_ptr : callerslot_ptr->GetConnectedCalls()) {
                                    auto call_name_length = ImGui::CalcTextSize(call_ptr->GetSlotsLabel().c_str()).x;
                                    max_call_width = std::max(call_name_length, max_call_width);
                                }
                            }
                        }
                    }
                }
                layer_item.group_ptr->SetGUIPosition(this->gui_graph_state.canvas, pos);
                auto group_size = layer_item.group_ptr->GetSize();
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


bool megamol::gui::Graph::connected_callslot(
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
            if (connected_callslot_ptr->gui_interfaceslot_ptr != nullptr) {
                if (this->contains_interfaceslot(groups, connected_callslot_ptr->gui_interfaceslot_ptr->uid)) {
                    retval = true;
                    break;
                }
            }
        }
    }
    return retval;
}


bool megamol::gui::Graph::connected_interfaceslot(
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
                if (connected_callslot_ptr->gui_interfaceslot_ptr != nullptr) {
                    if (this->contains_interfaceslot(groups, connected_callslot_ptr->gui_interfaceslot_ptr->uid)) {
                        retval = true;
                        break;
                    }
                }
            }
        }
    }
    return retval;
}


bool megamol::gui::Graph::contains_callslot(const ModulePtrVector_t& modules, ImGuiID callslot_uid) {

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


bool megamol::gui::Graph::contains_interfaceslot(const GroupPtrVector_t& groups, ImGuiID interfaceslot_uid) {

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


bool megamol::gui::Graph::contains_module(const ModulePtrVector_t& modules, ImGuiID module_uid) {

    for (auto& module_ptr : modules) {
        if (module_ptr->uid == module_uid) {
            return true;
        }
    }
    return false;
}


bool megamol::gui::Graph::contains_group(const GroupPtrVector_t& groups, ImGuiID group_uid) {

    for (auto& group_ptr : groups) {
        if (group_ptr->uid == group_uid) {
            return true;
        }
    }
    return false;
}


const std::string megamol::gui::Graph::generate_unique_group_name(void) {

    int new_name_id = 0;
    std::string new_name_prefix("Group_");
    for (auto& group : this->groups) {
        if (group->name.find(new_name_prefix) == 0) {
            std::string int_postfix = group->name.substr(new_name_prefix.length());
            try {
                int last_id = std::stoi(int_postfix);
                new_name_id = std::max(new_name_id, last_id);
            } catch (...) {}
        }
    }
    return std::string(new_name_prefix + std::to_string(new_name_id + 1));
}


const std::string megamol::gui::Graph::generate_unique_module_name(const std::string& module_name) {

    int new_name_id = 0;
    std::string new_name_prefix = module_name + "_";
    for (auto& mod : this->modules) {
        if (mod->name.find(new_name_prefix) == 0) {
            std::string int_postfix = mod->name.substr(new_name_prefix.length());
            try {
                int last_id = std::stoi(int_postfix);
                new_name_id = std::max(new_name_id, last_id);
            } catch (...) {}
        }
    }
    return std::string(new_name_prefix + std::to_string(new_name_id + 1));
}


const std::string megamol::gui::Graph::GenerateUniqueGraphEntryName(void) {

    int new_name_id = 0;
    std::string new_name_prefix("GraphEntry_");
    for (auto& module_ptr : this->modules) {
        if (module_ptr->graph_entry_name.find(new_name_prefix) == 0) {
            std::string int_postfix = module_ptr->graph_entry_name.substr(new_name_prefix.length());
            try {
                int last_id = std::stoi(int_postfix);
                new_name_id = std::max(new_name_id, last_id);
            } catch (...) {}
        }
    }
    return std::string(new_name_prefix + std::to_string(new_name_id + 1));
}
