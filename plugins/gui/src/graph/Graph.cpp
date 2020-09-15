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


megamol::gui::Graph::Graph(const std::string& graph_name)
    : uid(megamol::gui::GenerateUniqueID())
    , name(graph_name)
    , present()
    , modules()
    , calls()
    , groups()
    , dirty_flag(true)
    , sync_queue(nullptr)
    , running_state(vislib::math::Ternary::TRI_UNKNOWN) {

    this->sync_queue = std::make_shared<SyncQueue_t>();
    ASSERT(this->sync_queue != nullptr);
}


megamol::gui::Graph::~Graph(void) {

    this->present.ResetStatePointers();

    // 1) ! Delete all groups
    std::vector<ImGuiID> group_uids;
    for (auto& group_ptr : this->groups) {
        group_uids.emplace_back(group_ptr->uid);
    }
    for (auto& group_uid : group_uids) {
        this->DeleteGroup(group_uid);
    }

    // 2) Delete all modules
    std::vector<ImGuiID> module_uids;
    for (auto& module_ptr : this->modules) {
        module_uids.emplace_back(module_ptr->uid);
    }
    for (auto& module_uid : module_uids) {
        this->DeleteModule(module_uid, true);
    }

    // 3) Delete all calls
    std::vector<ImGuiID> call_uids;
    for (auto& call_ptr : this->calls) {
        call_uids.emplace_back(call_ptr->uid);
    }
    for (auto& call_uid : call_uids) {
        this->DeleteCall(call_uid);
    }
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
    } catch (std::exception e) {
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
                auto mod_ptr = std::make_shared<Module>(mod_uid);
                mod_ptr->class_name = mod.class_name;
                mod_ptr->description = mod.description;
                mod_ptr->plugin_name = mod.plugin_name;
                mod_ptr->is_view = mod.is_view;
                mod_ptr->name = this->generate_unique_module_name(mod.class_name);
                mod_ptr->main_view_name.clear();
                mod_ptr->present.label_visible = this->present.GetModuleLabelVisibility();

                for (auto& p : mod.parameters) {
                    Parameter param_slot(megamol::gui::GenerateUniqueID(), p.type, p.storage, p.minval, p.maxval);
                    param_slot.full_name = p.full_name;
                    param_slot.description = p.description;
                    param_slot.SetValueString(p.default_value, true, true);
                    param_slot.present.SetGUIVisible(p.gui_visibility);
                    param_slot.present.SetGUIReadOnly(p.gui_read_only);
                    param_slot.present.SetGUIPresentation(p.gui_presentation);
                    // Apply current global configurator parameter gui settings
                    // Do not apply global read-only and visibility.
                    param_slot.present.extended = this->present.param_extended_mode;

                    mod_ptr->parameters.emplace_back(param_slot);
                }

                for (auto& callslots_type : mod.callslots) {
                    for (auto& c : callslots_type.second) {
                        auto callslot_ptr = std::make_shared<CallSlot>(megamol::gui::GenerateUniqueID());
                        callslot_ptr->name = c.name;
                        callslot_ptr->description = c.description;
                        callslot_ptr->compatible_call_idxs = c.compatible_call_idxs;
                        callslot_ptr->type = c.type;
                        callslot_ptr->present.label_visible = this->present.GetCallSlotLabelVisibility();
                        callslot_ptr->ConnectParentModule(mod_ptr);

                        mod_ptr->AddCallSlot(callslot_ptr);
                    }
                }

                // Add data to queue for synchronization with core graph
                QueueData queue_data;
                queue_data.classname = mod_ptr->class_name;
                queue_data.id = mod_ptr->FullName();
                queue_data.graph_entry = mod_ptr->IsMainView();
                this->sync_queue->push(SyncQueueData_t(QueueChange::ADD_MODULE, queue_data));

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
    } catch (std::exception e) {
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

                if (!force && (*iter)->IsMainView() &&
                    this->NOT_SUPPORTED_RUNNING_GRAPH_ACTION("Delete entry point/ view instance")) {
                    return false;
                }

                this->present.ResetStatePointers();

                // 1) Reset module and call slot pointers in groups
                GroupPtr_t module_group_ptr = nullptr;
                ImGuiID delete_empty_group = GUI_INVALID_ID;
                for (auto& group_ptr : this->groups) {
                    if (group_ptr->ContainsModule(module_uid)) {
                        group_ptr->RemoveModule(module_uid);
                        module_group_ptr = group_ptr;
                    }
                    if (group_ptr->Empty()) {
                        delete_empty_group = group_ptr->uid;
                    }
                }

                // 2)  Delete calls
                for (auto& callslot_map : (*iter)->GetCallSlots()) {
                    for (auto& callslot_ptr : callslot_map.second) {
                        for (auto& call_ptr : callslot_ptr->GetConnectedCalls()) {
                            if (call_ptr != nullptr) {
                                auto call_uid = call_ptr->uid;
                                this->DeleteCall(call_uid);
                            }
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
                    "[GUI] Deleted module '%s' (uid %i) from  project '%s'.\n", (*iter)->class_name.c_str(),
                    (*iter)->uid, this->name.c_str());
#endif // GUI_VERBOSE

                // Add data to queue for synchronization with core graph
                QueueData queue_data;
                queue_data.classname = (*iter)->class_name;
                queue_data.id = (*iter)->FullName();
                this->sync_queue->push(SyncQueueData_t(QueueChange::DELETE_MODULE, queue_data));

                // 5) Delete module
                if ((*iter).use_count() > 1) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Unclean deletion. Found %i references pointing to module. [%s, %s, line %d]\n",
                        (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                }
                (*iter).reset();
                this->modules.erase(iter);

                // 6) Delete empty groups
                module_group_ptr.reset();
                if (delete_empty_group != GUI_INVALID_ID) {
                    this->DeleteGroup(delete_empty_group);
                }

                this->ForceSetDirty();
                return true;
            }
        }

    } catch (std::exception e) {
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

            ImGuiID interfaceslot_group_uid = interface_ptr->present.group.uid;
            ImGuiID callslot_group_uid = GUI_INVALID_ID;
            if (callslot_ptr->IsParentModuleConnected()) {
                callslot_group_uid = callslot_ptr->GetParentModule()->present.group.uid;
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
    } catch (std::exception e) {
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
        call_ptr->present.label_visible = this->present.GetCallLabelVisibility();

        return this->AddCall(call_ptr, callslot_1, callslot_2);

    } catch (std::exception e) {
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

        // Add data to queue for synchronization with core graph
        QueueData queue_data;
        queue_data.classname = call_ptr->class_name;
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
        this->sync_queue->push(SyncQueueData_t(QueueChange::ADD_CALL, queue_data));

        this->calls.emplace_back(call_ptr);
        this->ForceSetDirty();

#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Added call '%s' (uid %i) to project '%s'.\n",
            call_ptr->class_name.c_str(), call_ptr->uid, this->name.c_str());
#endif // GUI_VERBOSE

        // Add connected call slots to interface of group of the parent module
        if (callslot_1->IsParentModuleConnected() && callslot_2->IsParentModuleConnected()) {
            ImGuiID slot_1_parent_group_uid = callslot_1->GetParentModule()->present.group.uid;
            ImGuiID slot_2_parent_group_uid = callslot_2->GetParentModule()->present.group.uid;
            if (slot_1_parent_group_uid != slot_2_parent_group_uid) {
                if ((slot_1_parent_group_uid != GUI_INVALID_ID) &&
                    (callslot_1->present.group.interfaceslot_ptr == nullptr)) {
                    for (auto& group_ptr : this->groups) {
                        if (group_ptr->uid == slot_1_parent_group_uid) {
                            group_ptr->AddInterfaceSlot(callslot_1);
                        }
                    }
                }
                if ((slot_2_parent_group_uid != GUI_INVALID_ID) &&
                    (callslot_2->present.group.interfaceslot_ptr == nullptr)) {
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
                    if (caller->present.group.interfaceslot_ptr != nullptr) {
                        caller_uid = caller->present.group.interfaceslot_ptr->uid;
                    }
                }
                if (callee != nullptr) {
                    callee_uid = callee->uid;
                    if (callee->present.group.interfaceslot_ptr != nullptr) {
                        callee_uid = callee->present.group.interfaceslot_ptr->uid;
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
                    if (caller->present.group.interfaceslot_ptr != nullptr) {
                        caller_fits = (caller_uid == caller->present.group.interfaceslot_ptr->uid);
                    } else {
                        caller_fits = (caller_uid == caller->uid);
                    }
                }
                if (callee != nullptr) {
                    if (callee->present.group.interfaceslot_ptr != nullptr) {
                        callee_fits = (callee_uid == callee->present.group.interfaceslot_ptr->uid);
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

                    // Add data to queue for synchronization with core graph
                    QueueData queue_data;
                    queue_data.classname = (*iter)->class_name;
                    bool valid_ptr = false;
                    auto caller_ptr = (*iter)->GetCallSlot(megamol::gui::CallSlotType::CALLER);
                    if (caller_ptr != nullptr) {
                        if (caller_ptr->GetParentModule() != nullptr) {
                            queue_data.caller = caller_ptr->GetParentModule()->FullName() + "::" + caller_ptr->name;
                            valid_ptr = true;
                        }
                    }
                    if (!valid_ptr) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] Pointer to caller slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                            __LINE__);
                    }
                    valid_ptr = false;
                    auto callee_ptr = (*iter)->GetCallSlot(megamol::gui::CallSlotType::CALLEE);
                    if (callee_ptr != nullptr) {
                        if (callee_ptr->GetParentModule() != nullptr) {
                            queue_data.callee = callee_ptr->GetParentModule()->FullName() + "::" + callee_ptr->name;
                            valid_ptr = true;
                        }
                    }
                    if (!valid_ptr) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] Pointer to callee slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                            __LINE__);
                    }
                    this->sync_queue->push(SyncQueueData_t(QueueChange::DELETE_CALL, queue_data));

                    this->present.ResetStatePointers();

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
    } catch (std::exception e) {
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

    } catch (std::exception e) {
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

    try {
        for (auto iter = this->groups.begin(); iter != this->groups.end(); iter++) {
            if ((*iter)->uid == group_uid) {

                this->present.ResetStatePointers();

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

                this->present.ForceUpdate();
                return true;
            }
        }
    } catch (std::exception e) {
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
                    if (group_ptr->AddModule(module_ptr)) {
                        this->ForceSetDirty();
                        return existing_group_uid;
                    }
                }
            }
        }
    } catch (std::exception e) {
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


bool megamol::gui::Graph::UniqueModuleRename(const std::string& module_name) {

    for (auto& mod : this->modules) {
        if (module_name == mod->name) {
            mod->name = this->generate_unique_module_name(module_name);
            this->add_rename_module_sync_event(module_name, mod->name);
            this->present.ForceUpdate();
            return true;
        }
    }
    return false;
}


const std::string megamol::gui::Graph::GenerateUniqueMainViewName(void) {

    int new_name_id = 0;
    std::string new_name_prefix("Instance_");
    for (auto& module_ptr : this->modules) {
        if (module_ptr->main_view_name.find(new_name_prefix) == 0) {
            std::string int_postfix = module_ptr->main_view_name.substr(new_name_prefix.length());
            try {
                int last_id = std::stoi(int_postfix);
                new_name_id = std::max(new_name_id, last_id);
            } catch (...) {
            }
        }
    }
    return std::string(new_name_prefix + std::to_string(new_name_id + 1));
}


bool megamol::gui::Graph::StateFromJsonString(const std::string& in_json_string) {

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
                    GUIUtils::Utf8Decode(json_graph_id);
                    if (json_graph_id == GUI_JSON_TAG_PROJECT_GRAPH) {
                        auto config_state = content_item.value();
                        found = true;

                        // project_file (supports UTF-8)
                        if (config_state.at("project_file").is_string()) {
                            std::string filename = config_state.at("project_file").get<std::string>();
                            GUIUtils::Utf8Decode(filename);
                            this->SetFilename(filename);
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read 'project_file' as string. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }
                        // project_name (supports UTF-8)
                        if (config_state.at("project_name").is_string()) {
                            std::string projectname = config_state.at("project_name").get<std::string>();
                            GUIUtils::Utf8Decode(projectname);
                            this->name = projectname;
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read 'project_name' as string. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }
                        // show_parameter_sidebar
                        bool tmp_show_parameter_sidebar;
                        this->present.change_show_parameter_sidebar = false;
                        if (config_state.at("show_parameter_sidebar").is_boolean()) {
                            config_state.at("show_parameter_sidebar").get_to(tmp_show_parameter_sidebar);
                            this->present.change_show_parameter_sidebar = true;
                            this->present.show_parameter_sidebar = tmp_show_parameter_sidebar;
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read 'show_parameter_sidebar' as boolean. [%s, %s, line "
                                "%d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }
                        // parameter_sidebar_width
                        if (config_state.at("parameter_sidebar_width").is_number_float()) {
                            config_state.at("parameter_sidebar_width").get_to(this->present.parameter_sidebar_width);
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read first value of "
                                "'parameter_sidebar_width' as float. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }
                        // show_grid
                        if (config_state.at("show_grid").is_boolean()) {
                            config_state.at("show_grid").get_to(this->present.show_grid);
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read 'show_grid' as boolean. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }

                        // show_call_names
                        if (config_state.at("show_call_names").is_boolean()) {
                            config_state.at("show_call_names").get_to(this->present.show_call_names);
                            for (auto& call : this->GetCalls()) {
                                call->present.label_visible = this->present.show_call_names;
                            }
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read 'show_call_names' as boolean. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }
                        // show_slot_names
                        if (config_state.at("show_slot_names").is_boolean()) {
                            config_state.at("show_slot_names").get_to(this->present.show_slot_names);
                            for (auto& mod : this->GetModules()) {
                                for (auto& callslot_types : mod->GetCallSlots()) {
                                    for (auto& callslots : callslot_types.second) {
                                        callslots->present.label_visible = this->present.show_slot_names;
                                    }
                                }
                            }
                            for (auto& group_ptr : this->GetGroups()) {
                                for (auto& interfaceslots_map : group_ptr->GetInterfaceSlots()) {
                                    for (auto& interfaceslot_ptr : interfaceslots_map.second) {
                                        interfaceslot_ptr->present.label_visible = this->present.show_slot_names;
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
                            config_state.at("show_module_names").get_to(this->present.show_module_names);
                            for (auto& mod : this->GetModules()) {
                                mod->present.label_visible = this->present.show_module_names;
                            }
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read 'show_module_names' as boolean. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }
                        // params_visible
                        if (config_state.at("params_visible").is_boolean()) {
                            config_state.at("params_visible").get_to(this->present.params_visible);
                            /// Do not apply. Already refelcted in parameter gui state.
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read 'params_visible' as boolean. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }
                        // params_readonly
                        if (config_state.at("params_readonly").is_boolean()) {
                            config_state.at("params_readonly").get_to(this->present.params_readonly);
                            /// Do not apply. Already refelcted in parameter gui state.
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read 'params_readonly' as boolean. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }
                        // param_extended_mode
                        if (config_state.at("param_extended_mode").is_boolean()) {
                            config_state.at("param_extended_mode").get_to(this->present.param_extended_mode);
                            for (auto& module_ptr : this->GetModules()) {
                                for (auto& parameter : module_ptr->parameters) {
                                    parameter.present.extended = this->present.param_extended_mode;
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
                                config_state.at("canvas_scrolling")[0].get_to(
                                    this->present.graph_state.canvas.scrolling.x);
                            } else {
                                megamol::core::utility::log::Log::DefaultLog.WriteError(
                                    "[GUI] JSON state: Failed to read first value of 'canvas_scrolling' as float. [%s, "
                                    "%s, "
                                    "line %d]\n",
                                    __FILE__, __FUNCTION__, __LINE__);
                            }
                            if (config_state.at("canvas_scrolling")[1].is_number_float()) {
                                config_state.at("canvas_scrolling")[1].get_to(
                                    this->present.graph_state.canvas.scrolling.y);
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
                            config_state.at("canvas_zooming").get_to(this->present.graph_state.canvas.zooming);
                            this->present.reset_zooming = false;
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
                                        for (auto& module_ptr : this->GetModules()) {
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
                                                std::string callslot_name = callslot_item.value().get<std::string>();
                                                GUIUtils::Utf8Decode(callslot_name);
                                                calleslot_fullnames.emplace_back(callslot_name);
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
                                                        this->present.ResetStatePointers();
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
            this->present.update = true;
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[GUI] Read graph state for '%s' from JSON string.", this->name.c_str());
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


bool megamol::gui::Graph::StateToJSON(nlohmann::json& out_json, bool save_as_project_graph) {

    try {
        std::string filename = this->GetFilename();
        GUIUtils::Utf8Encode(filename);

        // For not running graphs save only file name of loaded project
        if (!save_as_project_graph) {
            out_json[GUI_JSON_TAG_GRAPHS][filename] = "";
        } else {

            out_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT_GRAPH]["project_file"] = filename;
            GUIUtils::Utf8Encode(this->name);
            out_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT_GRAPH]["project_name"] = this->name;
            GUIUtils::Utf8Decode(this->name);
            out_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT_GRAPH]["show_parameter_sidebar"] =
                this->present.show_parameter_sidebar;
            out_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT_GRAPH]["parameter_sidebar_width"] =
                this->present.parameter_sidebar_width;
            out_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT_GRAPH]["show_grid"] = this->present.show_grid;
            out_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT_GRAPH]["show_call_names"] =
                this->present.show_call_names;
            out_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT_GRAPH]["show_slot_names"] =
                this->present.show_slot_names;
            out_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT_GRAPH]["show_module_names"] =
                this->present.show_module_names;
            out_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT_GRAPH]["params_visible"] = this->present.params_visible;
            out_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT_GRAPH]["params_readonly"] =
                this->present.params_readonly;
            out_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT_GRAPH]["param_extended_mode"] =
                this->present.param_extended_mode;
            out_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT_GRAPH]["canvas_scrolling"] = {
                this->present.graph_state.canvas.scrolling.x, this->present.graph_state.canvas.scrolling.y};
            out_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT_GRAPH]["canvas_zooming"] =
                this->present.graph_state.canvas.zooming;

            // Module positions
            for (auto& module_ptr : this->GetModules()) {
                out_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT_GRAPH]["modules"][module_ptr->FullName()]
                        ["graph_position"] = {module_ptr->present.position.x, module_ptr->present.position.y};
            }
            // Group interface slots
            size_t interface_number = 0;
            for (auto& group_ptr : this->GetGroups()) {
                for (auto& interfaceslots_map : group_ptr->GetInterfaceSlots()) {
                    for (auto& interface_ptr : interfaceslots_map.second) {
                        std::string interface_label = "interface_slot_" + std::to_string(interface_number);
                        for (auto& callslot_ptr : interface_ptr->GetCallSlots()) {
                            std::string callslot_fullname;
                            if (callslot_ptr->IsParentModuleConnected()) {
                                callslot_fullname =
                                    callslot_ptr->GetParentModule()->FullName() + "::" + callslot_ptr->name;
                            }
                            GUIUtils::Utf8Encode(callslot_fullname);
                            out_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT_GRAPH]["interfaces"][group_ptr->name]
                                    [interface_label] += callslot_fullname;
                        }
                        interface_number++;
                    }
                }
            }
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Wrote graph state to JSON.");
#endif // GUI_VERBOSE
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


const std::string megamol::gui::Graph::generate_unique_group_name(void) {

    int new_name_id = 0;
    std::string new_name_prefix("Group_");
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


const std::string megamol::gui::Graph::generate_unique_module_name(const std::string& module_name) {

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


void megamol::gui::Graph::add_rename_module_sync_event(const std::string& current_name, const std::string& new_name) {

    auto queue = this->GetSyncQueue();
    megamol::gui::Graph::QueueData queue_data;
    queue_data.id = current_name;
    queue_data.rename_id = new_name;
    // Remove leading "::"
    if (queue_data.id.find_first_of("::") == 0) {
        queue_data.id = queue_data.id.substr(2);
    }
    if (queue_data.rename_id.find_first_of("::") == 0) {
        queue_data.rename_id = queue_data.rename_id.substr(2);
    }
    this->GetSyncQueue()->push(
        megamol::gui::Graph::SyncQueueData_t(megamol::gui::Graph::QueueChange::RENAME_MODULE, queue_data));
}


bool megamol::gui::Graph::NOT_SUPPORTED_RUNNING_GRAPH_ACTION(const std::string& log_action) {

    if (this->IsRunning()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] The action [%s] is not yet supported for the graph of the running project. Open project from file "
            "to make desired changes."
            "[%s, %s, line %d]\n",
            log_action.c_str(), __FILE__, __FUNCTION__, __LINE__);
        return true;
    }
    return false;
}
