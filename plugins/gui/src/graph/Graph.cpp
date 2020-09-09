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
    , group_name_uid(0)
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
        this->DeleteModule(module_uid);
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
                mod_ptr->is_view_instance = !this->IsMainViewSet();
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
                queue_data.graph_entry = mod_ptr->is_view_instance;
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


bool megamol::gui::Graph::DeleteModule(ImGuiID module_uid) {

    try {
        for (auto iter = this->modules.begin(); iter != this->modules.end(); iter++) {
            if ((*iter)->uid == module_uid) {

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
                            auto call_uid = call_ptr->uid;
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
            this->present.ForceUpdate();
            return true;
        }
    }
    return false;
}


bool megamol::gui::Graph::IsMainViewSet(void) {

    for (auto& mod : this->modules) {
        if (mod->is_view_instance) {
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
