/*
 * Graph.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "Graph.h"
#include "imgui_stdlib.h"
#include "vislib/math/Ternary.h"
#include "widgets/ButtonWidgets.h"
#ifdef MEGAMOL_USE_PROFILING
#define GRAPH_PROFILING_WINDOW_WIDTH (300.0f * megamol::gui::gui_scaling.Get())
#endif // MEGAMOL_USE_PROFILING

using namespace megamol;
using namespace megamol::gui;


megamol::gui::Graph::Graph(const std::string& graph_name)
        : uid(megamol::gui::GenerateUniqueID())
        , name(graph_name)
        , modules()
        , calls()
        , groups()
        , dirty_flag(true)
        , filenames()
        , sync_queue()
        , running(false)
        , gui_graph_state()
        , gui_update(true)
        , gui_show_grid(false)
        , gui_show_parameter_sidebar(false)
        , gui_change_show_parameter_sidebar(true)
        , gui_parameter_sidebar_width(300.0f)
        , gui_show_profiling_bar(false)
        , gui_change_show_profiling_bar(true)
        , gui_profiling_bar_height(300.0f)
        , gui_graph_layout(0)
        , gui_reset_zooming(true)
        , gui_increment_zooming(false)
        , gui_decrement_zooming(false)
        , gui_current_graph_entry_name()
        , gui_multiselect_start_pos()
        , gui_multiselect_end_pos()
        , gui_multiselect_done(false)
        , gui_canvas_hovered(false)
        , gui_current_font_scaling(1.0f)
        , gui_search_widget()
        , gui_parameters_splitter()
        , gui_profiling_splitter()
        , gui_rename_popup()
        , gui_tooltip()
#ifdef MEGAMOL_USE_PROFILING
        , gui_profiling_run_button()
        , profiling_list()
        , scroll_delta_time(std::chrono::system_clock::now())
        , gui_profiling_graph_link(true)
#endif // MEGAMOL_USE_PROFILING
{

    this->filenames.first.first = false;
    this->filenames.first.second = "";
    this->filenames.second.first = false;
    this->filenames.second.second = "";

    this->gui_graph_state.canvas.position = ImVec2(0.0f, 0.0f);
    this->gui_graph_state.canvas.size = ImVec2(1.0f, 1.0f);
    this->gui_graph_state.canvas.scrolling = ImVec2(0.0f, 0.0f);
    this->gui_graph_state.canvas.zooming = 1.0f;
    this->gui_graph_state.canvas.offset = ImVec2(0.0f, 0.0f);
    this->gui_graph_state.canvas.gui_font_ptr = nullptr;

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
    this->gui_graph_state.interact.module_show_label = true;

    this->gui_graph_state.interact.call_selected_uid = GUI_INVALID_ID;
    this->gui_graph_state.interact.call_hovered_uid = GUI_INVALID_ID;
    this->gui_graph_state.interact.call_show_label = true;
    this->gui_graph_state.interact.call_show_slots_label = false;
    this->gui_graph_state.interact.call_coloring_map = 0;
    this->gui_graph_state.interact.call_coloring_mode = 0;

    this->gui_graph_state.interact.slot_drag_drop_uids = UIDPair_t(GUI_INVALID_ID, GUI_INVALID_ID);

    this->gui_graph_state.interact.callslot_selected_uid = GUI_INVALID_ID;
    this->gui_graph_state.interact.callslot_hovered_uid = GUI_INVALID_ID;
    this->gui_graph_state.interact.callslot_add_group_uid = UIDPair_t(GUI_INVALID_ID, GUI_INVALID_ID);
    this->gui_graph_state.interact.callslot_remove_group_uid = UIDPair_t(GUI_INVALID_ID, GUI_INVALID_ID);
    this->gui_graph_state.interact.callslot_compat_ptr.reset();
    this->gui_graph_state.interact.callslot_show_label = false;

    this->gui_graph_state.interact.interfaceslot_selected_uid = GUI_INVALID_ID;
    this->gui_graph_state.interact.interfaceslot_hovered_uid = GUI_INVALID_ID;
    this->gui_graph_state.interact.interfaceslot_compat_ptr.reset();

    this->gui_graph_state.interact.parameters_extended_mode = false;
    this->gui_graph_state.interact.graph_is_running = false;
    this->gui_graph_state.interact.profiling_pause_update = false;
    this->gui_graph_state.interact.profiling_show = false;

    this->gui_graph_state.groups.clear();
    // this->gui_graph_state.hotkeys are already initialized
}

megamol::gui::Graph::~Graph() {

    this->Clear();
}


ModulePtr_t megamol::gui::Graph::AddModule(const std::string& class_name, const std::string& module_name,
    const std::string& group_name, const std::string& description, const std::string& plugin_name, bool is_view) {

    try {
        /// Parameters and CallSlots have to be added separately
        /// Add module to queue manually
        if (auto mod_ptr = std::make_shared<Module>(
                megamol::gui::GenerateUniqueID(), class_name, description, plugin_name, is_view)) {

            this->modules.emplace_back(mod_ptr);

            mod_ptr->SetName(module_name);
            this->AddGroupModule(group_name, mod_ptr, false);
            this->ForceSetDirty();

#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Added empty module to project.\n");
#endif // GUI_VERBOSE
            return mod_ptr;
        }

    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return nullptr;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return nullptr;
    }

    megamol::core::utility::log::Log::DefaultLog.WriteError(
        "[GUI] Unable to add module. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return nullptr;
}


ModulePtr_t megamol::gui::Graph::AddModule(const ModuleStockVector_t& stock_modules, const std::string& class_name,
    const std::string& module_name, const std::string& group_name) {

    try {
        for (auto& mod : stock_modules) {
            if (class_name == mod.class_name) {
                ImGuiID mod_uid = megamol::gui::GenerateUniqueID();
                auto mod_ptr =
                    std::make_shared<Module>(mod_uid, mod.class_name, mod.description, mod.plugin_name, mod.is_view);
                if (module_name.empty()) {
                    mod_ptr->SetName(this->generate_unique_module_name(mod.class_name));
                } else {
                    mod_ptr->SetName(module_name);
                }
                mod_ptr->SetGraphEntryName("");
                this->AddGroupModule(group_name, mod_ptr, false);

                for (auto& p : mod.parameters) {
                    Parameter parameter(megamol::gui::GenerateUniqueID(), p.type, p.storage, p.minval, p.maxval,
                        p.stepsize, p.param_name, p.description);
                    parameter.SetParentModuleName(mod_ptr->FullName());
                    parameter.SetValueString(p.default_value, true, true);
                    parameter.SetGUIVisible(p.gui_visibility);
                    parameter.SetGUIReadOnly(p.gui_read_only);
                    parameter.SetGUIPresentation(p.gui_presentation);
                    mod_ptr->Parameters().emplace_back(parameter);
                }

                for (auto& callslots_type : mod.callslots) {
                    for (auto& c : callslots_type.second) {
                        auto callslot_ptr = std::make_shared<CallSlot>(megamol::gui::GenerateUniqueID(), c.name,
                            c.description, c.compatible_call_idxs, c.type, c.necessity);
                        callslot_ptr->ConnectParentModule(mod_ptr);
                        mod_ptr->AddCallSlot(callslot_ptr);
                    }
                }

                QueueData queue_data;
                queue_data.class_name = mod_ptr->ClassName();
                queue_data.name_id = mod_ptr->FullName();
                this->PushSyncQueue(QueueAction::ADD_MODULE, queue_data);

                this->modules.emplace_back(mod_ptr);
                this->ForceSetDirty();

                // Automatically set new view module as graph entry, if no other entry point is set
                if (mod_ptr->IsView()) {
                    bool create_new_graph_entry = true;
                    for (auto& module_ptr : this->Modules()) {
                        if (module_ptr->IsView() && module_ptr->IsGraphEntry()) {
                            create_new_graph_entry = false;
                        }
                    }
                    if (create_new_graph_entry) {
                        Graph::QueueData entry_queue_data;
                        entry_queue_data.name_id = mod_ptr->FullName();
                        mod_ptr->SetGraphEntryName(this->GenerateUniqueGraphEntryName());
                        this->PushSyncQueue(Graph::QueueAction::CREATE_GRAPH_ENTRY, entry_queue_data);
                    }
                }

#ifdef GUI_VERBOSE
                megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                    "[GUI] Added module '%s' (uid %i) to project '%s'.\n", mod_ptr->ClassName().c_str(), mod_ptr->UID(),
                    this->name.c_str());
#endif // GUI_VERBOSE

                return mod_ptr;
            }
        }
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return nullptr;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return nullptr;
    }

    megamol::core::utility::log::Log::DefaultLog.WriteError(
        "[GUI] Unable to find module in stock: '%s'. [%s, %s, line %d]\n", class_name.c_str(), __FILE__, __FUNCTION__,
        __LINE__);
    return nullptr;
}


bool megamol::gui::Graph::DeleteModule(ImGuiID module_uid, bool use_queue) {

    try {
        for (auto iter = this->modules.begin(); iter != this->modules.end(); iter++) {
            if ((*iter)->UID() == module_uid) {

                this->ResetStatePointers();
                QueueData queue_data;

                // 1) Reset module and call slot pointers in groups
                auto current_full_name = (*iter)->FullName();
                GroupPtr_t module_group_ptr = nullptr;
                for (auto& group_ptr : this->groups) {
                    if (group_ptr->ContainsModule(module_uid)) {
                        module_group_ptr = group_ptr;
                        this->RemoveGroupModule(group_ptr->UID(), module_uid, use_queue, false);
                    }
                }

                // 2)  Delete calls
                for (auto& callslot_map : (*iter)->CallSlots()) {
                    for (auto& callslot_ptr : callslot_map.second) {
                        std::vector<ImGuiID> delete_call_uids;
                        for (auto& call_ptr : callslot_ptr->GetConnectedCalls()) {
                            if (call_ptr != nullptr) {
                                delete_call_uids.push_back(call_ptr->UID());
                            }
                        }
                        for (auto& call_uid : delete_call_uids) {
                            this->DeleteCall(call_uid);
                        }
                    }
                }

                // 3)  Remove call slots
                (*iter)->DeleteCallSlots();

                // 4) Automatically restore interfaceslots after connected calls are deleted
                if (module_group_ptr != nullptr) {
                    module_group_ptr->RestoreInterfaceslots();
                }


#ifdef GUI_VERBOSE
                megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                    "[GUI] Deleted module '%s' (uid %i) from  project '%s'.\n", (*iter)->FullName().c_str(),
                    (*iter)->UID(), this->name.c_str());
#endif // GUI_VERBOSE

                queue_data.name_id = (*iter)->FullName();
                if (use_queue) {
                    this->PushSyncQueue(QueueAction::DELETE_MODULE, queue_data);
                }

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


megamol::gui::ModulePtr_t megamol::gui::Graph::GetModule(ImGuiID module_uid) {

    if (module_uid != GUI_INVALID_ID) {
        for (auto& module_ptr : this->modules) {
            if (module_ptr->UID() == module_uid) {
                return module_ptr;
            }
        }
    }
    return nullptr;
}


ModulePtr_t megamol::gui::Graph::GetModule(const std::string& module_fullname) {

    for (auto& module_ptr : this->modules) {
        if (module_ptr->FullName() == module_fullname) {
            return module_ptr;
        }
    }
    return nullptr;
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
            if (auto callslot_ptr = module_ptr->CallSlotPtr(slot_1_uid)) {
                drag_callslot_ptr = callslot_ptr;
            }
            if (auto callslot_ptr = module_ptr->CallSlotPtr(slot_2_uid)) {
                drop_callslot_ptr = callslot_ptr;
            }
        }

        InterfaceSlotPtr_t drag_interfaceslot_ptr;
        InterfaceSlotPtr_t drop_interfaceslot_ptr;
        for (auto& group_ptr : this->groups) {
            if (auto interfaceslot_ptr = group_ptr->InterfaceSlotPtr(slot_1_uid)) {
                drag_interfaceslot_ptr = interfaceslot_ptr;
            }
            if (auto interfaceslot_ptr = group_ptr->InterfaceSlotPtr(slot_2_uid)) {
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

            ImGuiID interfaceslot_group_uid = interface_ptr->GroupUID();
            ImGuiID callslot_group_uid = GUI_INVALID_ID;
            if (callslot_ptr->IsParentModuleConnected()) {
                callslot_group_uid = callslot_ptr->GetParentModule()->GroupUID();
            }

            if (interfaceslot_group_uid == callslot_group_uid) {
                if (interface_ptr->AddCallSlot(callslot_ptr, interface_ptr)) {
                    CallSlotType compatible_callslot_type = (interface_ptr->GetCallSlotType() == CallSlotType::CALLEE)
                                                                ? (CallSlotType::CALLER)
                                                                : (CallSlotType::CALLEE);
                    // Get call slot the interface slot is connected to and add call for new added call slot
                    CallSlotPtr_t connect_callslot_ptr;
                    for (auto& interface_callslots_ptr : interface_ptr->CallSlots()) {
                        if (interface_callslots_ptr->UID() != callslot_ptr->UID()) {
                            for (auto& call_ptr : interface_callslots_ptr->GetConnectedCalls()) {
                                connect_callslot_ptr = call_ptr->CallSlotPtr(compatible_callslot_type);
                            }
                        }
                    }
                    if (connect_callslot_ptr != nullptr) {
                        if (!this->AddCall(stock_calls, callslot_ptr, connect_callslot_ptr)) {
                            interface_ptr->RemoveCallSlot(callslot_ptr->UID());
                        }
                    }
                }
            } else if (interfaceslot_group_uid != callslot_group_uid) {
                // Add calls to all call slots the call slots of the interface are connected to.
                for (auto& interface_callslots_ptr : interface_ptr->CallSlots()) {
                    this->AddCall(stock_calls, callslot_ptr, interface_callslots_ptr);
                }
            }
        }
        // InterfaceSlot <-> InterfaceSlot
        else if ((drag_interfaceslot_ptr != nullptr) && (drop_interfaceslot_ptr != nullptr)) {
            if (drag_interfaceslot_ptr->IsConnectionValid((*drop_interfaceslot_ptr))) {
                for (auto& drag_interface_callslots_ptr : drag_interfaceslot_ptr->CallSlots()) {
                    for (auto& drop_interface_callslots_ptr : drop_interfaceslot_ptr->CallSlots()) {
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


CallPtr_t megamol::gui::Graph::AddCall(
    const CallStockVector_t& stock_calls, CallSlotPtr_t callslot_1, CallSlotPtr_t callslot_2, bool use_queue) {

    try {
        if ((callslot_1 == nullptr) || (callslot_2 == nullptr)) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] Pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return nullptr;
        }
        if (!callslot_1->IsConnectionValid((*callslot_2))) {
            return nullptr;
        }
        auto compat_idx = CallSlot::GetCompatibleCallIndex(callslot_1, callslot_2);
        if (compat_idx == GUI_INVALID_ID) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] Unable to find index of compatible call. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return nullptr;
        }
        Call::StockCall call_stock_data = stock_calls[compat_idx];

        auto call_ptr = std::make_shared<Call>(megamol::gui::GenerateUniqueID(), call_stock_data.class_name,
            call_stock_data.description, call_stock_data.plugin_name, call_stock_data.functions);

        return this->ConnectCall(call_ptr, callslot_1, callslot_2, use_queue) ? call_ptr : nullptr;

    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return nullptr;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return nullptr;
    }
}


bool megamol::gui::Graph::ConnectCall(
    CallPtr_t& call_ptr, CallSlotPtr_t callslot_1, CallSlotPtr_t callslot_2, bool use_queue) {

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
    if ((callslot_1->Type() == CallSlotType::CALLER) && (callslot_1->CallsConnected())) {
        std::vector<ImGuiID> calls_uids;
        for (auto& call : callslot_1->GetConnectedCalls()) {
            calls_uids.emplace_back(call->UID());
        }
        for (auto& call_uid : calls_uids) {
            this->DeleteCall(call_uid);
        }
    }
    if ((callslot_2->Type() == CallSlotType::CALLER) && (callslot_2->CallsConnected())) {
        std::vector<ImGuiID> calls_uids;
        for (auto& call : callslot_2->GetConnectedCalls()) {
            calls_uids.emplace_back(call->UID());
        }
        for (auto& call_uid : calls_uids) {
            this->DeleteCall(call_uid);
        }
    }

    if (call_ptr->ConnectCallSlots(callslot_1, callslot_2) && callslot_1->ConnectCall(call_ptr) &&
        callslot_2->ConnectCall(call_ptr)) {

        QueueData queue_data;
        queue_data.class_name = call_ptr->ClassName();
        bool valid_ptr = false;
        auto caller_ptr = call_ptr->CallSlotPtr(megamol::gui::CallSlotType::CALLER);
        if (caller_ptr != nullptr) {
            if (caller_ptr->GetParentModule() != nullptr) {
                queue_data.caller = caller_ptr->GetParentModule()->FullName() + "::" + caller_ptr->Name();
                valid_ptr = true;
            }
        }
        if (!valid_ptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Pointer to caller slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        }
        valid_ptr = false;
        auto callee_ptr = call_ptr->CallSlotPtr(megamol::gui::CallSlotType::CALLEE);
        if (callee_ptr != nullptr) {
            if (callee_ptr->GetParentModule() != nullptr) {
                queue_data.callee = callee_ptr->GetParentModule()->FullName() + "::" + callee_ptr->Name();
                valid_ptr = true;
            }
        }
        if (!valid_ptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Pointer to callee slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        }
        if (use_queue) {
            this->PushSyncQueue(QueueAction::ADD_CALL, queue_data);
        }

        this->calls.emplace_back(call_ptr);
        this->ForceSetDirty();

#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Added call '%s' (uid %i) to project '%s'.\n",
            call_ptr->ClassName().c_str(), call_ptr->UID(), this->name.c_str());
#endif // GUI_VERBOSE

        // Add connected call slots to interface of group of the parent module
        if (callslot_1->IsParentModuleConnected() && callslot_2->IsParentModuleConnected()) {
            ImGuiID slot_1_parent_group_uid = callslot_1->GetParentModule()->GroupUID();
            ImGuiID slot_2_parent_group_uid = callslot_2->GetParentModule()->GroupUID();
            if (slot_1_parent_group_uid != slot_2_parent_group_uid) {
                if ((slot_1_parent_group_uid != GUI_INVALID_ID) && (callslot_1->InterfaceSlotPtr() == nullptr)) {
                    for (auto& group_ptr : this->groups) {
                        if (group_ptr->UID() == slot_1_parent_group_uid) {
                            group_ptr->AddInterfaceSlot(callslot_1);
                        }
                    }
                }
                if ((slot_2_parent_group_uid != GUI_INVALID_ID) && (callslot_2->InterfaceSlotPtr() == nullptr)) {
                    for (auto& group_ptr : this->groups) {
                        if (group_ptr->UID() == slot_2_parent_group_uid) {
                            group_ptr->AddInterfaceSlot(callslot_2);
                        }
                    }
                }
            }
        }

        return true;
    } else {
        this->DeleteCall(call_ptr->UID());
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Unable to create call. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
}


CallPtr_t megamol::gui::Graph::GetCall(
    std::string const& call_classname, std::string const& caller_fullname, std::string const& callee_fullname) {

    for (auto& call_ptr : this->calls) {
        auto caller_ptr = call_ptr->CallSlotPtr(CallSlotType::CALLER);
        auto callee_ptr = call_ptr->CallSlotPtr(CallSlotType::CALLEE);
        if ((caller_ptr != nullptr) && (callee_ptr != nullptr)) {
            std::string tmp_caller_fullname;
            std::string tmp_callee_fullname;
            if (caller_ptr->GetParentModule() != nullptr) {
                tmp_caller_fullname = caller_ptr->GetParentModule()->FullName() + "::" + caller_ptr->Name();
            }
            if (callee_ptr->GetParentModule() != nullptr) {
                tmp_callee_fullname = callee_ptr->GetParentModule()->FullName() + "::" + callee_ptr->Name();
            }
            if ((call_ptr->ClassName() == call_classname) && (tmp_caller_fullname == caller_fullname) &&
                (tmp_callee_fullname == callee_fullname)) {
                return call_ptr;
            }
        }
    }
    return nullptr;
}


bool megamol::gui::Graph::DeleteCall(ImGuiID call_uid, bool use_queue) {

    try {
        std::vector<ImGuiID> delete_calls_uids;
        delete_calls_uids.emplace_back(call_uid);

        // Also delete other calls, which are connceted to same interface slot and call slot
        ImGuiID caller_uid = GUI_INVALID_ID;
        ImGuiID callee_uid = GUI_INVALID_ID;
        for (auto& call_ptr : this->calls) {
            if (call_ptr->UID() == call_uid) {
                auto caller = call_ptr->CallSlotPtr(CallSlotType::CALLER);
                auto callee = call_ptr->CallSlotPtr(CallSlotType::CALLEE);
                if (caller != nullptr) {
                    caller_uid = caller->UID();
                    if (caller->InterfaceSlotPtr() != nullptr) {
                        caller_uid = caller->InterfaceSlotPtr()->UID();
                    }
                }
                if (callee != nullptr) {
                    callee_uid = callee->UID();
                    if (callee->InterfaceSlotPtr() != nullptr) {
                        callee_uid = callee->InterfaceSlotPtr()->UID();
                    }
                }
            }
        }
        for (auto& call_ptr : this->calls) {
            if (call_ptr->UID() != call_uid) {
                bool caller_fits = false;
                bool callee_fits = false;
                auto caller = call_ptr->CallSlotPtr(CallSlotType::CALLER);
                auto callee = call_ptr->CallSlotPtr(CallSlotType::CALLEE);
                if (caller != nullptr) {
                    if (caller->InterfaceSlotPtr() != nullptr) {
                        caller_fits = (caller_uid == caller->InterfaceSlotPtr()->UID());
                    } else {
                        caller_fits = (caller_uid == caller->UID());
                    }
                }
                if (callee != nullptr) {
                    if (callee->InterfaceSlotPtr() != nullptr) {
                        callee_fits = (callee_uid == callee->InterfaceSlotPtr()->UID());
                    } else {
                        callee_fits = (callee_uid == callee->UID());
                    }
                }
                if (caller_fits && callee_fits) {
                    delete_calls_uids.emplace_back(call_ptr->UID());
                }
            }
        }

        // Actual deletion of calls
        for (auto& delete_call_uid : delete_calls_uids) {
            for (auto iter = this->calls.begin(); iter != this->calls.end(); iter++) {
                if ((*iter)->UID() == delete_call_uid) {

                    QueueData queue_data;
                    bool valid_caller_ptr = false;
                    auto caller_ptr = (*iter)->CallSlotPtr(megamol::gui::CallSlotType::CALLER);
                    if (caller_ptr != nullptr) {
                        if (caller_ptr->GetParentModule() != nullptr) {
                            queue_data.caller = caller_ptr->GetParentModule()->FullName() + "::" + caller_ptr->Name();
                            valid_caller_ptr = true;
                        }
                    }
                    if (!valid_caller_ptr) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] Pointer to caller slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                            __LINE__);
                    }

                    bool valid_callee_ptr = false;
                    auto callee_ptr = (*iter)->CallSlotPtr(megamol::gui::CallSlotType::CALLEE);
                    if (callee_ptr != nullptr) {
                        if (callee_ptr->GetParentModule() != nullptr) {
                            queue_data.callee = callee_ptr->GetParentModule()->FullName() + "::" + callee_ptr->Name();
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

                    if (use_queue) {
                        this->PushSyncQueue(QueueAction::DELETE_CALL, queue_data);
                    }
                    this->ResetStatePointers();

                    (*iter)->DisconnectCallSlots();

                    if ((*iter).use_count() > 1) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] Unclean deletion. Found %i references pointing to call. [%s, %s, line %d]\n",
                            (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                    }
#ifdef GUI_VERBOSE
                    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                        "[GUI] Deleted call '%s' (uid %i) from  project '%s'.\n", (*iter)->ClassName().c_str(),
                        (*iter)->UID(), this->name.c_str());
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


bool megamol::gui::Graph::CallExists(
    std::string const& call_classname, std::string const& caller_fullname, std::string const& callee_fullname) {

    return (std::find_if(this->calls.begin(), this->calls.end(), [&](megamol::gui::CallPtr_t& call_ptr) {
        auto caller_ptr = call_ptr->CallSlotPtr(CallSlotType::CALLER);
        auto callee_ptr = call_ptr->CallSlotPtr(CallSlotType::CALLEE);
        if ((caller_ptr != nullptr) && (callee_ptr != nullptr)) {
            std::string tmp_caller_fullname;
            std::string tmp_callee_fullname;
            if (caller_ptr->GetParentModule() != nullptr) {
                tmp_caller_fullname = caller_ptr->GetParentModule()->FullName() + "::" + caller_ptr->Name();
            }
            if (callee_ptr->GetParentModule() != nullptr) {
                tmp_callee_fullname = callee_ptr->GetParentModule()->FullName() + "::" + callee_ptr->Name();
            }
            return ((call_ptr->ClassName() == call_classname) && (tmp_caller_fullname == caller_fullname) &&
                    (tmp_callee_fullname == callee_fullname));
        }
    }) != this->calls.end());
}


ImGuiID megamol::gui::Graph::AddGroup(const std::string& group_name) {

    try {
        ImGuiID group_id = megamol::gui::GenerateUniqueID();
        auto group_ptr = std::make_shared<Group>(group_id);
        group_ptr->SetName((group_name.empty()) ? (this->generate_unique_group_name()) : (group_name));
        this->groups.emplace_back(group_ptr);

#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Added group '%s' (uid %i) to project '%s'.\n",
            group_ptr->Name().c_str(), group_ptr->UID(), this->name.c_str());
#endif // GUI_VERBOSE

        this->ForceSetDirty();
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


megamol::gui::GroupPtr_t megamol::gui::Graph::GetGroup(ImGuiID group_uid) {

    if (group_uid != GUI_INVALID_ID) {
        for (auto& group_ptr : this->groups) {
            if (group_ptr->UID() == group_uid) {
                return group_ptr;
            }
        }
    }
    return nullptr;
}


bool megamol::gui::Graph::DeleteGroup(ImGuiID group_uid) {

    try {
        for (auto iter = this->groups.begin(); iter != this->groups.end(); iter++) {
            if ((*iter)->UID() == group_uid) {

                this->ResetStatePointers();

                if ((*iter).use_count() > 1) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Unclean deletion. Found %i references pointing to group. [%s, %s, line %d]\n",
                        (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                }
#ifdef GUI_VERBOSE
                megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                    "[GUI] Deleted group '%s' (uid %i) from  project '%s'.\n", (*iter)->Name().c_str(), (*iter)->UID(),
                    this->name.c_str());
#endif // GUI_VERBOSE
                (*iter).reset();
                this->groups.erase(iter);

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
        "[GUI] Invalid group uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}


ImGuiID megamol::gui::Graph::AddGroupModule(
    const std::string& group_name, const ModulePtr_t& module_ptr, bool use_queue) {

    // Only create new group if given name is not empty
    if (!group_name.empty()) {
        // Check if group with given name already exists
        ImGuiID existing_group_uid = GUI_INVALID_ID;
        for (auto& group_ptr : this->groups) {
            if (group_ptr->Name() == group_name) {
                if (existing_group_uid != GUI_INVALID_ID) {
                    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                        "[GUI] Warning: Found groups with same name. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                        __LINE__);
                }
                existing_group_uid = group_ptr->UID();
            }
        }
        // Create new group if there is no one with given name
        if (existing_group_uid == GUI_INVALID_ID) {
            existing_group_uid = this->AddGroup(group_name);
        }
        // Add module to group
        for (auto& group_ptr : this->groups) {
            if (group_ptr->UID() == existing_group_uid) {
                if (this->AddGroupModule(existing_group_uid, module_ptr, use_queue) != GUI_INVALID_ID) {
                    return existing_group_uid;
                }
            }
        }
    }
    return GUI_INVALID_ID;
}


ImGuiID megamol::gui::Graph::AddGroupModule(ImGuiID group_uid, const ModulePtr_t& module_ptr, bool use_queue) {

    try {
        // Add module to group
        for (auto& group_ptr : this->groups) {
            if (group_ptr->UID() == group_uid) {
                Graph::QueueData queue_data;
                queue_data.name_id = module_ptr->FullName();
                if (group_ptr->AddModule(module_ptr)) {
                    queue_data.rename_id = module_ptr->FullName();
                    if (use_queue) {
                        this->PushSyncQueue(Graph::QueueAction::RENAME_MODULE, queue_data);
                    }
                    this->ForceSetDirty();
                    return group_uid;
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


ImGuiID megamol::gui::Graph::RemoveGroupModule(
    ImGuiID group_uid, ImGuiID module_uid, bool use_queue, bool reset_interface) {

    try {
        // Remove module to group
        for (auto& group_ptr : this->groups) {
            if (group_ptr->UID() == group_uid) {
                for (auto module_ptr : this->modules) {
                    if (module_ptr->UID() == module_uid) {
                        Graph::QueueData queue_data;
                        queue_data.name_id = module_ptr->FullName();
                        if (group_ptr->RemoveModule(module_ptr->UID(), reset_interface)) {
                            queue_data.rename_id = module_ptr->FullName();
                            if (use_queue) {
                                this->PushSyncQueue(Graph::QueueAction::RENAME_MODULE, queue_data);
                            }
                            this->ForceSetDirty();
                            return group_uid;
                        }
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


void megamol::gui::Graph::Clear() {

    this->ResetStatePointers();

    // 1) Delete all modules
    std::vector<ImGuiID> module_uids;
    for (auto& module_ptr : this->modules) {
        module_uids.emplace_back(module_ptr->UID());
    }
    for (auto& module_uid : module_uids) {
        this->DeleteModule(module_uid);
    }

    // 2) Delete all calls
    std::vector<ImGuiID> call_uids;
    for (auto& call_ptr : this->calls) {
        call_uids.emplace_back(call_ptr->UID());
    }
    for (auto& call_uid : call_uids) {
        this->DeleteCall(call_uid);
    }

    // 3) Delete all groups
    std::vector<ImGuiID> group_uids;
    for (auto& group_ptr : this->groups) {
        group_uids.emplace_back(group_ptr->UID());
    }
    for (auto& group_uid : group_uids) {
        this->DeleteGroup(group_uid);
    }
}


bool megamol::gui::Graph::UniqueModuleRename(const std::string& module_full_name) {

    for (auto& mod : this->modules) {
        if (module_full_name == mod->FullName()) {
            mod->SetName(this->generate_unique_module_name(mod->Name()));

            QueueData queue_data;
            queue_data.name_id = module_full_name;
            queue_data.rename_id = mod->FullName();
            this->PushSyncQueue(QueueAction::RENAME_MODULE, queue_data);

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


bool Graph::ToggleGraphEntry(bool use_queue) {

    auto module_graph_entry_iter = this->modules.begin();
    // Search for first graph entry and set next view to graph entry (= graph entry point)
    for (auto module_iter = this->modules.begin(); module_iter != this->modules.end(); module_iter++) {
        if ((*module_iter)->IsView() && (*module_iter)->IsGraphEntry()) {
            // Remove all graph entries
            (*module_iter)->SetGraphEntryName("");
            if (this->IsRunning() && use_queue) {
                QueueData queue_data;
                queue_data.name_id = (*module_iter)->FullName();
                this->PushSyncQueue(Graph::QueueAction::REMOVE_GRAPH_ENTRY, queue_data);
            }
            // Save index of last found graph entry
            if (module_iter != this->modules.end()) {
                module_graph_entry_iter = module_iter + 1;
            }
        }
    }
    if (module_graph_entry_iter != this->modules.end()) {
        // Search for next graph entry
        for (auto module_iter = module_graph_entry_iter; module_iter != this->modules.end(); module_iter++) {
            if ((*module_iter)->IsView()) {
                (*module_iter)->SetGraphEntryName(this->GenerateUniqueGraphEntryName());
                if (this->IsRunning() && use_queue) {
                    QueueData queue_data;
                    queue_data.name_id = (*module_iter)->FullName();
                    this->PushSyncQueue(Graph::QueueAction::CREATE_GRAPH_ENTRY, queue_data);
                }
                break;
            }
        }
    }
    return true;
}


bool megamol::gui::Graph::AddGraphEntry(const ModulePtr_t& module_ptr, const std::string& name, bool use_queue) {

    if (module_ptr == nullptr)
        return false;

    // Remove all existing graph entries
    for (auto mod_ptr : this->modules) {
        mod_ptr->SetGraphEntryName("");
        if (this->IsRunning() && use_queue) {
            QueueData queue_data;
            queue_data.name_id = mod_ptr->FullName();
            this->PushSyncQueue(Graph::QueueAction::REMOVE_GRAPH_ENTRY, queue_data);
        }
    }

    if (module_ptr->IsView()) {
        module_ptr->SetGraphEntryName(name);
        if (this->IsRunning() && use_queue) {
            QueueData queue_data;
            queue_data.name_id = module_ptr->FullName();
            this->PushSyncQueue(Graph::QueueAction::CREATE_GRAPH_ENTRY, queue_data);
        }
    }

    return true;
}


bool megamol::gui::Graph::RemoveGraphEntry(const ModulePtr_t& module_ptr, bool use_queue) {

    if (module_ptr == nullptr)
        return false;

    module_ptr->SetGraphEntryName("");
    if (this->IsRunning() && use_queue) {
        QueueData queue_data;
        queue_data.name_id = module_ptr->FullName();
        this->PushSyncQueue(Graph::QueueAction::REMOVE_GRAPH_ENTRY, queue_data);
    }
    return true;
}


std::string megamol::gui::Graph::GetFilename() const {

    if (this->filenames.first.first) {
        // Return script path
        return this->filenames.first.second;
    } else if (this->filenames.second.first) {
        // Return saved file name
        return this->filenames.second.second;
    }
    return std::string();
}


void megamol::gui::Graph::SetFilename(const std::string& filename, bool saved_filename) {

    if (saved_filename) {
        if (filename != this->filenames.second.second) {
            this->filenames.second.second = filename;
            this->filenames.second.first = true;
            this->filenames.first.first = false;
        }
    } else {
        if (filename != this->filenames.first.second) {
            this->filenames.first.second = filename;
            this->filenames.first.first = true;
            this->filenames.second.first = false;
        }
    }
}


bool megamol::gui::Graph::PushSyncQueue(QueueAction action, const QueueData& in_data) {

    // Use sync queue only when interface to core graph is available
    if (!this->IsRunning())
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
        if (queue_data.name_id.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Graph sync queue action CREATE_GRAPH_ENTRY is missing data for 'name_id'. [%s, %s, line %d]\n",
                __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
    } break;
    case (QueueAction::REMOVE_GRAPH_ENTRY): {
        if (queue_data.name_id.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Graph sync queue action REMOVE_GRAPH_ENTRY is missing data for 'name_id'. [%s, %s, line %d]\n",
                __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
    } break;
    default: {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown graph sync queue action. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    } break;
    }

    this->sync_queue.emplace_back(SyncQueueData_t(action, queue_data));
    return true;
}


bool megamol::gui::Graph::PopSyncQueue(QueueAction& out_action, QueueData& out_data) {

    if (!this->sync_queue.empty()) {
        out_action = std::get<0>(this->sync_queue.front());
        out_data = std::get<1>(this->sync_queue.front());
        this->sync_queue.pop_front();
        return true;
    }
    return false;
}


megamol::gui::Graph::QueueData megamol::gui::Graph::FindQueueEntryByActionName(
    QueueAction action, const std::string& name) {

    for (auto& entry : this->sync_queue) {
        if ((std::get<0>(entry) == action) && (std::get<1>(entry).name_id == name)) {
            return std::get<1>(entry);
        }
    }
    return megamol::gui::Graph::QueueData();
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
                    if (json_graph_id == GUI_JSON_TAG_PROJECT) {
                        auto graph_state = content_item.value();

                        megamol::core::utility::get_json_value<std::string>(graph_state, {"project_name"}, &this->name);

                        bool tmp_show_parameter_sidebar = false;
                        this->gui_change_show_parameter_sidebar = false;
                        if (megamol::core::utility::get_json_value<bool>(
                                graph_state, {"show_parameter_sidebar"}, &tmp_show_parameter_sidebar)) {
                            this->gui_change_show_parameter_sidebar = true;
                            this->gui_show_parameter_sidebar = tmp_show_parameter_sidebar;
                        }
                        megamol::core::utility::get_json_value<float>(
                            graph_state, {"parameter_sidebar_width"}, &this->gui_parameter_sidebar_width);
#ifdef MEGAMOL_USE_PROFILING
                        bool tmp_show_profiling_bar = false;
                        this->gui_change_show_profiling_bar = false;
                        if (megamol::core::utility::get_json_value<bool>(
                                graph_state, {"show_profiling_bar"}, &tmp_show_profiling_bar)) {
                            this->gui_change_show_profiling_bar = true;
                            this->gui_show_profiling_bar = tmp_show_profiling_bar;
                        }
                        megamol::core::utility::get_json_value<float>(
                            graph_state, {"profiling_bar_height"}, &this->gui_profiling_bar_height);
#endif // MEGAMOL_USE_PROFILING
                        megamol::core::utility::get_json_value<bool>(graph_state, {"show_grid"}, &this->gui_show_grid);
                        megamol::core::utility::get_json_value<bool>(
                            graph_state, {"show_call_label"}, &this->gui_graph_state.interact.call_show_label);
                        megamol::core::utility::get_json_value<bool>(graph_state, {"show_call_slots_label"},
                            &this->gui_graph_state.interact.call_show_slots_label);
                        megamol::core::utility::get_json_value<unsigned int>(
                            graph_state, {"call_coloring_mode"}, &this->gui_graph_state.interact.call_coloring_mode);
                        megamol::core::utility::get_json_value<unsigned int>(
                            graph_state, {"call_coloring_map"}, &this->gui_graph_state.interact.call_coloring_map);
                        megamol::core::utility::get_json_value<bool>(
                            graph_state, {"show_module_label"}, &this->gui_graph_state.interact.module_show_label);
                        megamol::core::utility::get_json_value<bool>(
                            graph_state, {"show_slot_label"}, &this->gui_graph_state.interact.callslot_show_label);
                        megamol::core::utility::get_json_value<bool>(graph_state, {"param_extended_mode"},
                            &this->gui_graph_state.interact.parameters_extended_mode);
                        std::array<float, 2> canvas_scrolling = {0.0f, 0.0f};
                        megamol::core::utility::get_json_value<float>(
                            graph_state, {"canvas_scrolling"}, canvas_scrolling.data(), canvas_scrolling.size());
                        this->gui_graph_state.canvas.scrolling = ImVec2(canvas_scrolling[0], canvas_scrolling[1]);
                        if (megamol::core::utility::get_json_value<float>(
                                graph_state, {"canvas_zooming"}, &this->gui_graph_state.canvas.zooming)) {
                            this->gui_reset_zooming = false;
                        }

                        // modules
                        for (auto& module_item : graph_state.items()) {
                            if (module_item.key() == GUI_JSON_TAG_MODULES) {
                                for (auto& module_state : module_item.value().items()) {
                                    std::string module_fullname = module_state.key();
                                    auto position_item = module_state.value();
                                    std::array<float, 2> graph_position = {0.0f, 0.0f};
                                    megamol::core::utility::get_json_value<float>(module_state.value(),
                                        {"graph_position"}, graph_position.data(), graph_position.size());
                                    auto module_position = ImVec2(graph_position[0], graph_position[1]);

                                    // Apply graph position to module
                                    bool module_found = false;
                                    for (auto& module_ptr : this->Modules()) {
                                        if (module_ptr->FullName() == module_fullname) {
                                            module_ptr->SetPosition(module_position);
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
                                                for (auto& module_ptr : this->Modules()) {
                                                    if (module_ptr->FullName() == module_fullname) {
                                                        for (auto& callslot_map : module_ptr->CallSlots()) {
                                                            for (auto& callslot_ptr : callslot_map.second) {
                                                                if (callslot_ptr->Name() == callslot_name) {
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
                                                if (group_ptr->Name() == group_name) {
                                                    auto callslot_ptr = callslot_ptr_vector[0];
                                                    // First remove previously added interface slot which was
                                                    // automatically added during adding module to group
                                                    this->ResetStatePointers();
                                                    for (size_t i = 1; i < callslot_ptr_vector.size(); i++) {
                                                        if (group_ptr->InterfaceSlot_ContainsCallSlot(
                                                                callslot_ptr_vector[i]->UID())) {
                                                            group_ptr->InterfaceSlot_RemoveCallSlot(
                                                                callslot_ptr_vector[i]->UID(), true);
                                                        }
                                                    }
                                                    if (auto interfaceslot_ptr =
                                                            group_ptr->AddInterfaceSlot(callslot_ptr)) {
                                                        for (size_t i = 1; i < callslot_ptr_vector.size(); i++) {
                                                            interfaceslot_ptr->AddCallSlot(
                                                                callslot_ptr_vector[i], interfaceslot_ptr);
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

        this->gui_update = true;
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
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["project_name"] = this->name;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["show_parameter_sidebar"] =
            this->gui_show_parameter_sidebar;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["parameter_sidebar_width"] =
            this->gui_parameter_sidebar_width;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["show_profiling_bar"] = this->gui_show_profiling_bar;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["profiling_bar_height"] = this->gui_profiling_bar_height;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["show_grid"] = this->gui_show_grid;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["show_call_label"] =
            this->gui_graph_state.interact.call_show_label;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["show_call_slots_label"] =
            this->gui_graph_state.interact.call_show_slots_label;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["call_coloring_mode"] =
            this->gui_graph_state.interact.call_coloring_mode;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["call_coloring_map"] =
            this->gui_graph_state.interact.call_coloring_map;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["show_slot_label"] =
            this->gui_graph_state.interact.callslot_show_label;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["show_module_label"] =
            this->gui_graph_state.interact.module_show_label;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["param_extended_mode"] =
            this->gui_graph_state.interact.parameters_extended_mode;
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["canvas_scrolling"] = {
            this->gui_graph_state.canvas.scrolling.x, this->gui_graph_state.canvas.scrolling.y};
        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT]["canvas_zooming"] = this->gui_graph_state.canvas.zooming;

        // Write module positions
        for (auto& module_ptr : this->Modules()) {
            inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT][GUI_JSON_TAG_MODULES][module_ptr->FullName()]
                      ["graph_position"] = {module_ptr->Position().x, module_ptr->Position().y};
        }

        // Write group interface slots
        size_t interface_number = 0;
        for (auto& group_ptr : this->GetGroups()) {
            for (auto& interfaceslots_map : group_ptr->InterfaceSlots()) {
                for (auto& interface_ptr : interfaceslots_map.second) {
                    std::string interface_label = "interface_slot_" + std::to_string(interface_number);
                    for (auto& callslot_ptr : interface_ptr->CallSlots()) {
                        std::string callslot_fullname;
                        if (callslot_ptr->IsParentModuleConnected()) {
                            callslot_fullname =
                                callslot_ptr->GetParentModule()->FullName() + "::" + callslot_ptr->Name();
                        }
                        inout_json[GUI_JSON_TAG_GRAPHS][GUI_JSON_TAG_PROJECT][GUI_JSON_TAG_INTERFACES]
                                  [group_ptr->Name()][interface_label] += callslot_fullname;
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


void megamol::gui::Graph::Draw(GraphState_t& state) {

    try {
        if (ImGui::GetCurrentContext() == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return;
        }
        ImGuiIO& io = ImGui::GetIO();

        auto graph_uid = static_cast<int>(this->uid);
        ImGui::PushID(static_cast<int>(graph_uid));

        bool popup_rename = false;

        ImGuiTabItemFlags tab_flags = ImGuiTabItemFlags_None;
        if (this->IsDirty()) {
            tab_flags |= ImGuiTabItemFlags_UnsavedDocument;
        }

        bool open_value = true;
        // Hide close button of running project tab
        bool* tab_open = ((this->IsRunning()) ? (nullptr) : (&open_value));
        std::string graph_label = "  " + this->name + "  ###graph" + std::to_string(graph_uid);

        // Tab showing this graph
        if (ImGui::BeginTabItem(graph_label.c_str(), tab_open, tab_flags)) {

            // State Init ----------------------------

            // Only load gui_show_parameter_sidebar from project file for running graph
            if (this->gui_change_show_parameter_sidebar && this->IsRunning()) {
                state.show_parameter_sidebar = this->gui_show_parameter_sidebar;
            }
            this->gui_change_show_parameter_sidebar = false;
            this->gui_show_parameter_sidebar = state.show_parameter_sidebar;

            if (this->gui_graph_state.interact.profiling_show) {
                this->gui_graph_state.interact.profiling_show = false;
                this->gui_change_show_profiling_bar = true;
                this->gui_show_profiling_bar = true;
            }
            // Only load gui_show_profiling_bar from project file for running graph
            if (this->gui_change_show_profiling_bar && this->IsRunning()) {
                state.show_profiling_bar = this->gui_show_profiling_bar;
            }
            this->gui_change_show_profiling_bar = false;
            this->gui_show_profiling_bar = state.show_profiling_bar;

            this->gui_graph_state.interact.graph_is_running = this->IsRunning();
            this->gui_graph_state.hotkeys = state.hotkeys;
            this->gui_graph_state.groups.clear();
            for (auto& group : this->GetGroups()) {
                std::pair<ImGuiID, std::string> group_pair(group->UID(), group->Name());
                this->gui_graph_state.groups.emplace_back(group_pair);
            }
            this->gui_graph_state.interact.slot_drag_drop_uids = UIDPair_t(GUI_INVALID_ID, GUI_INVALID_ID);

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
                for (auto& module_ptr : this->Modules()) {
                    if (auto callslot_ptr = module_ptr->CallSlotPtr(slot_uid)) {
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
                    for (auto& group_ptr : this->GetGroups()) {
                        if (auto interfaceslot_ptr = group_ptr->InterfaceSlotPtr(slot_uid)) {
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
                    if (this->IsRunning()) {
                        state.global_graph_save = true;
                    } else {
                        state.configurator_graph_save = true;
                    }
                }

                if (ImGui::MenuItem("Rename")) {
                    popup_rename = true;
                }

                if (!this->GetFilename().empty()) {
                    ImGui::Separator();
                    ImGui::TextDisabled("File Name");
                    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 13.0f);
                    ImGui::TextUnformatted(this->GetFilename().c_str());
                    ImGui::PopTextWrapPos();
                }

                ImGui::EndPopup();
            }

            // Draw -----------------------------
            this->draw_menu(state);

            ImGui::BeginChild(
                "graph_tab", ImVec2(0.0f, 0.0f), false, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar);

            // Disable profiling bar for not running graphs
            if (!this->IsRunning() && this->gui_show_profiling_bar) {
                this->gui_show_profiling_bar = false;
            }

            // Draw splitter
            const auto splitter_width = this->gui_profiling_splitter.GetWidth();
            const auto cursor_position = ImGui::GetCursorScreenPos();
            const auto available_space = ImGui::GetContentRegionAvail();
            float graph_width_auto = available_space.x;
            this->gui_parameter_sidebar_width *= megamol::gui::gui_scaling.Get();
            if (this->gui_show_parameter_sidebar) {
                this->gui_parameters_splitter.Widget("parameters_splitter", true, 0.0f,
                    SplitterWidget::FixedSplitterSide::RIGHT_BOTTOM, graph_width_auto,
                    this->gui_parameter_sidebar_width, cursor_position);
            }
            float graph_height_auto = available_space.y;
#ifdef MEGAMOL_USE_PROFILING
            this->gui_profiling_bar_height *= megamol::gui::gui_scaling.Get();
            if (this->gui_show_profiling_bar) {
                this->gui_profiling_splitter.Widget("profiling_splitter", false, graph_width_auto,
                    SplitterWidget::FixedSplitterSide::RIGHT_BOTTOM, graph_height_auto, this->gui_profiling_bar_height,
                    cursor_position);
            }
#endif // MEGAMOL_USE_PROFILING

            // Draw content of splitter regions
            this->draw_canvas(cursor_position + ImVec2(0.0f, 0.0f), ImVec2(graph_width_auto, graph_height_auto), state);
            if (this->gui_show_parameter_sidebar) {
                this->draw_parameters(cursor_position + ImVec2(graph_width_auto + splitter_width, 0.0f),
                    ImVec2(this->gui_parameter_sidebar_width, available_space.y));
            }
#ifdef MEGAMOL_USE_PROFILING
            if (this->gui_show_profiling_bar) {
                this->draw_profiling(cursor_position + ImVec2(0.0f, graph_height_auto + splitter_width),
                    ImVec2(graph_width_auto, this->gui_profiling_bar_height));
            }
#endif // MEGAMOL_USE_PROFILING

            this->gui_parameter_sidebar_width /= megamol::gui::gui_scaling.Get();
            this->gui_profiling_bar_height /= megamol::gui::gui_scaling.Get();

            state.graph_selected_uid = this->uid;

            ImGui::EndChild();

            // State processing ---------------------
            this->ResetStatePointers();
            bool reset_state = false;
            // Add module renaming event to graph synchronization queue -----------
            if (!this->gui_graph_state.interact.module_rename.empty()) {
                Graph::QueueData queue_data;
                for (auto& str_pair : this->gui_graph_state.interact.module_rename) {
                    queue_data.name_id = str_pair.first;
                    queue_data.rename_id = str_pair.second;
                    this->PushSyncQueue(Graph::QueueAction::RENAME_MODULE, queue_data);
                }
                this->gui_graph_state.interact.module_rename.clear();
            }
            // Add module graph entry event to graph synchronization queue ----------
            if (this->gui_graph_state.interact.module_graphentry_changed != vislib::math::Ternary::TRI_UNKNOWN) {
                // Choose single selected view module
                ModulePtr_t selected_mod_ptr;
                if (this->gui_graph_state.interact.modules_selected_uids.size() == 1) {
                    for (auto& mod : this->Modules()) {
                        if ((this->gui_graph_state.interact.modules_selected_uids.front() == mod->UID()) &&
                            (mod->IsView())) {
                            selected_mod_ptr = mod;
                        }
                    }
                }
                if (selected_mod_ptr != nullptr) {
                    Graph::QueueData queue_data;
                    if (this->gui_graph_state.interact.module_graphentry_changed == vislib::math::Ternary::TRI_TRUE) {
                        // Remove all graph entries
                        for (auto& module_ptr : this->Modules()) {
                            if (module_ptr->IsView() && module_ptr->IsGraphEntry()) {
                                module_ptr->SetGraphEntryName("");
                                queue_data.name_id = module_ptr->FullName();
                                this->PushSyncQueue(Graph::QueueAction::REMOVE_GRAPH_ENTRY, queue_data);
                            }
                        }
                        // Add new graph entry
                        queue_data.name_id = selected_mod_ptr->FullName();
                        selected_mod_ptr->SetGraphEntryName(this->GenerateUniqueGraphEntryName());
                        this->PushSyncQueue(Graph::QueueAction::CREATE_GRAPH_ENTRY, queue_data);
                    } else {
                        queue_data.name_id = selected_mod_ptr->FullName();
                        selected_mod_ptr->SetGraphEntryName("");
                        this->PushSyncQueue(Graph::QueueAction::REMOVE_GRAPH_ENTRY, queue_data);
                    }
                }
                this->gui_graph_state.interact.module_graphentry_changed = vislib::math::Ternary::TRI_UNKNOWN;
            }
            // Add module to group ------------------------------------------------
            if (!this->gui_graph_state.interact.modules_add_group_uids.empty()) {
                ModulePtr_t module_ptr;
                ImGuiID new_group_uid = GUI_INVALID_ID;
                for (auto& uid_pair : this->gui_graph_state.interact.modules_add_group_uids) {
                    module_ptr.reset();
                    for (auto& mod : this->Modules()) {
                        if (mod->UID() == uid_pair.first) {
                            module_ptr = mod;
                        }
                    }
                    if (module_ptr != nullptr) {
                        // Add module to new or already existing group
                        // Create new group for multiple selected modules only once!
                        ImGuiID group_uid = GUI_INVALID_ID;
                        if ((uid_pair.second == GUI_INVALID_ID) && (new_group_uid == GUI_INVALID_ID)) {
                            new_group_uid = this->AddGroup();
                        }
                        if (uid_pair.second == GUI_INVALID_ID) {
                            group_uid = new_group_uid;
                        } else {
                            group_uid = uid_pair.second;
                        }

                        /// Without intermediate renaming via queue!
                        /// Would trigger megamol::gui::GraphCollection::NotifyRunningGraph_RenameModule for already renamed module!
                        std::string current_module_groupname = module_ptr->GroupName();
                        this->RemoveGroupModule(module_ptr->GroupUID(), module_ptr->UID(), false);
                        module_ptr->SetGroupName(current_module_groupname);
                        this->AddGroupModule(group_uid, module_ptr, true);
                    }
                }
                reset_state = true;
            }
            // Remove module from group -------------------------------------------
            if (!this->gui_graph_state.interact.modules_remove_group_uids.empty()) {
                for (auto& module_uid : this->gui_graph_state.interact.modules_remove_group_uids) {
                    for (auto& remove_group_ptr : this->GetGroups()) {
                        if (remove_group_ptr->ContainsModule(module_uid)) {
                            this->RemoveGroupModule(remove_group_ptr->UID(), module_uid, true);
                        }
                    }
                }
                reset_state = true;
            }
            // Create new interface slot for call slot ----------------------------
            ImGuiID callslot_uid = this->gui_graph_state.interact.callslot_add_group_uid.first;
            if (callslot_uid != GUI_INVALID_ID) {
                CallSlotPtr_t callslot_ptr = nullptr;
                for (auto& mod : this->Modules()) {
                    for (auto& callslot_map : mod->CallSlots()) {
                        for (auto& callslot : callslot_map.second) {
                            if (callslot->UID() == callslot_uid) {
                                callslot_ptr = callslot;
                            }
                        }
                    }
                }
                if (callslot_ptr != nullptr) {
                    ImGuiID module_uid = this->gui_graph_state.interact.callslot_add_group_uid.second;
                    if (module_uid != GUI_INVALID_ID) {
                        for (auto& group : this->GetGroups()) {
                            if (group->ContainsModule(module_uid)) {
                                group->AddInterfaceSlot(callslot_ptr, false);
                                this->ForceSetDirty();
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
                for (auto& mod : this->Modules()) {
                    for (auto& callslot_map : mod->CallSlots()) {
                        for (auto& callslot : callslot_map.second) {
                            if (callslot->UID() == callslot_uid) {
                                callslot_ptr = callslot;
                            }
                        }
                    }
                }
                ImGuiID module_uid = this->gui_graph_state.interact.callslot_remove_group_uid.second;
                if (module_uid != GUI_INVALID_ID) {
                    for (auto& group : this->GetGroups()) {
                        if (group->ContainsModule(module_uid)) {
                            if (group->InterfaceSlot_RemoveCallSlot(callslot_uid, true)) {
                                this->ForceSetDirty();
                                // Delete call which are connected outside the group
                                std::vector<ImGuiID> call_uids;
                                CallSlotType other_type = (callslot_ptr->Type() == CallSlotType::CALLEE)
                                                              ? (CallSlotType::CALLER)
                                                              : (CallSlotType::CALLEE);
                                for (auto& call_ptr : callslot_ptr->GetConnectedCalls()) {
                                    CallSlotPtr_t other_callslot_ptr = call_ptr->CallSlotPtr(other_type);
                                    if (other_callslot_ptr->IsParentModuleConnected()) {
                                        if (other_callslot_ptr->GetParentModule()->GroupUID() != group->UID()) {
                                            call_uids.emplace_back(call_ptr->UID());
                                        }
                                    }
                                }
                                for (auto& call_uid : call_uids) {
                                    this->DeleteCall(call_uid);
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
                    this->gui_graph_state.hotkeys[HOTKEY_CONFIGURATOR_DELETE_GRAPH_ITEM].is_pressed)) {
                if (!this->gui_graph_state.interact.modules_selected_uids.empty()) {
                    for (auto& module_uid : this->gui_graph_state.interact.modules_selected_uids) {
                        this->DeleteModule(module_uid);
                    }
                }
                if (this->gui_graph_state.interact.call_selected_uid != GUI_INVALID_ID) {
                    this->DeleteCall(this->gui_graph_state.interact.call_selected_uid);
                }
                if (this->gui_graph_state.interact.group_selected_uid != GUI_INVALID_ID) {
                    this->DeleteGroup(this->gui_graph_state.interact.group_selected_uid);
                }
                if (this->gui_graph_state.interact.interfaceslot_selected_uid != GUI_INVALID_ID) {
                    for (auto& group_ptr : this->GetGroups()) {
                        if (auto interfaceslot_ptr = group_ptr->InterfaceSlotPtr(
                                this->gui_graph_state.interact.interfaceslot_selected_uid)) {
                            // Delete all calls connected
                            std::vector<ImGuiID> call_uids;
                            for (auto& callslot_ptr : interfaceslot_ptr->CallSlots()) {
                                for (auto& call_ptr : callslot_ptr->GetConnectedCalls()) {
                                    auto caller = call_ptr->CallSlotPtr(CallSlotType::CALLER);
                                    auto callee = call_ptr->CallSlotPtr(CallSlotType::CALLEE);
                                    if (caller->IsParentModuleConnected() && callee->IsParentModuleConnected()) {
                                        if (caller->GetParentModule()->GroupUID() !=
                                            callee->GetParentModule()->GroupUID()) {
                                            call_uids.emplace_back(call_ptr->UID());
                                        }
                                    }
                                }
                            }
                            for (auto& call_uid : call_uids) {
                                this->DeleteCall(call_uid);
                            }
                            interfaceslot_ptr.reset();

                            group_ptr->DeleteInterfaceSlot(this->gui_graph_state.interact.interfaceslot_selected_uid);
                            this->ForceSetDirty();
                        }
                    }
                }

                reset_state = true;
            }
            // Delete empty group(s) ----------------------------------------------
            std::vector<ImGuiID> delete_empty_groups_uids;
            for (auto& group_ptr : this->GetGroups()) {
                if (group_ptr->Modules().empty()) {
                    delete_empty_groups_uids.emplace_back(group_ptr->UID());
                }
            }
            for (auto& group_uid : delete_empty_groups_uids) {
                if (this->DeleteGroup(group_uid)) {
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
            }

            // Layout graph -------------------------------------------------------
            /// One frame delay required for making sure canvas data is completely updated previously
            if (this->gui_graph_layout > 0) {
                if (this->gui_graph_layout > 1) {
                    this->layout_graph();
                    this->gui_graph_layout = 0;
                } else {
                    this->gui_graph_layout++;
                }
            }
            // Layout modules of selected group -----------------------------------
            if (this->gui_graph_state.interact.group_layout) {
                for (auto& group_ptr : this->GetGroups()) {
                    if (group_ptr->UID() == this->gui_graph_state.interact.group_selected_uid) {
                        ImVec2 init_position = ImVec2(FLT_MAX, FLT_MAX);
                        for (auto& module_ptr : group_ptr->Modules()) {
                            init_position.x = std::min(module_ptr->Position().x, init_position.x);
                            init_position.y = std::min(module_ptr->Position().y, init_position.y);
                        }
                        this->layout(group_ptr->Modules(), GroupPtrVector_t(), init_position);
                    }
                }
                this->gui_graph_state.interact.group_layout = false;
                this->gui_update = true;
            }
            // Layout selelected modules ------------------------------------------
            if (this->gui_graph_state.interact.modules_layout) {
                ImVec2 init_position = ImVec2(FLT_MAX, FLT_MAX);
                ModulePtrVector_t selected_modules;
                for (auto& module_ptr : this->Modules()) {
                    for (auto& selected_module_uid : this->gui_graph_state.interact.modules_selected_uids) {
                        if (module_ptr->UID() == selected_module_uid) {
                            init_position.x = std::min(module_ptr->Position().x, init_position.x);
                            init_position.y = std::min(module_ptr->Position().y, init_position.y);
                            selected_modules.emplace_back(module_ptr);
                        }
                    }
                }
                this->layout(selected_modules, GroupPtrVector_t(), init_position);
                this->gui_graph_state.interact.modules_layout = false;
            }
            // Set delete flag if tab was closed ----------------------------------
            if (!open_value) {
                state.graph_delete = true;
                state.graph_selected_uid = this->uid;
            }

            // Propoagate unhandeled hotkeys back to configurator state -----------
            state.hotkeys = this->gui_graph_state.hotkeys;

            // Rename pop-up ------------------------------------------------------
            if (this->gui_rename_popup.Rename("Rename Project", popup_rename, this->name)) {
                this->ForceSetDirty();
            }

            // Module Parameter Child -------------------------------------------------
            // Choose single selected view module
            ModulePtr_t selected_mod_ptr;
            if (this->gui_graph_state.interact.modules_selected_uids.size() == 1) {
                for (auto& mod : this->Modules()) {
                    if (this->gui_graph_state.interact.modules_selected_uids.front() == mod->UID()) {
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
                        ImGui::SetNextWindowPos(this->gui_graph_state.interact.module_param_child_position);
                        ImGui::SetNextWindowSize(ImVec2(10.0f, 10.0f));
                    }
                    auto popup_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar |
                                       ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar |
                                       ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove;
                    if (ImGui::BeginPopup(pop_up_id.c_str(), popup_flags)) {
                        // Draw parameters
                        selected_mod_ptr->DrawParameters(
                            "", false, false, Parameter::WidgetScope::LOCAL, nullptr, GUI_INVALID_ID, nullptr);

                        ImVec2 popup_pos = ImGui::GetWindowPos();
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
                        if (!param_popup_open &&
                            ((ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !module_parm_child_popup_hovered) ||
                                ImGui::IsKeyPressed(ImGuiKey_Escape))) {

                            this->gui_graph_state.interact.module_param_child_position = ImVec2(-1.0f, -1.0f);
                            // Reset module selection to prevent irrgular dragging
                            this->gui_graph_state.interact.modules_selected_uids.clear();
                            ImGui::CloseCurrentPopup();
                        }

                        // Save actual position since pop-up might be moved away from right edge
                        this->gui_graph_state.interact.module_param_child_position = popup_pos;

                        ImGui::EndPopup();
                    }
                }
            }

            ImGui::EndTabItem();

        } else {
            this->gui_canvas_hovered = false;
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


void Graph::DrawGlobalParameterWidgets(
    megamol::core::utility::PickingBuffer& picking_buffer, std::shared_ptr<TransferFunctionEditor> win_tfeditor_ptr) {

    for (auto& module_ptr : this->Modules()) {
        module_ptr->DrawParameters(
            "", false, false, Parameter::WidgetScope::GLOBAL, win_tfeditor_ptr, GUI_INVALID_ID, &picking_buffer);
    }
}


void megamol::gui::Graph::draw_menu(GraphState_t& state) {

    ImGuiStyle& style = ImGui::GetStyle();
    auto button_size = ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight());

    float child_height = ImGui::GetFrameHeightWithSpacing();
    auto child_flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NavFlattened | ImGuiWindowFlags_MenuBar;
    ImGui::BeginChild("graph_menu", ImVec2(0.0f, child_height), false, child_flags);

    ImGui::BeginMenuBar();

    // RUNNING
    if (megamol::gui::ButtonWidgets::OptionButton(ButtonWidgets::ButtonStyle::POINT_CIRCLE, "graph_running_button",
            ((this->running) ? ("Running") : ("Run")), this->running, this->running)) {
        if (!this->running) {
            state.new_running_graph_uid = this->uid;
        }
    }
    ImGui::Separator();

    // Choose single selected view module
    ModulePtr_t selected_mod_ptr;
    if (this->gui_graph_state.interact.modules_selected_uids.size() == 1) {
        for (auto& mod : this->Modules()) {
            if ((this->gui_graph_state.interact.modules_selected_uids.front() == mod->UID()) && (mod->IsView())) {
                selected_mod_ptr = mod;
            }
        }
    }
    // Graph Entry Checkbox
    const float min_text_width = 3.0f * ImGui::GetFrameHeightWithSpacing();
    if (selected_mod_ptr == nullptr) {
        gui_utils::PushReadOnly();
        bool is_graph_entry = false;
        this->gui_current_graph_entry_name.clear();
        megamol::gui::ButtonWidgets::ToggleButton("Entry", is_graph_entry);
        gui_utils::PopReadOnly();
    } else {
        bool is_graph_entry = selected_mod_ptr->IsGraphEntry();
        if (megamol::gui::ButtonWidgets::ToggleButton("Entry", is_graph_entry)) {
            Graph::QueueData queue_data;
            if (is_graph_entry) {
                // Remove all graph entries
                for (auto& module_ptr : this->Modules()) {
                    if (module_ptr->IsView() && module_ptr->IsGraphEntry()) {
                        module_ptr->SetGraphEntryName("");
                        queue_data.name_id = module_ptr->FullName();
                        this->PushSyncQueue(Graph::QueueAction::REMOVE_GRAPH_ENTRY, queue_data);
                    }
                }
                // Add new graph entry
                selected_mod_ptr->SetGraphEntryName(this->GenerateUniqueGraphEntryName());
                queue_data.name_id = selected_mod_ptr->FullName();
                this->PushSyncQueue(Graph::QueueAction::CREATE_GRAPH_ENTRY, queue_data);
            } else {
                selected_mod_ptr->SetGraphEntryName("");
                queue_data.name_id = selected_mod_ptr->FullName();
                this->PushSyncQueue(Graph::QueueAction::REMOVE_GRAPH_ENTRY, queue_data);
            }
        }
    }
    this->gui_tooltip.ToolTip("Set graph entry state of selected view module", ImGui::GetItemID(), 0.5f, 5.0f);
    ImGui::Separator();

    // GRAPH LAYOUT
    if (ImGui::Button("Layout")) {
        this->gui_graph_layout = 1;
    }
    this->gui_tooltip.ToolTip("Align modules of graph in regular layout", ImGui::GetItemID(), 0.5f, 5.0f);
    ImGui::Separator();

    // GRID
    if (megamol::gui::ButtonWidgets::OptionButton(
            ButtonWidgets::ButtonStyle::GRID, "grid_option_button", "Grid", this->gui_show_grid, false)) {
        this->gui_show_grid = !this->gui_show_grid;
    }
    ImGui::Separator();

    // Module and Call LABELS
    auto cursor_pos = ImGui::GetCursorScreenPos();
    if (megamol::gui::ButtonWidgets::OptionButton(
            ButtonWidgets::ButtonStyle::LINES, "labels_option_button", "Labels", false, false)) {
        ImGui::OpenPopup("module_call_labels_button_context");
        ImGui::SetNextWindowPos(cursor_pos + ImVec2(0.0f, ImGui::GetFrameHeight()));
    }
    if (ImGui::BeginPopup("module_call_labels_button_context", ImGuiPopupFlags_MouseButtonLeft)) {
        // MODULES
        if (ImGui::BeginMenu("Modules")) {
            if (ImGui::MenuItem("Name", nullptr, &this->gui_graph_state.interact.module_show_label)) {
                this->gui_update = true;
            }
            if (ImGui::MenuItem("Slots", nullptr, &this->gui_graph_state.interact.callslot_show_label)) {
                this->gui_update = true;
            }
            ImGui::EndMenu();
        }
        // CALLS
        if (ImGui::BeginMenu("Calls")) {
            if (ImGui::MenuItem("Name", nullptr, &this->gui_graph_state.interact.call_show_label)) {
                this->gui_update = true;
            }
            if (ImGui::MenuItem("Slots", nullptr, &this->gui_graph_state.interact.call_show_slots_label)) {
                this->gui_update = true;
            }
            ImGui::EndMenu();
        }
        ImGui::EndPopup();
    }
    ImGui::Separator();

    // Coloring
    cursor_pos = ImGui::GetCursorScreenPos();
    if (megamol::gui::ButtonWidgets::OptionButton(
            ButtonWidgets::ButtonStyle::POINTS, "coloring_option_button", "Coloring", false, false)) {
        ImGui::OpenPopup("coloring_button_context");
        ImGui::SetNextWindowPos(cursor_pos + ImVec2(0.0f, ImGui::GetFrameHeight()));
    }
    if (ImGui::BeginPopup("coloring_button_context", ImGuiPopupFlags_MouseButtonLeft)) {
        if (ImGui::BeginMenu("Calls")) {
            ImGui::TextDisabled("Mode");
            if (ImGui::RadioButton("Per module coloring", (this->gui_graph_state.interact.call_coloring_mode == 1))) {
                this->gui_graph_state.interact.call_coloring_mode = 1;
            }
            if (ImGui::RadioButton(
                    "Per call slot coloring", (this->gui_graph_state.interact.call_coloring_mode == 0))) {
                this->gui_graph_state.interact.call_coloring_mode = 0;
            }
            ImGui::Separator();
            ImGui::TextDisabled("Map");
            if (ImGui::RadioButton("Monochrome", (this->gui_graph_state.interact.call_coloring_map == 0))) {
                this->gui_graph_state.interact.call_coloring_map = 0;
            }
            if (ImGui::RadioButton("Set3Map(12)", (this->gui_graph_state.interact.call_coloring_map == 1))) {
                this->gui_graph_state.interact.call_coloring_map = 1;
            }
            if (ImGui::RadioButton("PairedMap(12)", (this->gui_graph_state.interact.call_coloring_map == 2))) {
                this->gui_graph_state.interact.call_coloring_map = 2;
            }
            ImGui::EndMenu();
        }
        ImGui::EndPopup();
    }
    ImGui::Separator();

    // SCROLLING
    const float scroll_fac = 10.0f;
    ImGui::Text(
        "Scrolling: %.2f, %.2f", this->gui_graph_state.canvas.scrolling.x, this->gui_graph_state.canvas.scrolling.y);
    ImGui::TextUnformatted("H:");
    if (ImGui::Button("+###hor_incr_scrolling", button_size)) {
        this->gui_graph_state.canvas.scrolling.x += scroll_fac;
        this->gui_update = true;
    }
    if (ImGui::Button("-###hor_decr_scrolling", button_size)) {
        this->gui_graph_state.canvas.scrolling.x -= scroll_fac;
        this->gui_update = true;
    }
    ImGui::TextUnformatted("V:");
    if (ImGui::Button("+###vert_incr_scrolling", button_size)) {
        this->gui_graph_state.canvas.scrolling.y += scroll_fac;
        this->gui_update = true;
    }
    if (ImGui::Button("-###vert_decr_scrolling", button_size)) {
        this->gui_graph_state.canvas.scrolling.y -= scroll_fac;
        this->gui_update = true;
    }
    if (ImGui::Button("Reset###reset_scrolling")) {
        this->gui_graph_state.canvas.scrolling = ImVec2(0.0f, 0.0f);
        this->gui_update = true;
    }
    this->gui_tooltip.Marker("Middle Mouse Button");
    ImGui::Separator();

    // ZOOMING
    ImGui::Text("Zooming: %.2f", this->gui_graph_state.canvas.zooming);
    if (ImGui::Button("+###incr_zooming", button_size)) {
        this->gui_increment_zooming = true;
    }
    if (ImGui::Button("-###decr_zooming", button_size)) {
        this->gui_decrement_zooming = true;
    }
    if (ImGui::Button("Reset###reset_zooming")) {
        this->gui_reset_zooming = true;
    }
    this->gui_tooltip.Marker("Mouse Wheel");
    ImGui::Separator();

    ImGui::EndMenuBar();

    ImGui::EndChild();
}


void megamol::gui::Graph::draw_canvas(ImVec2 position, ImVec2 size, GraphState_t& state) {

    ImGuiIO& io = ImGui::GetIO();

    // Load font for canvas
    ImFont* gui_font_ptr = io.FontDefault;
    ImFont* graph_font_ptr = nullptr;
    auto scalings_count = static_cast<int>(state.graph_zoom_font_scalings.size());
    if (scalings_count == 0) {
        throw std::invalid_argument("Bug: Array for graph fonts is empty.");
    } else if (scalings_count == 1) {
        graph_font_ptr = io.Fonts->Fonts[0];
        this->gui_current_font_scaling = state.graph_zoom_font_scalings[0];
    } else {
        for (int i = 0; i < scalings_count; i++) {
            bool apply = false;
            if (i == 0) {
                if (this->gui_graph_state.canvas.zooming <= state.graph_zoom_font_scalings[i]) {
                    apply = true;
                }
            } else if (i == (scalings_count - 1)) {
                if (this->gui_graph_state.canvas.zooming >= state.graph_zoom_font_scalings[i]) {
                    apply = true;
                }
            } else {
                if ((state.graph_zoom_font_scalings[i - 1] < this->gui_graph_state.canvas.zooming) &&
                    (this->gui_graph_state.canvas.zooming < state.graph_zoom_font_scalings[i + 1])) {
                    apply = true;
                }
            }
            if (apply) {
                graph_font_ptr = io.Fonts->Fonts[i];
                this->gui_current_font_scaling = state.graph_zoom_font_scalings[i];
                break;
            }
        }
    }
    if ((graph_font_ptr == nullptr) || (gui_font_ptr == nullptr)) {
        throw std::invalid_argument("Bug: Required fonts not available.");
    }
    this->gui_graph_state.canvas.gui_font_ptr = gui_font_ptr;
    ImGui::PushFont(graph_font_ptr);

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(1, 1));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    auto child_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove;
    ImGui::SetNextWindowPos(position);
    ImGui::BeginChild("canvas", size, true, child_flags);

    this->gui_canvas_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_None); // Ignores Pop-Ups like Context-Menu

    // UPDATE CANVAS -----------------------------------------------------------

    // Update canvas position
    ImVec2 new_position = ImGui::GetWindowPos();
    if (this->gui_graph_state.canvas.position != new_position) {
        this->gui_update = true;
    }
    this->gui_graph_state.canvas.position = new_position;
    // Update canvas size
    ImVec2 new_size = ImGui::GetWindowSize();
    if (this->gui_graph_state.canvas.size != new_size) {
        this->gui_update = true;
    }
    this->gui_graph_state.canvas.size = new_size;
    // Update canvas offset
    ImVec2 new_offset = this->gui_graph_state.canvas.position +
                        (this->gui_graph_state.canvas.scrolling * this->gui_graph_state.canvas.zooming);
    if (this->gui_graph_state.canvas.offset != new_offset) {
        this->gui_update = true;
    }
    this->gui_graph_state.canvas.offset = new_offset;

    // Update position and size of modules (and call slots) and groups.
    if (this->gui_update) {
        for (auto& mod : this->Modules()) {
            mod->InstantUpdate(this->gui_graph_state);
        }
        for (auto& group : this->GetGroups()) {
            group->Update(this->gui_graph_state.canvas);
        }
        this->gui_update = false;
    }

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    assert(drawList != nullptr);

    drawList->PushClipRect(this->gui_graph_state.canvas.position,
        this->gui_graph_state.canvas.position + this->gui_graph_state.canvas.size, true);
    ImGui::PushClipRect(this->gui_graph_state.canvas.position,
        this->gui_graph_state.canvas.position + this->gui_graph_state.canvas.size, true);

    // GRID --------------------------------------
    if (this->gui_show_grid) {
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
        auto phase = static_cast<PresentPhase>(p);

        // 1] GROUPS and INTERFACE SLOTS --------------------------------------
        for (auto& group_ptr : this->GetGroups()) {
            group_ptr->Draw(phase, this->gui_graph_state);

            // 2] MODULES and CALL SLOTS (of group) ---------------------------
            for (auto& module_ptr : this->Modules()) {
                if (module_ptr->GroupUID() == group_ptr->UID()) {
                    module_ptr->Draw(phase, this->gui_graph_state);
                }
            }

            // 3] CALLS (of group) --------------------------------------------
            for (auto& module_ptr : this->Modules()) {
                if (module_ptr->GroupUID() == group_ptr->UID()) {
                    /// Check only for calls of caller slots for considering each call only once
                    for (auto& callslots_ptr : module_ptr->CallSlots(CallSlotType::CALLER)) {
                        for (auto& call_ptr : callslots_ptr->GetConnectedCalls()) {

                            bool caller_group = false;
                            auto caller_ptr = call_ptr->CallSlotPtr(CallSlotType::CALLER);
                            if (caller_ptr->IsParentModuleConnected()) {
                                if (caller_ptr->GetParentModule()->GroupUID() == group_ptr->UID()) {
                                    caller_group = true;
                                }
                            }
                            bool callee_group = false;
                            auto callee_ptr = call_ptr->CallSlotPtr(CallSlotType::CALLER);
                            if (callee_ptr->IsParentModuleConnected()) {
                                if (callee_ptr->GetParentModule()->GroupUID() == group_ptr->UID()) {
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
        for (auto& module_ptr : this->Modules()) {
            if (module_ptr->GroupUID() == GUI_INVALID_ID) {
                module_ptr->Draw(phase, this->gui_graph_state);
            }
        }

        // 5] CALLS  (non group members) --------------------------------------
        /// (connected to call slots which are not part of module which is group member)
        for (auto& call_ptr : this->Calls()) {
            bool caller_group = false;
            auto caller_ptr = call_ptr->CallSlotPtr(CallSlotType::CALLER);
            if (caller_ptr->IsParentModuleConnected()) {
                if (caller_ptr->GetParentModule()->GroupUID() != GUI_INVALID_ID) {
                    caller_group = true;
                }
            }
            bool callee_group = false;
            auto callee_ptr = call_ptr->CallSlotPtr(CallSlotType::CALLER);
            if (callee_ptr->IsParentModuleConnected()) {
                if (callee_ptr->GetParentModule()->GroupUID() != GUI_INVALID_ID) {
                    callee_group = true;
                }
            }
            if ((!caller_group) && (!callee_group)) {
                call_ptr->Draw(phase, this->gui_graph_state);
            }
        }
    }

    // Multiselection ----------------------------
    this->draw_canvas_multiselection();

    // Dragged CALL ------------------------------
    this->draw_canvas_dragged_call();

    drawList->PopClipRect();
    ImGui::PopClipRect();

    // Zooming and Scaling ----------------------
    // Must be checked inside this canvas child window!
    // Check at the end of drawing for being applied in next frame when font scaling matches zooming.
    if ((ImGui::IsWindowHovered() && !ImGui::IsAnyItemActive()) || this->gui_reset_zooming ||
        this->gui_increment_zooming || this->gui_decrement_zooming) {

        // Scrolling (2 = Middle Mouse Button)
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
            this->gui_graph_state.canvas.scrolling = this->gui_graph_state.canvas.scrolling +
                                                     ImGui::GetIO().MouseDelta / this->gui_graph_state.canvas.zooming;
            this->gui_update = true;
        }

        // Zooming (Mouse Wheel) + Reset
        float mouse_wheel_value = megamol::gui::gui_mouse_wheel;
        if (((isfinite(mouse_wheel_value)) && (mouse_wheel_value != 0)) || this->gui_reset_zooming ||
            this->gui_increment_zooming || this->gui_decrement_zooming) {

            float last_zooming = this->gui_graph_state.canvas.zooming;
            // Center mouse position as init value
            ImVec2 current_mouse_pos =
                this->gui_graph_state.canvas.offset -
                (this->gui_graph_state.canvas.position + this->gui_graph_state.canvas.size * 0.5f);
            const float zoom_fac = 1.1f; // = 10%

            if (this->gui_reset_zooming) {
                this->gui_graph_state.canvas.zooming = 1.0f;
                last_zooming = 1.0f;
                this->gui_reset_zooming = false;
            } else if (this->gui_increment_zooming) {
                this->gui_graph_state.canvas.zooming *= zoom_fac;
                this->gui_increment_zooming = false;
            } else if (this->gui_decrement_zooming) {
                this->gui_graph_state.canvas.zooming /= zoom_fac;
                this->gui_decrement_zooming = false;
            } else if ((isfinite(mouse_wheel_value)) && (mouse_wheel_value != 0)) {
                const float factor = this->gui_graph_state.canvas.zooming / 10.0f;
                this->gui_graph_state.canvas.zooming += (mouse_wheel_value * factor);
                current_mouse_pos = this->gui_graph_state.canvas.offset - ImGui::GetMousePos();
            }
            // Limit zooming
            const float zooming_minimum = 0.01f;
            this->gui_graph_state.canvas.zooming = (this->gui_graph_state.canvas.zooming <= zooming_minimum)
                                                       ? zooming_minimum
                                                       : (this->gui_graph_state.canvas.zooming);
            // Compensate zooming shift of origin
            ImVec2 scrolling_diff = (this->gui_graph_state.canvas.scrolling * last_zooming) -
                                    (this->gui_graph_state.canvas.scrolling * this->gui_graph_state.canvas.zooming);
            this->gui_graph_state.canvas.scrolling += (scrolling_diff / this->gui_graph_state.canvas.zooming);
            // Move origin away from mouse position
            ImVec2 new_mouse_position = (current_mouse_pos / last_zooming) * this->gui_graph_state.canvas.zooming;
            this->gui_graph_state.canvas.scrolling +=
                ((new_mouse_position - current_mouse_pos) / this->gui_graph_state.canvas.zooming);

            this->gui_update = true;
        }
    }

    ImGui::EndChild();

    // FONT scaling
    float font_scaling = this->gui_graph_state.canvas.zooming / this->gui_current_font_scaling;
    // Update when scaling of font has changed due to project tab switching
    if (ImGui::GetFont()->Scale != font_scaling) {
        this->gui_update = true;
    }
    // Font scaling is applied next frame after ImGui::Begin()
    // Font for graph should not be the currently used font of the gui.
    ImGui::GetFont()->Scale = font_scaling;

    ImGui::PopFont();
}


void megamol::gui::Graph::draw_parameters(ImVec2 position, ImVec2 size) {

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(1, 1));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    auto child_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove;
    ImGui::SetNextWindowPos(position);
    ImGui::BeginChild("parameters", size, false, child_flags);

    // ------------------------------------------------------------------------

    float search_child_height = ImGui::GetFrameHeightWithSpacing() * 3.5f;
    child_flags =
        ImGuiWindowFlags_AlwaysUseWindowPadding | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NavFlattened;
    ImGui::BeginChild("parameter_search_child", ImVec2(size.x, search_child_height), false, child_flags);

    ImGui::TextUnformatted("Parameters");
    this->gui_tooltip.Marker("Show parameters of selected modules");
    ImGui::Separator();

    // Mode
    megamol::gui::ButtonWidgets::ExtendedModeButton(
        "parameter_search_child", this->gui_graph_state.interact.parameters_extended_mode);

    // Parameter Search
    if (this->gui_graph_state.hotkeys[HOTKEY_CONFIGURATOR_PARAMETER_SEARCH].is_pressed) {
        this->gui_search_widget.SetSearchFocus();
    }
    std::string help_text = "[" +
                            this->gui_graph_state.hotkeys[HOTKEY_CONFIGURATOR_PARAMETER_SEARCH].keycode.ToString() +
                            "] Set keyboard focus to search input field.\n"
                            "Case insensitive substring search in module and parameter names.";
    this->gui_search_widget.Widget("graph_parameter_search", help_text);


    ImGui::EndChild();

    // ------------------------------------------------------------------------

    child_flags = ImGuiWindowFlags_AlwaysVerticalScrollbar | ImGuiWindowFlags_NavFlattened |
                  ImGuiWindowFlags_AlwaysUseWindowPadding;
    ImGui::BeginChild("parameter_param_frame_child", ImVec2(size.x, 0.0f), false, child_flags);

    if (!this->gui_graph_state.interact.modules_selected_uids.empty()) {
        // Get module groups
        std::map<std::string, std::vector<ModulePtr_t>> group_map;
        for (auto& module_uid : this->gui_graph_state.interact.modules_selected_uids) {
            // Get pointer to currently selected module(s)
            if (auto module_ptr = this->GetModule(module_uid)) {
                auto group_name = module_ptr->GroupName();
                if (!group_name.empty()) {
                    group_map["::" + group_name].emplace_back(module_ptr);
                } else {
                    group_map[""].emplace_back(module_ptr);
                }
            }
        }
        for (auto& group : group_map) {
            std::string search_string = this->gui_search_widget.GetSearchString();
            bool indent = false;
            bool group_header_open = group.first.empty();
            if (!group_header_open) {
                group_header_open =
                    gui_utils::GroupHeader(megamol::gui::HeaderType::MODULE_GROUP, group.first, search_string);
                indent = true;
                ImGui::Indent();
            }
            if (group_header_open) {
                for (auto& module_ptr : group.second) {
                    ImGui::PushID(static_cast<int>(module_ptr->UID()));
                    std::string module_label = module_ptr->FullName();

                    // Draw module header
                    bool module_header_open =
                        gui_utils::GroupHeader(megamol::gui::HeaderType::MODULE, module_label, search_string);
                    // Module description as hover tooltip
                    this->gui_tooltip.ToolTip(
                        module_ptr->Description(), ImGui::GetID(module_label.c_str()), 0.5f, 5.0f);

                    // Draw parameters
                    if (module_header_open) {
                        module_ptr->DrawParameters(search_string,
                            this->gui_graph_state.interact.parameters_extended_mode, true,
                            Parameter::WidgetScope::LOCAL, nullptr, GUI_INVALID_ID, nullptr);
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

    // ------------------------------------------------------------------------

    ImGui::EndChild();
    ImGui::PopStyleVar(2);
}


void megamol::gui::Graph::draw_canvas_grid() const {

    ImGuiStyle& style = ImGui::GetStyle();

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    assert(drawList != nullptr);

    /// COLOR_GRID
    const ImU32 COLOR_GRID = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Border]);

    const float GRID_SIZE = (64.0f * megamol::gui::gui_scaling.Get()) * this->gui_graph_state.canvas.zooming;
    ImVec2 relative_offset = this->gui_graph_state.canvas.offset - this->gui_graph_state.canvas.position;

    for (float x = fmodf(relative_offset.x, GRID_SIZE); x < this->gui_graph_state.canvas.size.x; x += GRID_SIZE) {
        drawList->AddLine(ImVec2(x, 0.0f) + this->gui_graph_state.canvas.position,
            ImVec2(x, this->gui_graph_state.canvas.size.y) + this->gui_graph_state.canvas.position, COLOR_GRID);
    }

    for (float y = fmodf(relative_offset.y, GRID_SIZE); y < this->gui_graph_state.canvas.size.y; y += GRID_SIZE) {
        drawList->AddLine(ImVec2(0.0f, y) + this->gui_graph_state.canvas.position,
            ImVec2(this->gui_graph_state.canvas.size.x, y) + this->gui_graph_state.canvas.position, COLOR_GRID);
    }
}


void megamol::gui::Graph::draw_canvas_dragged_call() {

    if (const ImGuiPayload* payload = ImGui::GetDragDropPayload()) {
        if (payload->IsDataType(GUI_DND_CALLSLOT_UID_TYPE)) {
            auto* selected_slot_uid_ptr = (ImGuiID*)payload->Data;
            if (selected_slot_uid_ptr == nullptr) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Pointer to drag and drop payload data is nullptr. [%s, %s, line %d]\n", __FILE__,
                    __FUNCTION__, __LINE__);
                return;
            }

            ImGuiStyle& style = ImGui::GetStyle();
            ImDrawList* drawList = ImGui::GetWindowDrawList();

            /// COLOR_CALL_CURVE
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
                for (auto& module_ptr : this->Modules()) {
                    if (auto callslot_ptr = module_ptr->CallSlotPtr((*selected_slot_uid_ptr))) {
                        selected_callslot_ptr = callslot_ptr;
                    }
                }
                if (selected_callslot_ptr != nullptr) {
                    p1 = selected_callslot_ptr->Position();
                    found_valid_slot = true;
                }
                if (!found_valid_slot) {
                    InterfaceSlotPtr_t selected_interfaceslot_ptr;
                    for (auto& group_ptr : this->GetGroups()) {
                        if (auto interfaceslot_ptr = group_ptr->InterfaceSlotPtr((*selected_slot_uid_ptr))) {
                            selected_interfaceslot_ptr = interfaceslot_ptr;
                        }
                    }
                    if (selected_interfaceslot_ptr != nullptr) {
                        p1 = selected_interfaceslot_ptr->Position();
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
                        drawList->AddBezierCubic(p1, p1 + ImVec2(+50, 0), p2 + ImVec2(-50, 0), p2, COLOR_CALL_CURVE,
                            GUI_LINE_THICKNESS * this->gui_graph_state.canvas.zooming);
                    }
                }
            }
        }
    }
}


void megamol::gui::Graph::draw_canvas_multiselection() {

    bool no_graph_item_selected = ((this->gui_graph_state.interact.callslot_selected_uid == GUI_INVALID_ID) &&
                                   (this->gui_graph_state.interact.call_selected_uid == GUI_INVALID_ID) &&
                                   (this->gui_graph_state.interact.modules_selected_uids.empty()) &&
                                   (this->gui_graph_state.interact.interfaceslot_selected_uid == GUI_INVALID_ID) &&
                                   (this->gui_graph_state.interact.group_selected_uid == GUI_INVALID_ID));

    if (no_graph_item_selected && ImGui::IsWindowHovered() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {

        this->gui_multiselect_end_pos = ImGui::GetMousePos();
        this->gui_multiselect_done = true;

        ImGuiStyle& style = ImGui::GetStyle();
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        assert(drawList != nullptr);

        /// COLOR_MULTISELECT_BACKGROUND
        ImVec4 tmpcol = style.Colors[ImGuiCol_FrameBg];
        tmpcol.w = 0.2f; // alpha
        const ImU32 COLOR_MULTISELECT_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);
        /// COLOR_MULTISELECT_BORDER
        const ImU32 COLOR_MULTISELECT_BORDER = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Border]);

        drawList->AddRectFilled(this->gui_multiselect_start_pos, this->gui_multiselect_end_pos,
            COLOR_MULTISELECT_BACKGROUND, GUI_RECT_CORNER_RADIUS, ImDrawFlags_RoundCornersAll);

        float border = (1.0f * megamol::gui::gui_scaling.Get());
        drawList->AddRect(this->gui_multiselect_start_pos, this->gui_multiselect_end_pos, COLOR_MULTISELECT_BORDER,
            GUI_RECT_CORNER_RADIUS, ImDrawFlags_RoundCornersAll, border);
    } else if (this->gui_multiselect_done && ImGui::IsWindowHovered() &&
               ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        ImVec2 outer_rect_min = ImVec2(std::min(this->gui_multiselect_start_pos.x, this->gui_multiselect_end_pos.x),
            std::min(this->gui_multiselect_start_pos.y, this->gui_multiselect_end_pos.y));
        ImVec2 outer_rect_max = ImVec2(std::max(this->gui_multiselect_start_pos.x, this->gui_multiselect_end_pos.x),
            std::max(this->gui_multiselect_start_pos.y, this->gui_multiselect_end_pos.y));
        ImVec2 inner_rect_min, inner_rect_max;
        ImVec2 module_size;
        this->gui_graph_state.interact.modules_selected_uids.clear();
        for (auto& module_ptr : this->Modules()) {
            bool group_member = (module_ptr->GroupUID() != GUI_INVALID_ID);
            if (!group_member || (group_member && !module_ptr->IsHidden())) {
                module_size = module_ptr->Size() * this->gui_graph_state.canvas.zooming;
                inner_rect_min =
                    this->gui_graph_state.canvas.offset + module_ptr->Position() * this->gui_graph_state.canvas.zooming;
                inner_rect_max = inner_rect_min + module_size;
                if (((outer_rect_min.x < inner_rect_max.x) && (outer_rect_max.x > inner_rect_min.x) &&
                        (outer_rect_min.y < inner_rect_max.y) && (outer_rect_max.y > inner_rect_min.y))) {
                    this->gui_graph_state.interact.modules_selected_uids.emplace_back(module_ptr->UID());
                }
            }
        }
        this->gui_multiselect_done = false;
    } else {
        this->gui_multiselect_start_pos = ImGui::GetMousePos();
    }
}


void megamol::gui::Graph::layout_graph() {

    ImVec2 init_position = megamol::gui::Module::GetDefaultModulePosition(this->gui_graph_state.canvas);

    /// 1] Layout all grouped modules
    for (auto& group_ptr : this->GetGroups()) {
        this->layout(group_ptr->Modules(), GroupPtrVector_t(), init_position);
        group_ptr->Update(this->gui_graph_state.canvas);
    }

    /// 2] Layout ungrouped modules and groups
    ModulePtrVector_t ungrouped_modules;
    for (auto& module_ptr : this->Modules()) {
        if (module_ptr->GroupUID() == GUI_INVALID_ID) {
            ungrouped_modules.emplace_back(module_ptr);
        }
    }
    this->layout(ungrouped_modules, this->GetGroups(), init_position);
}


void megamol::gui::Graph::layout(const ModulePtrVector_t& ms, const GroupPtrVector_t& gs, ImVec2 init_position) {

    struct LayoutItem {
        ModulePtr_t module_ptr;
        GroupPtr_t group_ptr;
        bool considered;
        LayoutItem() : module_ptr(nullptr), group_ptr(nullptr), considered(false) {}
        ~LayoutItem() = default;
    };

    std::vector<std::vector<LayoutItem>> layers;
    layers.clear();

    // Fill first layer with graph elements having no connected callee slots
    layers.emplace_back();
    for (auto& group_ptr : gs) {
        bool any_connected_callee = false;
        for (auto& interfaceslot_ptr : group_ptr->InterfaceSlots(CallSlotType::CALLEE)) {
            if (this->connected_interfaceslot(ms, gs, interfaceslot_ptr)) {
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
    for (auto& module_ptr : ms) {
        bool any_connected_callee = false;
        for (auto& calleeslot_ptr : module_ptr->CallSlots(CallSlotType::CALLEE)) {
            if (this->connected_callslot(ms, gs, calleeslot_ptr)) {
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
                for (auto& callerslot_ptr : layer_item.module_ptr->CallSlots(CallSlotType::CALLER)) {
                    if (this->connected_callslot(ms, gs, callerslot_ptr)) {
                        callerslots.emplace_back(callerslot_ptr);
                    }
                }
            } else if (layer_item.group_ptr != nullptr) {
                for (auto& interfaceslot_slot : layer_item.group_ptr->InterfaceSlots(CallSlotType::CALLER)) {
                    for (auto& callerslot_ptr : interfaceslot_slot->CallSlots()) {
                        if (this->connected_callslot(ms, gs, callerslot_ptr)) {
                            callerslots.emplace_back(callerslot_ptr);
                        }
                    }
                }
            }
            for (auto& callerslot_ptr : callerslots) {
                if (callerslot_ptr->CallsConnected()) {
                    for (auto& call_ptr : callerslot_ptr->GetConnectedCalls()) {
                        if (call_ptr->CallSlotPtr(CallSlotType::CALLEE)->IsParentModuleConnected()) {

                            auto add_module_ptr = call_ptr->CallSlotPtr(CallSlotType::CALLEE)->GetParentModule();
                            if (this->contains_module(ms, add_module_ptr->UID())) {
                                // Add module only if not already  Prevents cyclic dependency
                                bool module_already_added = false;
                                for (auto& previous_layer : layers) {
                                    for (auto& previous_layer_item : previous_layer) {
                                        if (previous_layer_item.module_ptr != nullptr) {
                                            if (previous_layer_item.module_ptr->UID() == add_module_ptr->UID()) {
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
                            } else if (add_module_ptr->GroupUID() != GUI_INVALID_ID) {
                                ImGuiID group_uid = add_module_ptr->GroupUID(); // != GUI_INVALID_ID
                                if (this->contains_group(gs, group_uid)) {
                                    GroupPtr_t add_group_ptr;
                                    for (auto& group_ptr : gs) {
                                        if (group_ptr->UID() == group_uid) {
                                            add_group_ptr = group_ptr;
                                        }
                                    }
                                    if (add_group_ptr != nullptr) {
                                        // Add group only if not already  Prevents cyclic dependency
                                        bool group_already_added = false;
                                        for (auto& previous_layer : layers) {
                                            for (auto& previous_layer_item : previous_layer) {
                                                if (previous_layer_item.group_ptr != nullptr) {
                                                    if (previous_layer_item.group_ptr->UID() == add_group_ptr->UID()) {
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
                for (auto& callerslot_ptr : layers[i][j].module_ptr->CallSlots(CallSlotType::CALLER)) {
                    callerslots.emplace_back(callerslot_ptr);
                }
            } else if (layers[i][j].group_ptr != nullptr) {
                for (auto& interfaceslot_slot : layers[i][j].group_ptr->InterfaceSlots(CallSlotType::CALLER)) {
                    for (auto& callerslot_ptr : interfaceslot_slot->CallSlots()) {
                        callerslots.emplace_back(callerslot_ptr);
                    }
                }
            }
            // Collect all connected callee slots
            CallSlotPtrVector_t current_calleeslots;
            for (auto& callerslot_ptr : callerslots) {
                for (auto& call_ptr : callerslot_ptr->GetConnectedCalls()) {
                    auto calleeslot_ptr = call_ptr->CallSlotPtr(CallSlotType::CALLEE);
                    current_calleeslots.emplace_back(calleeslot_ptr);
                }
            }

            // Search for connected graph elements lying in same or lower layer and move graph element
            for (size_t k = 0; k <= i; k++) {
                for (size_t m = 0; m < layers[k].size(); m++) {
                    if (!layers[k][m].considered) {
                        CallSlotPtrVector_t other_calleeslots;
                        if (layers[k][m].module_ptr != nullptr) {
                            for (auto& calleeslot_ptr : layers[k][m].module_ptr->CallSlots(CallSlotType::CALLEE)) {
                                other_calleeslots.emplace_back(calleeslot_ptr);
                            }
                        } else if (layers[k][m].group_ptr != nullptr) {
                            for (auto& interfaceslot_slot :
                                layers[k][m].group_ptr->InterfaceSlots(CallSlotType::CALLEE)) {
                                for (auto& calleeslot_ptr : interfaceslot_slot->CallSlots()) {
                                    other_calleeslots.emplace_back(calleeslot_ptr);
                                }
                            }
                        }
                        for (auto& current_calleeslot_ptr : current_calleeslots) {
                            for (auto& other_calleeslot_ptr : other_calleeslots) {
                                if (current_calleeslot_ptr->UID() == other_calleeslot_ptr->UID()) {
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

        if (this->gui_graph_state.interact.call_show_label || this->gui_graph_state.interact.call_show_slots_label) {
            max_call_width = 0.0f;
        }
        max_graph_element_width = 0.0f;
        pos.y = init_position.y;
        bool found_layer_item = false;

        size_t layer_items_count = layers[i].size();
        for (size_t j = 0; j < layer_items_count; j++) {
            auto layer_item = layers[i][j];

            if (layer_item.module_ptr != nullptr) {
                if (this->gui_graph_state.interact.call_show_label) {
                    for (auto& callerslot_ptr : layer_item.module_ptr->CallSlots(CallSlotType::CALLER)) {
                        if (callerslot_ptr->CallsConnected() && this->connected_callslot(ms, gs, callerslot_ptr)) {
                            for (auto& call_ptr : callerslot_ptr->GetConnectedCalls()) {
                                auto call_name_length = ImGui::CalcTextSize(call_ptr->ClassName().c_str()).x;
                                max_call_width = std::max(call_name_length, max_call_width);
                            }
                        }
                    }
                }
                if (this->gui_graph_state.interact.call_show_slots_label) {
                    for (auto& callerslot_ptr : layer_item.module_ptr->CallSlots(CallSlotType::CALLER)) {
                        if (callerslot_ptr->CallsConnected() && this->connected_callslot(ms, gs, callerslot_ptr)) {
                            for (auto& call_ptr : callerslot_ptr->GetConnectedCalls()) {
                                auto call_name_length = ImGui::CalcTextSize(call_ptr->SlotsLabel().c_str()).x;
                                max_call_width = std::max(call_name_length, max_call_width);
                            }
                        }
                    }
                }
                layer_item.module_ptr->SetPosition(pos);
                auto module_size = layer_item.module_ptr->Size();
                pos.y += (module_size.y + GUI_GRAPH_BORDER);
                max_graph_element_width = std::max(module_size.x, max_graph_element_width);
                found_layer_item = true;

            } else if (layer_item.group_ptr != nullptr) {
                if (this->gui_graph_state.interact.call_show_label) {
                    for (auto& interfaceslot_slot : layer_item.group_ptr->InterfaceSlots(CallSlotType::CALLER)) {
                        for (auto& callerslot_ptr : interfaceslot_slot->CallSlots()) {
                            if (callerslot_ptr->CallsConnected() && this->connected_callslot(ms, gs, callerslot_ptr)) {
                                for (auto& call_ptr : callerslot_ptr->GetConnectedCalls()) {
                                    auto call_name_length = ImGui::CalcTextSize(call_ptr->ClassName().c_str()).x;
                                    max_call_width = std::max(call_name_length, max_call_width);
                                }
                            }
                        }
                    }
                }
                if (this->gui_graph_state.interact.call_show_slots_label) {
                    for (auto& interfaceslot_slot : layer_item.group_ptr->InterfaceSlots(CallSlotType::CALLER)) {
                        for (auto& callerslot_ptr : interfaceslot_slot->CallSlots()) {
                            if (callerslot_ptr->CallsConnected() && this->connected_callslot(ms, gs, callerslot_ptr)) {
                                for (auto& call_ptr : callerslot_ptr->GetConnectedCalls()) {
                                    auto call_name_length = ImGui::CalcTextSize(call_ptr->SlotsLabel().c_str()).x;
                                    max_call_width = std::max(call_name_length, max_call_width);
                                }
                            }
                        }
                    }
                }
                layer_item.group_ptr->SetPosition(this->gui_graph_state, pos);
                auto group_size = layer_item.group_ptr->Size();
                pos.y += (group_size.y + GUI_GRAPH_BORDER);
                max_graph_element_width = std::max(group_size.x, max_graph_element_width);
                found_layer_item = true;
            }
        }
        if (found_layer_item) {
            pos.x += (max_graph_element_width + max_call_width + (2.0f * GUI_GRAPH_BORDER));
        }
    }

    this->gui_update = true;
}


bool megamol::gui::Graph::connected_callslot(
    const ModulePtrVector_t& ms, const GroupPtrVector_t& gs, const CallSlotPtr_t& callslot_ptr) const {

    bool retval = false;
    for (auto& call_ptr : callslot_ptr->GetConnectedCalls()) {
        CallSlotType type =
            (callslot_ptr->Type() == CallSlotType::CALLER) ? (CallSlotType::CALLEE) : (CallSlotType::CALLER);
        auto connected_callslot_ptr = call_ptr->CallSlotPtr(type);
        if (connected_callslot_ptr != nullptr) {
            if (this->contains_callslot(ms, connected_callslot_ptr->UID())) {
                retval = true;
                break;
            }
            if (connected_callslot_ptr->InterfaceSlotPtr() != nullptr) {
                if (this->contains_interfaceslot(gs, connected_callslot_ptr->InterfaceSlotPtr()->UID())) {
                    retval = true;
                    break;
                }
            }
        }
    }
    return retval;
}


bool megamol::gui::Graph::connected_interfaceslot(
    const ModulePtrVector_t& ms, const GroupPtrVector_t& gs, const InterfaceSlotPtr_t& interfaceslot_ptr) const {

    bool retval = false;
    for (auto& callslot_ptr : interfaceslot_ptr->CallSlots()) {
        for (auto& call_ptr : callslot_ptr->GetConnectedCalls()) {
            CallSlotType type =
                (callslot_ptr->Type() == CallSlotType::CALLER) ? (CallSlotType::CALLEE) : (CallSlotType::CALLER);
            auto connected_callslot_ptr = call_ptr->CallSlotPtr(type);
            if (connected_callslot_ptr != nullptr) {
                if (this->contains_callslot(ms, connected_callslot_ptr->UID())) {
                    retval = true;
                    break;
                }
                if (connected_callslot_ptr->InterfaceSlotPtr() != nullptr) {
                    if (this->contains_interfaceslot(gs, connected_callslot_ptr->InterfaceSlotPtr()->UID())) {
                        retval = true;
                        break;
                    }
                }
            }
        }
    }
    return retval;
}


bool megamol::gui::Graph::contains_callslot(const ModulePtrVector_t& ms, ImGuiID callslot_uid) const {

    for (auto& module_ptr : ms) {
        for (auto& callslots_map : module_ptr->CallSlots()) {
            for (auto& callslot_ptr : callslots_map.second) {
                if (callslot_ptr->UID() == callslot_uid) {
                    return true;
                }
            }
        }
    }
    return false;
}


bool megamol::gui::Graph::contains_interfaceslot(const GroupPtrVector_t& gs, ImGuiID interfaceslot_uid) const {

    for (auto& group_ptr : gs) {
        for (auto& interfaceslots_map : group_ptr->InterfaceSlots()) {
            for (auto& interfaceslot_ptr : interfaceslots_map.second) {
                if (interfaceslot_ptr->UID() == interfaceslot_uid) {
                    return true;
                }
            }
        }
    }
    return false;
}


bool megamol::gui::Graph::contains_module(const ModulePtrVector_t& ms, ImGuiID module_uid) const {

    for (auto& module_ptr : ms) {
        if (module_ptr->UID() == module_uid) {
            return true;
        }
    }
    return false;
}


bool megamol::gui::Graph::contains_group(const GroupPtrVector_t& gs, ImGuiID group_uid) const {

    for (auto& group_ptr : gs) {
        if (group_ptr->UID() == group_uid) {
            return true;
        }
    }
    return false;
}


std::string megamol::gui::Graph::generate_unique_group_name() const {

    int new_name_id = 0;
    std::string new_name_prefix("Group_");
    for (auto& group : this->groups) {
        if (group->Name().find(new_name_prefix) == 0) {
            std::string int_postfix = group->Name().substr(new_name_prefix.length());
            int last_id = 0;
            std::istringstream(int_postfix) >> last_id; // 0 if failed
            new_name_id = std::max(new_name_id, last_id);
        }
    }
    return std::string(new_name_prefix + std::to_string(new_name_id + 1));
}


std::string megamol::gui::Graph::generate_unique_module_name(const std::string& module_name) const {

    int new_name_id = 0;
    std::string new_name_prefix = module_name + "_";
    for (auto& mod : this->modules) {
        if (mod->Name().find(new_name_prefix) == 0) {
            std::string int_postfix = mod->Name().substr(new_name_prefix.length());
            int last_id = 0;
            std::istringstream(int_postfix) >> last_id; // 0 if failed
            new_name_id = std::max(new_name_id, last_id);
        }
    }
    return std::string(new_name_prefix + std::to_string(new_name_id + 1));
}


std::string megamol::gui::Graph::GenerateUniqueGraphEntryName() {

    int new_name_id = 0;
    std::string new_name_prefix("GraphEntry_");
    for (auto& module_ptr : this->modules) {
        if (module_ptr->GraphEntryName().find(new_name_prefix) == 0) {
            std::string int_postfix = module_ptr->GraphEntryName().substr(new_name_prefix.length());
            int last_id = 0;
            std::istringstream(int_postfix) >> last_id; // 0 if failed
            new_name_id = std::max(new_name_id, last_id);
        }
    }
    return std::string(new_name_prefix + std::to_string(new_name_id + 1));
}


#ifdef MEGAMOL_USE_PROFILING

void megamol::gui::Graph::draw_profiling(ImVec2 position, ImVec2 size) {

    ImGuiStyle& style = ImGui::GetStyle();
    ImGuiIO& io = ImGui::GetIO();

    ImGui::SetNextWindowPos(position);

    float child_height = ImGui::GetFrameHeightWithSpacing();
    auto child_flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NavFlattened | ImGuiWindowFlags_MenuBar;
    ImGui::BeginChild("profiling_menu", ImVec2(size.x, child_height), false, child_flags);

    ImGui::BeginMenuBar();
    ImGui::TextUnformatted("Profiling");

    ImGui::Separator();
    // Lazy loading of run button textures
    if (!this->gui_profiling_run_button.IsLoaded()) {
        this->gui_profiling_run_button.LoadTextureFromFile(
            GUI_FILENAME_TEXTURE_TRANSPORT_ICON_PAUSE, GUI_FILENAME_TEXTURE_TRANSPORT_ICON_PLAY);
    }

    this->gui_profiling_run_button.ToggleButton(this->gui_graph_state.interact.profiling_pause_update,
        "Pause update of profiling values globally", "Continue updating of profiling values",
        ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight()));
    ImGui::TextUnformatted(((this->gui_graph_state.interact.profiling_pause_update) ? ("Run") : ("Pause")));

    ImGui::Separator();
    if (ImGui::Button(((this->gui_profiling_graph_link) ? ("Hide") : ("Show")))) {
        this->gui_profiling_graph_link = !this->gui_profiling_graph_link;
    }
    ImGui::TextUnformatted("Graph Links");

    ImGui::Separator();

    ImGui::EndMenuBar();
    ImGui::EndChild();

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(1, 1));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

    float delta_height = ImGui::GetFrameHeightWithSpacing() + style.ItemSpacing.y;
    auto child_position = position + ImVec2(0.0f, delta_height);
    auto child_size = size + ImVec2(0.0f, -delta_height);
    ImGui::SetNextWindowPos(child_position);
    ImGui::BeginChild("profiling_child", child_size, false, ImGuiWindowFlags_AlwaysHorizontalScrollbar);
    auto window_position = ImGui::GetCursorScreenPos();

    // Sync profiling list
    auto iterp = this->profiling_list.begin();
    while (iterp != this->profiling_list.end()) {
        const auto m_ptr = iterp->first.lock();
        const auto c_ptr = iterp->second.lock();
        if ((m_ptr != nullptr && !m_ptr->ShowProfiling()) || (c_ptr != nullptr && !c_ptr->ShowProfiling()) ||
            (c_ptr == nullptr && m_ptr == nullptr)) {
            iterp = this->profiling_list.erase(iterp);
        } else {
            iterp++;
        }
    }
    for (const auto& m_ptr : this->modules) {
        if (m_ptr->ShowProfiling()) {
            bool found = false;
            for (const auto& p : this->profiling_list) {
                if (p.first.lock() == m_ptr) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                this->profiling_list.emplace_back(m_ptr, std::weak_ptr<gui::Call>());
            }
        }
    }
    for (auto& c_ptr : this->calls) {
        if (c_ptr->ShowProfiling()) {
            bool found = false;
            for (const auto& p : this->profiling_list) {
                if (p.second.lock() == c_ptr) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                this->profiling_list.emplace_back(std::weak_ptr<gui::Module>(), c_ptr);
            }
        }
    }

    // Set same max number of module uid decimal digits for drag and drop.
    const size_t dnd_size = 16 * sizeof(char);
    unsigned int drop_target_index = 0;
    ImGuiID drop_uid = 0;
    int drop_index = 0;

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    assert(drawList != nullptr);

    drawList->PushClipRect(this->gui_graph_state.canvas.position, position + size, false);

    // Draw profiling data
    for (auto& p : this->profiling_list) {

        const auto m_ptr = p.first.lock();
        const auto c_ptr = p.second.lock();

        std::string uid_str;
        std::string label;
        if (m_ptr != nullptr) {
            uid_str = std::to_string(m_ptr->UID());
            label = "Module: " + m_ptr->FullName();
        } else if (c_ptr != nullptr) {
            uid_str = std::to_string(c_ptr->UID());
            label = "Call: " + c_ptr->ClassName();
        }
        ImGui::BeginChild(("##profiling_tab_bar" + uid_str).c_str(), ImVec2(GRAPH_PROFILING_WINDOW_WIDTH, 0.0f), true,
            ImGuiWindowFlags_NoMove);

        // Draw line to graph item
        if (this->gui_profiling_graph_link) {
            ImVec2 start =
                ImGui::GetCursorScreenPos() + ImVec2(ImGui::GetFrameHeight() / 2.0f, ImGui::GetFrameHeight() / 2.0f);
            if (window_position.y > start.y) {
                start.y = window_position.y;
            }
            ImVec2 end;
            if (m_ptr != nullptr) {
                end = m_ptr->GetProfilingButtonPosition();
                if (m_ptr->IsHidden()) {
                    auto group_uid = m_ptr->GroupUID();
                    for (auto group_ptr : this->groups) {
                        if (group_ptr->UID() == group_uid) {
                            end = group_ptr->GetPositionBottomCenter();
                        }
                    }
                }
            } else if (c_ptr != nullptr) {
                end = c_ptr->GetProfilingButtonPosition();
                if (c_ptr->IsHidden()) {
                    auto callerslot_ptr = c_ptr->CallSlotPtr(CallSlotType::CALLER);
                    if (callerslot_ptr != nullptr) {
                        if (callerslot_ptr->IsParentModuleConnected()) {
                            auto group_uid = callerslot_ptr->GetParentModule()->GroupUID();
                            for (auto group_ptr : this->groups) {
                                if (group_ptr->UID() == group_uid) {
                                    end = group_ptr->GetPositionBottomCenter();
                                }
                            }
                        }
                    }
                }
            }
            if (window_position.y < end.y) {
                end.y = window_position.y;
            }
            auto tmpcol = style.Colors[ImGuiCol_PlotLines];
            tmpcol = ImVec4(tmpcol.x, tmpcol.y, tmpcol.z, 0.5f);
            const ImU32 COLOR_CALL_CURVE = ImGui::ColorConvertFloat4ToU32(tmpcol);
            drawList->AddBezierCubic(start, start + ImVec2(0.0f, (-100.0f * megamol::gui::gui_scaling.Get())),
                end + ImVec2(0.0f, (100.0f * megamol::gui::gui_scaling.Get())), end, COLOR_CALL_CURVE,
                GUI_LINE_THICKNESS);
        }
        // Synchronize highlighting of hovered/active call/module with profiling header
        bool push_style_color = false;
        if (c_ptr != nullptr) {
            if (c_ptr->IsSelected() || c_ptr->IsHovered()) {
                push_style_color = true;
            }
        }
        if (m_ptr != nullptr) {
            if (m_ptr->IsSelected() || m_ptr->IsHovered()) {
                push_style_color = true;
            }
        }
        if (push_style_color) {
            ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        }
        // Header
        ImGui::CollapsingHeader(label.c_str(), ImGuiTreeNodeFlags_Bullet);
        if (push_style_color) {
            ImGui::PopStyleColor(1);
        }
        this->gui_tooltip.ToolTip("[Drag and drop] Horizontally sort profiling data.", ImGui::GetItemID(), 0.5f, 5.0f);
        // Synchronize highlighting of hovered/active profiling header with call/module
        if (ImGui::IsItemActivated()) {
            if (c_ptr != nullptr)
                c_ptr->SetActive();
            if (m_ptr != nullptr)
                m_ptr->SetActive();
        }
        if (ImGui::IsItemHovered()) {
            if (c_ptr != nullptr)
                c_ptr->SetHovered();
            if (m_ptr != nullptr)
                m_ptr->SetHovered();
        }

        // Drag source
        if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
            std::string payload;
            payload = uid_str;
            assert((payload.size() * sizeof(char)) <= dnd_size);
            ImGui::SetDragDropPayload("DND_COPY_PROFILING_DATA", payload.c_str(), dnd_size);
            ImGui::CollapsingHeader(label.c_str(), ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_Bullet);
            ImGui::EndDragDropSource();
        }
        ImGui::Separator();
        ImGui::TextUnformatted("Region Name:");
        if (m_ptr != nullptr) {
            m_ptr->DrawProfiling(this->gui_graph_state);
        } else if (c_ptr != nullptr) {
            c_ptr->DrawProfiling(this->gui_graph_state);
        }
        ImGui::EndChild();

        // Drop target
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("DND_COPY_PROFILING_DATA")) {
                assert(payload->DataSize == (dnd_size * sizeof(char)));
                std::string payload_id = (const char*)payload->Data;
                try {
                    auto uid_payload = static_cast<ImGuiID>(std::strtol(payload_id.c_str(), nullptr, 10));
                    if (errno == ERANGE) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] Failed to read dragged payload: '%s'. [%s, %s, line %d]\n", std::strerror(errno),
                            __FILE__, __FUNCTION__, __LINE__);
                    } else {
                        drop_uid = uid_payload;
                        drop_index = drop_target_index;
                    }
                } catch (...) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Failed to drop profiling data. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                }
            }
            ImGui::EndDragDropTarget();
        }
        drop_target_index++;
        ImGui::SameLine(0.0f);
    }

    drawList->PopClipRect();

    // Apply drop
    if (drop_uid != 0) {
        for (auto iterp = this->profiling_list.begin(); iterp != this->profiling_list.end(); iterp++) {
            const auto m_ptr = iterp->first.lock();
            const auto c_ptr = iterp->second.lock();
            if (((m_ptr != nullptr) && (m_ptr->UID() == drop_uid)) ||
                ((c_ptr != nullptr) && (c_ptr->UID() == drop_uid))) {
                auto tmp = (*iterp);
                this->profiling_list.erase(iterp);
                this->profiling_list.insert(this->profiling_list.begin() + drop_index, tmp);
                break;
            }
        }
    }

    // Auto scroll during profiling drag
    auto mouse_pos = ImGui::GetMousePos();
    const float mouse_delta = 50.0f * megamol::gui::gui_scaling.Get();
    if (const ImGuiPayload* payload = ImGui::GetDragDropPayload()) {
        auto min_win_x = ImGui::GetWindowPos().x;
        auto max_win_x = min_win_x + ImGui::GetWindowSize().x;
        auto time_delta = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - this->scroll_delta_time)
                              .count();
        if (mouse_pos.x <= (min_win_x + mouse_delta)) {
            ImGui::SetScrollX(ImGui::GetScrollX() - static_cast<float>(time_delta));
        } else if (mouse_pos.x >= (max_win_x - mouse_delta)) {
            ImGui::SetScrollX(ImGui::GetScrollX() + static_cast<float>(time_delta));
        }
    }
    this->scroll_delta_time = std::chrono::system_clock::now();

    ImGui::EndChild();

    ImGui::PopStyleVar(2);
}


#endif // MEGAMOL_USE_PROFILING
