/*
 * GraphCollection.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "GraphCollection.h"
#include "mmcore/view/AbstractView.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::GraphCollection::GraphCollection(void) : graphs(), modules_stock(), calls_stock(), graph_name_uid(0) {}


megamol::gui::GraphCollection::~GraphCollection(void) {}


bool megamol::gui::GraphCollection::AddEmptyProject(void) {

    ImGuiID graph_uid = this->AddGraph(GraphCoreInterface::NO_INTERFACE);
    if (graph_uid != GUI_INVALID_ID) {
        /// Setup new empty graph ...
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to create new graph. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }

    return false;
}


ImGuiID megamol::gui::GraphCollection::AddGraph(GraphCoreInterface graph_core_interface) {

    ImGuiID retval = GUI_INVALID_ID;

    try {
        GraphPtr_t graph_ptr = std::make_shared<Graph>(this->generate_unique_graph_name(), graph_core_interface);
        if (graph_ptr != nullptr) {
            this->graphs.emplace_back(graph_ptr);
            retval = graph_ptr->uid;
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[GUI] Added graph %s' (uid %i). \n", graph_ptr->name.c_str(), graph_ptr->uid);
#endif // GUI_VERBOSE
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

    return retval;
}


bool megamol::gui::GraphCollection::DeleteGraph(ImGuiID in_graph_uid) {

    for (auto iter = this->graphs.begin(); iter != this->graphs.end(); iter++) {
        if ((*iter)->uid == in_graph_uid) {
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[GUI] Deleted graph %s' (uid %i). \n", (*iter)->name.c_str(), (*iter)->uid);
#endif // GUI_VERBOSE

            if ((*iter).use_count() > 1) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Unclean deletion. Found %i references pointing to graph. [%s, %s, line %d]\n",
                    (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
            }
            (*iter).reset();
            this->graphs.erase(iter);

            return true;
        }
    }
    return false;
}


bool megamol::gui::GraphCollection::GetGraph(ImGuiID in_graph_uid, megamol::gui::GraphPtr_t& out_graph_ptr) {

    if (in_graph_uid != GUI_INVALID_ID) {
        for (auto& graph_ptr : this->graphs) {
            if (graph_ptr->uid == in_graph_uid) {
                out_graph_ptr = graph_ptr;
                return true;
            }
        }
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Unable to find graph for given graph uid: %i [%s, %s, line %d]\n", in_graph_uid, __FILE__,
            __FUNCTION__, __LINE__);
    }
    return false;
}


bool megamol::gui::GraphCollection::LoadCallStock(const megamol::core::CoreInstance* core_instance) {

    if (core_instance == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Pointer to Core Instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Load only once
    if (!this->calls_stock.empty()) {
        return true;
        // megamol::core::utility::log::Log::DefaultLog.WriteWarn("[GUI] Call stock already exists. Deleting exiting
        // stock and recreating new one. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }

    bool retval = true;
    this->calls_stock.clear();

    try {
        std::string plugin_name;
        auto start_time = std::chrono::system_clock::now();

        // CALLS ------------------------------------------------------------------
        /// ! Get calls before getting modules for having calls in place for setting compatible call indices of slots!
        if (this->calls_stock.empty()) {

            // Get plugin calls (get prior to core calls for being  able to find duplicates in core instance call desc.
            // manager)
            const std::vector<core::utility::plugins::AbstractPluginInstance::ptr_type>& plugins =
                core_instance->Plugins().GetPlugins();
            for (core::utility::plugins::AbstractPluginInstance::ptr_type plugin : plugins) {
                plugin_name = plugin->GetAssemblyName();
                for (auto& c_desc : plugin->GetCallDescriptionManager()) {
                    Call::StockCall call;
                    call.plugin_name = plugin_name;
                    this->get_call_stock_data(call, c_desc);
                    this->calls_stock.emplace_back(call);
                }
            }

            // Get core calls
            plugin_name = "Core"; // (core_instance->GetAssemblyName() = "")
            for (auto& c_desc : core_instance->GetCallDescriptionManager()) {
                std::string class_name(c_desc->ClassName());
                if (std::find_if(this->calls_stock.begin(), this->calls_stock.end(),
                        [class_name](const Call::StockCall& call) { return (call.class_name == class_name); }) ==
                    this->calls_stock.end()) {
                    Call::StockCall call;
                    call.plugin_name = plugin_name;
                    this->get_call_stock_data(call, c_desc);
                    this->calls_stock.emplace_back(call);
                }
            }
        }

        auto delta_time =
            static_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - start_time).count();


        megamol::core::utility::log::Log::DefaultLog.WriteInfo(
            "[GUI] Reading %i available calls ... DONE (duration: %.3f seconds)\n", this->calls_stock.size(),
            delta_time);

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


bool megamol::gui::GraphCollection::LoadModuleStock(const megamol::core::CoreInstance* core_instance) {

    if (core_instance == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Pointer to Core Instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (this->calls_stock.empty()) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Load call stock before getting modules for having calls in place for "
            "setting compatible call indices of slots. [%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Load only once
    if (!this->modules_stock.empty()) {
        return true;
        // megamol::core::utility::log::Log::DefaultLog.WriteWarn("[GUI] Module stock already exists. Deleting exiting
        // stock and recreating new one. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }

    bool retval = true;
    this->modules_stock.clear();

    try {
        std::string plugin_name;
        auto start_time = std::chrono::system_clock::now();
#ifdef GUI_VERBOSE
        auto module_load_time = std::chrono::system_clock::now();
#endif // GUI_VERBOSE
       // MODULES ----------------------------------------------------------------
        if (this->modules_stock.empty()) {

            // Get plugin modules (get prior to core modules for being  able to find duplicates in core instance module
            // desc. manager)
            const std::vector<core::utility::plugins::AbstractPluginInstance::ptr_type>& plugins =
                core_instance->Plugins().GetPlugins();
            for (core::utility::plugins::AbstractPluginInstance::ptr_type plugin : plugins) {
                plugin_name = plugin->GetAssemblyName();
                for (auto& m_desc : plugin->GetModuleDescriptionManager()) {
                    std::string class_name(m_desc->ClassName());
                    Module::StockModule mod;
                    mod.plugin_name = plugin_name;
                    this->get_module_stock_data(mod, m_desc);
                    this->modules_stock.emplace_back(mod);
#ifdef GUI_VERBOSE
                    auto module_load_time_count =
                        static_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - module_load_time)
                            .count();
                    module_load_time = std::chrono::system_clock::now();
                    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                        "[GUI] Reading module '%s' ... DONE (duration: %.3f seconds)\n", class_name.c_str(),
                        module_load_time_count);
#endif // GUI_VERBOSE
                }
            }

            // Get core modules
            plugin_name = "Core"; // (core_instance->GetAssemblyName() = "")
            for (auto& m_desc : core_instance->GetModuleDescriptionManager()) {
                std::string class_name(m_desc->ClassName());
                if (std::find_if(this->modules_stock.begin(), this->modules_stock.end(),
                        [class_name](const Module::StockModule& mod) { return (mod.class_name == class_name); }) ==
                    this->modules_stock.end()) {
                    Module::StockModule mod;
                    mod.plugin_name = plugin_name;
                    this->get_module_stock_data(mod, m_desc);
                    this->modules_stock.emplace_back(mod);
#ifdef GUI_VERBOSE
                    auto module_load_time_count =
                        static_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - module_load_time)
                            .count();
                    module_load_time = std::chrono::system_clock::now();
                    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                        "[GUI] Reading module '%s' ... DONE (duration: %.3f seconds)\n", class_name.c_str(),
                        module_load_time_count);
#endif // GUI_VERBOSE
                }
            }

            // Sorting module by alphabetically ascending class names.
            std::sort(this->modules_stock.begin(), this->modules_stock.end(),
                [](Module::StockModule mod1, Module::StockModule mod2) {
                    std::string a_str(mod1.class_name);
                    for (auto& c : a_str)
                        c = std::toupper(c);
                    std::string b_str(mod2.class_name);
                    for (auto& c : b_str)
                        c = std::toupper(c);
                    return (a_str < b_str);
                });
        }

        auto delta_time =
            static_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - start_time).count();

        megamol::core::utility::log::Log::DefaultLog.WriteInfo(
            "[GUI] Reading %i available modules ... DONE (duration: %.3f seconds)\n", this->modules_stock.size(),
            delta_time);

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


bool megamol::gui::GraphCollection::LoadUpdateProjectFromCore(ImGuiID& inout_graph_uid,
    megamol::core::CoreInstance* core_instance, megamol::core::MegaMolGraph* megamol_graph, bool sync) {

    bool created_new_graph = false;
    ImGuiID valid_graph_id = inout_graph_uid;
    if (valid_graph_id == GUI_INVALID_ID) {
        // Create new graph
        GraphCoreInterface graph_core_interface = GraphCoreInterface::NO_INTERFACE;
        if (sync) {
            graph_core_interface = (megamol_graph != nullptr) ? (GraphCoreInterface::MEGAMOL_GRAPH)
                                                              : (GraphCoreInterface::CORE_INSTANCE_GRAPH);
        }
        valid_graph_id = this->AddGraph(graph_core_interface);
        if (valid_graph_id == GUI_INVALID_ID) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Failed to create new graph. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        created_new_graph = true;
    }

    GraphPtr_t graph_ptr;
    if (!this->GetGraph(valid_graph_id, graph_ptr)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to find graph for given uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    if (this->AddUpdateProjectFromCore(valid_graph_id, core_instance, megamol_graph, false)) {
        inout_graph_uid = valid_graph_id;
        if (created_new_graph) {
            graph_ptr->present.SetLayoutGraph();
            graph_ptr->ResetDirty();
        }
        return true;
    }
    // else { this->DeleteGraph(valid_graph_id); }

    return false;
}


bool megamol::gui::GraphCollection::AddUpdateProjectFromCore(ImGuiID in_graph_uid,
    megamol::core::CoreInstance* core_instance, megamol::core::MegaMolGraph* megamol_graph, bool use_stock) {

    /// TODO
    ///    - Add call(s)       - Core Instance
    ///    - Delete call(s)    - Core Instance

    // Apply updates from core graph -> gui graph
    try {
        GraphPtr_t graph_ptr;
        if (!this->GetGraph(in_graph_uid, graph_ptr)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Unable to find graph for given uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        const bool use_megamol_graph = (megamol_graph != nullptr);
        const bool use_core_instance = (core_instance != nullptr);
        if (use_megamol_graph == use_core_instance) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Ambiguous references to the graph. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        // Collect new call connections to temporary data structure and apply in the end,
        // since not all call slots might be available for being connected while modules are created consecutively.
        struct CallData {
            std::string call_class_name;
            std::string caller_module_full_name;
            std::string caller_module_callslot_name;
            std::string callee_module_full_name;
            std::string callee_module_callslot_name;
        };
        std::vector<CallData> call_data;

        bool gui_graph_changed = false;

        // ADD new modules to GUI graph -------------------------------------------------
        std::vector<megamol::core::Module*> add_module_list;
        std::map<std::string, std::string> new_view_instances;
        if (use_megamol_graph) {
            for (auto& module_inst : megamol_graph->ListModules()) {
                std::string module_fullname =
                    std::string(module_inst.modulePtr->Name().PeekBuffer()); /// Check only 'Name()'!
                if (!graph_ptr->ModuleExists(module_fullname)) {
                    add_module_list.emplace_back(module_inst.modulePtr.get());
                    if (module_inst.isGraphEntryPoint) {
                        new_view_instances[module_fullname] = graph_ptr->GenerateUniqueGraphEntryName();
                    }
                }
            }
        } else if (use_core_instance) {
            const auto module_func = [&, this](megamol::core::Module* mod) {
                std::string module_fullname = std::string(mod->FullName().PeekBuffer());
                if (!graph_ptr->ModuleExists(module_fullname)) {
                    add_module_list.emplace_back(mod);
                }
            };
            core_instance->EnumModulesNoLock(nullptr, module_func);

            if (!add_module_list.empty()) {
                vislib::sys::AutoLock lock(core_instance->ModuleGraphRoot()->ModuleGraphLock());
                megamol::core::AbstractNamedObjectContainer::const_ptr_type anoc =
                    megamol::core::AbstractNamedObjectContainer::dynamic_pointer_cast(core_instance->ModuleGraphRoot());
                for (auto ano = anoc->ChildList_Begin(); ano != anoc->ChildList_End(); ++ano) {
                    auto vi = dynamic_cast<megamol::core::ViewInstance*>(ano->get());
                    if ((vi != nullptr) && (vi->View() != nullptr)) {
                        new_view_instances[std::string(vi->View()->FullName().PeekBuffer())] =
                            std::string(vi->Name().PeekBuffer());
                    }
                }
            }
        }
        for (auto& module_ptr : add_module_list) {
            std::string full_name;
            if (use_megamol_graph) {
                full_name = (module_ptr->Name().PeekBuffer()); /// Check only 'Name()'!
            } else if (use_core_instance) {
                full_name = (module_ptr->FullName().PeekBuffer());
            }
            std::string class_name(module_ptr->ClassName());
            std::string module_name;
            std::string module_namespace;
            if (!this->project_separate_name_and_namespace(full_name, module_namespace, module_name)) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Core Project: Invalid module name '%s'. [%s, %s, line %d]\n", full_name.c_str(), __FILE__,
                    __FUNCTION__, __LINE__);
            }
            /// DEBUG
            /// megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] >>>> Class: '%s' NameSpace: '%s' Name:
            /// '%s'.\n", mod->ClassName(), module_namespace.c_str(), module_name.c_str());

            // Ensure unique module name is not yet assigned
            graph_ptr->UniqueModuleRename(module_name);

            // Create new module
            ImGuiID moduel_uid = GUI_INVALID_ID;
            ModulePtr_t new_module_ptr;
            if (use_stock) {
                moduel_uid = graph_ptr->AddModule(this->modules_stock, class_name);
                graph_ptr->GetModule(moduel_uid, new_module_ptr);
            } else {
                moduel_uid = graph_ptr->AddEmptyModule();
                if (graph_ptr->GetModule(moduel_uid, new_module_ptr)) {
                    new_module_ptr->class_name = class_name;
                    new_module_ptr->description = "[n/a]";
                    /// XXX ModuleDescriptionManager is only available via core instance graph yet.
                    if (use_core_instance) {
                        auto mod_desc =
                            core_instance->GetModuleDescriptionManager().Find(vislib::StringA(class_name.c_str()));
                        if (mod_desc != nullptr) {
                            new_module_ptr->description = std::string(mod_desc->Description());
                        }
                    }
                    new_module_ptr->plugin_name = "[n/a]";

                    /// XXX VIEW TEST
                    core::view::AbstractView* viewptr = dynamic_cast<core::view::AbstractView*>(module_ptr);
                    new_module_ptr->is_view = (viewptr != nullptr);
                    new_module_ptr->present.label_visible = graph_ptr->present.GetModuleLabelVisibility();
                }
            }
            // Check for successfully created module
            if (new_module_ptr != nullptr) {
                gui_graph_changed = true;
                // Set remaining module data
                new_module_ptr->name = module_name;
                // Check if module is view instance
                auto view_inst_iter = new_view_instances.find(full_name);
                if (view_inst_iter != new_view_instances.end()) {
                    new_module_ptr->graph_entry_name = view_inst_iter->second;
                    Graph::QueueData queue_data;
                    // Remove all graph entries
                    // for (auto module_ptr : graph_ptr->GetModules()) {
                    //    if (module_ptr->is_view && module_ptr->IsGraphEntry()) {
                    //        module_ptr->graph_entry_name.clear();
                    //        queue_data.name_id = module_ptr->FullName();
                    //        graph_ptr->PushSyncQueue(Graph::QueueAction::REMOVE_graph_entry, queue_data);
                    //    }
                    //}
                    // Add new graph entry
                    queue_data.name_id = new_module_ptr->FullName();
                    graph_ptr->PushSyncQueue(Graph::QueueAction::CREATE_GRAPH_ENTRY, queue_data);
                }
                // Add module to group
                graph_ptr->AddGroupModule(module_namespace, new_module_ptr);
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Unable to get created module. [%s, %s, line %d]\n", full_name.c_str(), __FILE__,
                    __FUNCTION__, __LINE__);
                continue;
            }

            megamol::core::AbstractNamedObjectContainer::child_list_type::const_iterator se =
                module_ptr->ChildList_End();
            for (megamol::core::AbstractNamedObjectContainer::child_list_type::const_iterator si =
                     module_ptr->ChildList_Begin();
                 si != se; ++si) {

                // Parameters
                auto param_slot = dynamic_cast<megamol::core::param::ParamSlot*>((*si).get());
                if (param_slot != nullptr) {
                    std::string param_full_name(param_slot->Name().PeekBuffer());

                    if (use_stock) {
                        for (auto& parameter : new_module_ptr->parameters) {
                            if (this->case_insensitive_str_comp(parameter.full_name, param_full_name)) {
                                megamol::gui::Parameter::ReadNewCoreParameterToExistingParameter(
                                    (*param_slot), parameter, true, false, false);
                            }
                        }
                    } else {
                        std::shared_ptr<Parameter> param_ptr;
                        megamol::gui::Parameter::ReadNewCoreParameterToNewParameter(
                            (*param_slot), param_ptr, false, false, true);
                        new_module_ptr->parameters.emplace_back((*param_ptr));
                    }
                }

                // Add call slots if required and collect call connection data from core instance graph
                std::shared_ptr<core::CallerSlot> caller_slot =
                    std::dynamic_pointer_cast<megamol::core::CallerSlot>((*si));
                if (caller_slot) {
                    if (use_core_instance) {
                        const megamol::core::Call* call = caller_slot->CallAs<megamol::core::Call>();
                        if (call != nullptr) {
                            CallData cd;
                            cd.call_class_name = std::string(call->ClassName());
                            cd.caller_module_full_name =
                                std::string(call->PeekCallerSlot()->Parent()->FullName().PeekBuffer());
                            cd.caller_module_callslot_name = std::string(call->PeekCallerSlot()->Name().PeekBuffer());
                            cd.callee_module_full_name =
                                std::string(call->PeekCalleeSlot()->Parent()->FullName().PeekBuffer());
                            cd.callee_module_callslot_name = std::string(call->PeekCalleeSlot()->Name().PeekBuffer());
                            call_data.emplace_back(cd);
                        }
                    }
                    if (!use_stock) {
                        auto callslot_ptr = std::make_shared<CallSlot>(megamol::gui::GenerateUniqueID());
                        callslot_ptr->name = std::string(caller_slot->Name().PeekBuffer());
                        callslot_ptr->description = std::string(caller_slot->Description().PeekBuffer());
                        callslot_ptr->compatible_call_idxs = this->get_compatible_caller_idxs(caller_slot.get());
                        callslot_ptr->type = CallSlotType::CALLER;
                        callslot_ptr->present.label_visible = graph_ptr->present.GetSlotLabelVisibility();
                        callslot_ptr->ConnectParentModule(new_module_ptr);

                        new_module_ptr->AddCallSlot(callslot_ptr);
                    }
                }
                std::shared_ptr<core::CalleeSlot> callee_slot =
                    std::dynamic_pointer_cast<megamol::core::CalleeSlot>((*si));
                if (callee_slot && !use_stock) {
                    auto callslot_ptr = std::make_shared<CallSlot>(megamol::gui::GenerateUniqueID());
                    callslot_ptr->name = std::string(callee_slot->Name().PeekBuffer());
                    callslot_ptr->description = std::string(callee_slot->Description().PeekBuffer());
                    callslot_ptr->compatible_call_idxs = this->get_compatible_callee_idxs(callee_slot.get());
                    callslot_ptr->type = CallSlotType::CALLEE;
                    callslot_ptr->present.label_visible = graph_ptr->present.GetSlotLabelVisibility();
                    callslot_ptr->ConnectParentModule(new_module_ptr);

                    new_module_ptr->AddCallSlot(callslot_ptr);
                }
            }
        }

        // REMOVE deleted modules from GUI graph ----------------------------------------
        std::map<std::string, ImGuiID> delete_module_map;
        for (auto& module_ptr : graph_ptr->GetModules()) {
            delete_module_map[module_ptr->FullName()] = module_ptr->uid;
        }
        for (auto& module_map : delete_module_map) {
            if (use_megamol_graph) {
                if (!megamol_graph->FindModule(module_map.first)) {
                    graph_ptr->DeleteModule(module_map.second, true);
                    gui_graph_changed = true;
                }
            } else if (use_core_instance) {
                bool found_module = false;
                std::function<void(megamol::core::Module*)> fun = [&](megamol::core::Module* mod) {
                    found_module = true;
                };
                core_instance->FindModuleNoLock(module_map.first, fun);
                if (!found_module) {
                    graph_ptr->DeleteModule(module_map.second, true);
                    gui_graph_changed = true;
                }
            }
        }

        // Collect current call information of gui graph ----------------------
        struct CallInfo {
            std::string class_name;
            std::string from;
            std::string to;
            ImGuiID uid;
        };
        std::vector<CallInfo> gui_graph_call_info;
        for (auto& call_ptr : graph_ptr->GetCalls()) {
            CallInfo call_info;
            call_info.class_name = call_ptr->class_name;
            call_info.uid = call_ptr->uid;
            bool valid_callslot = false;
            auto caller_ptr = call_ptr->GetCallSlot(megamol::gui::CallSlotType::CALLER);
            if (caller_ptr != nullptr) {
                if (caller_ptr->GetParentModule() != nullptr) {
                    call_info.from = caller_ptr->GetParentModule()->FullName() + "::" + caller_ptr->name;
                    valid_callslot = true;
                }
            }
            if (!valid_callslot) {
#ifdef GUI_VERBOSE
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Caller slot is not valid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
#endif // GUI_VERBOSE
                continue;
            }
            valid_callslot = false;
            auto callee_ptr = call_ptr->GetCallSlot(megamol::gui::CallSlotType::CALLEE);
            if (callee_ptr != nullptr) {
                if (callee_ptr->GetParentModule() != nullptr) {
                    call_info.to = callee_ptr->GetParentModule()->FullName() + "::" + callee_ptr->name;
                    valid_callslot = true;
                }
            }
            if (!valid_callslot) {
#ifdef GUI_VERBOSE
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Callee slot is not valid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
#endif // GUI_VERBOSE
                continue;
            }
            gui_graph_call_info.push_back(call_info);
        }

        // ADD new calls to GUI graph ---------------------------------------------------
        if (use_megamol_graph) {
            for (auto& call : megamol_graph->ListCalls()) {
                auto call_ptr = call.callPtr;
                if (call_ptr == nullptr)
                    continue;
                bool add_new_call = true;
                for (auto& call_info : gui_graph_call_info) {
                    if ((call_info.class_name == call.request.className) &&
                        this->case_insensitive_str_comp(call_info.from, call.request.from) &&
                        this->case_insensitive_str_comp(call_info.to, call.request.to)) {
                        add_new_call = false;
                    }
                }
                if (!add_new_call)
                    continue;

                auto call_class_name = call.request.className;
                std::string call_caller_name;
                std::string call_caller_parent_name;
                if (!this->project_separate_name_and_namespace(
                        call.request.from, call_caller_parent_name, call_caller_name)) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Core Project: Invalid call slot name '%s'. [%s, %s, line %d]\n",
                        call.request.from.c_str(), __FILE__, __FUNCTION__, __LINE__);
                }
                std::string call_callee_name;
                std::string call_callee_parent_name;
                if (!this->project_separate_name_and_namespace(
                        call.request.to, call_callee_parent_name, call_callee_name)) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Core Project: Invalid call slot name '%s'. [%s, %s, line %d]\n", call.request.to.c_str(),
                        __FILE__, __FUNCTION__, __LINE__);
                }
                // Full module name required
                call_callee_parent_name = "::" + call_callee_parent_name;
                call_caller_parent_name = "::" + call_caller_parent_name;
                CallData cd;
                cd.call_class_name = call_class_name;
                cd.caller_module_full_name = call_caller_parent_name;
                cd.caller_module_callslot_name = call_caller_name;
                cd.callee_module_full_name = call_callee_parent_name;
                cd.callee_module_callslot_name = call_callee_name;
                call_data.emplace_back(cd);
            }
        } else if (use_core_instance) {
            /// TODO
            /// How to list calls in core instance graph?
        }

        // REMOVE deleted calls from GUI graph ------------------------------------------
        for (auto& call_info : gui_graph_call_info) {
            if (use_megamol_graph) {
                if (!megamol_graph->FindCall(call_info.from, call_info.to)) {
                    graph_ptr->DeleteCall(call_info.uid);
                    gui_graph_changed = true;
                }
            } else if (use_core_instance) {
                /// TODO
                /// How to list calls in core instance graph?
            }
        }

        // Actually create new calls from collected call connection data ----------------
        for (auto& cd : call_data) {
            CallSlotPtr_t callslot_1 = nullptr;
            for (auto& mod : graph_ptr->GetModules()) {
                if (this->case_insensitive_str_comp(mod->FullName(), cd.caller_module_full_name)) {
                    for (auto& callslot : mod->GetCallSlots(CallSlotType::CALLER)) {
                        if (this->case_insensitive_str_comp(callslot->name, cd.caller_module_callslot_name)) {
                            callslot_1 = callslot;
                        }
                    }
                }
            }
            CallSlotPtr_t callslot_2 = nullptr;
            for (auto& mod : graph_ptr->GetModules()) {
                if (this->case_insensitive_str_comp(mod->FullName(), cd.callee_module_full_name)) {
                    for (auto& callslot : mod->GetCallSlots(CallSlotType::CALLEE)) {
                        if (this->case_insensitive_str_comp(callslot->name, cd.callee_module_callslot_name)) {
                            callslot_2 = callslot;
                        }
                    }
                }
            }
            if (graph_ptr->AddCall(this->GetCallsStock(), callslot_1, callslot_2)) {
                gui_graph_changed = true;
            }
        }

        if (gui_graph_changed) {
            graph_ptr->ClearSyncQueue();
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[GUI] Successfully loaded/updated project '%s' from running MegaMol.\n", graph_ptr->name.c_str());
        }
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        this->DeleteGraph(in_graph_uid);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        this->DeleteGraph(in_graph_uid);
        return false;
    }

    return true;
}


ImGuiID megamol::gui::GraphCollection::LoadAddProjectFromFile(
    ImGuiID in_graph_uid, const std::string& project_filename) {

    std::string projectstr;
    if (!megamol::core::utility::FileUtils::ReadFile(project_filename, projectstr)) {
        return false;
    }
    GUIUtils::Utf8Decode(projectstr);

    const std::string luacmd_view("mmCreateView");
    const std::string luacmd_module("mmCreateModule");
    const std::string luacmd_param("mmSetParamValue");
    const std::string luacmd_call("mmCreateCall");

    GraphPtr_t graph_ptr;
    this->GetGraph(in_graph_uid, graph_ptr);
    ImGuiID retval = in_graph_uid;
    // Create new graph if necessary
    if (graph_ptr == nullptr) {
        ImGuiID new_graph_uid = this->AddGraph(GraphCoreInterface::NO_INTERFACE);
        if (new_graph_uid == GUI_INVALID_ID) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Load Project File '%s': Unable to create new graph. [%s, %s, line %d]\n",
                project_filename.c_str(), __FILE__, __FUNCTION__, __LINE__);
            return GUI_INVALID_ID;
        }
        if (!this->GetGraph(new_graph_uid, graph_ptr)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Unable to get pointer to last added graph. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
            return GUI_INVALID_ID;
        }
        retval = new_graph_uid;
    }

    try {
        std::stringstream content(projectstr);
        std::vector<std::string> lines;

        // Prepare and read lines
        std::string tmpline;
        // Split lines at new line
        while (std::getline(content, tmpline, '\n')) {
            // Remove leading spaces
            if (!tmpline.empty()) {
                while (tmpline.front() == ' ') {
                    tmpline.erase(tmpline.begin());
                }
            }
            lines.emplace_back(tmpline);
        }

        size_t lines_count = lines.size();
        for (unsigned int i = 0; i < lines_count; i++) {
            // Lua command must start at beginning after removing leading spaces
            if (lines[i].rfind(luacmd_view, 0) == 0) {

                size_t arg_count = 3;
                std::vector<std::string> args;
                if (!this->read_project_command_arguments(lines[i], arg_count, args)) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Load Project File '%s' line %i: Error parsing lua command '%s' "
                        "requiring %i arguments. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), luacmd_view.c_str(), arg_count, __FILE__, __FUNCTION__,
                        __LINE__);
                    retval = GUI_INVALID_ID;
                    continue;
                }

                std::string view_instance(args[0]);
                std::string view_class_name(args[1]);
                std::string view_full_name(args[2]);
                std::string view_namespace;
                std::string view_name;
                if (!this->project_separate_name_and_namespace(view_full_name, view_namespace, view_name)) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Load Project File '%s' line %i: Invalid view name argument "
                        "(3rd) in lua command '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), luacmd_view.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    retval = GUI_INVALID_ID;
                    continue;
                }

                /// DEBUG
                /// megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                ///     "[GUI] >>>> Instance: '%s' Class: '%s' NameSpace: '%s' Name: '%s' ConfPos: %f, %f.\n",
                ///     view_instance.c_str(), view_class_name.c_str(), view_namespace.c_str(), view_name.c_str());

                // Add module and set as view instance
                if (graph_ptr->AddModule(this->modules_stock, view_class_name) == GUI_INVALID_ID) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Load Project File '%s' line %i: Unable to add new module '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), view_class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    retval = GUI_INVALID_ID;
                    continue;
                }
                auto graph_module = graph_ptr->GetModules().back();

                Graph::QueueData queue_data;
                queue_data.name_id = graph_module->FullName();

                graph_ptr->UniqueModuleRename(view_name);
                graph_module->name = view_name;
                graph_ptr->AddGroupModule(view_namespace, graph_module);
                graph_module->graph_entry_name = view_instance;

                queue_data.rename_id = graph_module->FullName();
                graph_ptr->PushSyncQueue(Graph::QueueAction::RENAME_MODULE, queue_data);

                // Remove all graph entries
                // for (auto module_ptr : graph_ptr->GetModules()) {
                //    if (module_ptr->is_view && module_ptr->IsGraphEntry()) {
                //        module_ptr->graph_entry_name.clear();
                //        queue_data.name_id = module_ptr->FullName();
                //        graph_ptr->PushSyncQueue(Graph::QueueAction::REMOVE_graph_entry, queue_data);
                //    }
                //}

                // Add new graph entry
                queue_data.name_id = queue_data.rename_id;
                graph_ptr->PushSyncQueue(Graph::QueueAction::CREATE_GRAPH_ENTRY, queue_data);

                // Allow only one graph entry at a time
                /// XXX TODO reset other graph entry
            }
        }

        // Save filename for graph
        graph_ptr->SetFilename(project_filename);

        // Find and create modules
        for (unsigned int i = 0; i < lines_count; i++) {
            // Lua command must start at beginning after removeing leading spaces
            if (lines[i].rfind(luacmd_module, 0) == 0) {

                size_t arg_count = 2;
                std::vector<std::string> args;
                if (!this->read_project_command_arguments(lines[i], arg_count, args)) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Load Project File '%s' line %i: Error parsing lua command '%s' "
                        "requiring %i arguments. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), luacmd_module.c_str(), arg_count, __FILE__, __FUNCTION__,
                        __LINE__);
                    retval = GUI_INVALID_ID;
                    continue;
                }

                std::string module_class_name(args[0]);
                std::string module_full_name(args[1]);
                std::string module_namespace;
                std::string module_name;
                if (!this->project_separate_name_and_namespace(module_full_name, module_namespace, module_name)) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Load Project File '%s' line %i: Invalid module name argument "
                        "(2nd) in lua command '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), luacmd_module.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    retval = GUI_INVALID_ID;
                    continue;
                }

                /// DEBUG
                /// megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] >>>> Class: '%s' NameSpace: '%s' Name:
                /// '%s' ConfPos: %f, %f.\n",
                ///    module_class_name.c_str(), module_namespace.c_str(), module_name.c_str());

                // Add module
                if (graph_ptr != nullptr) {
                    if (graph_ptr->AddModule(this->modules_stock, module_class_name) == GUI_INVALID_ID) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] Load Project File '%s' line %i: Unable to add new module '%s'. [%s, %s, line %d]\n",
                            project_filename.c_str(), (i + 1), module_class_name.c_str(), __FILE__, __FUNCTION__,
                            __LINE__);
                        retval = GUI_INVALID_ID;
                        continue;
                    }
                    auto graph_module = graph_ptr->GetModules().back();

                    Graph::QueueData queue_data;
                    queue_data.name_id = graph_module->FullName();

                    graph_ptr->UniqueModuleRename(module_name);
                    graph_module->name = module_name;
                    graph_ptr->AddGroupModule(module_namespace, graph_module);

                    queue_data.rename_id = graph_module->FullName();
                    graph_ptr->PushSyncQueue(Graph::QueueAction::RENAME_MODULE, queue_data);
                }
            }
        }

        // Find and create calls
        for (unsigned int i = 0; i < lines_count; i++) {
            // Lua command must start at beginning after removing leading spaces
            if (lines[i].rfind(luacmd_call, 0) == 0) {

                size_t arg_count = 3;
                std::vector<std::string> args;
                if (!this->read_project_command_arguments(lines[i], arg_count, args)) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Load Project File '%s' line %i: Error parsing lua command '%s' "
                        "requiring %i arguments. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), luacmd_call.c_str(), arg_count, __FILE__, __FUNCTION__,
                        __LINE__);
                    retval = GUI_INVALID_ID;
                    continue;
                }

                std::string call_class_name(args[0]);
                std::string caller_slot_full_name(args[1]);
                std::string callee_slot_full_name(args[2]);

                std::string caller_slot_name;
                std::string caller_slot_namespace;
                if (!this->project_separate_name_and_namespace(
                        caller_slot_full_name, caller_slot_namespace, caller_slot_name)) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Load Project File '%s' line %i: Invalid caller slot name "
                        "argument (2nd) in lua command '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), luacmd_call.c_str(), __FILE__, __FUNCTION__, __LINE__);
                }

                std::string callee_slot_name;
                std::string callee_slot_namespace;
                if (!this->project_separate_name_and_namespace(
                        callee_slot_full_name, callee_slot_namespace, callee_slot_name)) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Load Project File '%s' line %i: Invalid callee slot name "
                        "argument (3nd) in lua command '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), luacmd_call.c_str(), __FILE__, __FUNCTION__, __LINE__);
                }

                /// DEBUG
                /// megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                ///    "[GUI] >>>> Call Name: '%s' CALLER Module: '%s' Slot: '%s' - CALLEE Module: '%s' Slot: '%s'.\n",
                ///    call_class_name.c_str(), caller_slot_namespace.c_str(), caller_slot_name.c_str(),
                ///    callee_slot_namespace.c_str(), callee_slot_name.c_str());

                // Searching for call
                if (graph_ptr != nullptr) {
                    std::string module_full_name;
                    size_t module_name_idx = std::string::npos;
                    std::string callee_name, caller_name;
                    CallSlotPtr_t callee_slot, caller_slot;

                    for (auto& mod : graph_ptr->GetModules()) {
                        module_full_name = mod->FullName() + "::";
                        // Caller
                        module_name_idx = caller_slot_full_name.find(module_full_name);
                        if (module_name_idx != std::string::npos) {
                            for (auto& callslot_map : mod->GetCallSlots()) {
                                for (auto& callslot : callslot_map.second) {
                                    if (this->case_insensitive_str_comp(caller_slot_name, callslot->name)) {
                                        caller_slot = callslot;
                                    }
                                }
                            }
                        }
                        // Callee
                        module_name_idx = callee_slot_full_name.find(module_full_name);
                        if (module_name_idx != std::string::npos) {
                            for (auto& callslot_map : mod->GetCallSlots()) {
                                for (auto& callslot : callslot_map.second) {
                                    if (this->case_insensitive_str_comp(callee_slot_name, callslot->name)) {
                                        callee_slot = callslot;
                                    }
                                }
                            }
                        }
                    }

                    if (callee_slot == nullptr) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] Load Project File '%s' line %i: Unable to find callee slot '%s' "
                            "for creating call '%s'. [%s, %s, line %d]\n",
                            project_filename.c_str(), (i + 1), callee_slot_full_name.c_str(), call_class_name.c_str(),
                            __FILE__, __FUNCTION__, __LINE__);
                        retval = GUI_INVALID_ID;
                        continue;
                    }
                    if (caller_slot == nullptr) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] Load Project File '%s' line %i: Unable to find caller slot '%s' "
                            "for creating call '%s'. [%s, %s, line %d]\n",
                            project_filename.c_str(), (i + 1), caller_slot_full_name.c_str(), call_class_name.c_str(),
                            __FILE__, __FUNCTION__, __LINE__);
                        retval = GUI_INVALID_ID;
                        continue;
                    }


                    // Add call
                    if (!graph_ptr->AddCall(this->calls_stock, caller_slot, callee_slot)) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] Load Project File '%s' line %i: Unable to add new call '%s'. [%s, %s, line %d]\n",
                            project_filename.c_str(), (i + 1), call_class_name.c_str(), __FILE__, __FUNCTION__,
                            __LINE__);
                        retval = GUI_INVALID_ID;
                        continue;
                    }
                }
            }
        }

        // Find and create parameters
        for (unsigned int i = 0; i < lines_count; i++) {
            // Lua command must start at beginning after removeing leading spaces
            if (lines[i].rfind(luacmd_param, 0) == 0) {
                const std::string start_delimieter("[=[");
                const std::string end_delimieter("]=]");

                std::string param_line = lines[i];

                size_t first_bracket_idx = param_line.find('(');
                if (first_bracket_idx == std::string::npos) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Load Project File '%s' line %i: Missing opening brackets for '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), luacmd_param.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    retval = GUI_INVALID_ID;
                    continue;
                }
                size_t first_delimiter_idx = param_line.find(',');
                if (first_delimiter_idx == std::string::npos) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Load Project File '%s' line %i: Missing argument delimiter ',' for '%s'. [%s, %s, line "
                        "%d]\n",
                        project_filename.c_str(), (i + 1), luacmd_param.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    retval = GUI_INVALID_ID;
                    continue;
                }

                std::string param_slot_full_name =
                    param_line.substr(first_bracket_idx + 1, (first_delimiter_idx - first_bracket_idx - 1));
                if ((param_slot_full_name.front() != '"') || (param_slot_full_name.back() != '"')) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Load Project File '%s' line %i: Parameter name argument should "
                        "be enclosed in '\"' for '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), luacmd_param.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    retval = GUI_INVALID_ID;
                    continue;
                }
                param_slot_full_name = param_slot_full_name.substr(1, param_slot_full_name.size() - 2);

                /// DEBUG
                /// megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] >>>> %s\n",
                /// param_slot_full_name.c_str());

                // Copy multi line parameter values into one string
                auto value_start_idx = param_line.find(start_delimieter);
                if (value_start_idx == std::string::npos) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Load Project File '%s' line %i: Unable to find parameter value "
                        "start delimiter '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), start_delimieter.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    retval = GUI_INVALID_ID;
                    continue;
                }
                bool found_end_delimiter = true;
                if (param_line.find(end_delimieter) == std::string::npos) {
                    found_end_delimiter = false;
                    for (unsigned int j = (i + 1); j < lines_count; j++) {
                        param_line += (lines[j] + '\n');
                        if (lines[j].find(end_delimieter) != std::string::npos) {
                            found_end_delimiter = true;
                            break;
                        }
                    }
                }
                if (!found_end_delimiter) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Load Project File '%s' line %i: Unable to find parameter value "
                        "end delimiter '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), end_delimieter.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    retval = GUI_INVALID_ID;
                    continue;
                }
                std::string value_str = param_line.substr(value_start_idx + start_delimieter.size(),
                    (param_line.find(end_delimieter)) - value_start_idx - end_delimieter.size());

                /// DEBUG
                /// megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] >>>> '%s'\n", value_str.c_str());

                // Searching for parameter
                if (graph_ptr != nullptr) {
                    std::string module_full_name;
                    size_t module_name_idx = std::string::npos;
                    for (auto& module_ptr : graph_ptr->GetModules()) {
                        module_full_name = module_ptr->FullName() + "::";
                        module_name_idx = param_slot_full_name.find(module_full_name);
                        if (module_name_idx != std::string::npos) {
                            std::string param_full_name =
                                param_slot_full_name.substr(module_name_idx + module_full_name.size());
                            for (auto& parameter : module_ptr->parameters) {
                                if (this->case_insensitive_str_comp(parameter.full_name, param_full_name)) {
                                    parameter.SetValueString(value_str);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Load gui state from file
        if (!this->load_state_from_file(project_filename, graph_ptr->uid)) {
            // Layout graph if no positions for modules could be found in state.
            graph_ptr->present.SetLayoutGraph();
        }
        graph_ptr->ResetDirty();

        megamol::core::utility::log::Log::DefaultLog.WriteInfo(
            "[GUI] Successfully loaded project '%s' from file '%s'.\n", graph_ptr->name.c_str(),
            project_filename.c_str());

    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    }

    return retval;
}


bool megamol::gui::GraphCollection::SaveProjectToFile(
    ImGuiID in_graph_uid, const std::string& project_filename, const std::string& state_json) {

    try {
        for (auto& graph_ptr : this->graphs) {
            if (graph_ptr->uid == in_graph_uid) {

                // Some pre-checks --------------------------------------------
                bool found_error = false;
                for (auto& mod_1 : graph_ptr->GetModules()) {
                    for (auto& mod_2 : graph_ptr->GetModules()) {
                        if ((mod_1 != mod_2) && (mod_1->FullName() == mod_2->FullName())) {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] Save Project File '%s': Found non unique module name: %s [%s, %s, line %d]\n",
                                project_filename.c_str(), mod_1->FullName().c_str(), __FILE__, __FUNCTION__, __LINE__);
                            found_error = true;
                        }
                    }
                }
                if (found_error)
                    return false;

                // Serialze graph to string -----------------------------------
                std::string projectstr;
                std::stringstream confInstances, confModules, confCalls, confParams, confGUIState;
                for (auto& module_ptr : graph_ptr->GetModules()) {
                    if (module_ptr->IsGraphEntry()) {
                        confInstances << "mmCreateView(\"" << module_ptr->graph_entry_name << "\",\""
                                      << module_ptr->class_name << "\",\"" << module_ptr->FullName() << "\") \n";
                    } else {
                        confModules << "mmCreateModule(\"" << module_ptr->class_name << "\",\""
                                    << module_ptr->FullName() << "\") \n";
                    }

                    for (auto& parameter : module_ptr->parameters) {
                        // - Write all parameters for running graph (default value is not available)
                        // - For other graphs only write parameters with other values than the default
                        // - Ignore button parameters
                        if ((graph_ptr->HasCoreInterface() || parameter.DefaultValueMismatch()) &&
                            (parameter.type != Param_t::BUTTON)) {
                            // Encode to UTF-8 string
                            vislib::StringA valueString;
                            vislib::UTF8Encoder::Encode(
                                valueString, vislib::StringA(parameter.GetValueString().c_str()));
                            confParams << "mmSetParamValue(\"" << module_ptr->FullName() << "::" << parameter.full_name
                                       << "\",[=[" << std::string(valueString.PeekBuffer()) << "]=])\n";
                        }
                    }

                    for (auto& caller_slot : module_ptr->GetCallSlots(CallSlotType::CALLER)) {
                        for (auto& call : caller_slot->GetConnectedCalls()) {
                            if (call->IsConnected()) {
                                confCalls << "mmCreateCall(\"" << call->class_name << "\",\""
                                          << call->GetCallSlot(CallSlotType::CALLER)->GetParentModule()->FullName()
                                          << "::" << call->GetCallSlot(CallSlotType::CALLER)->name << "\",\""
                                          << call->GetCallSlot(CallSlotType::CALLEE)->GetParentModule()->FullName()
                                          << "::" << call->GetCallSlot(CallSlotType::CALLEE)->name << "\")\n";
                            }
                        }
                    }
                }
                if (!state_json.empty()) {
                    confGUIState << (GUI_PROJECT_GUI_STATE_START_TAG) << state_json << (GUI_PROJECT_GUI_STATE_END_TAG);
                }

                projectstr = confInstances.str() + "\n" + confModules.str() + "\n" + confCalls.str() + "\n" +
                             confParams.str() + "\n" + confGUIState.str();

                graph_ptr->ResetDirty();
                if (megamol::core::utility::FileUtils::WriteFile(project_filename, projectstr)) {
                    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                        "[GUI] Successfully saved project '%s' to file '%s'.\n", graph_ptr->name.c_str(),
                        project_filename.c_str());

                    // Save filename for graph
                    graph_ptr->SetFilename(project_filename);

                    return true;
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

    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
        "[GUI] Invalid graph uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}


bool megamol::gui::GraphCollection::get_module_stock_data(
    Module::StockModule& mod, const std::shared_ptr<const megamol::core::factories::ModuleDescription> mod_desc) {

    mod.class_name = std::string(mod_desc->ClassName());
    mod.description = std::string(mod_desc->Description());
    mod.is_view = false;
    mod.parameters.clear();
    mod.callslots.clear();
    mod.callslots.emplace(CallSlotType::CALLER, std::vector<CallSlot::StockCallSlot>());
    mod.callslots.emplace(CallSlotType::CALLEE, std::vector<CallSlot::StockCallSlot>());
    /// mod.plugin_name is not (yet) available in mod_desc (set from AbstractAssemblyInstance or
    /// AbstractPluginInstance).

    if (this->calls_stock.empty()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Call list is empty. Call read_call_data() prior to that. [%s, %s, line %d]\n", __FILE__,
            __FUNCTION__, __LINE__);
        return false;
    }

    try {
        /// Following code is adapted from megamol::core::job::job::PluginsStateFileGeneratorJob.cpp

        /// DEBUG
        // megamol::core::utility::log::Log::DefaultLog.WriteInfo(
        //    "[GUI] [DEBUG] Temporary creating module '%s'...", mod_desc->ClassName());

        megamol::core::Module::ptr_type new_mod(mod_desc->CreateModule(nullptr));
        if (new_mod == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Unable to create module '%s'. [%s, %s, line %d]\n", mod_desc->ClassName(), __FILE__,
                __FUNCTION__, __LINE__);
            return false;
        }
        megamol::core::RootModuleNamespace::ptr_type root_mod_ns =
            std::make_shared<megamol::core::RootModuleNamespace>();
        root_mod_ns->AddChild(new_mod);

        /// DEBUG
        // megamol::core::utility::log::Log::DefaultLog.WriteInfo(
        //    "[GUI] [DEBUG] Created temporary module '%s'.", mod_desc->ClassName());

        /// XXX VIEW TEST
        std::shared_ptr<const core::view::AbstractView> viewptr =
            std::dynamic_pointer_cast<const core::view::AbstractView>(new_mod);

        mod.is_view = (viewptr != nullptr);

        std::vector<std::shared_ptr<core::param::ParamSlot>> paramSlots;
        std::vector<std::shared_ptr<core::CallerSlot>> callerSlots;
        std::vector<std::shared_ptr<core::CalleeSlot>> calleeSlots;

        core::Module::child_list_type::iterator ano_end = new_mod->ChildList_End();
        for (core::Module::child_list_type::iterator ano_i = new_mod->ChildList_Begin(); ano_i != ano_end; ++ano_i) {
            std::shared_ptr<core::param::ParamSlot> p_ptr = std::dynamic_pointer_cast<core::param::ParamSlot>(*ano_i);
            if (p_ptr != nullptr)
                paramSlots.push_back(p_ptr);
            std::shared_ptr<core::CallerSlot> cr_ptr = std::dynamic_pointer_cast<core::CallerSlot>(*ano_i);
            if (cr_ptr != nullptr)
                callerSlots.push_back(cr_ptr);
            std::shared_ptr<core::CalleeSlot> ce_ptr = std::dynamic_pointer_cast<core::CalleeSlot>(*ano_i);
            if (ce_ptr != nullptr)
                calleeSlots.push_back(ce_ptr);
        }

        // Param Slots
        for (std::shared_ptr<core::param::ParamSlot> param_slot : paramSlots) {
            if (param_slot == nullptr)
                continue;
            Parameter::StockParameter psd;
            if (megamol::gui::Parameter::ReadNewCoreParameterToStockParameter((*param_slot), psd)) {
                mod.parameters.emplace_back(psd);
            }
        }

        // CallerSlots
        for (std::shared_ptr<core::CallerSlot> caller_slot : callerSlots) {
            CallSlot::StockCallSlot csd;
            csd.name = std::string(caller_slot->Name().PeekBuffer());
            csd.description = std::string(caller_slot->Description().PeekBuffer());
            csd.compatible_call_idxs = this->get_compatible_caller_idxs(caller_slot.get());
            csd.type = CallSlotType::CALLER;

            mod.callslots[csd.type].emplace_back(csd);
        }

        // CalleeSlots
        for (std::shared_ptr<core::CalleeSlot> callee_slot : calleeSlots) {
            CallSlot::StockCallSlot csd;
            csd.name = std::string(callee_slot->Name().PeekBuffer());
            csd.description = std::string(callee_slot->Description().PeekBuffer());
            csd.compatible_call_idxs = this->get_compatible_callee_idxs(callee_slot.get());
            csd.type = CallSlotType::CALLEE;

            mod.callslots[csd.type].emplace_back(csd);
        }

        paramSlots.clear();
        callerSlots.clear();
        calleeSlots.clear();
        root_mod_ns->RemoveChild(new_mod);
        new_mod->SetAllCleanupMarks();
        new_mod->PerformCleanup();
        new_mod = nullptr;

        /// DEBUG
        // megamol::core::utility::log::Log::DefaultLog.WriteInfo(
        //    "[GUI] [DEBUG] Removed temporary module '%s'.", mod_desc->ClassName());

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


bool megamol::gui::GraphCollection::get_call_stock_data(
    Call::StockCall& call, const std::shared_ptr<const megamol::core::factories::CallDescription> call_desc) {

    try {
        call.class_name = std::string(call_desc->ClassName());
        call.description = std::string(call_desc->Description());
        call.functions.clear();
        for (unsigned int i = 0; i < call_desc->FunctionCount(); ++i) {
            call.functions.emplace_back(call_desc->FunctionName(i));
        }
        /// call.plugin_name is not (yet) available in call_desc (set from AbstractAssemblyInstance or
        /// AbstractPluginInstance).

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


bool megamol::gui::GraphCollection::read_project_command_arguments(
    const std::string& line, size_t arg_count, std::vector<std::string>& out_args) {

    /// Can be used for mmCreateView, mmCreateModule and mmCreateCall lua commands

    // (Leaving current line in original state)
    std::string args_str = line;
    out_args.clear();

    // Searching for command delimiter
    auto start = args_str.find('(');
    auto end = args_str.rfind(')');
    if ((start == std::string::npos) || (end == std::string::npos)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI]  Missing opening and/or closing bracket(s). [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    args_str = args_str.substr(start + 1, (end - start) - 1);

    // Getting arguments
    std::string arg;
    const char args_delim(',');
    const std::string args_string_delim("\"");
    size_t str_delim_idx_1 = std::string::npos;
    size_t str_delim_idx_2 = std::string::npos;
    size_t delim_idx = std::string::npos;

    for (size_t i = 0; i < arg_count; i++) {
        str_delim_idx_1 = args_str.find(args_string_delim);
        str_delim_idx_2 = str_delim_idx_1 + args_str.substr(str_delim_idx_1 + 1).find(args_string_delim);
        if ((str_delim_idx_1 == std::string::npos) || (str_delim_idx_2 == std::string::npos)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Missing argument string delimiter '%s' at position %i. [%s, %s, line %d]\n",
                args_string_delim.c_str(), (i + 1), __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        arg = args_str.substr(str_delim_idx_1 + 1, str_delim_idx_2 - str_delim_idx_1);
        out_args.emplace_back(arg);
        args_str = args_str.substr(str_delim_idx_2 + 2);
        if (i < (arg_count - 1)) {
            delim_idx = args_str.find(args_delim);
            if (delim_idx == std::string::npos) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Missing argument  delimiter '%c' at position %i. [%s, %s, line %d]\n", args_delim, (i + 1),
                    __FILE__, __FUNCTION__, __LINE__);
                return false;
            }
            args_str = args_str.substr(delim_idx + 1);
        }
    }

    if (out_args.size() != arg_count) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Argument count error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


ImVec2 megamol::gui::GraphCollection::project_read_confpos(const std::string& line) {

    ImVec2 conf_pos(FLT_MAX, FLT_MAX);

    std::string x_start_tag("--confPos={X=");
    std::string y_start_tag(",Y=");
    std::string end_tag("}");
    size_t x_start_idx = line.find(x_start_tag);
    size_t y_start_idx = line.find(y_start_tag);
    size_t end_idx = line.find(end_tag);
    if ((x_start_idx != std::string::npos) && (y_start_idx != std::string::npos) && (end_idx != std::string::npos)) {
        size_t x_length = ((y_start_idx) - (x_start_idx + x_start_tag.length()));
        size_t y_length = ((end_idx) - (y_start_idx + y_start_tag.length()));
        float x = 0.0f;
        float y = 0.0f;
        try {
            std::string val_str = line.substr(x_start_idx + x_start_tag.length(), x_length);
            x = std::stof(val_str);
        } catch (std::invalid_argument& e) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI]  Error while reading x value of confPos: %s [%s, %s, line %d]\n", e.what(), __FILE__,
                __FUNCTION__, __LINE__);
            return conf_pos;
        }
        try {
            std::string val_str = line.substr(y_start_idx + y_start_tag.length(), y_length);
            y = std::stof(val_str);
        } catch (std::invalid_argument& e) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI]  Error while reading y value of confPos: %s [%s, %s, line %d]\n", e.what(), __FILE__,
                __FUNCTION__, __LINE__);
            return conf_pos;
        }
        conf_pos = ImVec2(x, y);
    }

    return conf_pos;
}


bool megamol::gui::GraphCollection::project_separate_name_and_namespace(
    const std::string& full_name, std::string& name_space, std::string& name) {

    name = full_name;
    name_space = "";
    size_t delimiter_index = full_name.rfind("::");
    if (delimiter_index != std::string::npos) {
        name = full_name.substr(delimiter_index + 2);
        name_space = full_name.substr(0, delimiter_index);
        if (!name_space.empty()) {
            if (name_space.find("::") == 0) {
                name_space = name_space.substr(2);
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Invalid namespace in argument. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                return false;
            }
        }
    }
    if (name.empty()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Invalid name in argument. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


std::vector<size_t> megamol::gui::GraphCollection::get_compatible_callee_idxs(
    const megamol::core::CalleeSlot* callee_slot) {

    std::vector<size_t> retval;
    retval.clear();
    if (callee_slot == nullptr)
        return retval;

    SIZE_T callbackCount = callee_slot->GetCallbackCount();
    std::vector<std::string> callNames, funcNames;
    std::set<std::string> uniqueCallNames, completeCallNames;
    for (SIZE_T i = 0; i < callbackCount; ++i) {
        uniqueCallNames.insert(callee_slot->GetCallbackCallName(i));
        callNames.push_back(callee_slot->GetCallbackCallName(i));
        funcNames.push_back(callee_slot->GetCallbackFuncName(i));
    }
    size_t ll = callNames.size();
    assert(ll == funcNames.size());
    for (std::string callName : uniqueCallNames) {
        bool found_call = false;
        Call::StockCall call;
        for (auto& c : this->calls_stock) {
            if (callName == c.class_name) {
                call = c;
                found_call = true;
                break;
            }
        }
        bool allFound = true;
        if (found_call) {
            for (auto& func_name : call.functions) {
                bool found = false;
                for (size_t j = 0; j < ll; ++j) {
                    if ((callNames[j] == callName) && (funcNames[j] == func_name)) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    allFound = false;
                    break;
                }
            }
        } else {
            allFound = false;
        }
        if (allFound) {
            completeCallNames.insert(callName);
        }
    }
    for (std::string callName : completeCallNames) {
        size_t calls_cnt = this->calls_stock.size();
        for (size_t idx = 0; idx < calls_cnt; ++idx) {
            // Case-Insensitive call slot comparison
            if (this->case_insensitive_str_comp(this->calls_stock[idx].class_name, callName)) {
                retval.emplace_back(idx);
            }
        }
    }

    return retval;
}


std::vector<size_t> megamol::gui::GraphCollection::get_compatible_caller_idxs(
    const megamol::core::CallerSlot* caller_slot) {

    std::vector<size_t> retval;
    retval.clear();
    if (caller_slot == nullptr)
        return retval;

    SIZE_T callCount = caller_slot->GetCompCallCount();
    for (SIZE_T i = 0; i < callCount; ++i) {
        std::string comp_call_class_name = std::string(caller_slot->GetCompCallClassName(i));
        size_t calls_cnt = this->calls_stock.size();
        for (size_t idx = 0; idx < calls_cnt; ++idx) {
            // Case-Insensitive call slot comparison
            if (this->case_insensitive_str_comp(this->calls_stock[idx].class_name, comp_call_class_name)) {
                retval.emplace_back(idx);
            }
        }
    }

    return retval;
}


std::string megamol::gui::GraphCollection::GetUpdatedGUIState(ImGuiID graph_id, const std::string& filename) {

    nlohmann::json state_json;

    // Try to load existing gui state from file
    std::string state_str;
    if (megamol::core::utility::FileUtils::ReadFile(filename, state_str, true)) {
        state_str = GUIUtils::ExtractGUIState(state_str);
        if (!state_str.empty()) {
            state_json = nlohmann::json::parse(state_str);
            if (!state_json.is_object()) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Invalid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                return std::string("");
            }
        }
    }

    GraphPtr_t graph_ptr;
    if (this->GetGraph(graph_id, graph_ptr)) {
        // Write/replace GUI_JSON_TAG_PROJECT graph state
        try {
            state_json[GUI_JSON_TAG_GRAPHS].erase(GUI_JSON_TAG_PROJECT);
        } catch (...) {}
        graph_ptr->StateToJSON(state_json);

        // Write/replace GUI state of parameters (groups)
        try {
            state_json.erase(GUI_JSON_TAG_GUISTATE_PARAMETERS);
        } catch (...) {}
        for (auto& module_ptr : graph_ptr->GetModules()) {
            std::string module_full_name = module_ptr->FullName();
            // Parameter Groups
            module_ptr->present.param_groups.StateToJSON(state_json, module_full_name);
            // Parameters
            for (auto& param : module_ptr->parameters) {
                std::string param_full_name = module_full_name + "::" + param.full_name;
                param.present.StateToJSON(state_json, param_full_name);
            }
        }
        state_str = state_json.dump(); // No line feed
        return state_str;
    }
    return std::string("");
}


bool megamol::gui::GraphCollection::load_state_from_file(const std::string& filename, ImGuiID graph_id) {

    std::string state_str;
    if (megamol::core::utility::FileUtils::ReadFile(filename, state_str, true)) {
        state_str = GUIUtils::ExtractGUIState(state_str);
        if (state_str.empty())
            return false;
        nlohmann::json json;
        json = nlohmann::json::parse(state_str);
        if (!json.is_object()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Invalid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        GraphPtr_t graph_ptr;
        if (this->GetGraph(graph_id, graph_ptr)) {

            // Read GUI state of parameters (groups)
            for (auto& module_ptr : graph_ptr->GetModules()) {
                std::string module_full_name = module_ptr->FullName();
                // Parameter Groups
                module_ptr->present.param_groups.StateFromJSON(json, module_full_name);
                // Parameters
                for (auto& param : module_ptr->parameters) {
                    std::string param_full_name = module_full_name + "::" + param.full_name;
                    param.present.StateFromJSON(json, param_full_name);
                    param.present.ForceSetGUIStateDirty();
                }
            }

            // Read GUI_JSON_TAG_PROJECT graph state
            if (graph_ptr->StateFromJSON(json)) {
                // Disable layouting if graph state was found
                graph_ptr->present.SetLayoutGraph(false);
            }

            return true;
        }
    }
    return false;
}
