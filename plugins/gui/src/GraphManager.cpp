/*
 * GraphManager.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "GraphManager.h"


using namespace megamol;
using namespace megamol::gui;
using vislib::sys::Log;


megamol::gui::GraphManager::GraphManager(void) : graphs(), modules_stock(), calls_stock(), generated_uid(0) {}


megamol::gui::GraphManager::~GraphManager(void) {}


bool megamol::gui::GraphManager::AddGraph(std::string name) {

    this->graphs.emplace_back(std::make_shared<Graph>(this->get_unique_id(), name));

    return true;
}


bool megamol::gui::GraphManager::DeleteGraph(int graph_uid) {

    for (auto iter = this->graphs.begin(); iter != this->graphs.end(); iter++) {
        if ((*iter)->GetUID() == graph_uid) {

            vislib::sys::Log::DefaultLog.WriteWarn("Found %i references pointing to graph. [%s, %s, line %d]\n",
                (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
            assert((*iter).use_count() == 1);

            (*iter) = nullptr;
            this->graphs.erase(iter);
            return true;
        }
    }

    return false;
}


const GraphManager::GraphsType& megamol::gui::GraphManager::GetGraphs(void) { return this->graphs; }


bool megamol::gui::GraphManager::UpdateModulesCallsStock(const megamol::core::CoreInstance* core_instance) {

    bool retval = true;
    if (core_instance == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to Core Instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    if (!this->calls_stock.empty() || !this->modules_stock.empty()) {
        vislib::sys::Log::DefaultLog.WriteWarn("Modules and calls stock already exists. Deleting exiting stock and "
                                               "recreating new one. [%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
    }
    this->calls_stock.clear();
    this->modules_stock.clear();

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
                    Graph::StockCall call;
                    call.plugin_name = plugin_name;
                    this->get_call_stock_data(call, c_desc);
                    this->calls_stock.emplace_back(call);
                }
            }

            // Get core calls
            plugin_name = "Core"; // (core_instance->GetAssemblyName() = "")
            for (auto& c_desc : core_instance->GetCallDescriptionManager()) {
                std::string class_name = std::string(c_desc->ClassName());
                if (std::find_if(this->calls_stock.begin(), this->calls_stock.end(),
                        [class_name](const Graph::StockCall& call) { return (call.class_name == class_name); }) ==
                    this->calls_stock.end()) {
                    Graph::StockCall call;
                    call.plugin_name = plugin_name;
                    this->get_call_stock_data(call, c_desc);
                    this->calls_stock.emplace_back(call);
                }
            }
        }

        // MODULES ----------------------------------------------------------------
        if (this->modules_stock.empty()) {

            // Get plugin modules (get prior to core modules for being  able to find duplicates in core instance module
            // desc. manager)
            const std::vector<core::utility::plugins::AbstractPluginInstance::ptr_type>& plugins =
                core_instance->Plugins().GetPlugins();
            for (core::utility::plugins::AbstractPluginInstance::ptr_type plugin : plugins) {
                plugin_name = plugin->GetAssemblyName();
                for (auto& m_desc : plugin->GetModuleDescriptionManager()) {
                    Graph::StockModule mod;
                    mod.plugin_name = plugin_name;
                    this->get_module_stock_data(mod, m_desc);
                    this->modules_stock.emplace_back(mod);
                }
            }

            // Get core modules
            plugin_name = "Core"; // (core_instance->GetAssemblyName() = "")
            for (auto& m_desc : core_instance->GetModuleDescriptionManager()) {
                std::string class_name = std::string(m_desc->ClassName());
                if (std::find_if(this->modules_stock.begin(), this->modules_stock.end(),
                        [class_name](const Graph::StockModule& mod) { return (mod.class_name == class_name); }) ==
                    this->modules_stock.end()) {
                    Graph::StockModule mod;
                    mod.plugin_name = plugin_name;
                    this->get_module_stock_data(mod, m_desc);
                    this->modules_stock.emplace_back(mod);
                }
            }

            // Sorting module by alphabetically ascending class names.
            std::sort(this->modules_stock.begin(), this->modules_stock.end(),
                [](Graph::StockModule mod1, Graph::StockModule mod2) {
                    std::vector<std::string> v;
                    v.clear();
                    v.emplace_back(mod1.class_name);
                    v.emplace_back(mod2.class_name);
                    std::sort(v.begin(), v.end());
                    return (v.front() != mod2.class_name);
                });
        }

        auto delta_time =
            static_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - start_time).count();
        vislib::sys::Log::DefaultLog.WriteInfo(
            "Reading available modules and calls ... DONE (duration: %.3f seconds)\n",
            delta_time); // [%s, %s, line %d]\n", delta_time, __FILE__, __FUNCTION__, __LINE__);
    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return retval;
}


bool megamol::gui::GraphManager::get_module_stock_data(
    Graph::StockModule& mod, const std::shared_ptr<const megamol::core::factories::ModuleDescription> mod_desc) {

    /// mod.plugin_name is not available in mod_desc (set from AbstractAssemblyInstance or AbstractPluginInstance).
    mod.class_name = std::string(mod_desc->ClassName());
    mod.description = std::string(mod_desc->Description());
    mod.is_view = false;
    mod.param_slots.clear();
    mod.call_slots.clear();
    mod.call_slots.emplace(Graph::CallSlotType::CALLER, std::vector<Graph::StockCallSlot>());
    mod.call_slots.emplace(Graph::CallSlotType::CALLEE, std::vector<Graph::StockCallSlot>());

    if (this->calls_stock.empty()) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Call list is empty. Call read_call_data() prior to that. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
        return false;
    }

    try {
        // SLOTS ------------------------------------------------------------------
        /// (Following code is adapted from megamol::core::job::job::PluginsStateFileGeneratorJob.cpp)

        megamol::core::Module::ptr_type new_mod(mod_desc->CreateModule(nullptr));
        if (new_mod == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError("Unable to create module: %s. [%s, %s, line %d]\n",
                mod_desc->ClassName(), __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        megamol::core::RootModuleNamespace::ptr_type root_mod_ns =
            std::make_shared<megamol::core::RootModuleNamespace>();
        root_mod_ns->AddChild(new_mod);

        std::shared_ptr<const core::view::AbstractView> viewptr =
            std::dynamic_pointer_cast<const core::view::AbstractView>(new_mod);
        mod.is_view = (viewptr != nullptr);

        std::vector<std::shared_ptr<core::param::ParamSlot>> paramSlots;
        std::vector<std::shared_ptr<core::CallerSlot>> callerSlots;
        std::vector<std::shared_ptr<core::CalleeSlot>> calleeSlots;

        core::Module::child_list_type::iterator ano_end = new_mod->ChildList_End();
        for (core::Module::child_list_type::iterator ano_i = new_mod->ChildList_Begin(); ano_i != ano_end; ++ano_i) {
            std::shared_ptr<core::param::ParamSlot> p_ptr = std::dynamic_pointer_cast<core::param::ParamSlot>(*ano_i);
            if (p_ptr != nullptr) paramSlots.push_back(p_ptr);
            std::shared_ptr<core::CallerSlot> cr_ptr = std::dynamic_pointer_cast<core::CallerSlot>(*ano_i);
            if (cr_ptr != nullptr) callerSlots.push_back(cr_ptr);
            std::shared_ptr<core::CalleeSlot> ce_ptr = std::dynamic_pointer_cast<core::CalleeSlot>(*ano_i);
            if (ce_ptr != nullptr) calleeSlots.push_back(ce_ptr);
        }

        // Param Slots
        for (std::shared_ptr<core::param::ParamSlot> param_slot : paramSlots) {
            Graph::StockParamSlot psd;
            psd.class_name = std::string(param_slot->Name().PeekBuffer());
            psd.description = std::string(param_slot->Description().PeekBuffer());

            if (auto* p_ptr = param_slot->Param<core::param::ButtonParam>()) {
                psd.type = Graph::ParamType::BUTTON;
            } else if (auto* p_ptr = param_slot->Param<core::param::BoolParam>()) {
                psd.type = Graph::ParamType::BOOL;
            } else if (auto* p_ptr = param_slot->Param<core::param::ColorParam>()) {
                psd.type = Graph::ParamType::COLOR;
            } else if (auto* p_ptr = param_slot->Param<core::param::EnumParam>()) {
                psd.type = Graph::ParamType::ENUM;
            } else if (auto* p_ptr = param_slot->Param<core::param::FilePathParam>()) {
                psd.type = Graph::ParamType::FILEPATH;
            } else if (auto* p_ptr = param_slot->Param<core::param::FlexEnumParam>()) {
                psd.type = Graph::ParamType::FLEXENUM;
            } else if (auto* p_ptr = param_slot->Param<core::param::FloatParam>()) {
                psd.type = Graph::ParamType::FLOAT;
            } else if (auto* p_ptr = param_slot->Param<core::param::IntParam>()) {
                psd.type = Graph::ParamType::INT;
            } else if (auto* p_ptr = param_slot->Param<core::param::StringParam>()) {
                psd.type = Graph::ParamType::STRING;
            } else if (auto* p_ptr = param_slot->Param<core::param::TernaryParam>()) {
                psd.type = Graph::ParamType::TERNARY;
            } else if (auto* p_ptr = param_slot->Param<core::param::TransferFunctionParam>()) {
                psd.type = Graph::ParamType::TRANSFERFUNCTION;
            } else if (auto* p_ptr = param_slot->Param<core::param::Vector2fParam>()) {
                psd.type = Graph::ParamType::VECTOR2F;
            } else if (auto* p_ptr = param_slot->Param<core::param::Vector3fParam>()) {
                psd.type = Graph::ParamType::VECTOR3F;
            } else if (auto* p_ptr = param_slot->Param<core::param::Vector4fParam>()) {
                psd.type = Graph::ParamType::VECTOR4F;
            } else {
                psd.type = Graph::ParamType::UNKNOWN;
            }

            mod.param_slots.emplace_back(psd);
        }

        // CallerSlots
        for (std::shared_ptr<core::CallerSlot> caller_slot : callerSlots) {
            Graph::StockCallSlot csd;
            csd.name = std::string(caller_slot->Name().PeekBuffer());
            csd.description = std::string(caller_slot->Description().PeekBuffer());
            csd.compatible_call_idxs.clear();
            csd.type = Graph::CallSlotType::CALLER;

            SIZE_T callCount = caller_slot->GetCompCallCount();
            for (SIZE_T i = 0; i < callCount; ++i) {
                std::string comp_call_class_name = std::string(caller_slot->GetCompCallClassName(i));
                size_t calls_cnt = this->calls_stock.size();
                for (size_t idx = 0; idx < calls_cnt; ++idx) {
                    if (this->calls_stock[idx].class_name == comp_call_class_name) {
                        csd.compatible_call_idxs.emplace_back(idx);
                    }
                }
            }

            mod.call_slots[csd.type].emplace_back(csd);
        }

        // CalleeSlots
        for (std::shared_ptr<core::CalleeSlot> callee_slot : calleeSlots) {
            Graph::StockCallSlot csd;
            csd.name = std::string(callee_slot->Name().PeekBuffer());
            csd.description = std::string(callee_slot->Description().PeekBuffer());
            csd.compatible_call_idxs.clear();
            csd.type = Graph::CallSlotType::CALLEE;

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
                Graph::StockCall call;
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
                    if (this->calls_stock[idx].class_name == callName) {
                        csd.compatible_call_idxs.emplace_back(idx);
                    }
                }
            }

            mod.call_slots[csd.type].emplace_back(csd);
        }

        paramSlots.clear();
        callerSlots.clear();
        calleeSlots.clear();
        root_mod_ns->RemoveChild(new_mod);
        new_mod->SetAllCleanupMarks();
        new_mod->PerformCleanup();
        new_mod = nullptr;

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


bool megamol::gui::GraphManager::get_call_stock_data(
    Graph::StockCall& call, const std::shared_ptr<const megamol::core::factories::CallDescription> call_desc) {

    try {
        /// call.plugin_name is not available in call_desc (set from AbstractAssemblyInstance or
        /// AbstractPluginInstance).
        call.class_name = std::string(call_desc->ClassName());
        call.description = std::string(call_desc->Description());
        call.functions.clear();
        for (unsigned int i = 0; i < call_desc->FunctionCount(); ++i) {
            call.functions.emplace_back(call_desc->FunctionName(i));
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


int megamol::gui::GraphManager::GetCompatibleCallIndex(
    Graph::CallSlotPtrType call_slot_1, Graph::CallSlotPtrType call_slot_2) {

    if ((call_slot_1 != nullptr) && (call_slot_2 != nullptr)) {
        if ((call_slot_1 != call_slot_2) && (call_slot_1->GetParentModule() != call_slot_2->GetParentModule()) &&
            (call_slot_1->type != call_slot_2->type)) {
            // Return first found compatible call index
            for (auto& selected_comp_call_slots : call_slot_1->compatible_call_idxs) {
                for (auto& current_comp_call_slots : call_slot_2->compatible_call_idxs) {
                    if (selected_comp_call_slots == current_comp_call_slots) {
                        // Show only comaptible calls for unconnected caller slots
                        if ((call_slot_1->type == Graph::CallSlotType::CALLER) && (call_slot_1->CallsConnected())) {
                            return -1;
                        } else if ((call_slot_2->type == Graph::CallSlotType::CALLER) &&
                                   (call_slot_2->CallsConnected())) {
                            return -1;
                        }
                        return current_comp_call_slots;
                    }
                }
            }
        }
    }
    return -1;
}


int megamol::gui::GraphManager::GetCompatibleCallIndex(
    Graph::CallSlotPtrType call_slot, Graph::StockCallSlot stock_call_slot) {

    if (call_slot != nullptr) {
        if (call_slot->type != stock_call_slot.type) {
            // Return first found compatible call index
            for (auto& selected_comp_call_slots : call_slot->compatible_call_idxs) {
                for (auto& current_comp_call_slots : stock_call_slot.compatible_call_idxs) {
                    if (selected_comp_call_slots == current_comp_call_slots) {
                        return current_comp_call_slots;
                    }
                }
            }
        }
    }
    return -1;
}


bool megamol::gui::GraphManager::PROTOTYPE_SaveGraph(
    int graph_id, std::string project_filename, megamol::core::CoreInstance* core_instance) {

    if (core_instance == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to CoreInstance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    std::string confstr;
    std::stringstream confInstances, confModules, confCalls, confParams;
    bool already_found_main_view = false;
    bool found_graph = false;

    try {
        // Search for top most view
        for (auto& graph : this->graphs) {
            if (graph->GetUID() == graph_id) {
                for (auto& mod : graph->GetGraphModules()) {
                    // Check for not connected calle_slots
                    bool is_main_view = false;
                    if (mod->is_view) {
                        bool callee_not_connected = false;
                        for (auto& call_slots : mod->GetCallSlots(Graph::CallSlotType::CALLEE)) {
                            if (!call_slots->CallsConnected()) {
                                callee_not_connected = true;
                            }
                        }
                        if (!callee_not_connected) {
                            if (!already_found_main_view) {
                                confInstances << "mmCreateView(\"" << mod->instance << "\",\"" << mod->class_name
                                              << "\",\"" << mod->full_name << "\")\n";
                                already_found_main_view = true;
                                is_main_view = true;
                            } else {
                                vislib::sys::Log::DefaultLog.WriteError(
                                    "Only one main view is allowed. Found another one. [%s, %s, line %d]\n", __FILE__,
                                    __FUNCTION__, __LINE__);
                            }
                        }
                    }
                    if (!is_main_view) {
                        confModules << "mmCreateModule(\"" << mod->class_name << "\",\"" << mod->full_name << "\")\n";
                    }

                    /*
                    for (auto& p : m.param_slots) {
                        if (p.type != GraphManager::ParamType::BUTTON) {
                            std::string val = slot->Parameter()->ValueString().PeekBuffer();
                            // Encode to UTF-8 string
                            vislib::StringA valueString;
                            vislib::UTF8Encoder::Encode(valueString, vislib::StringA(val.c_str()));
                            val = valueString.PeekBuffer();
                            confParams << "mmSetParamValue(\"" << p.full_name << "\",[=[" << val << "]=])\n";
                        }
                    }
                    */

                    for (auto& cr : mod->GetCallSlots(Graph::CallSlotType::CALLER)) {
                        for (auto& call : cr->GetConnectedCalls()) {
                            if (call->IsConnected()) {
                                confCalls
                                    << "mmCreateCall(\"" << call->class_name << "\",\""
                                    << call->GetCallSlot(Graph::CallSlotType::CALLER)->GetParentModule()->full_name
                                    << "::" << call->GetCallSlot(Graph::CallSlotType::CALLER)->name << "\",\""
                                    << call->GetCallSlot(Graph::CallSlotType::CALLEE)->GetParentModule()->full_name
                                    << "::" << call->GetCallSlot(Graph::CallSlotType::CALLEE)->name << "\")\n";
                            }
                        }
                    }
                }

                confstr = confInstances.str() + "\n" + confModules.str() + "\n" + confCalls.str() + "\n" +
                          confParams.str() + "\n";
                found_graph = true;
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

    if (!found_graph) {
        vislib::sys::Log::DefaultLog.WriteWarn("..... [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    try {
        std::ofstream file;
        file.open(project_filename);
        if (file.good()) {
            file << confstr.c_str();
            file.close();
        } else {
            vislib::sys::Log::DefaultLog.WriteError(
                "Unable to create project file. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            file.close();
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
