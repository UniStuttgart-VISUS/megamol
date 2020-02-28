/*
 * GraphManager.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "GraphManager.h"


using namespace megamol;
using namespace megamol::gui::configurator;


megamol::gui::configurator::GraphManager::GraphManager(void) : graphs(), modules_stock(), calls_stock() {}


megamol::gui::configurator::GraphManager::~GraphManager(void) {}


bool megamol::gui::configurator::GraphManager::AddGraph(std::string name) {

    Graph graph(name);
    this->graphs.emplace_back(std::make_shared<Graph>(graph));

    return true;
}


bool megamol::gui::configurator::GraphManager::DeleteGraph(int graph_uid) {

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


const GraphManager::GraphsType& megamol::gui::configurator::GraphManager::GetGraphs(void) { return this->graphs; }


const GraphManager::GraphPtrType megamol::gui::configurator::GraphManager::GetGraph(int graph_uid) {

    for (auto iter = this->graphs.begin(); iter != this->graphs.end(); iter++) {
        if ((*iter)->GetUID() == graph_uid) {
            return (*iter);
        }
    }
    vislib::sys::Log::DefaultLog.WriteWarn(
        "Invalif graph uid. Returning nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return nullptr;
}


bool megamol::gui::configurator::GraphManager::UpdateModulesCallsStock(
    const megamol::core::CoreInstance* core_instance) {

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
                    Call::StockCall call;
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
                        [class_name](const Call::StockCall& call) { return (call.class_name == class_name); }) ==
                    this->calls_stock.end()) {
                    Call::StockCall call;
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
                    Module::StockModule mod;
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
                        [class_name](const Module::StockModule& mod) { return (mod.class_name == class_name); }) ==
                    this->modules_stock.end()) {
                    Module::StockModule mod;
                    mod.plugin_name = plugin_name;
                    this->get_module_stock_data(mod, m_desc);
                    this->modules_stock.emplace_back(mod);
                }
            }

            // Sorting module by alphabetically ascending class names.
            std::sort(this->modules_stock.begin(), this->modules_stock.end(),
                [](Module::StockModule mod1, Module::StockModule mod2) {
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
            "Reading available modules and calls ... DONE (duration: %.3f seconds)\n", delta_time);

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


bool megamol::gui::configurator::GraphManager::LoadCurrentCoreProject(
    std::string name, megamol::core::CoreInstance* core_instance) {

    try {
        // Temporary data structure holding call connection data
        struct CallData {
            int compat_call_idx;
            std::string caller_module_full_name;
            std::string caller_module_call_slot_name;
            std::string callee_module_full_name;
            std::string callee_module_call_slot_name;
        };
        std::vector<CallData> call_data;

        // Create new graph
        this->AddGraph(name);
        auto graph = this->graphs.back();

        // Search for view instance
        std::map<std::string, std::string> view_instances;
        vislib::sys::AutoLock lock(core_instance->ModuleGraphRoot()->ModuleGraphLock());
        megamol::core::AbstractNamedObjectContainer::const_ptr_type anoc =
            megamol::core::AbstractNamedObjectContainer::dynamic_pointer_cast(core_instance->ModuleGraphRoot());
        for (auto ano = anoc->ChildList_Begin(); ano != anoc->ChildList_End(); ++ano) {
            auto vi = dynamic_cast<megamol::core::ViewInstance*>(ano->get());
            if ((vi != nullptr) && (vi->View() != nullptr)) {
                std::string vin = std::string(vi->Name().PeekBuffer());
                view_instances[std::string(vi->View()->FullName().PeekBuffer())] = vin;
            }
        }

        // Create modules and get additional module information.
        // Load call connections to temporary data structure since not all call slots are yet available for being
        // connected.
        const auto module_func = [&, this](megamol::core::Module* mod) {
            // Creating new module
            graph->AddModule(this->modules_stock, std::string(mod->ClassName()));
            auto graph_module = graph->GetGraphModules().back();
            graph_module->name = std::string(mod->Name().PeekBuffer());
            graph_module->full_name = std::string(mod->FullName().PeekBuffer());

            if (view_instances.find(std::string(mod->FullName().PeekBuffer())) != view_instances.end()) {
                // Instance Name
                graph->SetName(view_instances[std::string(mod->FullName().PeekBuffer())]);
                graph_module->is_view_instance = true;
            }

            megamol::core::AbstractNamedObjectContainer::child_list_type::const_iterator se = mod->ChildList_End();
            for (megamol::core::AbstractNamedObjectContainer::child_list_type::const_iterator si =
                     mod->ChildList_Begin();
                 si != se; ++si) {

                // Parameter
                const auto param_slot = dynamic_cast<megamol::core::param::ParamSlot*>((*si).get());
                if (param_slot != nullptr) {
                    std::string param_class_name = std::string(param_slot->Name().PeekBuffer());
                    for (auto& param : graph_module->parameters) {
                        if (param.class_name == param_class_name) {
                            param.full_name = std::string(param_slot->FullName().PeekBuffer());
                            if (auto* p_ptr = param_slot->Param<core::param::ButtonParam>()) {
                                param.SetStorage(p_ptr->GetKeyCode());
                            } else if (auto* p_ptr = param_slot->Param<core::param::BoolParam>()) {
                                param.SetValue(p_ptr->Value());
                            } else if (auto* p_ptr = param_slot->Param<core::param::ColorParam>()) {
                                param.SetValue(p_ptr->Value());
                            } else if (auto* p_ptr = param_slot->Param<core::param::EnumParam>()) {
                                param.SetValue(p_ptr->Value());
                                Parameter::EnumStorageType map;
                                auto param_map = p_ptr->getMap();
                                auto iter = param_map.GetConstIterator();
                                while (iter.HasNext()) {
                                    auto pair = iter.Next();
                                    map.emplace(pair.Key(), std::string(pair.Value().PeekBuffer()));
                                }
                                param.SetStorage(map);
                            } else if (auto* p_ptr = param_slot->Param<core::param::FilePathParam>()) {
                                param.SetValue(std::string(p_ptr->Value().PeekBuffer()));
                            } else if (auto* p_ptr = param_slot->Param<core::param::FlexEnumParam>()) {
                                param.SetValue(p_ptr->Value());
                                param.SetStorage(p_ptr->getStorage());
                            } else if (auto* p_ptr = param_slot->Param<core::param::FloatParam>()) {
                                param.SetValue(p_ptr->Value());
                                param.SetMinValue(p_ptr->MinValue());
                                param.SetMaxValue(p_ptr->MaxValue());
                            } else if (auto* p_ptr = param_slot->Param<core::param::IntParam>()) {
                                param.SetValue(p_ptr->Value());
                                param.SetMinValue(p_ptr->MinValue());
                                param.SetMaxValue(p_ptr->MaxValue());
                            } else if (auto* p_ptr = param_slot->Param<core::param::StringParam>()) {
                                param.SetValue(std::string(p_ptr->Value().PeekBuffer()));
                            } else if (auto* p_ptr = param_slot->Param<core::param::TernaryParam>()) {
                                param.SetValue(p_ptr->Value());
                            } else if (auto* p_ptr = param_slot->Param<core::param::TransferFunctionParam>()) {
                                param.SetValue(p_ptr->Value());
                            } else if (auto* p_ptr = param_slot->Param<core::param::Vector2fParam>()) {
                                auto val = p_ptr->Value();
                                param.SetValue(glm::vec2(val.X(), val.Y()));
                                auto min = p_ptr->MinValue();
                                param.SetMinValue(glm::vec2(min.X(), min.Y()));
                                auto max = p_ptr->MaxValue();
                                param.SetMaxValue(glm::vec2(max.X(), max.Y()));
                            } else if (auto* p_ptr = param_slot->Param<core::param::Vector3fParam>()) {
                                auto val = p_ptr->Value();
                                param.SetValue(glm::vec3(val.X(), val.Y(), val.Z()));
                                auto min = p_ptr->MinValue();
                                param.SetMinValue(glm::vec3(min.X(), min.Y(), min.Z()));
                                auto max = p_ptr->MaxValue();
                                param.SetMaxValue(glm::vec3(max.X(), max.Y(), max.Z()));
                            } else if (auto* p_ptr = param_slot->Param<core::param::Vector4fParam>()) {
                                auto val = p_ptr->Value();
                                param.SetValue(glm::vec4(val.X(), val.Y(), val.Z(), val.W()));
                                auto min = p_ptr->MinValue();
                                param.SetMinValue(glm::vec4(min.X(), min.Y(), min.Z(), min.W()));
                                auto max = p_ptr->MaxValue();
                                param.SetMaxValue(glm::vec4(max.X(), max.Y(), max.Z(), max.W()));
                            } else {
                                vislib::sys::Log::DefaultLog.WriteError(
                                    "Found unknown parameter type. Please extend parameter types in the configurator. "
                                    "[%s, %s, line %d]\n",
                                    __FILE__, __FUNCTION__, __LINE__);
                            }
                        }
                    }
                }

                // Collect call connection data
                const auto caller_slot = dynamic_cast<megamol::core::CallerSlot*>((*si).get());
                if (caller_slot) {
                    const megamol::core::Call* call =
                        const_cast<megamol::core::CallerSlot*>(caller_slot)->CallAs<megamol::core::Call>();
                    if (call != nullptr) {
                        CallData cd;

                        cd.compat_call_idx = GUI_INVALID_ID;
                        size_t calls_count = this->calls_stock.size();
                        for (size_t i = 0; i < calls_count; i++) {
                            if (this->calls_stock[i].class_name == std::string(call->ClassName())) {
                                cd.compat_call_idx = static_cast<int>(i);
                            }
                        }
                        cd.caller_module_full_name =
                            std::string(call->PeekCallerSlot()->Parent()->FullName().PeekBuffer());
                        cd.caller_module_call_slot_name = std::string(call->PeekCallerSlot()->Name().PeekBuffer());
                        cd.callee_module_full_name =
                            std::string(call->PeekCalleeSlot()->Parent()->FullName().PeekBuffer());
                        cd.callee_module_call_slot_name = std::string(call->PeekCalleeSlot()->Name().PeekBuffer());

                        call_data.emplace_back(cd);
                    }
                }
            }
        };
        core_instance->EnumModulesNoLock(nullptr, module_func);

        // Create calls
        for (auto& cd : call_data) {
            CallSlotPtrType call_slot_1 = nullptr;
            for (auto& mod : graph->GetGraphModules()) {
                if (mod->full_name == cd.caller_module_full_name) {
                    for (auto call_slot : mod->GetCallSlots(CallSlot::CallSlotType::CALLER)) {
                        if (call_slot->name == cd.caller_module_call_slot_name) {
                            call_slot_1 = call_slot;
                        }
                    }
                }
            }
            CallSlotPtrType call_slot_2 = nullptr;
            for (auto& mod : graph->GetGraphModules()) {
                if (mod->full_name == cd.callee_module_full_name) {
                    for (auto call_slot : mod->GetCallSlots(CallSlot::CallSlotType::CALLEE)) {
                        if (call_slot->name == cd.callee_module_call_slot_name) {
                            call_slot_2 = call_slot;
                        }
                    }
                }
            }
            graph->AddCall(this->GetCallsStock(), cd.compat_call_idx, call_slot_1, call_slot_2);
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


bool megamol::gui::configurator::GraphManager::PROTOTYPE_SaveGraph(
    int graph_id, std::string project_filename, megamol::core::CoreInstance* core_instance) {

    if (core_instance == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to CoreInstance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    std::string confstr;
    std::stringstream confInstances, confModules, confCalls, confParams;
    GraphPtrType found_graph = nullptr;

    try {
        // Search for top most view
        for (auto& graph : this->graphs) {
            if (graph->GetUID() == graph_id) {
                for (auto& mod : graph->GetGraphModules()) {

                    std::string instance_name = graph->GetName();
                    std::string instance = "::" + instance_name + "::";
                    if (mod->is_view_instance) {
                        confInstances << "mmCreateView(\"" << instance_name << "\",\"" << mod->class_name << "\",\""
                                      << instance << mod->name << "\")\n";
                    } else {
                        confModules << "mmCreateModule(\"" << mod->class_name << "\",\"" << instance << mod->name
                                    << "\")\n";
                    }

                    for (auto& param_slot : mod->parameters) {
                        if (param_slot.type != Parameter::ParamType::BUTTON) {
                            // Encode to UTF-8 string
                            vislib::StringA valueString;
                            vislib::UTF8Encoder::Encode(
                                valueString, vislib::StringA(param_slot.GetValueString().c_str()));
                            confParams << "mmSetParamValue(\"" << instance << mod->name << "::" << param_slot.class_name
                                       << "\",[=[" << std::string(valueString.PeekBuffer()) << "]=])\n";
                        }
                    }

                    for (auto& caller_slot : mod->GetCallSlots(CallSlot::CallSlotType::CALLER)) {
                        for (auto& call : caller_slot->GetConnectedCalls()) {
                            if (call->IsConnected()) {
                                confCalls << "mmCreateCall(\"" << call->class_name << "\",\"" << instance
                                          << call->GetCallSlot(CallSlot::CallSlotType::CALLER)->GetParentModule()->name
                                          << "::" << call->GetCallSlot(CallSlot::CallSlotType::CALLER)->name << "\",\""
                                          << instance
                                          << call->GetCallSlot(CallSlot::CallSlotType::CALLEE)->GetParentModule()->name
                                          << "::" << call->GetCallSlot(CallSlot::CallSlotType::CALLEE)->name << "\")\n";
                            }
                        }
                    }
                }

                confstr = confInstances.str() + "\n" + confModules.str() + "\n" + confCalls.str() + "\n" +
                          confParams.str() + "\n";
                found_graph = graph;
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

    if (found_graph == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Invalid graph uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
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

    if (found_graph != nullptr) {
        found_graph->ResetDirty();
    }
    return true;
}


bool megamol::gui::configurator::GraphManager::get_module_stock_data(
    Module::StockModule& mod, const std::shared_ptr<const megamol::core::factories::ModuleDescription> mod_desc) {

    /// mod.plugin_name is not available in mod_desc (set from AbstractAssemblyInstance or AbstractPluginInstance).
    mod.class_name = std::string(mod_desc->ClassName());
    mod.description = std::string(mod_desc->Description());
    mod.is_view = false;
    mod.parameters.clear();
    mod.call_slots.clear();
    mod.call_slots.emplace(CallSlot::CallSlotType::CALLER, std::vector<CallSlot::StockCallSlot>());
    mod.call_slots.emplace(CallSlot::CallSlotType::CALLEE, std::vector<CallSlot::StockCallSlot>());

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

            Parameter::StockParameter psd;
            psd.class_name = std::string(param_slot->Name().PeekBuffer());
            psd.description = std::string(param_slot->Description().PeekBuffer());

            // Set parameter type
            if (auto* p_ptr = param_slot->Param<core::param::ButtonParam>()) {
                psd.type = Parameter::ParamType::BUTTON;
            } else if (auto* p_ptr = param_slot->Param<core::param::BoolParam>()) {
                psd.type = Parameter::ParamType::BOOL;
            } else if (auto* p_ptr = param_slot->Param<core::param::ColorParam>()) {
                psd.type = Parameter::ParamType::COLOR;
            } else if (auto* p_ptr = param_slot->Param<core::param::EnumParam>()) {
                psd.type = Parameter::ParamType::ENUM;
            } else if (auto* p_ptr = param_slot->Param<core::param::FilePathParam>()) {
                psd.type = Parameter::ParamType::FILEPATH;
            } else if (auto* p_ptr = param_slot->Param<core::param::FlexEnumParam>()) {
                psd.type = Parameter::ParamType::FLEXENUM;
            } else if (auto* p_ptr = param_slot->Param<core::param::FloatParam>()) {
                psd.type = Parameter::ParamType::FLOAT;
            } else if (auto* p_ptr = param_slot->Param<core::param::IntParam>()) {
                psd.type = Parameter::ParamType::INT;
            } else if (auto* p_ptr = param_slot->Param<core::param::StringParam>()) {
                psd.type = Parameter::ParamType::STRING;
            } else if (auto* p_ptr = param_slot->Param<core::param::TernaryParam>()) {
                psd.type = Parameter::ParamType::TERNARY;
            } else if (auto* p_ptr = param_slot->Param<core::param::TransferFunctionParam>()) {
                psd.type = Parameter::ParamType::TRANSFERFUNCTION;
            } else if (auto* p_ptr = param_slot->Param<core::param::Vector2fParam>()) {
                psd.type = Parameter::ParamType::VECTOR2F;
            } else if (auto* p_ptr = param_slot->Param<core::param::Vector3fParam>()) {
                psd.type = Parameter::ParamType::VECTOR3F;
            } else if (auto* p_ptr = param_slot->Param<core::param::Vector4fParam>()) {
                psd.type = Parameter::ParamType::VECTOR4F;
            } else {
                vislib::sys::Log::DefaultLog.WriteError("Found unknown parameter type. Please extend parameter types "
                                                        "in the configurator. [%s, %s, line %d]\n",
                    __FILE__, __FUNCTION__, __LINE__);
                psd.type = Parameter::ParamType::UNKNOWN;
            }

            mod.parameters.emplace_back(psd);
        }

        // CallerSlots
        for (std::shared_ptr<core::CallerSlot> caller_slot : callerSlots) {
            CallSlot::StockCallSlot csd;
            csd.name = std::string(caller_slot->Name().PeekBuffer());
            csd.description = std::string(caller_slot->Description().PeekBuffer());
            csd.compatible_call_idxs.clear();
            csd.type = CallSlot::CallSlotType::CALLER;

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
            CallSlot::StockCallSlot csd;
            csd.name = std::string(callee_slot->Name().PeekBuffer());
            csd.description = std::string(callee_slot->Description().PeekBuffer());
            csd.compatible_call_idxs.clear();
            csd.type = CallSlot::CallSlotType::CALLEE;

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


bool megamol::gui::configurator::GraphManager::get_call_stock_data(
    Call::StockCall& call, const std::shared_ptr<const megamol::core::factories::CallDescription> call_desc) {

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


// GRAPH MANAGET PRESENTATION ####################################################

megamol::gui::configurator::GraphManager::Presentation::Presentation(void)
    : presented_graph(nullptr), delete_graph_uid(GUI_INVALID_ID) {}


megamol::gui::configurator::GraphManager::Presentation::~Presentation(void) {}


bool megamol::gui::configurator::GraphManager::Presentation::GUI_Present(GraphManager& graph_manager, float child_width,
    ImFont* graph_font, HotkeyData paramter_search, HotkeyData delete_graph_element) {

    try {
        if (ImGui::GetCurrentContext() == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        const auto child_flags = ImGuiWindowFlags_None;

        ImGui::BeginChild("graph_child_window", ImVec2(child_width, 0.0f), true, child_flags);

        // Assuming only one closed tab/graph per frame.
        bool open_close_unsaved_popup = false;

        // Draw Graphs
        ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_Reorderable;
        ImGui::BeginTabBar("Graphs", tab_bar_flags);
        for (auto& graph : graph_manager.GetGraphs()) {

            bool delete_graph = false;
            if (graph->GUI_Present(child_width, graph_font, paramter_search, delete_graph_element, delete_graph)) {
                this->presented_graph = graph;
            }

            // Do not delete graph while looping through graphs list
            if (delete_graph) {
                this->delete_graph_uid = graph->GetUID();
                if (graph->IsDirty()) {
                    open_close_unsaved_popup = true;
                }
            }
        }
        ImGui::EndTabBar();

        // Delete marked graph when tab closed and
        if ((this->delete_graph_uid != GUI_INVALID_ID) && this->close_unsaved_popup(open_close_unsaved_popup)) {
            this->presented_graph = nullptr;
            graph_manager.DeleteGraph(delete_graph_uid);
            this->delete_graph_uid = GUI_INVALID_ID;
        }

        ImGui::EndChild();
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


bool megamol::gui::configurator::GraphManager::Presentation::close_unsaved_popup(bool open_popup) {

    bool retval = true;
    std::string save_project_label = "Warning: Closing Unsaved Project";

    if (open_popup) {
        ImGui::OpenPopup(save_project_label.c_str());
    }
    if (ImGui::BeginPopupModal(save_project_label.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        retval = false;

        ImGui::Text("Discard changes?");

        if (ImGui::Button("YES")) {
            ImGui::CloseCurrentPopup();
            retval = true;
        }
        ImGui::SameLine(0.0f, 100.0f);
        if (ImGui::Button("NO")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    return retval;
}