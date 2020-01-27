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

    Graph graph(this->get_unique_id(), name);

    graph.gui.slot_radius = 8.0f;
    graph.gui.canvas_position = ImVec2(0.0f, 0.0f);
    graph.gui.canvas_size = ImVec2(0.0f, 0.0f);
    graph.gui.canvas_scrolling = ImVec2(0.0f, 0.0f);
    graph.gui.canvas_zooming = 1.0f;
    graph.gui.show_grid = true;
    graph.gui.show_call_names = true;
    graph.gui.show_modules_small = false;
    graph.gui.selected_module_uid = -1;
    graph.gui.selected_call_uid = -1;
    graph.gui.hovered_slot_uid = -1;
    graph.gui.selected_slot_ptr = nullptr;
    graph.gui.process_selected_slot = 0;
    graph.gui.update_layout = true;

    this->graphs.emplace_back(std::make_shared<Graph>(graph));

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


const GraphManager::GraphPtrType megamol::gui::GraphManager::GetGraph(int graph_uid) {

    for (auto iter = this->graphs.begin(); iter != this->graphs.end(); iter++) {
        if ((*iter)->GetUID() == graph_uid) {
            return (*iter);
        }
    }
    vislib::sys::Log::DefaultLog.WriteWarn(
        "Invalif graph uid. Returning nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return nullptr;
}


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


bool megamol::gui::GraphManager::LoadCurrentCoreProject(std::string name, megamol::core::CoreInstance* core_instance) {

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
                    const auto button_param = param_slot->Param<megamol::core::param::ButtonParam>();
                    if (button_param == nullptr) {
                        std::string value_string = param_slot->Parameter()->ValueString().PeekBuffer();
                        // Encode to UTF-8 string
                        vislib::StringA valueString;
                        vislib::UTF8Encoder::Encode(valueString, vislib::StringA(value_string.c_str()));
                        value_string = valueString.PeekBuffer();

                        std::string param_class_name = std::string(param_slot->Name().PeekBuffer());
                        std::string param_full_name = std::string(param_slot->FullName().PeekBuffer());
                        for (auto& param : graph_module->param_slots) {
                            if (param.class_name == param_class_name) {
                                param.full_name = param_full_name;


                                param.value_string = value_string;
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

                        cd.compat_call_idx = -1;
                        int calls_count = this->calls_stock.size();
                        for (int i = 0; i < calls_count; i++) {
                            if (this->calls_stock[i].class_name == std::string(call->ClassName())) {
                                cd.compat_call_idx = i;
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
            Graph::CallSlotPtrType call_slot_1 = nullptr;
            for (auto& mod : graph->GetGraphModules()) {
                if (mod->full_name == cd.caller_module_full_name) {
                    for (auto call_slot : mod->GetCallSlots(Graph::CallSlotType::CALLER)) {
                        if (call_slot->name == cd.caller_module_call_slot_name) {
                            call_slot_1 = call_slot;
                        }
                    }
                }
            }
            Graph::CallSlotPtrType call_slot_2 = nullptr;
            for (auto& mod : graph->GetGraphModules()) {
                if (mod->full_name == cd.callee_module_full_name) {
                    for (auto call_slot : mod->GetCallSlots(Graph::CallSlotType::CALLEE)) {
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

                    for (auto& param_slot : mod->param_slots) {
                        if (param_slot.type != Graph::ParamType::BUTTON) {
                            // Encode to UTF-8 string
                            vislib::StringA valueString;
                            vislib::UTF8Encoder::Encode(valueString, vislib::StringA(param_slot.value_string.c_str()));
                            confParams << "mmSetParamValue(\"" << instance << mod->name << "::" << param_slot.class_name
                                       << "\",[=[" << std::string(valueString.PeekBuffer()) << "]=])\n";
                        }
                    }

                    for (auto& caller_slot : mod->GetCallSlots(Graph::CallSlotType::CALLER)) {
                        for (auto& call : caller_slot->GetConnectedCalls()) {
                            if (call->IsConnected()) {
                                confCalls << "mmCreateCall(\"" << call->class_name << "\",\"" << instance
                                          << call->GetCallSlot(Graph::CallSlotType::CALLER)->GetParentModule()->name
                                          << "::" << call->GetCallSlot(Graph::CallSlotType::CALLER)->name << "\",\""
                                          << instance
                                          << call->GetCallSlot(Graph::CallSlotType::CALLEE)->GetParentModule()->name
                                          << "::" << call->GetCallSlot(Graph::CallSlotType::CALLEE)->name << "\")\n";
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
                vislib::sys::Log::DefaultLog.WriteError("Found unknown parameter type. Please extend parameter types "
                                                        "in the configurator. [%s, %s, line %d]\n",
                    __FILE__, __FUNCTION__, __LINE__);
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
