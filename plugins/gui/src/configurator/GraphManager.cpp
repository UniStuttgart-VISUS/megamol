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
using namespace megamol::gui::configurator;


megamol::gui::configurator::GraphManager::GraphManager(void) : graphs(), modules_stock(), calls_stock() {}


megamol::gui::configurator::GraphManager::~GraphManager(void) {}


bool megamol::gui::configurator::GraphManager::AddGraph(std::string name) {

    try {
        Graph graph(name);
        this->graphs.emplace_back(std::make_shared<Graph>(graph));
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


bool megamol::gui::configurator::GraphManager::DeleteGraph(ImGuiID graph_uid) {

    for (auto iter = this->graphs.begin(); iter != this->graphs.end(); iter++) {
        if ((*iter)->GetUID() == graph_uid) {

            vislib::sys::Log::DefaultLog.WriteWarn("Found %i references pointing to graph. [%s, %s, line %d]\n",
                (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
            assert((*iter).use_count() == 1);
            (*iter) = nullptr;
            this->graphs.erase(iter);
            // vislib::sys::Log::DefaultLog.WriteInfo("Deleted graph: %s [%s, %s, line %d]\n",
            //    (*iter)->GetName().c_str(), __FILE__, __FUNCTION__, __LINE__);

            return true;
        }
    }

    return false;
}


const GraphManager::GraphsType& megamol::gui::configurator::GraphManager::GetGraphs(void) { return this->graphs; }


const GraphManager::GraphPtrType megamol::gui::configurator::GraphManager::GetGraph(ImGuiID graph_uid) {

    for (auto iter = this->graphs.begin(); iter != this->graphs.end(); iter++) {
        if ((*iter)->GetUID() == graph_uid) {
            return (*iter);
        }
    }
    // vislib::sys::Log::DefaultLog.WriteWarn(
    //    "Invalid graph uid. Returning nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
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
                    std::string a_str = mod1.class_name;
                    for (auto& c : a_str) c = std::toupper(c);
                    std::string b_str = mod2.class_name;
                    for (auto& c : b_str) c = std::toupper(c);
                    return (a_str < b_str);
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


bool megamol::gui::configurator::GraphManager::LoadProjectCore(
    const std::string& name, megamol::core::CoreInstance* core_instance) {

    // Create new graph
    bool retval = this->AddGraph(name);
    auto graph_ptr = this->GetGraphs().back();

    if (retval && (graph_ptr != nullptr)) {
        return this->AddProjectCore(graph_ptr->GetUID(), core_instance);
    }

    vislib::sys::Log::DefaultLog.WriteError(
        "Failed to create new graph: %s [%s, %s, line %d]\n", name.c_str(), __FILE__, __FUNCTION__, __LINE__);
    return false;
}


bool megamol::gui::configurator::GraphManager::AddProjectCore(
    ImGuiID graph_uid, megamol::core::CoreInstance* core_instance) {

    try {
        auto graph_ptr = this->GetGraph(graph_uid);
        if (graph_ptr == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Unable to find graph for given uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        // Temporary data structure holding call connection data
        struct CallData {
            std::string caller_module_full_name;
            std::string caller_module_call_slot_name;
            std::string callee_module_full_name;
            std::string callee_module_call_slot_name;
        };
        std::vector<CallData> call_data;

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
            // Ensure unique module name is not yet assigned
            std::string module_name = std::string(mod->Name().PeekBuffer());
            if (graph_ptr->RenameAssignedModuleName(module_name)) {
                vislib::sys::Log::DefaultLog.WriteWarn(
                    "Renamed existing module '%s' while adding module with same name. "
                    "This is required for successful unambiguous parameter addressing which uses the module name. [%s, "
                    "%s, line %d]\n",
                    module_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
            }
            // Creating new module
            graph_ptr->AddModule(this->modules_stock, std::string(mod->ClassName()));
            auto graph_module = graph_ptr->GetGraphModules().back();
            graph_module->name = module_name;
            std::string full_name = std::string(mod->FullName().PeekBuffer());
            graph_module->name_space = full_name.substr(0, full_name.find(graph_module->name) - 2);
            graph_module->is_view_instance = false;

            if (view_instances.find(std::string(mod->FullName().PeekBuffer())) != view_instances.end()) {
                // Instance Name
                graph_ptr->SetName(view_instances[std::string(mod->FullName().PeekBuffer())]);
                graph_module->is_view_instance = true;
            }

            megamol::core::AbstractNamedObjectContainer::child_list_type::const_iterator se = mod->ChildList_End();
            for (megamol::core::AbstractNamedObjectContainer::child_list_type::const_iterator si =
                     mod->ChildList_Begin();
                 si != se; ++si) {

                // Parameter
                const auto param_slot = dynamic_cast<megamol::core::param::ParamSlot*>((*si).get());
                if (param_slot != nullptr) {
                    std::string param_full_name = std::string(param_slot->Name().PeekBuffer());
                    for (auto& param : graph_module->parameters) {
                        if (param.full_name == param_full_name) {
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
                                    "Found unknown parameter type. Please extend parameter types for the configurator. "
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
            for (auto& mod : graph_ptr->GetGraphModules()) {
                if (mod->FullName() == cd.caller_module_full_name) {
                    for (auto call_slot : mod->GetCallSlots(CallSlot::CallSlotType::CALLER)) {
                        if (call_slot->name == cd.caller_module_call_slot_name) {
                            call_slot_1 = call_slot;
                        }
                    }
                }
            }
            CallSlotPtrType call_slot_2 = nullptr;
            for (auto& mod : graph_ptr->GetGraphModules()) {
                if (mod->FullName() == cd.callee_module_full_name) {
                    for (auto call_slot : mod->GetCallSlots(CallSlot::CallSlotType::CALLEE)) {
                        if (call_slot->name == cd.callee_module_call_slot_name) {
                            call_slot_2 = call_slot;
                        }
                    }
                }
            }
            graph_ptr->AddCall(this->GetCallsStock(), call_slot_1, call_slot_2);
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


bool megamol::gui::configurator::GraphManager::LoadAddProjectFile(
    ImGuiID graph_uid, const std::string& project_filename) {

    std::string projectstr;
    if (!file::ReadFile(project_filename, projectstr)) return false;

    const std::string lua_view = "mmCreateView";
    const std::string lua_module = "mmCreateModule";
    const std::string lua_param = "mmSetParamValue";
    const std::string lua_call = "mmCreateCall";

    GraphPtrType graph_ptr;
    if (graph_uid != GUI_INVALID_ID) {
        graph_ptr = this->GetGraph(graph_uid);
        if (graph_ptr == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Unable to find graph for given uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
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

        // First find main view for graph creation and view creation.
        bool found_main_view = false;
        size_t lines_count = lines.size();
        for (unsigned int i = 0; i < lines_count; i++) {
            // Lua command must start at beginning after removeing leading spaces
            if (lines[i].rfind(lua_view, 0) == 0) {

                size_t arg_count = 3;
                std::vector<std::string> args;
                if (!this->readLuaProjectCommandArguments(lines[i], arg_count, args)) {
                    vislib::sys::Log::DefaultLog.WriteError("Project File '%s' line %i: Error parsing lua command '%s' "
                                                            "requiring %i arguments. [%s, %s, line %d]\n",
                        project_filename.c_str(), i, lua_view.c_str(), arg_count, __FILE__, __FUNCTION__, __LINE__);
                    return false;
                }

                std::string view_instance = args[0];
                std::string view_class_name = args[1];
                std::string view_full_name = args[2];
                std::string view_name_prefix;
                std::string view_name;
                if (!this->separateNameAndPrefix(view_full_name, view_name_prefix, view_name)) {
                    vislib::sys::Log::DefaultLog.WriteError("Project File '%s' line %i: Invalid view name argument "
                                                            "(3rd) in lua command '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), i, lua_view.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    return false;
                }
                ImVec2 module_pos = this->readLuaProjectConfPos(lines[i]);

                /// DEBUG
                // vislib::sys::Log::DefaultLog.WriteInfo(
                //     ">>>> Instance: '%s' Class: '%s' NameSpace: '%s' Name: '%s' ConfPos: %f, %f.\n",
                //     view_instance.c_str(), view_class_name.c_str(), view_name_space.c_str(), view_name.c_str(),
                //     module_pos.x, module_pos.y);

                // Create new graph
                if (graph_uid == GUI_INVALID_ID) {
                    if (!this->AddGraph(view_instance)) {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "Project File '%s' line %i: Unable to create new graph '%s'. [%s, %s, line %d]\n",
                            project_filename.c_str(), i, view_instance.c_str(), __FILE__, __FUNCTION__, __LINE__);
                        return false;
                    }
                    graph_ptr = this->GetGraphs().back();
                    if (graph_ptr == nullptr) {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "Unable to get pointer to last added graph. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                            __LINE__);
                        return false;
                    }
                }

                // Ensure unique module name is not yet assigned
                if (graph_ptr->RenameAssignedModuleName(view_name)) {
                    vislib::sys::Log::DefaultLog.WriteWarn(
                        "Project File '%s' line %i: Renamed existing module '%s' while adding module with same name. "
                        "This is required for successful unambiguous parameter addressing which uses the module name. "
                        "[%s, %s, line %d]\n",
                        project_filename.c_str(), i, view_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                }

                // Add module and set as view instance
                if (!graph_ptr->AddModule(this->modules_stock, view_class_name)) {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "Project File '%s' line %i: Unable to add new module '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), i, view_class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    return false;
                }
                auto graph_module = graph_ptr->GetGraphModules().back();
                graph_module->name = view_name;
                graph_module->name_space = view_name_prefix;
                graph_module->is_view_instance = true;
                graph_module->GUI_SetPosition(module_pos);

                found_main_view = true;
            }
        }
        if (!found_main_view) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Project File '%s': Missing main view lua command '%s'. [%s, %s, line %d]\n", project_filename.c_str(),
                lua_view.c_str(), __FILE__, __FUNCTION__, __LINE__);
        }

        // Find and create modules
        for (unsigned int i = 0; i < lines_count; i++) {
            // Lua command must start at beginning after removeing leading spaces
            if (lines[i].rfind(lua_module, 0) == 0) {

                size_t arg_count = 2;
                std::vector<std::string> args;
                if (!this->readLuaProjectCommandArguments(lines[i], arg_count, args)) {
                    vislib::sys::Log::DefaultLog.WriteError("Project File '%s' line %i: Error parsing lua command '%s' "
                                                            "requiring %i arguments. [%s, %s, line %d]\n",
                        project_filename.c_str(), i, lua_module.c_str(), arg_count, __FILE__, __FUNCTION__, __LINE__);
                    return false;
                }

                std::string module_class_name = args[0];
                std::string module_full_name = args[1];
                std::string module_name_prefix;
                std::string module_name;
                if (!this->separateNameAndPrefix(module_full_name, module_name_prefix, module_name)) {
                    vislib::sys::Log::DefaultLog.WriteError("Project File '%s' line %i: Invalid module name argument "
                                                            "(2nd) in lua command '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), i, lua_module.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    return false;
                }
                ImVec2 module_pos = this->readLuaProjectConfPos(lines[i]);

                /// DEBUG
                // vislib::sys::Log::DefaultLog.WriteInfo(">>>> Class: '%s' NameSpace: '%s' Name: '%s' ConfPos: %f,
                // %f.\n",
                //     module_class_name.c_str(), module_name_space.c_str(), module_name.c_str(), module_pos.x,
                //     module_pos.y);

                // Add module
                if (graph_ptr != nullptr) {

                    // Ensure unique module name is not yet assigned
                    if (graph_ptr->RenameAssignedModuleName(module_name)) {
                        vislib::sys::Log::DefaultLog.WriteWarn(
                            "Project File '%s' line %i: Renamed existing module '%s' while adding module with same "
                            "name. "
                            "This is required for successful unambiguous parameter addressing which uses the module "
                            "name. [%s, %s, line %d]\n",
                            project_filename.c_str(), i, module_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    }

                    if (!graph_ptr->AddModule(this->modules_stock, module_class_name)) {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "Project File '%s' line %i: Unable to add new module '%s'. [%s, %s, line %d]\n",
                            project_filename.c_str(), i, module_class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                        return false;
                    }
                    auto graph_module = graph_ptr->GetGraphModules().back();
                    graph_module->name = module_name;
                    graph_module->name_space = module_name_prefix;
                    graph_module->is_view_instance = false;
                    graph_module->GUI_SetPosition(module_pos);
                }
            }
        }

        // Find and create calls
        for (unsigned int i = 0; i < lines_count; i++) {
            // Lua command must start at beginning after removeing leading spaces
            if (lines[i].rfind(lua_call, 0) == 0) {

                size_t arg_count = 3;
                std::vector<std::string> args;
                if (!this->readLuaProjectCommandArguments(lines[i], arg_count, args)) {
                    vislib::sys::Log::DefaultLog.WriteError("Project File '%s' line %i: Error parsing lua command '%s' "
                                                            "requiring %i arguments. [%s, %s, line %d]\n",
                        project_filename.c_str(), i, lua_call.c_str(), arg_count, __FILE__, __FUNCTION__, __LINE__);
                    return false;
                }

                std::string call_class_name = args[0];
                std::string caller_slot_full_name = args[1];
                std::string callee_slot_full_name = args[2];

                std::string caller_slot_name;
                std::string caller_slot_prefix;
                if (!this->separateNameAndPrefix(caller_slot_full_name, caller_slot_prefix, caller_slot_name)) {
                    vislib::sys::Log::DefaultLog.WriteError("Project File '%s' line %i: Invalid caller slot name "
                                                            "argument (2nd) in lua command '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), i, lua_call.c_str(), __FILE__, __FUNCTION__, __LINE__);
                }

                std::string callee_slot_name;
                std::string callee_slot_prefix;
                if (!this->separateNameAndPrefix(callee_slot_full_name, callee_slot_prefix, callee_slot_name)) {
                    vislib::sys::Log::DefaultLog.WriteError("Project File '%s' line %i: Invalid callee slot name "
                                                            "argument (3nd) in lua command '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), i, lua_call.c_str(), __FILE__, __FUNCTION__, __LINE__);
                }

                /// DEBUG
                // vislib::sys::Log::DefaultLog.WriteInfo(
                //     ">>>> Call Name: '%s' CALLER Module: '%s' Slot: '%s' - CALLEE Module: '%s' Slot: '%s'.\n",
                //     call_class_name.c_str(), caller_module_name.c_str(), caller_slot_name.c_str(),
                //     callee_module_name.c_str(), callee_slot_name.c_str());

                // Searching for call
                if (graph_ptr != nullptr) {
                    std::string module_full_name;
                    size_t module_name_idx = std::string::npos;
                    std::string callee_name, caller_name;
                    CallSlotPtrType callee_slot, caller_slot;

                    for (auto& mod : graph_ptr->GetGraphModules()) {
                        module_full_name = mod->FullName() + "::";
                        // Caller
                        module_name_idx = caller_slot_full_name.find(module_full_name);
                        if (module_name_idx != std::string::npos) {
                            for (auto& call_slot_map : mod->GetCallSlots()) {
                                for (auto& call_slot : call_slot_map.second) {
                                    if (caller_slot_name == call_slot->name) {
                                        caller_slot = call_slot;
                                    }
                                }
                            }
                        }
                        // Callee
                        module_name_idx = callee_slot_full_name.find(module_full_name);
                        if (module_name_idx != std::string::npos) {
                            for (auto& call_slot_map : mod->GetCallSlots()) {
                                for (auto& call_slot : call_slot_map.second) {
                                    if (callee_slot_name == call_slot->name) {
                                        callee_slot = call_slot;
                                    }
                                }
                            }
                        }
                    }

                    if ((callee_slot == nullptr) || (caller_slot == nullptr)) {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "Project File '%s' line %i: Unable to find all call slots "
                            "for creating call '%s'. [%s, %s, line %d]\n",
                            project_filename.c_str(), i, call_class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                        return false;
                    }

                    // Add call
                    if (!graph_ptr->AddCall(this->calls_stock, caller_slot, callee_slot)) {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "Project File '%s' line %i: Unable to add new call '%s'. [%s, %s, line %d]\n",
                            project_filename.c_str(), i, call_class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                        return false;
                    }
                }
            }
        }

        // Find and create parameters
        for (unsigned int i = 0; i < lines_count; i++) {
            // Lua command must start at beginning after removeing leading spaces
            if (lines[i].rfind(lua_param, 0) == 0) {
                const std::string start_delimieter = "[=[";
                const std::string end_delimieter = "]=]";

                std::string param_line = lines[i];

                size_t first_bracket_idx = param_line.find('(');
                if (first_bracket_idx == std::string::npos) {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "Project File '%s' line %i: Missing opening brackets for '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), i, lua_param.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    return false;
                }
                size_t first_delimiter_idx = param_line.find(',');
                if (first_delimiter_idx == std::string::npos) {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "Project File '%s' line %i: Missing argument delimiter ',' for '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), i, lua_param.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    return false;
                }

                std::string param_slot_full_name =
                    param_line.substr(first_bracket_idx + 1, (first_delimiter_idx - first_bracket_idx - 1));
                if ((param_slot_full_name.front() != '"') || (param_slot_full_name.back() != '"')) {
                    vislib::sys::Log::DefaultLog.WriteError("Project File '%s' line %i: Parameter name argument should "
                                                            "be enclosed in '\"' for '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), i, lua_param.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    return false;
                }
                param_slot_full_name = param_slot_full_name.substr(1, param_slot_full_name.size() - 2);

                /// DEBUG
                // vislib::sys::Log::DefaultLog.WriteInfo(">>>> %s\n", param_slot_full_name.c_str());

                // Copy multi line parameter values into one string
                auto value_start_idx = param_line.find(start_delimieter);
                if (value_start_idx == std::string::npos) {
                    vislib::sys::Log::DefaultLog.WriteError("Project File '%s' line %i: Unable to find parameter value "
                                                            "start delimiter '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), i, start_delimieter.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    return false;
                }
                bool found_end_delimiter = true;
                if (param_line.find(end_delimieter) == std::string::npos) {
                    found_end_delimiter = false;
                    for (unsigned int j = (i + 1); j < lines_count; j++) {
                        param_line += lines[j];
                        if (lines[j].find(end_delimieter) != std::string::npos) {
                            found_end_delimiter = true;
                            break;
                        }
                    }
                }
                if (!found_end_delimiter) {
                    vislib::sys::Log::DefaultLog.WriteError("Project File '%s' line %i: Unable to find parameter value "
                                                            "end delimiter '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), i, end_delimieter.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    return false;
                }
                std::string value_str = param_line.substr(value_start_idx + start_delimieter.size(),
                    (param_line.find(end_delimieter)) - value_start_idx - end_delimieter.size());


                /// DEBUG
                // vislib::sys::Log::DefaultLog.WriteInfo(">>>> '%s'\n", value_str.c_str());

                // Searching for parameter
                if (graph_ptr != nullptr) {
                    std::string module_full_name;
                    size_t module_name_idx = std::string::npos;
                    for (auto& mod : graph_ptr->GetGraphModules()) {
                        module_full_name = mod->FullName() + "::";
                        module_name_idx = param_slot_full_name.find(module_full_name);
                        if (module_name_idx != std::string::npos) {
                            std::string param_full_name =
                                param_slot_full_name.substr(module_name_idx + module_full_name.size());
                            for (auto& param : mod->parameters) {
                                if (param.full_name == param_full_name) {
                                    param.SetValueString(value_str);
                                }
                            }
                        }
                    }
                }
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

    return true;
}


bool megamol::gui::configurator::GraphManager::SaveProjectFile(ImGuiID graph_id, const std::string& project_filename) {

    std::string projectstr;
    std::stringstream confInstances, confModules, confCalls, confParams;
    GraphPtrType found_graph = nullptr;

    try {
        // Search for top most view
        for (auto& graph : this->graphs) {
            if (graph->GetUID() == graph_id) {

                bool found_error = false;
                bool found_instance = false;
                for (auto& mod_1 : graph->GetGraphModules()) {
                    for (auto& mod_2 : graph->GetGraphModules()) {
                        if ((mod_1 != mod_2) && (mod_1->FullName() == mod_2->FullName())) {
                            vislib::sys::Log::DefaultLog.WriteError(
                                "Save Project >>> Found non unique module name: %s [%s, %s, line %d]\n",
                                mod_1->FullName().c_str(), __FILE__, __FUNCTION__, __LINE__);
                            found_error = true;
                        }
                    }
                    if (mod_1->is_view_instance) {
                        if (found_instance) {
                            vislib::sys::Log::DefaultLog.WriteError(
                                "Save Project >>> Found multiple view instances. [%s, %s, line %d]\n", __FILE__,
                                __FUNCTION__, __LINE__);
                            found_error = true;
                        }
                        found_instance = true;
                    }
                }
                if (!found_instance) {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "Save Project >>> Could not find required main view. [%s, %s, line %d]\n", __FILE__,
                        __FUNCTION__, __LINE__);
                    found_error = true;
                }
                if (found_error) return false;

                for (auto& mod : graph->GetGraphModules()) {
                    std::string instance_name = graph->GetName();
                    if (mod->is_view_instance) {
                        confInstances << "mmCreateView(\"" << instance_name << "\",\"" << mod->class_name << "\",\""
                                      << mod->FullName() << "\") "
                                      << this->writeLuaProjectConfPos(mod->GUI_GetPosition()) << "\n";
                    } else {
                        confModules << "mmCreateModule(\"" << mod->class_name << "\",\"" << mod->FullName() << "\") "
                                    << this->writeLuaProjectConfPos(mod->GUI_GetPosition()) << "\n";
                    }

                    for (auto& param_slot : mod->parameters) {
                        // Only write parameters with other values than the default
                        if (param_slot
                                .DefaultValueMismatch()) { // && (param_slot.type != Parameter::ParamType::BUTTON)) {
                            // Encode to UTF-8 string
                            vislib::StringA valueString;
                            vislib::UTF8Encoder::Encode(
                                valueString, vislib::StringA(param_slot.GetValueString().c_str()));
                            confParams << "mmSetParamValue(\"" << mod->FullName() << "::" << param_slot.full_name
                                       << "\",[=[" << std::string(valueString.PeekBuffer()) << "]=])\n";
                        }
                    }

                    for (auto& caller_slot : mod->GetCallSlots(CallSlot::CallSlotType::CALLER)) {
                        for (auto& call : caller_slot->GetConnectedCalls()) {
                            if (call->IsConnected()) {
                                confCalls
                                    << "mmCreateCall(\"" << call->class_name << "\",\""
                                    << call->GetCallSlot(CallSlot::CallSlotType::CALLER)->GetParentModule()->FullName()
                                    << "::" << call->GetCallSlot(CallSlot::CallSlotType::CALLER)->name << "\",\""
                                    << call->GetCallSlot(CallSlot::CallSlotType::CALLEE)->GetParentModule()->FullName()
                                    << "::" << call->GetCallSlot(CallSlot::CallSlotType::CALLEE)->name << "\")\n";
                            }
                        }
                    }
                }


                projectstr = confInstances.str() + "\n" + confModules.str() + "\n" + confCalls.str() + "\n" +
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

    if (found_graph != nullptr) {
        found_graph->ResetDirty();
    }

    return file::WriteFile(project_filename, projectstr);
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
            psd.full_name = std::string(param_slot->Name().PeekBuffer());
            psd.description = std::string(param_slot->Description().PeekBuffer());

            // Set parameter type
            if (auto* p_ptr = param_slot->Param<core::param::ButtonParam>()) {
                psd.type = Parameter::ParamType::BUTTON;
                psd.storage = p_ptr->GetKeyCode();
            } else if (auto* p_ptr = param_slot->Param<core::param::BoolParam>()) {
                psd.type = Parameter::ParamType::BOOL;
                psd.default_value = std::string(p_ptr->ValueString().PeekBuffer());
            } else if (auto* p_ptr = param_slot->Param<core::param::ColorParam>()) {
                psd.type = Parameter::ParamType::COLOR;
                psd.default_value = std::string(p_ptr->ValueString().PeekBuffer());
            } else if (auto* p_ptr = param_slot->Param<core::param::EnumParam>()) {
                psd.type = Parameter::ParamType::ENUM;
                psd.default_value = std::string(p_ptr->ValueString().PeekBuffer());
                Parameter::EnumStorageType map;
                auto psd_map = p_ptr->getMap();
                auto iter = psd_map.GetConstIterator();
                while (iter.HasNext()) {
                    auto pair = iter.Next();
                    map.emplace(pair.Key(), std::string(pair.Value().PeekBuffer()));
                }
                psd.storage = map;
            } else if (auto* p_ptr = param_slot->Param<core::param::FilePathParam>()) {
                psd.type = Parameter::ParamType::FILEPATH;
                psd.default_value = std::string(p_ptr->ValueString().PeekBuffer());
            } else if (auto* p_ptr = param_slot->Param<core::param::FlexEnumParam>()) {
                psd.type = Parameter::ParamType::FLEXENUM;
                psd.default_value = std::string(p_ptr->ValueString().PeekBuffer());
                psd.storage = p_ptr->getStorage();
            } else if (auto* p_ptr = param_slot->Param<core::param::FloatParam>()) {
                psd.type = Parameter::ParamType::FLOAT;
                psd.default_value = std::string(p_ptr->ValueString().PeekBuffer());
                psd.minval = p_ptr->MinValue();
                psd.maxval = p_ptr->MaxValue();
            } else if (auto* p_ptr = param_slot->Param<core::param::IntParam>()) {
                psd.type = Parameter::ParamType::INT;
                psd.default_value = std::string(p_ptr->ValueString().PeekBuffer());
                psd.minval = p_ptr->MinValue();
                psd.maxval = p_ptr->MaxValue();
            } else if (auto* p_ptr = param_slot->Param<core::param::StringParam>()) {
                psd.type = Parameter::ParamType::STRING;
                psd.default_value = std::string(p_ptr->ValueString().PeekBuffer());
            } else if (auto* p_ptr = param_slot->Param<core::param::TernaryParam>()) {
                psd.type = Parameter::ParamType::TERNARY;
                psd.default_value = std::string(p_ptr->ValueString().PeekBuffer());
            } else if (auto* p_ptr = param_slot->Param<core::param::TransferFunctionParam>()) {
                psd.type = Parameter::ParamType::TRANSFERFUNCTION;
                psd.default_value = std::string(p_ptr->ValueString().PeekBuffer());
            } else if (auto* p_ptr = param_slot->Param<core::param::Vector2fParam>()) {
                psd.type = Parameter::ParamType::VECTOR2F;
                psd.default_value = std::string(p_ptr->ValueString().PeekBuffer());
                auto min = p_ptr->MinValue();
                psd.minval = glm::vec2(min.X(), min.Y());
                auto max = p_ptr->MaxValue();
                psd.maxval = glm::vec2(max.X(), max.Y());
            } else if (auto* p_ptr = param_slot->Param<core::param::Vector3fParam>()) {
                psd.type = Parameter::ParamType::VECTOR3F;
                psd.default_value = std::string(p_ptr->ValueString().PeekBuffer());
                auto min = p_ptr->MinValue();
                psd.minval = glm::vec3(min.X(), min.Y(), min.Z());
                auto max = p_ptr->MaxValue();
                psd.maxval = glm::vec3(max.X(), max.Y(), max.Z());
            } else if (auto* p_ptr = param_slot->Param<core::param::Vector4fParam>()) {
                psd.type = Parameter::ParamType::VECTOR4F;
                psd.default_value = std::string(p_ptr->ValueString().PeekBuffer());
                auto min = p_ptr->MinValue();
                psd.minval = glm::vec4(min.X(), min.Y(), min.Z(), min.W());
                auto max = p_ptr->MaxValue();
                psd.maxval = glm::vec4(max.X(), max.Y(), max.Z(), max.W());
            } else {
                vislib::sys::Log::DefaultLog.WriteError("Found unknown parameter type. Please extend parameter types "
                                                        "for the configurator. [%s, %s, line %d]\n",
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


bool megamol::gui::configurator::GraphManager::readLuaProjectCommandArguments(
    const std::string& line, size_t arg_count, std::vector<std::string>& out_args) {

    // (Leaving current line in original state)
    std::string args_str = line;
    out_args.clear();

    // Searching for command delimiter
    auto start = args_str.find('(');
    auto end = args_str.rfind(')');
    if ((start == std::string::npos) || (end == std::string::npos)) {
        vislib::sys::Log::DefaultLog.WriteError(
            " Missing opening and/or closing bracket(s). [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    args_str = args_str.substr(start + 1, (end - start) - 1);

    // Getting arguments
    std::string arg;
    const char args_delim = ',';
    const std::string args_string_delim = "\"";
    size_t str_delim_idx_1 = std::string::npos;
    size_t str_delim_idx_2 = std::string::npos;
    size_t delim_idx = std::string::npos;

    for (size_t i = 0; i < arg_count; i++) {
        str_delim_idx_1 = args_str.find(args_string_delim);
        str_delim_idx_2 = str_delim_idx_1 + args_str.substr(str_delim_idx_1 + 1).find(args_string_delim);
        if ((str_delim_idx_1 == std::string::npos) || (str_delim_idx_2 == std::string::npos)) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Missing argument string delimiter '%s' at position %i. [%s, %s, line %d]\n", args_string_delim.c_str(),
                (i + 1), __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        arg = args_str.substr(str_delim_idx_1 + 1, str_delim_idx_2 - str_delim_idx_1);
        out_args.emplace_back(arg);
        args_str = args_str.substr(str_delim_idx_2 + 2);
        if (i < (arg_count - 1)) {
            delim_idx = args_str.find(args_delim);
            if (delim_idx == std::string::npos) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "Missing argument  delimiter '%c' at position %i. [%s, %s, line %d]\n", args_delim, (i + 1),
                    __FILE__, __FUNCTION__, __LINE__);
                return false;
            }
            args_str = args_str.substr(delim_idx + 1);
        }
    }

    if (out_args.size() != arg_count) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Argument count error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


ImVec2 megamol::gui::configurator::GraphManager::readLuaProjectConfPos(const std::string& line) {

    ImVec2 conf_idx;

    std::string x_start_tag = "--confPos={X=";
    std::string y_start_tag = ",Y=";
    std::string end_tag = "}";
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
        } catch (std::invalid_argument e) {
            vislib::sys::Log::DefaultLog.WriteError(" Error while reading x value of confPos: %s [%s, %s, line %d]\n",
                e.what(), __FILE__, __FUNCTION__, __LINE__);
            return conf_idx;
        }
        try {
            std::string val_str = line.substr(y_start_idx + y_start_tag.length(), y_length);
            y = std::stof(val_str);
        } catch (std::invalid_argument e) {
            vislib::sys::Log::DefaultLog.WriteError(" Error while reading y value of confPos: %s [%s, %s, line %d]\n",
                e.what(), __FILE__, __FUNCTION__, __LINE__);
            return conf_idx;
        }
        conf_idx = ImVec2(x, y);
    }

    return conf_idx;
}


std::string megamol::gui::configurator::GraphManager::writeLuaProjectConfPos(const ImVec2& pos) {

    std::stringstream conf_idx;
    /// XXX Should position values be int for compatibility with old configurator?
    conf_idx << " --confPos={X=" << pos.x << ",Y=" << pos.y << "} ";
    return conf_idx.str();
}


bool megamol::gui::configurator::GraphManager::separateNameAndPrefix(
    const std::string& full_name, std::string& name_space, std::string& name) {

    name = full_name;
    name_space = "";
    size_t delimiter_index = full_name.rfind(':');
    if (delimiter_index != std::string::npos) {
        name = full_name.substr(delimiter_index + 1);
        name_space = full_name.substr(0, delimiter_index - 1);
    }

    if (name.empty()) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Invalid name in argument. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


// GRAPH MANAGET PRESENTATION ####################################################

megamol::gui::configurator::GraphManager::Presentation::Presentation(void)
    : delete_graph_uid(GUI_INVALID_ID), utils() {}


megamol::gui::configurator::GraphManager::Presentation::~Presentation(void) {}


ImGuiID megamol::gui::configurator::GraphManager::Presentation::Present(
    megamol::gui::configurator::GraphManager& inout_graph_manager, float in_child_width, ImFont* in_graph_font,
    HotKeyArrayType& inout_hotkeys) {

    ImGuiID retval = GUI_INVALID_ID;

    try {
        if (ImGui::GetCurrentContext() == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        const auto child_flags = ImGuiWindowFlags_None;

        ImGui::BeginChild("graph_child_window", ImVec2(in_child_width, 0.0f), true, child_flags);

        // Assuming only one closed tab/graph per frame.
        bool popup_close_unsaved = false;

        // Draw Graphs
        ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_Reorderable;
        ImGui::BeginTabBar("Graphs", tab_bar_flags);

        for (auto& graph : inout_graph_manager.GetGraphs()) {

            // Draw graph
            bool delete_graph = false;
            auto id = graph->GUI_Present(in_child_width, in_graph_font, inout_hotkeys, delete_graph);
            if (id != GUI_INVALID_ID) {
                retval = id;
            }

            // Do not delete graph while looping through graphs list
            if (delete_graph) {
                this->delete_graph_uid = retval;
                if (graph->IsDirty()) {
                    popup_close_unsaved = true;
                }
            }

            // Catch call drop event and create new call
            if (const ImGuiPayload* payload = ImGui::GetDragDropPayload()) {
                if (payload->IsDataType(GUI_DND_CALL_UID_TYPE) && payload->IsDelivery()) {
                    ImGuiID* dragged_call_slot_uid_ptr = (ImGuiID*)payload->Data;

                    auto drag_call_slot_uid = (*dragged_call_slot_uid_ptr);
                    auto drop_call_slot_uid = graph->GUI_GetDropCallSlot();
                    CallSlotPtrType drag_call_slot_ptr;
                    CallSlotPtrType drop_call_slot_ptr;
                    for (auto& mods : graph->GetGraphModules()) {
                        CallSlotPtrType call_slot_ptr = mods->GetCallSlot(drag_call_slot_uid);
                        if (call_slot_ptr != nullptr) {
                            drag_call_slot_ptr = call_slot_ptr;
                        }
                        call_slot_ptr = mods->GetCallSlot(drop_call_slot_uid);
                        if (call_slot_ptr != nullptr) {
                            drop_call_slot_ptr = call_slot_ptr;
                        }
                    }
                    graph->AddCall(inout_graph_manager.calls_stock, drag_call_slot_ptr, drop_call_slot_ptr);
                }
            }

            /*

            // Drag source
            label.resize(dnd_size);
            if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
                ImGui::SetDragDropPayload(
                    "DND_COPY_MODULE_PARAMETERS", label.c_str(), (label.size() * sizeof(char)));
                ImGui::Text(label.c_str());
                ImGui::EndDragDropSource();
            }

            // Drop target
            ImGui::Dummy(ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetFontSize()));
            if (ImGui::BeginDragDropTarget()) {
                if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("DND_COPY_MODULE_PARAMETERS")) {

                    IM_ASSERT(payload->DataSize == (dnd_size * sizeof(char)));
                    std::string payload_id = (const char*)payload->Data;

                    // Insert dragged module name only if not contained in list
                    if (!this->considerModule(payload_id, wc.param_modules_list)) {
                        wc.param_modules_list.emplace_back(payload_id);
                    }
                }
                ImGui::EndDragDropTarget();
            }

            */
        }
        ImGui::EndTabBar();

        // Delete marked graph when tab is closed and unsaved changes should be discarded.
        bool confirmed = false;
        bool aborted = false;
        bool popup_open = this->utils.MinimalPopUp(
            "Closing Unsaved Project", popup_close_unsaved, "Discard changes?", "Yes", confirmed, "No", aborted);
        if (this->delete_graph_uid != GUI_INVALID_ID) {
            if (aborted) {
                this->delete_graph_uid = GUI_INVALID_ID;
            } else if (confirmed || !popup_open) {
                inout_graph_manager.DeleteGraph(delete_graph_uid);
                this->delete_graph_uid = GUI_INVALID_ID;
            }
        }

        ImGui::EndChild();
    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    }

    return retval;
}
