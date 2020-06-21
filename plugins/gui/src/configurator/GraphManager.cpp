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


// GRAPH MANAGER PRESENTATION ####################################################

megamol::gui::configurator::GraphManagerPresentation::GraphManagerPresentation(void)
    : utils(), file_utils(), graph_delete_uid(GUI_INVALID_ID) {}


megamol::gui::configurator::GraphManagerPresentation::~GraphManagerPresentation(void) {}


void megamol::gui::configurator::GraphManagerPresentation::Present(
    megamol::gui::configurator::GraphManager& inout_graph_manager, GraphStateType& state) {

    try {
        if (ImGui::GetCurrentContext() == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return;
        }

        const auto child_flags = ImGuiWindowFlags_None;
        ImGui::BeginChild("graph_child_window", ImVec2(state.graph_width, 0.0f), true, child_flags);

        // Assuming only one closed tab/graph per frame.
        bool popup_close_unsaved = false;

        // Draw Graphs
        ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_Reorderable;
        ImGui::BeginTabBar("Graphs", tab_bar_flags);

        for (auto& graph : inout_graph_manager.GetGraphs()) {

            // Draw graph
            graph->GUI_Present(state);

            // Do not delete graph while looping through graphs list
            if (state.graph_delete) {
                this->graph_delete_uid = state.graph_selected_uid;
                if (graph->IsDirty()) {
                    popup_close_unsaved = true;
                }
                state.graph_delete = false;
            }

            // Catch call drop event and create new call(s) ...
            if (const ImGuiPayload* payload = ImGui::GetDragDropPayload()) {
                if (payload->IsDataType(GUI_DND_CALLSLOT_UID_TYPE) && payload->IsDelivery()) {
                    ImGuiID* dragged_slot_uid_ptr = (ImGuiID*)payload->Data;
                    auto drag_slot_uid = (*dragged_slot_uid_ptr);
                    auto drop_slot_uid = graph->GUI_GetDropSlot();
                    graph->AddCall(inout_graph_manager.GetCallsStock(), drag_slot_uid, drop_slot_uid);
                }
            }
        }
        ImGui::EndTabBar();

        // Save selected graph
        this->SaveProjectToFile(state.graph_save, inout_graph_manager, state);
        state.graph_save = false;

        // Delete selected graph when tab is closed and unsaved changes should be discarded.
        bool confirmed = false;
        bool aborted = false;
        bool popup_open = this->utils.MinimalPopUp(
            "Closing unsaved Project", popup_close_unsaved, "Discard changes?", "Yes", confirmed, "No", aborted);
        if (this->graph_delete_uid != GUI_INVALID_ID) {
            if (aborted) {
                this->graph_delete_uid = GUI_INVALID_ID;
            } else if (confirmed || !popup_open) {
                inout_graph_manager.DeleteGraph(graph_delete_uid);
                this->graph_delete_uid = GUI_INVALID_ID;
                state.graph_selected_uid = GUI_INVALID_ID;
            }
        }

        ImGui::EndChild();

    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


void megamol::gui::configurator::GraphManagerPresentation::SaveProjectToFile(
    bool open_popup, GraphManager& inout_graph_manager, GraphStateType& state) {

    bool confirmed, aborted;
    bool popup_failed = false;
    std::string project_filename;
    GraphPtrType graph_ptr;
    if (inout_graph_manager.GetGraph(state.graph_selected_uid, graph_ptr)) {
        project_filename = graph_ptr->GetFilename();
    }
    if (this->file_utils.FileBrowserPopUp(
            FileUtils::FileBrowserFlag::SAVE, "Save Editor Project", open_popup, project_filename)) {
        popup_failed = !inout_graph_manager.SaveProjectToFile(state.graph_selected_uid, project_filename);
    }
    this->utils.MinimalPopUp("Failed to Save Project", popup_failed, "See console log output for more information.", "",
        confirmed, "Cancel", aborted);
}

// GRAPH MANAGER ##############################################################

megamol::gui::configurator::GraphManager::GraphManager(void)
    : graphs(), modules_stock(), calls_stock(), graph_name_uid(0) {}


megamol::gui::configurator::GraphManager::~GraphManager(void) {}


ImGuiID megamol::gui::configurator::GraphManager::AddGraph(void) {

    ImGuiID retval = GUI_INVALID_ID;

    try {
        GraphPtrType graph_ptr = std::make_shared<Graph>(this->generate_unique_graph_name());
        if (graph_ptr != nullptr) {
            this->graphs.emplace_back(graph_ptr);
            retval = graph_ptr->uid;
#ifdef GUI_VERBOSE
            vislib::sys::Log::DefaultLog.WriteInfo(
                "[Configurator] Added graph %s' (uid %i). \n", graph_ptr->name.c_str(), graph_ptr->uid);
#endif // GUI_VERBOSE
        }
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


bool megamol::gui::configurator::GraphManager::DeleteGraph(ImGuiID graph_uid) {

    for (auto iter = this->graphs.begin(); iter != this->graphs.end(); iter++) {
        if ((*iter)->uid == graph_uid) {
#ifdef GUI_VERBOSE
            vislib::sys::Log::DefaultLog.WriteInfo(
                "[Configurator] Deleted graph %s' (uid %i). \n", (*iter)->name.c_str(), (*iter)->uid);
#endif // GUI_VERBOSE

            if ((*iter).use_count() > 1) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "Unclean deletion. Found %i references pointing to graph. [%s, %s, line %d]\n", (*iter).use_count(),
                    __FILE__, __FUNCTION__, __LINE__);
            }

            (*iter).reset();
            this->graphs.erase(iter);

            return true;
        }
    }

    return false;
}


bool megamol::gui::configurator::GraphManager::GetGraph(
    ImGuiID graph_uid, megamol::gui::configurator::GraphPtrType& out_graph_ptr) {

    if (graph_uid != GUI_INVALID_ID) {
        for (auto& graph_ptr : this->graphs) {
            if (graph_ptr->uid == graph_uid) {
                out_graph_ptr = graph_ptr;
                return true;
            }
        }
    }
    return false;
}


bool megamol::gui::configurator::GraphManager::LoadModulesCallsStock(
    const megamol::core::CoreInstance* core_instance) {

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
    
    bool retval = true;    
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
            "[Configurator] Reading available modules (#%i) and calls (#%i) ... DONE (duration: %.3f seconds)\n",
            this->modules_stock.size(), this->calls_stock.size(), delta_time);

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


ImGuiID megamol::gui::configurator::GraphManager::LoadUpdateProjectFromCore(ImGuiID graph_uid, megamol::core::CoreInstance* core_instance, ParamInterfaceMapType& inout_param_interface_map) {
    
    if (core_instance == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to core instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
    }
            
    GraphPtrType graph_ptr;
    ImGuiID current_graph_id = graph_uid;
    
    if (current_graph_id == GUI_INVALID_ID) {
        // Create new graph
        current_graph_id = this->AddGraph();
        if (current_graph_id == GUI_INVALID_ID) {
            vislib::sys::Log::DefaultLog.WriteError("Failed to create new graph. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return GUI_INVALID_ID;
        }
    }
    if (!this->GetGraph(current_graph_id, graph_ptr)) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to find graph for given uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    }
    
    if (this->AddProjectFromCore(current_graph_id, core_instance, false)) {
        
        
        /// TODO create inout_param_interface_map
        // inout_param_interface_map.clear()
        
        
        
        return graph_ptr->uid;
    }
    return GUI_INVALID_ID;
}


ImGuiID megamol::gui::configurator::GraphManager::LoadProjectFromCore(megamol::core::CoreInstance* core_instance) {

    if (core_instance == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to core instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
    }
            
    GraphPtrType graph_ptr;
    
    // Create new graph
    ImGuiID new_graph_id = this->AddGraph();
    if (new_graph_id == GUI_INVALID_ID) {
        vislib::sys::Log::DefaultLog.WriteError("Failed to create new graph. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    }
    if (!this->GetGraph(new_graph_id, graph_ptr)) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to find graph for given uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    }

    if (this->AddProjectFromCore(new_graph_id, core_instance, true)) {
        return graph_ptr->uid;
    }
    return GUI_INVALID_ID;
}


bool megamol::gui::configurator::GraphManager::AddProjectFromCore(
    ImGuiID graph_uid, megamol::core::CoreInstance* core_instance, bool use_stock) {

    try {
        if (core_instance == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Pointer to core instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                return false;
        }
            
        GraphPtrType graph_ptr;
        if (!this->GetGraph(graph_uid, graph_ptr)) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Unable to find graph for given uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        // Temporary data structure holding call connection data
        struct CallData {
            std::string caller_module_full_name;
            std::string caller_module_callslot_name;
            std::string callee_module_full_name;
            std::string callee_module_callslot_name;
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
            std::string full_name = std::string(mod->FullName().PeekBuffer());
            std::string module_name;
            std::string module_namespace;
            if (!this->project_separate_name_and_namespace(full_name, module_namespace, module_name)) {
                vislib::sys::Log::DefaultLog.WriteError("Core Project: Invalid module name '%s'. [%s, %s, line %d]\n",
                    full_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
            }

            /// DEBUG
            /// vislib::sys::Log::DefaultLog.WriteInfo(">>>> Class: '%s' NameSpace: '%s' Name: '%s'.\n",
            /// mod->ClassName(), module_namespace.c_str(), module_name.c_str());

            // Ensure unique module name is not yet assigned
            if (graph_ptr->UniqueModuleRename(module_name)) {
                vislib::sys::Log::DefaultLog.WriteWarn(
                    "Renamed existing module '%s' while adding module with same name. "
                    "This is required for successful unambiguous parameter addressing which uses the module name. [%s, "
                    "%s, line %d]\n",
                    module_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
            }

            // Creating new module
            graph_ptr->AddModule(this->modules_stock, std::string(mod->ClassName()));
            auto graph_module = graph_ptr->GetModules().back();
            graph_module->name = module_name;
            graph_module->is_view_instance = false;

            if (use_stock) {
                graph_ptr->AddGroupModule(module_namespace, graph_module);
            }
            else {
                
                /// TODO
                
            }

            if (view_instances.find(std::string(mod->FullName().PeekBuffer())) != view_instances.end()) {
                // Instance Name
                graph_ptr->name = view_instances[std::string(mod->FullName().PeekBuffer())];
                graph_module->is_view_instance = (graph_ptr->IsMainViewSet()) ? (false) : (true);
            }

            megamol::core::AbstractNamedObjectContainer::child_list_type::const_iterator se = mod->ChildList_End();
            for (megamol::core::AbstractNamedObjectContainer::child_list_type::const_iterator si =
                     mod->ChildList_Begin();
                 si != se; ++si) {

                // Parameter
                const auto param_slot = dynamic_cast<megamol::core::param::ParamSlot*>((*si).get());
                if (param_slot != nullptr) {
                    std::string param_full_name = std::string(param_slot->Name().PeekBuffer());
                    
                    if (use_stock) {
                        for (auto& parameter : graph_module->parameters) {
                            if (parameter.full_name == param_full_name) {
                                megamol::gui::configurator::ReadCoreParameter((*param_slot), parameter, full_name);
                            }
                        }
                    }
                    else {
                        
                        /// TODO
                        
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
                        cd.caller_module_callslot_name = std::string(call->PeekCallerSlot()->Name().PeekBuffer());
                        cd.callee_module_full_name =
                            std::string(call->PeekCalleeSlot()->Parent()->FullName().PeekBuffer());
                        cd.callee_module_callslot_name = std::string(call->PeekCalleeSlot()->Name().PeekBuffer());

                        call_data.emplace_back(cd);
                    }
                }
            }
        };
        core_instance->EnumModulesNoLock(nullptr, module_func);

        // Create calls
        for (auto& cd : call_data) {
            CallSlotPtrType callslot_1 = nullptr;
            for (auto& mod : graph_ptr->GetModules()) {
                if (mod->FullName() == cd.caller_module_full_name) {
                    for (auto& callslot : mod->GetCallSlots(CallSlotType::CALLER)) {
                        if (callslot->name == cd.caller_module_callslot_name) {
                            callslot_1 = callslot;
                        }
                    }
                }
            }
            CallSlotPtrType callslot_2 = nullptr;
            for (auto& mod : graph_ptr->GetModules()) {
                if (mod->FullName() == cd.callee_module_full_name) {
                    for (auto& callslot : mod->GetCallSlots(CallSlotType::CALLEE)) {
                        if (callslot->name == cd.callee_module_callslot_name) {
                            callslot_2 = callslot;
                        }
                    }
                }
            }
            
            if (use_stock) {
                graph_ptr->AddCall(this->GetCallsStock(), callslot_1, callslot_2);
            }
            else {
                
                /// TODO
                
            }            
        }

        graph_ptr->GUI_SetLayoutGraph();
        graph_ptr->ResetDirty();

        vislib::sys::Log::DefaultLog.WriteInfo(
            "[Configurator] Successfully loaded project '%s' from running MegaMol.\n", graph_ptr->name.c_str());

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


ImGuiID megamol::gui::configurator::GraphManager::LoadAddProjectFromFile(
    ImGuiID graph_uid, const std::string& project_filename) {

    std::string projectstr;
    if (!FileUtils::ReadFile(project_filename, projectstr)) return false;

    const std::string lua_view = "mmCreateView";
    const std::string lua_module = "mmCreateModule";
    const std::string lua_param = "mmSetParamValue";
    const std::string lua_call = "mmCreateCall";

    GraphPtrType graph_ptr;
    this->GetGraph(graph_uid, graph_ptr);
    ImGuiID retval = graph_uid;

    try {
        bool found_configurator_positions = false;

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
                if (!this->read_project_command_arguments(lines[i], arg_count, args)) {
                    vislib::sys::Log::DefaultLog.WriteError("Project File '%s' line %i: Error parsing lua command '%s' "
                                                            "requiring %i arguments. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), lua_view.c_str(), arg_count, __FILE__, __FUNCTION__,
                        __LINE__);
                    return GUI_INVALID_ID;
                }

                std::string view_instance = args[0];
                std::string view_class_name = args[1];
                std::string view_full_name = args[2];
                std::string view_namespace;
                std::string view_name;
                if (!this->project_separate_name_and_namespace(view_full_name, view_namespace, view_name)) {
                    vislib::sys::Log::DefaultLog.WriteError("Project File '%s' line %i: Invalid view name argument "
                                                            "(3rd) in lua command '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), lua_view.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    return GUI_INVALID_ID;
                }

                /// DEPRECATED
                /*
                ImVec2 module_pos = this->project_read_confpos(lines[i]);
                if ((module_pos.x != FLT_MAX) && (module_pos.x != FLT_MAX)) found_configurator_positions = true;
                */
                /// DEPRECATED

                /// DEBUG
                /// vislib::sys::Log::DefaultLog.WriteInfo(
                ///     ">>>> Instance: '%s' Class: '%s' NameSpace: '%s' Name: '%s' ConfPos: %f, %f.\n",
                ///     view_instance.c_str(), view_class_name.c_str(), view_namespace.c_str(), view_name.c_str());

                // Create new graph
                if (graph_ptr == nullptr) {
                    ImGuiID new_graph_uid = this->AddGraph();
                    if (new_graph_uid == GUI_INVALID_ID) {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "Project File '%s' line %i: Unable to create new graph '%s'. [%s, %s, line %d]\n",
                            project_filename.c_str(), (i + 1), view_instance.c_str(), __FILE__, __FUNCTION__, __LINE__);
                        return GUI_INVALID_ID;
                    }
                    if (!this->GetGraph(new_graph_uid, graph_ptr)) {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "Unable to get pointer to last added graph. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                            __LINE__);
                        return GUI_INVALID_ID;
                    }
                    graph_ptr->name = view_instance;
                    retval = new_graph_uid;
                }

                // Ensure unique module name is not yet assigned
                if (graph_ptr->UniqueModuleRename(view_name)) {
                    vislib::sys::Log::DefaultLog.WriteWarn(
                        "Project File '%s' line %i: Renamed existing module '%s' while adding module with same name. "
                        "This is required for successful unambiguous parameter addressing which uses the module name. "
                        "[%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), view_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                }

                // Add module and set as view instance
                if (graph_ptr->AddModule(this->modules_stock, view_class_name) == GUI_INVALID_ID) {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "Project File '%s' line %i: Unable to add new module '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), view_class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    return GUI_INVALID_ID;
                }
                auto graph_module = graph_ptr->GetModules().back();
                graph_module->name = view_name;
                graph_module->is_view_instance = (graph_ptr->IsMainViewSet()) ? (false) : (true);
                graph_ptr->AddGroupModule(view_namespace, graph_module);

                /// DEPRECATED
                /*
                graph_module->GUI_SetPosition(module_pos);
                */
                /// DEPRECATED

                found_main_view = true;
            }
        }
        if (!found_main_view) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Project File '%s': Missing main view lua command '%s'. [%s, %s, line %d]\n", project_filename.c_str(),
                lua_view.c_str(), __FILE__, __FUNCTION__, __LINE__);
        }
        // Save filename for graph (before saving states!)
        graph_ptr->SetFilename(project_filename);

        // Find and create modules
        for (unsigned int i = 0; i < lines_count; i++) {
            // Lua command must start at beginning after removeing leading spaces
            if (lines[i].rfind(lua_module, 0) == 0) {

                size_t arg_count = 2;
                std::vector<std::string> args;
                if (!this->read_project_command_arguments(lines[i], arg_count, args)) {
                    vislib::sys::Log::DefaultLog.WriteError("Project File '%s' line %i: Error parsing lua command '%s' "
                                                            "requiring %i arguments. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), lua_module.c_str(), arg_count, __FILE__, __FUNCTION__,
                        __LINE__);
                    return GUI_INVALID_ID;
                }

                std::string module_class_name = args[0];
                std::string module_full_name = args[1];
                std::string module_namespace;
                std::string module_name;
                if (!this->project_separate_name_and_namespace(module_full_name, module_namespace, module_name)) {
                    vislib::sys::Log::DefaultLog.WriteError("Project File '%s' line %i: Invalid module name argument "
                                                            "(2nd) in lua command '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), lua_module.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    return GUI_INVALID_ID;
                }

                /// DEPRECATED
                /*
                ImVec2 module_pos = this->project_read_confpos(lines[i]);
                if ((module_pos.x != FLT_MAX) && (module_pos.x != FLT_MAX)) found_configurator_positions = true;
                */
                /// DEPRECATED

                /// DEBUG
                /// vislib::sys::Log::DefaultLog.WriteInfo(">>>> Class: '%s' NameSpace: '%s' Name: '%s' ConfPos: %f,
                /// %f.\n",
                ///    module_class_name.c_str(), module_namespace.c_str(), module_name.c_str());

                // Add module
                if (graph_ptr != nullptr) {

                    // Ensure unique module name is not yet assigned
                    if (graph_ptr->UniqueModuleRename(module_name)) {
                        vislib::sys::Log::DefaultLog.WriteWarn(
                            "Project File '%s' line %i: Renamed existing module '%s' while adding module with same "
                            "name. "
                            "This is required for successful unambiguous parameter addressing which uses the module "
                            "name. [%s, %s, line %d]\n",
                            project_filename.c_str(), (i + 1), module_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    }

                    if (graph_ptr->AddModule(this->modules_stock, module_class_name) == GUI_INVALID_ID) {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "Project File '%s' line %i: Unable to add new module '%s'. [%s, %s, line %d]\n",
                            project_filename.c_str(), (i + 1), module_class_name.c_str(), __FILE__, __FUNCTION__,
                            __LINE__);
                        return GUI_INVALID_ID;
                    }
                    auto graph_module = graph_ptr->GetModules().back();
                    graph_module->name = module_name;
                    graph_module->is_view_instance = false;
                    graph_ptr->AddGroupModule(module_namespace, graph_module);

                    /// DEPRECATED
                    /*
                    graph_module->GUI_SetPosition(module_pos);
                    */
                    /// DEPRECATED
                }
            }
        }

        // Find and create calls
        for (unsigned int i = 0; i < lines_count; i++) {
            // Lua command must start at beginning after removing leading spaces
            if (lines[i].rfind(lua_call, 0) == 0) {

                size_t arg_count = 3;
                std::vector<std::string> args;
                if (!this->read_project_command_arguments(lines[i], arg_count, args)) {
                    vislib::sys::Log::DefaultLog.WriteError("Project File '%s' line %i: Error parsing lua command '%s' "
                                                            "requiring %i arguments. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), lua_call.c_str(), arg_count, __FILE__, __FUNCTION__,
                        __LINE__);
                    return GUI_INVALID_ID;
                }

                std::string call_class_name = args[0];
                std::string caller_slot_full_name = args[1];
                std::string callee_slot_full_name = args[2];

                std::string caller_slot_name;
                std::string caller_slot_namespace;
                if (!this->project_separate_name_and_namespace(
                        caller_slot_full_name, caller_slot_namespace, caller_slot_name)) {
                    vislib::sys::Log::DefaultLog.WriteError("Project File '%s' line %i: Invalid caller slot name "
                                                            "argument (2nd) in lua command '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), lua_call.c_str(), __FILE__, __FUNCTION__, __LINE__);
                }

                std::string callee_slot_name;
                std::string callee_slot_namespace;
                if (!this->project_separate_name_and_namespace(
                        callee_slot_full_name, callee_slot_namespace, callee_slot_name)) {
                    vislib::sys::Log::DefaultLog.WriteError("Project File '%s' line %i: Invalid callee slot name "
                                                            "argument (3nd) in lua command '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), lua_call.c_str(), __FILE__, __FUNCTION__, __LINE__);
                }

                /// DEBUG
                /// vislib::sys::Log::DefaultLog.WriteInfo(
                ///    ">>>> Call Name: '%s' CALLER Module: '%s' Slot: '%s' - CALLEE Module: '%s' Slot: '%s'.\n",
                ///    call_class_name.c_str(), caller_slot_namespace.c_str(), caller_slot_name.c_str(),
                ///    callee_slot_namespace.c_str(), callee_slot_name.c_str());

                // Searching for call
                if (graph_ptr != nullptr) {
                    std::string module_full_name;
                    size_t module_name_idx = std::string::npos;
                    std::string callee_name, caller_name;
                    CallSlotPtrType callee_slot, caller_slot;

                    for (auto& mod : graph_ptr->GetModules()) {
                        module_full_name = mod->FullName() + "::";
                        // Caller
                        module_name_idx = caller_slot_full_name.find(module_full_name);
                        if (module_name_idx != std::string::npos) {
                            for (auto& callslot_map : mod->GetCallSlots()) {
                                for (auto& callslot : callslot_map.second) {
                                    if (caller_slot_name == callslot->name) {
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
                                    if (callee_slot_name == callslot->name) {
                                        callee_slot = callslot;
                                    }
                                }
                            }
                        }
                    }

                    if (callee_slot == nullptr) {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "Project File '%s' line %i: Unable to find callee slot '%s' "
                            "for creating call '%s'. [%s, %s, line %d]\n",
                            project_filename.c_str(), (i + 1), callee_slot_full_name.c_str(), call_class_name.c_str(),
                            __FILE__, __FUNCTION__, __LINE__);
                        return GUI_INVALID_ID;
                    }
                    if (caller_slot == nullptr) {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "Project File '%s' line %i: Unable to find caller slot '%s' "
                            "for creating call '%s'. [%s, %s, line %d]\n",
                            project_filename.c_str(), (i + 1), caller_slot_full_name.c_str(), call_class_name.c_str(),
                            __FILE__, __FUNCTION__, __LINE__);
                        return GUI_INVALID_ID;
                    }


                    // Add call
                    if (!graph_ptr->AddCall(this->calls_stock, caller_slot, callee_slot)) {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "Project File '%s' line %i: Unable to add new call '%s'. [%s, %s, line %d]\n",
                            project_filename.c_str(), (i + 1), call_class_name.c_str(), __FILE__, __FUNCTION__,
                            __LINE__);
                        return GUI_INVALID_ID;
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
                        project_filename.c_str(), (i + 1), lua_param.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    return GUI_INVALID_ID;
                }
                size_t first_delimiter_idx = param_line.find(',');
                if (first_delimiter_idx == std::string::npos) {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "Project File '%s' line %i: Missing argument delimiter ',' for '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), lua_param.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    return GUI_INVALID_ID;
                }

                std::string param_slot_full_name =
                    param_line.substr(first_bracket_idx + 1, (first_delimiter_idx - first_bracket_idx - 1));
                if ((param_slot_full_name.front() != '"') || (param_slot_full_name.back() != '"')) {
                    vislib::sys::Log::DefaultLog.WriteError("Project File '%s' line %i: Parameter name argument should "
                                                            "be enclosed in '\"' for '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), lua_param.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    return GUI_INVALID_ID;
                }
                param_slot_full_name = param_slot_full_name.substr(1, param_slot_full_name.size() - 2);

                /// DEBUG
                /// vislib::sys::Log::DefaultLog.WriteInfo(">>>> %s\n", param_slot_full_name.c_str());

                // Copy multi line parameter values into one string
                auto value_start_idx = param_line.find(start_delimieter);
                if (value_start_idx == std::string::npos) {
                    vislib::sys::Log::DefaultLog.WriteError("Project File '%s' line %i: Unable to find parameter value "
                                                            "start delimiter '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), start_delimieter.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    return GUI_INVALID_ID;
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
                        project_filename.c_str(), (i + 1), end_delimieter.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    return GUI_INVALID_ID;
                }
                std::string value_str = param_line.substr(value_start_idx + start_delimieter.size(),
                    (param_line.find(end_delimieter)) - value_start_idx - end_delimieter.size());

                /// DEBUG
                /// vislib::sys::Log::DefaultLog.WriteInfo(">>>> '%s'\n", value_str.c_str());

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
                                if (parameter.full_name == param_full_name) {
                                    parameter.SetValueString(value_str);

                                    // Reading state parameters
                                    /// XXX State parameters have no newline formatting
                                    if (module_ptr->class_name == GUI_MODULE_NAME) {
                                        if (parameter.full_name == GUI_GUI_STATE_PARAM_NAME) {
                                            // Reading gui state param containing parameter gui states
                                            this->parameters_gui_state_from_json_string(graph_ptr, value_str);
                                        }
                                        if (parameter.full_name == GUI_CONFIGURATOR_STATE_PARAM_NAME) {
                                            // Reading configurator state param containing a graph state
                                            /// ! Needs filename set for graph because this is how the graph state is
                                            /// found inside the JSON state
                                            if (graph_ptr->GUI_StateFromJsonString(value_str)) {
                                                found_configurator_positions = true;
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

        // Layout graph if no positions for modules could be found in state.
        if (!found_configurator_positions) {
            graph_ptr->GUI_SetLayoutGraph();
        }
        graph_ptr->ResetDirty();

        vislib::sys::Log::DefaultLog.WriteInfo("[Configurator] Successfully loaded project '%s' from file '%s'.\n",
            graph_ptr->name.c_str(), project_filename.c_str());

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


bool megamol::gui::configurator::GraphManager::SaveProjectToFile(ImGuiID graph_uid, const std::string& project_filename) {

    std::string projectstr;
    std::stringstream confInstances, confModules, confCalls, confParams;
    GraphPtrType found_graph_ptr = nullptr;

    bool wrote_graph_state = false;
    bool wrote_parameter_gui_state = false;

    try {
        for (auto& graph_ptr : this->graphs) {
            if (graph_ptr->uid == graph_uid) {

                // Some pre-checks
                bool found_error = false;
                bool found_instance = false;
                for (auto& mod_1 : graph_ptr->GetModules()) {
                    for (auto& mod_2 : graph_ptr->GetModules()) {
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

                // Save filename for graph (before saving states!)
                graph_ptr->SetFilename(project_filename);

                // Serialze graph to string
                for (auto& module_ptr : graph_ptr->GetModules()) {
                    std::string instance_name = graph_ptr->name;
                    if (module_ptr->is_view_instance) {
                        confInstances << "mmCreateView(\"" << instance_name << "\",\"" << module_ptr->class_name
                                      << "\",\"" << module_ptr->FullName() << "\") \n";
                    } else {
                        confModules << "mmCreateModule(\"" << module_ptr->class_name << "\",\""
                                    << module_ptr->FullName() << "\") \n";
                    }

                    for (auto& parameter : module_ptr->parameters) {

                        // Writing state parameters
                        /// ! Needs filename set for graph because this is how the graph state is found
                        /// inside the JSON state
                        if (module_ptr->class_name == GUI_MODULE_NAME) {
                            // Store graph state to state parameter of configurator
                            if (!wrote_graph_state && (parameter.full_name == GUI_CONFIGURATOR_STATE_PARAM_NAME)) {
                                // Replacing exisiting graph state with new one and leaving rest untouched
                                std::string new_configurator_graph_state;
                                this->replace_graph_state(
                                    graph_ptr, parameter.GetValueString(), new_configurator_graph_state);
                                parameter.SetValue(new_configurator_graph_state);
                                wrote_graph_state = true;
                            }
                            // Store parameter gui states to state parameter of gui
                            if (!wrote_parameter_gui_state && (parameter.full_name == GUI_GUI_STATE_PARAM_NAME)) {
                                // Replacing exisiting parameter gui state with new one and leaving rest untouched
                                std::string new_parameter_gui_state;
                                this->replace_parameter_gui_state(
                                    graph_ptr, parameter.GetValueString(), new_parameter_gui_state);
                                parameter.SetValue(new_parameter_gui_state);
                                wrote_parameter_gui_state = true;
                            }
                        }

                        // Only write parameters with other values than the default
                        if (parameter.DefaultValueMismatch()) {
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

                projectstr = confInstances.str() + "\n" + confModules.str() + "\n" + confCalls.str() + "\n" +
                             confParams.str() + "\n";

                found_graph_ptr = graph_ptr;
            }
        }

        if (found_graph_ptr != nullptr) {
            found_graph_ptr->ResetDirty();
            if (FileUtils::WriteFile(project_filename, projectstr)) {

                vislib::sys::Log::DefaultLog.WriteInfo("[Configurator] Successfully saved project '%s' to file '%s'.\n",
                    found_graph_ptr->name.c_str(), project_filename.c_str());
                return true;
            }
        } else {
            vislib::sys::Log::DefaultLog.WriteWarn(
                "Invalid graph uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
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

    return false;
}


bool megamol::gui::configurator::GraphManager::get_module_stock_data(
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
        vislib::sys::Log::DefaultLog.WriteError(
            "Call list is empty. Call read_call_data() prior to that. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
        return false;
    }

    try {
        /// Following code is adapted from megamol::core::job::job::PluginsStateFileGeneratorJob.cpp

        // vislib::sys::Log::DefaultLog.WriteInfo("[Configurator] Creating module '%s' ...", mod_desc->ClassName());

        megamol::core::Module::ptr_type new_mod(mod_desc->CreateModule(nullptr));
        if (new_mod == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError("Unable to create module '%s'. [%s, %s, line %d]\n",
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
            if (param_slot == nullptr) {
                break;
            }
            Parameter::StockParameter psd;
            if (megamol::gui::configurator::ReadCoreParameter((*param_slot), psd)) {
                mod.parameters.emplace_back(psd);
            }
        }

        // CallerSlots
        for (std::shared_ptr<core::CallerSlot> caller_slot : callerSlots) {
            CallSlot::StockCallSlot csd;
            csd.name = std::string(caller_slot->Name().PeekBuffer());
            csd.description = std::string(caller_slot->Description().PeekBuffer());
            csd.compatible_call_idxs.clear();
            csd.type = CallSlotType::CALLER;

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

            mod.callslots[csd.type].emplace_back(csd);
        }

        // CalleeSlots
        for (std::shared_ptr<core::CalleeSlot> callee_slot : calleeSlots) {
            CallSlot::StockCallSlot csd;
            csd.name = std::string(callee_slot->Name().PeekBuffer());
            csd.description = std::string(callee_slot->Description().PeekBuffer());
            csd.compatible_call_idxs.clear();
            csd.type = CallSlotType::CALLEE;

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

            mod.callslots[csd.type].emplace_back(csd);
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
        call.class_name = std::string(call_desc->ClassName());
        call.description = std::string(call_desc->Description());
        call.functions.clear();
        for (unsigned int i = 0; i < call_desc->FunctionCount(); ++i) {
            call.functions.emplace_back(call_desc->FunctionName(i));
        }
        /// call.plugin_name is not (yet) available in call_desc (set from AbstractAssemblyInstance or
        /// AbstractPluginInstance).

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


bool megamol::gui::configurator::GraphManager::read_project_command_arguments(
    const std::string& line, size_t arg_count, std::vector<std::string>& out_args) {

    /// Can be used for mmCreateView, mmCreateModule and mmCreateCall lua commands

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


ImVec2 megamol::gui::configurator::GraphManager::project_read_confpos(const std::string& line) {

    ImVec2 conf_pos(FLT_MAX, FLT_MAX);

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
            return conf_pos;
        }
        try {
            std::string val_str = line.substr(y_start_idx + y_start_tag.length(), y_length);
            y = std::stof(val_str);
        } catch (std::invalid_argument e) {
            vislib::sys::Log::DefaultLog.WriteError(" Error while reading y value of confPos: %s [%s, %s, line %d]\n",
                e.what(), __FILE__, __FUNCTION__, __LINE__);
            return conf_pos;
        }
        conf_pos = ImVec2(x, y);
    }

    return conf_pos;
}


bool megamol::gui::configurator::GraphManager::project_separate_name_and_namespace(
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
                vislib::sys::Log::DefaultLog.WriteError(
                    "Invalid namespace in argument. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                return false;
            }
        }
    }

    if (name.empty()) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Invalid name in argument. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


/// ! Implementation should be duplicate to Core-Version
/// megamol::gui::GUIWindows::parameters_gui_state_from_json_string()
bool megamol::gui::configurator::GraphManager::parameters_gui_state_from_json_string(
    const GraphPtrType& graph_ptr, const std::string& in_json_string) {

    try {
        if (in_json_string.empty()) {
            return false;
        }
        if (graph_ptr == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Pointer to graph is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        bool found = false;
        bool valid = true;
        nlohmann::json json;
        json = nlohmann::json::parse(in_json_string);
        if (!json.is_object()) {
            vislib::sys::Log::DefaultLog.WriteError(
                "State is no valid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        for (auto& header_item : json.items()) {
            if (header_item.key() == GUI_JSON_TAG_GUISTATE_PARAMETERS) {
                found = true;
                for (auto& config_item : header_item.value().items()) {
                    std::string json_param_name = config_item.key();
                    auto gui_state = config_item.value();
                    valid = true;

                    // gui_visibility
                    bool gui_visibility;
                    if (gui_state.at("gui_visibility").is_boolean()) {
                        gui_state.at("gui_visibility").get_to(gui_visibility);
                    } else {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "JSON state: Failed to read 'gui_visibility' as boolean. [%s, %s, line %d]\n", __FILE__,
                            __FUNCTION__, __LINE__);
                        valid = false;
                    }

                    // gui_read-only
                    bool gui_read_only;
                    if (gui_state.at("gui_read-only").is_boolean()) {
                        gui_state.at("gui_read-only").get_to(gui_read_only);
                    } else {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "JSON state: Failed to read 'gui_read-only' as boolean. [%s, %s, line %d]\n", __FILE__,
                            __FUNCTION__, __LINE__);
                        valid = false;
                    }

                    // gui_presentation_mode
                    PresentType gui_presentation_mode;
                    if (gui_state.at("gui_presentation_mode").is_number_integer()) {
                        gui_presentation_mode =
                            static_cast<PresentType>(gui_state.at("gui_presentation_mode").get<int>());
                    } else {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "JSON state: Failed to read 'gui_presentation_mode' as integer. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        valid = false;
                    }

                    if (valid) {
                        for (auto& module_ptr : graph_ptr->GetModules()) {
                            for (auto& parameter : module_ptr->parameters) {
                                if (parameter.full_name == json_param_name) {
                                    parameter.GUI_SetVisibility(gui_visibility);
                                    parameter.GUI_SetReadOnly(gui_read_only);
                                    parameter.GUI_SetPresentation(gui_presentation_mode);
                                }
                            }
                        }
                    }
                }
            }
        }

        if (found) {
#ifdef GUI_VERBOSE
            vislib::sys::Log::DefaultLog.WriteInfo("[Configurator] Read parameter gui state from JSON string.");
#endif // GUI_VERBOSE
        } else {
#ifdef GUI_VERBOSE
            vislib::sys::Log::DefaultLog.WriteWarn(
                "Could not find parameter gui state in JSON. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
#endif // GUI_VERBOSE
            return false;
        }

    } catch (nlohmann::json::type_error& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::invalid_iterator& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::out_of_range& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::other_error& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Unknown Error - Unable to parse JSON string. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


/// ! Implementation should be duplicate to Core-Version megamol::gui::GUIWindows::parameters_gui_state_to_json()
bool megamol::gui::configurator::GraphManager::parameters_gui_state_to_json(
    const GraphPtrType& graph_ptr, nlohmann::json& out_json) {

    try {
        if (graph_ptr == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Pointer to graph is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        /// Append to given json
        // out_json.clear();

        for (auto& module_ptr : graph_ptr->GetModules()) {
            for (auto& parameter : module_ptr->parameters) {
                out_json[GUI_JSON_TAG_GUISTATE_PARAMETERS][parameter.full_name]["gui_visibility"] =
                    parameter.GUI_GetVisibility();
                out_json[GUI_JSON_TAG_GUISTATE_PARAMETERS][parameter.full_name]["gui_read-only"] =
                    parameter.GUI_GetReadOnly();
                out_json[GUI_JSON_TAG_GUISTATE_PARAMETERS][parameter.full_name]["gui_presentation_mode"] =
                    static_cast<int>(parameter.GUI_GetPresentation());
            }
        }
#ifdef GUI_VERBOSE
        vislib::sys::Log::DefaultLog.WriteInfo("[Configurator] Wrote parameter gui state to JSON.");
#endif // GUI_VERBOSE

    } catch (nlohmann::json::type_error& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::invalid_iterator& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::out_of_range& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::other_error& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Unknown Error - Unable to write JSON of state. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


bool megamol::gui::configurator::GraphManager::replace_graph_state(
    const GraphPtrType& graph_ptr, const std::string& in_json_string, std::string& out_json_string) {

    try {
        nlohmann::json json;
        if (!in_json_string.empty()) {
            json = nlohmann::json::parse(in_json_string);
            if (!json.is_object()) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "State is no valid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                return false;
            }
            std::string json_graph_id = graph_ptr->GetFilename(); /// = graph filename
            json[GUI_JSON_TAG_GRAPHS].erase(json_graph_id);
        }
        if (graph_ptr->GUI_StateToJSON(json)) {
            out_json_string = json.dump(2);
        } else {
            return false;
        }

    } catch (nlohmann::json::type_error& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::invalid_iterator& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::out_of_range& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::other_error& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Unknown Error - Unable to parse JSON string. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


bool megamol::gui::configurator::GraphManager::replace_parameter_gui_state(
    const GraphPtrType& graph_ptr, const std::string& in_json_string, std::string& out_json_string) {

    try {
        nlohmann::json json;
        if (!in_json_string.empty()) {
            json = nlohmann::json::parse(in_json_string);
            if (!json.is_object()) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "State is no valid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                return false;
            }
            json.erase(GUI_JSON_TAG_GUISTATE_PARAMETERS);
        }
        if (this->parameters_gui_state_to_json(graph_ptr, json)) {
            out_json_string = json.dump(2);
        } else {
            return false;
        }

    } catch (nlohmann::json::type_error& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::invalid_iterator& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::out_of_range& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::other_error& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Unknown Error - Unable to parse JSON string. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}
