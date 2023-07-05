/*
 * GraphCollection.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "GraphCollection.h"
#include "mmcore/utility/FileUtils.h"
#include "mmcore/utility/buildinfo/BuildInfo.h"
#include "mmcore/view/AbstractViewInterface.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::GraphCollection::GraphCollection()
        : graphs()
        , modules_stock()
        , calls_stock()
        , graph_name_uid(0)
        , gui_file_browser()
        , gui_graph_delete_uid(GUI_INVALID_ID)
        , created_running_graph(false)
        , initialized_syncing(false) {}


bool megamol::gui::GraphCollection::AddEmptyProject() {

    ImGuiID graph_uid = this->AddGraph();
    if (graph_uid != GUI_INVALID_ID) {
        /// Setup new empty graph ...
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to create new graph. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }

    return false;
}


ImGuiID megamol::gui::GraphCollection::AddGraph() {

    ImGuiID retval = GUI_INVALID_ID;

    try {
        GraphPtr_t graph_ptr = std::make_shared<Graph>(this->generate_unique_graph_name());
        if (graph_ptr != nullptr) {
            this->graphs.emplace_back(graph_ptr);
            retval = graph_ptr->UID();
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[GUI] Added graph %s' (uid %i). \n", graph_ptr->Name().c_str(), graph_ptr->UID());
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
        if ((*iter)->UID() == in_graph_uid) {
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[GUI] Deleted graph %s' (uid %i). \n", (*iter)->Name().c_str(), (*iter)->UID());
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


megamol::gui::GraphPtr_t megamol::gui::GraphCollection::GetGraph(ImGuiID in_graph_uid) {

    if (in_graph_uid != GUI_INVALID_ID) {
        for (auto& graph_ptr : this->graphs) {
            if (graph_ptr->UID() == in_graph_uid) {
                return graph_ptr;
            }
        }
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Unable to find graph for given graph uid: %i [%s, %s, line %d]\n", in_graph_uid, __FILE__,
            __FUNCTION__, __LINE__);
    }
    return nullptr;
}


megamol::gui::GraphPtr_t megamol::gui::GraphCollection::GetRunningGraph() {

    for (auto& graph_ptr : this->graphs) {
        if (graph_ptr->IsRunning()) {
            // Return first found running graph
            return graph_ptr;
        }
    }
    // megamol::core::utility::log::Log::DefaultLog.WriteWarn("[GUI] Unable to find running graph. [%s, %s, line %d]\n",
    // __FILE__, __FUNCTION__, __LINE__);
    return nullptr;
}


bool megamol::gui::GraphCollection::load_call_stock(const megamol::frontend_resources::PluginsResource& pluginsRes) {

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
            for (auto& plugin : pluginsRes.plugins) {
                plugin_name = plugin->GetObjectFactoryName();
                for (auto& c_desc : plugin->GetCallDescriptionManager()) {
                    Call::StockCall call;
                    if (this->get_call_stock_data(call, c_desc, plugin_name)) {
                        this->calls_stock.emplace_back(call);
                    }
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


bool megamol::gui::GraphCollection::load_module_stock(const megamol::frontend_resources::PluginsResource& pluginsRes) {

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
            for (auto& plugin : pluginsRes.plugins) {
                plugin_name = plugin->GetObjectFactoryName();
                for (auto& m_desc : plugin->GetModuleDescriptionManager()) {
                    Module::StockModule mod;
                    if (this->get_module_stock_data(mod, m_desc, plugin_name)) {
                        this->modules_stock.emplace_back(mod);
                    }
#ifdef GUI_VERBOSE
                    auto module_load_time_count =
                        static_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - module_load_time)
                            .count();
                    module_load_time = std::chrono::system_clock::now();
                    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                        "[GUI] Reading module '%s' ... DONE (duration: %.3f seconds)\n", mod.class_name.c_str(),
                        module_load_time_count);
#endif // GUI_VERBOSE
                }
            }

            // Sorting module by alphabetically ascending class names.
            std::sort(this->modules_stock.begin(), this->modules_stock.end(),
                [](Module::StockModule& mod1, Module::StockModule& mod2) {
                    std::string a_str(mod1.class_name);
                    core::utility::string::ToUpperAscii(a_str);
                    std::string b_str(mod2.class_name);
                    core::utility::string::ToUpperAscii(b_str);
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


void megamol::gui::GraphCollection::SetLuaFunc(lua_func_type* func) {

    this->input_lua_func = func;
}


bool megamol::gui::GraphCollection::InitializeGraphSynchronisation(
    const megamol::frontend_resources::PluginsResource& pluginsRes) {

    // Load all known calls from core instance ONCE
    if (!this->load_call_stock(pluginsRes)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Failed to load call stock once. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Load all known modules from core instance ONCE
    if (!this->load_module_stock(pluginsRes)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Failed to load module stock once. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Create inital running graph
    auto graph_ptr = this->GetRunningGraph();
    ImGuiID valid_graph_id = (graph_ptr != nullptr) ? (graph_ptr->UID()) : (GUI_INVALID_ID);
    if (valid_graph_id == GUI_INVALID_ID) {

        valid_graph_id = this->AddGraph();
        if (valid_graph_id == GUI_INVALID_ID) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Failed to create new graph. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        graph_ptr = this->GetGraph(valid_graph_id);
        if (graph_ptr == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Unable to find graph for given uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        } else {
            graph_ptr->SetRunning(true);
            this->created_running_graph = true;
            this->initialized_syncing = true;
        }
    }

    return this->initialized_syncing;
}


bool megamol::gui::GraphCollection::SynchronizeGraphs(megamol::core::MegaMolGraph& megamol_graph) {

    if (!this->initialized_syncing) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Graph synchronization not initialized. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    bool synced = false;
    bool sync_success = true;
    auto graph_ptr = this->GetRunningGraph();

    /// Convenience: Layout new graph initally after changes have been propagated from core to GUI
    if ((graph_ptr != nullptr) && (this->created_running_graph)) {
        this->created_running_graph = false;
        graph_ptr->SetLayoutGraph();
        graph_ptr->ResetDirty();
    }

    // Propagate all changes from the GUI graph to the MegaMol graph
    if (graph_ptr != nullptr) {
        bool graph_sync_success = true;

        Graph::QueueAction action;
        Graph::QueueData data;
        while (graph_ptr->PopSyncQueue(action, data)) {
            synced = true;

            switch (action) {
            case (Graph::QueueAction::ADD_MODULE): {
                auto created = std::get<0>(
                    (*input_lua_func)("mmCreateModule([=[" + data.class_name + "]=],[=[" + data.name_id + "]=])"));
                graph_sync_success &= created;
#ifdef MEGAMOL_USE_PROFILING
                if (created) {
                    auto gui_module_ptr = graph_ptr->GetModule(data.name_id);
                    auto graph_module_ptr = megamol_graph.FindModule(data.name_id).get();
                    gui_module_ptr->SetProfilingData(graph_module_ptr, perf_manager);
                    module_to_module[graph_module_ptr] = gui_module_ptr;
                }
#endif
            } break;
            case (Graph::QueueAction::RENAME_MODULE): {
                graph_sync_success &= std::get<0>(
                    (*input_lua_func)("mmRenameModule([=[" + data.name_id + "]=],[=[" + data.rename_id + "]=])"));
            } break;
            case (Graph::QueueAction::DELETE_MODULE): {
#ifdef MEGAMOL_USE_PROFILING
                module_to_module.erase(megamol_graph.FindModule(data.name_id).get());
#endif
                graph_sync_success &= std::get<0>((*input_lua_func)("mmDeleteModule([=[" + data.name_id + "]=])"));
            } break;
            case (Graph::QueueAction::ADD_CALL): {
                auto created = std::get<0>((*input_lua_func)(
                    "mmCreateCall([=[" + data.class_name + "]=],[=[" + data.caller + "]=],[=[" + data.callee + "]=])"));
                graph_sync_success &= created;
#ifdef MEGAMOL_USE_PROFILING
                if (created) {
                    auto gui_call_ptr = graph_ptr->GetCall(data.class_name, data.caller, data.callee);
                    auto graph_call_ptr = megamol_graph.FindCall(data.caller, data.callee).get();
                    gui_call_ptr->SetProfilingData(graph_call_ptr, graph_call_ptr->GetCallbackCount());
                    call_to_call[graph_call_ptr] = gui_call_ptr;
                }
#endif
            } break;
            case (Graph::QueueAction::DELETE_CALL): {
#ifdef MEGAMOL_USE_PROFILING
                call_to_call.erase(megamol_graph.FindCall(data.caller, data.callee).get());
#endif
                graph_sync_success &=
                    std::get<0>((*input_lua_func)("mmDeleteCall([=[" + data.caller + "]=],[=[" + data.callee + "]=])"));
            } break;
            case (Graph::QueueAction::CREATE_GRAPH_ENTRY): {
                // megamol currently does not handle well having multiple entrypoints active
                (*input_lua_func)("mmRemoveAllGraphEntryPoints()\n"
                                  "mmSetGraphEntryPoint([=[" +
                                  data.name_id + "]=])");
            } break;
            case (Graph::QueueAction::REMOVE_GRAPH_ENTRY): {
                (*input_lua_func)("mmRemoveGraphEntryPoint([=[" + data.name_id + "]=])");
            } break;
            default:
                break;
            }
        }

        if (!graph_sync_success) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Failed to synchronize gui graph with core graph. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
        }

        sync_success &= graph_sync_success;
    }

    // Propagate all changed parameter values from the GUI graph to the MegaMol graph
    if (graph_ptr != nullptr) {

        bool param_sync_success = true;
        for (auto& module_ptr : graph_ptr->Modules()) {
            for (auto& p : module_ptr->Parameters()) {

                // Try to connect gui parameter to newly created parameter of core modules
                if (p.CoreParamPtr() == nullptr) {
                    auto module_name = module_ptr->FullName();
                    megamol::core::Module* core_module_ptr = megamol_graph.FindModule(module_name).get();
                    // Connect pointer of new parameters of core module to parameters in gui module
                    if (core_module_ptr != nullptr) {
                        auto se = core_module_ptr->ChildList_End();
                        for (auto si = core_module_ptr->ChildList_Begin(); si != se; ++si) {
                            auto param_slot = dynamic_cast<megamol::core::param::ParamSlot*>((*si).get());
                            if (param_slot != nullptr) {
                                std::string param_full_name(param_slot->FullName().PeekBuffer());
                                for (auto& parameter : module_ptr->Parameters()) {
                                    if (gui_utils::CaseInsensitiveStringEqual(parameter.FullName(), param_full_name)) {
                                        megamol::gui::Parameter::ReadNewCoreParameterToExistingParameter(
                                            (*param_slot), parameter, true, false, true);
                                    }
                                }
                            }
                        }
                    }
#ifdef GUI_VERBOSE
                    if (p.CoreParamPtr() == nullptr) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] Unable to connect core parameter to gui parameter. [%s, %s, line %d]\n", __FILE__,
                            __FUNCTION__, __LINE__);
                    }
#endif // GUI_VERBOSE
                }

                if (p.CoreParamPtr() != nullptr) {
                    // Write changed gui state to core parameter
                    if (p.IsGUIStateDirty()) {
                        p.ResetGUIStateDirty(); // ! Reset before calling lua cmd because of instantly triggered subscription callback
                        // TODO what gets logged in the historian here?
                        param_sync_success &= megamol::gui::Parameter::WriteCoreParameterGUIState(p, p.CoreParamPtr());
                    }
                    // Write changed parameter value to core parameter
                    if (p.IsValueDirty()) {
                        p.ResetValueDirty(); // ! Reset before calling lua cmd because of instantly triggered subscription callback
                        param_sync_success &= std::get<0>((*input_lua_func)(
                            "mmSetParamValue([=[" + p.FullName() + "]=],[=[" + p.GetValueString() + "]=])"));
                    }
                }
            }
        }
#ifdef GUI_VERBOSE
        if (!param_sync_success) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] Failed to synchronize parameter values. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        }
#endif // GUI_VERBOSE
        sync_success &= param_sync_success;
    }

    return sync_success;
}


bool megamol::gui::GraphCollection::LoadOrAddProjectFromFile(
    ImGuiID in_graph_uid, const std::string& project_filename) {

    std::string loaded_project;
    if (!megamol::core::utility::FileUtils::ReadFile(std::filesystem::u8path(project_filename), loaded_project)) {
        return false;
    }

    const std::string luacmd_view("mmCreateView");
    const std::string luacmd_module("mmCreateModule");
    const std::string luacmd_param("mmSetParamValue");
    const std::string luacmd_call("mmCreateCall");

    GraphPtr_t graph_ptr = this->GetGraph(in_graph_uid);
    bool retval = (in_graph_uid != GUI_INVALID_ID);
    // Create new graph if necessary
    if (graph_ptr == nullptr) {
        ImGuiID new_graph_uid = this->AddGraph();
        if (new_graph_uid == GUI_INVALID_ID) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Load Project File '%s': Unable to create new graph. [%s, %s, line %d]\n",
                project_filename.c_str(), __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        graph_ptr = this->GetGraph(new_graph_uid);
        if (graph_ptr == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Unable to get pointer to last added graph. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
            return false;
        }
        retval = (new_graph_uid != GUI_INVALID_ID);
    }

    try {
        std::stringstream content(loaded_project);
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
                    retval = false;
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
                    retval = false;
                    continue;
                }

                /// DEBUG
                /// megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                ///     "[GUI] >>>> Instance: '%s' Class: '%s' NameSpace: '%s' Name: '%s' ConfPos: %f, %f.\n",
                ///     view_instance.c_str(), view_class_name.c_str(), view_namespace.c_str(), view_name.c_str());

                // First, rename existing modules with same name
                graph_ptr->UniqueModuleRename(view_full_name);
                // Add module and set as view instance
                auto graph_module =
                    graph_ptr->AddModule(this->modules_stock, view_class_name, view_name, view_namespace);
                if (graph_module == nullptr) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Load Project File '%s' line %i: Unable to add new module '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), view_class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    retval = false;
                    continue;
                }

                // Add new graph entry
                graph_ptr->AddGraphEntry(graph_module, view_instance);
            }
        }

        // Save filename for graph
        graph_ptr->SetFilename(project_filename, true);

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
                    retval = false;
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
                    retval = false;
                    continue;
                }

                /// DEBUG
                /// megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] >>>> Class: '%s' NameSpace: '%s' Name:
                /// '%s' ConfPos: %f, %f.\n",
                ///    module_class_name.c_str(), module_namespace.c_str(), module_name.c_str());


                // First, rename existing modules with same name
                graph_ptr->UniqueModuleRename(module_full_name);
                // Add module
                if (graph_ptr != nullptr) {
                    auto graph_module =
                        graph_ptr->AddModule(this->modules_stock, module_class_name, module_name, module_namespace);
                    if (graph_module == nullptr) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] Load Project File '%s' line %i: Unable to add new module '%s'. [%s, %s, line %d]\n",
                            project_filename.c_str(), (i + 1), module_class_name.c_str(), __FILE__, __FUNCTION__,
                            __LINE__);
                        retval = false;
                        continue;
                    }
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
                    retval = false;
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

                    for (auto& mod : graph_ptr->Modules()) {
                        module_full_name = mod->FullName() + "::";
                        // Caller
                        module_name_idx = caller_slot_full_name.find(module_full_name);
                        if (module_name_idx != std::string::npos) {
                            for (auto& callslot_map : mod->CallSlots()) {
                                for (auto& callslot : callslot_map.second) {
                                    if (gui_utils::CaseInsensitiveStringEqual(caller_slot_name, callslot->Name())) {
                                        caller_slot = callslot;
                                    }
                                }
                            }
                        }
                        // Callee
                        module_name_idx = callee_slot_full_name.find(module_full_name);
                        if (module_name_idx != std::string::npos) {
                            for (auto& callslot_map : mod->CallSlots()) {
                                for (auto& callslot : callslot_map.second) {
                                    if (gui_utils::CaseInsensitiveStringEqual(callee_slot_name, callslot->Name())) {
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
                        retval = false;
                        continue;
                    }
                    if (caller_slot == nullptr) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] Load Project File '%s' line %i: Unable to find caller slot '%s' "
                            "for creating call '%s'. [%s, %s, line %d]\n",
                            project_filename.c_str(), (i + 1), caller_slot_full_name.c_str(), call_class_name.c_str(),
                            __FILE__, __FUNCTION__, __LINE__);
                        retval = false;
                        continue;
                    }


                    // Add call
                    if (!graph_ptr->AddCall(this->calls_stock, caller_slot, callee_slot)) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] Load Project File '%s' line %i: Unable to add new call '%s'. [%s, %s, line %d]\n",
                            project_filename.c_str(), (i + 1), call_class_name.c_str(), __FILE__, __FUNCTION__,
                            __LINE__);
                        retval = false;
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
                    retval = false;
                    continue;
                }
                size_t first_delimiter_idx = param_line.find(',');
                if (first_delimiter_idx == std::string::npos) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Load Project File '%s' line %i: Missing argument delimiter ',' for '%s'. [%s, %s, line "
                        "%d]\n",
                        project_filename.c_str(), (i + 1), luacmd_param.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    retval = false;
                    continue;
                }

                std::string param_slot_full_name =
                    param_line.substr(first_bracket_idx + 1, (first_delimiter_idx - first_bracket_idx - 1));
                if ((param_slot_full_name.front() != '"') || (param_slot_full_name.back() != '"')) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Load Project File '%s' line %i: Parameter name argument should "
                        "be enclosed in '\"' for '%s'. [%s, %s, line %d]\n",
                        project_filename.c_str(), (i + 1), luacmd_param.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    retval = false;
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
                    retval = false;
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
                    retval = false;
                    continue;
                }
                std::string value_str = param_line.substr(value_start_idx + start_delimieter.size(),
                    (param_line.find(end_delimieter)) - value_start_idx - end_delimieter.size());

                /// DEBUG
                /// megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] >>>> '%s'\n", value_str.c_str());

                // Searching for parameter
                if (graph_ptr != nullptr) {
                    for (auto& module_ptr : graph_ptr->Modules()) {
                        for (auto& parameter : module_ptr->Parameters()) {
                            if (gui_utils::CaseInsensitiveStringEqual(parameter.FullName(), param_slot_full_name)) {
                                parameter.SetValueString(value_str);
                            }
                        }
                    }
                }
            }
        }

        // Load gui state from file
        if (!this->load_state_from_file(project_filename, graph_ptr->UID())) {
            // Layout graph if no positions for modules could be found in state.
            graph_ptr->SetLayoutGraph();
        }
        graph_ptr->ResetDirty();

        megamol::core::utility::log::Log::DefaultLog.WriteInfo(
            "[GUI] Successfully loaded project '%s'.\n", graph_ptr->Name().c_str());

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


bool megamol::gui::GraphCollection::SaveProjectToFile(ImGuiID in_graph_uid, const std::string& project_filename,
    const std::string& state_json, bool write_all_param_values) {

    /// Should be same as: megamol::core::MegaMolGraph_Convenience::SerializeGraph()
    try {
        for (auto& graph_ptr : this->graphs) {
            if (graph_ptr->UID() == in_graph_uid) {

                // Some pre-checks --------------------------------------------
                bool found_error = false;
                for (auto& mod_1 : graph_ptr->Modules()) {
                    for (auto& mod_2 : graph_ptr->Modules()) {
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
                std::stringstream confInstances, confModules, confCalls, confParams;
                for (auto& module_ptr : graph_ptr->Modules()) {
                    if (module_ptr->IsGraphEntry()) {
                        confInstances << "mmCreateView(\"" << module_ptr->GraphEntryName() << "\",\""
                                      << module_ptr->ClassName() << "\",\"" << module_ptr->FullName() << "\")\n";
                    } else {
                        confModules << "mmCreateModule(\"" << module_ptr->ClassName() << "\",\""
                                    << module_ptr->FullName() << "\")\n";
                    }

                    for (auto& parameter : module_ptr->Parameters()) {
                        // Either write_all_param_values or only write parameters with values deviating from the default
                        // Button parameters are always ignored
                        if ((write_all_param_values || parameter.DefaultValueMismatch()) &&
                            (parameter.Type() != ParamType_t::BUTTON)) {
                            confParams << "mmSetParamValue(\"" << parameter.FullName() << "\",[=["
                                       << parameter.GetValueString() << "]=])\n";
                        }
                    }

                    for (auto& caller_slot : module_ptr->CallSlots(CallSlotType::CALLER)) {
                        for (auto& call : caller_slot->GetConnectedCalls()) {
                            if (call->IsConnected()) {
                                confCalls << "mmCreateCall(\"" << call->ClassName() << "\",\""
                                          << call->CallSlotPtr(CallSlotType::CALLER)->GetParentModule()->FullName()
                                          << "::" << call->CallSlotPtr(CallSlotType::CALLER)->Name() << "\",\""
                                          << call->CallSlotPtr(CallSlotType::CALLEE)->GetParentModule()->FullName()
                                          << "::" << call->CallSlotPtr(CallSlotType::CALLEE)->Name() << "\")\n";
                            }
                        }
                    }
                }

                projectstr = std::string("mmCheckVersion(\"") + megamol::core::utility::buildinfo::MEGAMOL_GIT_HASH() +
                             "\")\n" + confInstances.str() + "\n" + confModules.str() + "\n" + confCalls.str() + "\n" +
                             confParams.str() + "\n" + state_json;

                graph_ptr->ResetDirty();
                if (megamol::core::utility::FileUtils::WriteFile(
                        std::filesystem::u8path(project_filename), projectstr)) {
                    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                        "[GUI] Successfully saved project '%s'.\n", graph_ptr->Name().c_str());

                    // Save filename for graph
                    graph_ptr->SetFilename(project_filename, true);

                    return true;
                } else {
                    return false;
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


bool megamol::gui::GraphCollection::get_module_stock_data(Module::StockModule& out_mod,
    std::shared_ptr<const megamol::core::factories::ModuleDescription> mod_desc, const std::string& plugin_name) {

    out_mod.class_name = std::string(mod_desc->ClassName());
    out_mod.description = std::string(mod_desc->Description());
    out_mod.is_view = false;
    out_mod.parameters.clear();
    out_mod.callslots.clear();
    out_mod.callslots.emplace(CallSlotType::CALLER, std::vector<CallSlot::StockCallSlot>());
    out_mod.callslots.emplace(CallSlotType::CALLEE, std::vector<CallSlot::StockCallSlot>());
    out_mod.plugin_name = plugin_name;

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

        megamol::core::Module::ptr_type new_mod(mod_desc->CreateModule(""));
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

        /// XXX Better check if module is a 'view':
        std::shared_ptr<const core::view::AbstractViewInterface> viewptr =
            std::dynamic_pointer_cast<const core::view::AbstractViewInterface>(new_mod);

        out_mod.is_view = (viewptr != nullptr);

        std::vector<std::shared_ptr<core::param::ParamSlot>> paramSlots;
        std::vector<std::shared_ptr<core::CallerSlot>> callerSlots;
        std::vector<std::shared_ptr<core::CalleeSlot>> calleeSlots;

        auto ano_end = new_mod->ChildList_End();
        for (auto ano_i = new_mod->ChildList_Begin(); ano_i != ano_end; ++ano_i) {
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
        for (auto& param_slot : paramSlots) {
            if (param_slot == nullptr)
                continue;
            Parameter::StockParameter psd;
            if (megamol::gui::Parameter::ReadNewCoreParameterToStockParameter((*param_slot), psd)) {
                out_mod.parameters.emplace_back(psd);
            }
        }

        // CallerSlots
        for (auto& caller_slot : callerSlots) {
            CallSlot::StockCallSlot csd;
            csd.name = std::string(caller_slot->Name().PeekBuffer());
            csd.description = std::string(caller_slot->Description().PeekBuffer());
            csd.compatible_call_idxs = this->get_compatible_caller_idxs(caller_slot.get());
            csd.type = CallSlotType::CALLER;
            csd.necessity = caller_slot->GetNecessity();

            out_mod.callslots[csd.type].emplace_back(csd);
        }

        // CalleeSlots
        for (auto& callee_slot : calleeSlots) {
            CallSlot::StockCallSlot csd;
            csd.name = std::string(callee_slot->Name().PeekBuffer());
            csd.description = std::string(callee_slot->Description().PeekBuffer());
            csd.compatible_call_idxs = this->get_compatible_callee_idxs(callee_slot.get());
            csd.type = CallSlotType::CALLEE;
            csd.necessity = callee_slot->GetNecessity();

            out_mod.callslots[csd.type].emplace_back(csd);
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


bool megamol::gui::GraphCollection::get_call_stock_data(Call::StockCall& out_call,
    std::shared_ptr<const megamol::core::factories::CallDescription> call_desc, const std::string& plugin_name) {

    try {
        out_call.class_name = std::string(call_desc->ClassName());
        out_call.description = std::string(call_desc->Description());
        out_call.functions.clear();
        for (unsigned int i = 0; i < call_desc->FunctionCount(); ++i) {
            out_call.functions.emplace_back(call_desc->FunctionName(i));
        }
        out_call.plugin_name = plugin_name;

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
    const std::string& line, size_t arg_count, std::vector<std::string>& out_args) const {

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


bool megamol::gui::GraphCollection::project_separate_name_and_namespace(
    const std::string& full_name, std::string& name_space, std::string& name) const {

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
        callNames.emplace_back(callee_slot->GetCallbackCallName(i));
        funcNames.emplace_back(callee_slot->GetCallbackFuncName(i));
    }
    size_t ll = callNames.size();
    assert(ll == funcNames.size());
    for (auto& callName : uniqueCallNames) {
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
    for (auto& callName : completeCallNames) {
        size_t calls_cnt = this->calls_stock.size();
        for (size_t idx = 0; idx < calls_cnt; ++idx) {
            // Case-Insensitive call slot comparison
            if (gui_utils::CaseInsensitiveStringEqual(this->calls_stock[idx].class_name, callName)) {
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
            if (gui_utils::CaseInsensitiveStringEqual(this->calls_stock[idx].class_name, comp_call_class_name)) {
                retval.emplace_back(idx);
            }
        }
    }

    return retval;
}


std::string megamol::gui::GraphCollection::get_state(ImGuiID graph_id, const std::string& filename) {

    nlohmann::json state_json;

    // Try to load existing gui state from file
    std::string loaded_state;
    if (megamol::core::utility::FileUtils::ReadFile(std::filesystem::u8path(filename), loaded_state)) {
        loaded_state =
            gui_utils::ExtractTaggedString(loaded_state, GUI_START_TAG_SET_GUI_STATE, GUI_END_TAG_SET_GUI_STATE);
        if (!loaded_state.empty()) {
            state_json = nlohmann::json::parse(loaded_state);
            if (!state_json.is_object()) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Invalid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                return std::string("");
            }
        }
    }

    if (auto graph_ptr = this->GetGraph(graph_id)) {
        // Write/replace GUI_JSON_TAG_PROJECT graph state
        try {
            state_json[GUI_JSON_TAG_GRAPHS].erase(GUI_JSON_TAG_PROJECT);
        } catch (...) {}
        graph_ptr->StateToJSON(state_json);

        // Write/replace GUI state of parameters (groups)
        try {
            state_json.erase(GUI_JSON_TAG_GUISTATE_PARAMETERS);
        } catch (...) {}
        for (auto& module_ptr : graph_ptr->Modules()) {
            module_ptr->StateToJSON(state_json);
        }
        loaded_state = state_json.dump(); // No line feed

        loaded_state =
            std::string(GUI_START_TAG_SET_GUI_STATE) + loaded_state + std::string(GUI_END_TAG_SET_GUI_STATE) + "\n";

        return loaded_state;
    }
    return std::string("");
}


bool megamol::gui::GraphCollection::load_state_from_file(const std::string& filename, ImGuiID graph_id) {

    std::string loaded_state;
    if (megamol::core::utility::FileUtils::ReadFile(std::filesystem::u8path(filename), loaded_state)) {
        loaded_state =
            gui_utils::ExtractTaggedString(loaded_state, GUI_START_TAG_SET_GUI_STATE, GUI_END_TAG_SET_GUI_STATE);
        if (loaded_state.empty())
            return false;
        nlohmann::json json;
        json = nlohmann::json::parse(loaded_state);
        if (!json.is_object()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Invalid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        if (auto graph_ptr = this->GetGraph(graph_id)) {

            // Read GUI state of parameters (groups)
            for (auto& module_ptr : graph_ptr->Modules()) {
                module_ptr->StateFromJSON(json);
            }

            // Read GUI_JSON_TAG_PROJECT graph state
            if (graph_ptr->StateFromJSON(json)) {
                // Disable layouting if graph state was found
                graph_ptr->SetLayoutGraph(false);
            }

            return true;
        }
    }
    return false;
}


void megamol::gui::GraphCollection::Draw(GraphState_t& state) {

    try {
        if (ImGui::GetCurrentContext() == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return;
        }

        ImGui::BeginChild("graph_tab_indow", ImVec2(state.graph_width, 0.0f), false,
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar);

        // Assuming only one closed tab/graph per frame.
        bool popup_close_unsaved = false;

        // Draw Graphs --------------------------------------------------------
        ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_Reorderable;
        ImGui::BeginTabBar("Graphs", tab_bar_flags);

        for (auto& graph_ptr : this->GetGraphs()) {

            graph_ptr->Draw(state);

            // Do not delete graph while looping through graphs list
            if (state.graph_delete) {
                this->gui_graph_delete_uid = state.graph_selected_uid;
                if (graph_ptr->IsDirty()) {
                    popup_close_unsaved = true;
                }
                state.graph_delete = false;
            }

            // Catch call drop event and create new call
            auto drag_drop_uids = graph_ptr->ConsumeDragAndDropSlots();
            if ((drag_drop_uids.first != GUI_INVALID_ID) && (drag_drop_uids.second != GUI_INVALID_ID)) {
                graph_ptr->AddCall(this->GetCallsStock(), drag_drop_uids.first, drag_drop_uids.second);
            }
        }
        ImGui::EndTabBar();

        // Process changed running graph --------------------------------------
        this->change_running_graph(state.new_running_graph_uid);
        state.new_running_graph_uid = GUI_INVALID_ID;

        // Save selected graph in configurator --------------------------------
        this->save_graph_dialog(state.graph_selected_uid, state.configurator_graph_save);

        // Delete selected graph when tab is closed and unsaved changes should be discarded
        bool confirmed = false;
        bool aborted = false;
        bool popup_open = PopUps::Minimal(
            "Closing unsaved Project", popup_close_unsaved, "Discard changes?", "Yes", confirmed, "No", aborted);
        if (this->gui_graph_delete_uid != GUI_INVALID_ID) {
            if (aborted) {
                this->gui_graph_delete_uid = GUI_INVALID_ID;
            } else if (confirmed || !popup_open) {
                this->DeleteGraph(this->gui_graph_delete_uid);
                this->gui_graph_delete_uid = GUI_INVALID_ID;
                state.graph_selected_uid = GUI_INVALID_ID;
            }
        }

        ImGui::EndChild();

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

#ifdef MEGAMOL_USE_PROFILING

void megamol::gui::GraphCollection::AppendPerformanceData(
    const frontend_resources::PerformanceManager::frame_info& fi) {
    auto frame = fi.frame;
    for (auto& e : fi.entries) {
        auto p = perf_manager->lookup_parent_pointer(e.handle);
        auto t = perf_manager->lookup_parent_type(e.handle);
        if (t == frontend_resources::PerformanceManager::parent_type::CALL) {
            auto c = static_cast<megamol::core::Call*>(p);
            // printf("looking up call map for @ %p = %s \n", c, c->GetDescriptiveText().c_str());
            if (call_to_call[p].lock() != nullptr) { // XXX Consider delayed clean-up
                call_to_call[p].lock()->AppendPerformanceData(frame, e);
            }
        } else if (t == frontend_resources::PerformanceManager::parent_type::USER_REGION) {
            // Region in a Module
            if (module_to_module[p].lock() != nullptr) { // XXX Consider delayed clean-up
                module_to_module[p].lock()->AppendPerformanceData(frame, e);
            }
        }
    }
}

#endif


bool megamol::gui::GraphCollection::NotifyRunningGraph_AddModule(core::ModuleInstance_t const& module_inst) {

    if (!this->initialized_syncing) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Graph synchronization not initialized. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    if (auto graph_ptr = this->GetRunningGraph()) {

        std::string full_name(module_inst.request.id);

        if (graph_ptr->ModuleExists(full_name)) {
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Module already exists: '%s' [%s, %s, line %d]\n", full_name.c_str(), __FILE__, __FUNCTION__,
                __LINE__);
#endif // GUI_VERBOSE
            /// Error tolerance to ignore redundant changes that have been triggered by the GUI
            return true;
        }

        std::string class_name(module_inst.modulePtr->ClassName());
        std::string module_name;
        std::string module_namespace;
        if (!this->project_separate_name_and_namespace(full_name, module_namespace, module_name)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Invalid module name: '%s' [%s, %s, line %d]\n", full_name.c_str(), __FILE__, __FUNCTION__,
                __LINE__);
            return false;
        }

        ImGuiID moduel_uid = GUI_INVALID_ID;
        ModulePtr_t gui_module_ptr;
        /// XXX ModuleDescriptionManager is only available via core instance graph yet:
        std::string module_description = "[n/a]";
        std::string module_plugin = "[n/a]";
        /// XXX Better check if module is a 'view':
        auto viewptr = dynamic_cast<core::view::AbstractViewInterface*>(module_inst.modulePtr.get());
        bool is_view = (viewptr != nullptr);

        if (auto gui_module_ptr = graph_ptr->AddModule(
                class_name, module_name, module_namespace, module_description, module_plugin, is_view)) {

            // Set remaining module data
            if (module_inst.isGraphEntryPoint) {
                graph_ptr->AddGraphEntry(gui_module_ptr, graph_ptr->GenerateUniqueGraphEntryName(), false);
            }

#ifdef MEGAMOL_USE_PROFILING
            // TODO set some stuff here so I can find which regions are which!?
            gui_module_ptr->SetProfilingData(module_inst.modulePtr.get(), perf_manager);
            module_to_module[module_inst.modulePtr.get()] = gui_module_ptr;
#endif

            auto se = module_inst.modulePtr->ChildList_End();
            for (auto si = module_inst.modulePtr->ChildList_Begin(); si != se; ++si) {

                // Add parameters
                auto param_slot = dynamic_cast<megamol::core::param::ParamSlot*>((*si).get());
                if (param_slot != nullptr) {
                    std::string param_full_name(param_slot->FullName().PeekBuffer());
                    std::shared_ptr<Parameter> param_ptr;
                    // This is the default value of the parameter since changed values are propagated separately via parameter subscription
                    megamol::gui::Parameter::ReadNewCoreParameterToNewParameter(
                        (*param_slot), param_ptr, true, false, true, gui_module_ptr->FullName());
                    gui_module_ptr->Parameters().emplace_back((*param_ptr));
                }

                // Add call slots
                std::shared_ptr<core::CallerSlot> caller_slot =
                    std::dynamic_pointer_cast<megamol::core::CallerSlot>((*si));
                if (caller_slot) {
                    auto callslot_ptr = std::make_shared<CallSlot>(megamol::gui::GenerateUniqueID(),
                        std::string(caller_slot->Name().PeekBuffer()),
                        std::string(caller_slot->Description().PeekBuffer()),
                        this->get_compatible_caller_idxs(caller_slot.get()), CallSlotType::CALLER,
                        caller_slot->GetNecessity());
                    callslot_ptr->ConnectParentModule(gui_module_ptr);
                    gui_module_ptr->AddCallSlot(callslot_ptr);
                }
                std::shared_ptr<core::CalleeSlot> callee_slot =
                    std::dynamic_pointer_cast<megamol::core::CalleeSlot>((*si));
                if (callee_slot) {
                    auto callslot_ptr = std::make_shared<CallSlot>(megamol::gui::GenerateUniqueID(),
                        std::string(callee_slot->Name().PeekBuffer()),
                        std::string(callee_slot->Description().PeekBuffer()),
                        this->get_compatible_callee_idxs(callee_slot.get()), CallSlotType::CALLEE,
                        callee_slot->GetNecessity());
                    callslot_ptr->ConnectParentModule(gui_module_ptr);
                    gui_module_ptr->AddCallSlot(callslot_ptr);
                }
            }

            return true;
        }

        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to create module: '%s' [%s, %s, line %d]\n", full_name.c_str(), __FILE__, __FUNCTION__,
            __LINE__);
        return false;
    }
    return false;
}


bool megamol::gui::GraphCollection::NotifyRunningGraph_DeleteModule(core::ModuleInstance_t const& module_inst) {

    if (!this->initialized_syncing) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Graph synchronization not initialized. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    if (auto graph_ptr = this->GetRunningGraph()) {

        for (auto& module_ptr : graph_ptr->Modules()) {
            if (module_ptr->FullName() == module_inst.request.id) {
#ifdef MEGAMOL_USE_PROFILING
                this->module_to_module.erase(module_ptr->GetProfilingParent());
#endif
                bool success = graph_ptr->DeleteModule(module_ptr->UID(), false);
                return success;
            }
        }

#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Could not find module for deletion: '%s' [%s, %s, line %d]\n", module_inst.request.id.c_str(),
            __FILE__, __FUNCTION__, __LINE__);
#endif // GUI_VERBOSE
        /// Error tolerance to ignore redundant changes that have been triggered by the GUI
        return true;
    }
    return false;
}


bool megamol::gui::GraphCollection::NotifyRunningGraph_RenameModule(
    std::string const& old_name, std::string const& new_name, core::ModuleInstance_t const& module_inst) {

    if (!this->initialized_syncing) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Graph synchronization not initialized. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    if (auto graph_ptr = this->GetRunningGraph()) {

        if (graph_ptr->ModuleExists(new_name)) {
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Module already exists: '%s' [%s, %s, line %d]\n", module_inst.request.id.c_str(), __FILE__,
                __FUNCTION__, __LINE__);
#endif // GUI_VERBOSE
            /// Error tolerance to ignore redundant changes that have been triggered by the GUI
            return true;
        }

        for (auto& module_ptr : graph_ptr->Modules()) {
            if (module_ptr->FullName() == old_name) {
                module_ptr->SetName(new_name);
                module_ptr->Update();
                return true;
            }
        }

#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to find module for renaming: '%s' [%s, %s, line %d]\n", module_inst.request.id.c_str(),
            __FILE__, __FUNCTION__, __LINE__);
#endif // GUI_VERBOSE
        /// Error tolerance to ignore redundant changes that have been triggered by the GUI
        return true;
    }
    return false;
}


bool megamol::gui::GraphCollection::NotifyRunningGraph_AddParameters(
    std::vector<megamol::frontend_resources::ModuleGraphSubscription::ParamSlotPtr> const& param_slots) {

    /// XXX Unused
    return true;
}


bool megamol::gui::GraphCollection::NotifyRunningGraph_RemoveParameters(
    std::vector<megamol::frontend_resources::ModuleGraphSubscription::ParamSlotPtr> const& param_slots) {

    /// XXX Unused
    return true;
}


bool megamol::gui::GraphCollection::NotifyRunningGraph_ParameterChanged(
    megamol::frontend_resources::ModuleGraphSubscription::ParamSlotPtr const& param_slot,
    std::string const& new_value) {

    if (!this->initialized_syncing) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Graph synchronization not initialized. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    if (auto graph_ptr = this->GetRunningGraph()) {

        for (auto& module_ptr : graph_ptr->Modules()) {
            for (auto& p : module_ptr->Parameters()) {
                if (param_slot->Parameter() == p.CoreParamPtr()) {
                    bool success =
                        megamol::gui::Parameter::ReadCoreParameterToParameter(param_slot->Parameter(), p, false, false);
                    return success;
                }
            }
        }

        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Could not find parameter: '%s' [%s, %s, line %d]\n", param_slot->FullName().PeekBuffer(), __FILE__,
            __FUNCTION__, __LINE__);
    }
    return false;
}


bool megamol::gui::GraphCollection::NotifyRunningGraph_AddCall(core::CallInstance_t const& call_inst) {

    if (!this->initialized_syncing) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Graph synchronization not initialized. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    if (auto graph_ptr = this->GetRunningGraph()) {

        if (graph_ptr->CallExists(call_inst.request.className, call_inst.request.from, call_inst.request.to)) {
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Call already exists: '%s' [%s, %s, line %d]\n", call_inst.callPtr->ClassName(), __FILE__,
                __FUNCTION__, __LINE__);
#endif // GUI_VERBOSE
            /// Error tolerance to ignore redundant changes that have been triggered by the GUI
            return true;
        }

        std::string call_caller_name;
        std::string call_caller_parent_name;
        if (!this->project_separate_name_and_namespace(
                call_inst.request.from, call_caller_parent_name, call_caller_name)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Core Project: Invalid call slot name '%s'. [%s, %s, line %d]\n", call_inst.request.from.c_str(),
                __FILE__, __FUNCTION__, __LINE__);
        }
        std::string call_callee_name;
        std::string call_callee_parent_name;
        if (!this->project_separate_name_and_namespace(
                call_inst.request.to, call_callee_parent_name, call_callee_name)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Core Project: Invalid call slot name '%s'. [%s, %s, line %d]\n", call_inst.request.to.c_str(),
                __FILE__, __FUNCTION__, __LINE__);
        }
        call_callee_parent_name = "::" + call_callee_parent_name;
        call_caller_parent_name = "::" + call_caller_parent_name;

        CallSlotPtr_t callslot_1 = nullptr;
        for (auto& mod : graph_ptr->Modules()) {
            if (gui_utils::CaseInsensitiveStringEqual(mod->FullName(), call_caller_parent_name)) {
                for (auto& callslot : mod->CallSlots(CallSlotType::CALLER)) {
                    if (gui_utils::CaseInsensitiveStringEqual(callslot->Name(), call_caller_name)) {
                        callslot_1 = callslot;
                    }
                }
            }
        }
        CallSlotPtr_t callslot_2 = nullptr;
        for (auto& mod : graph_ptr->Modules()) {
            if (gui_utils::CaseInsensitiveStringEqual(mod->FullName(), call_callee_parent_name)) {
                for (auto& callslot : mod->CallSlots(CallSlotType::CALLEE)) {
                    if (gui_utils::CaseInsensitiveStringEqual(callslot->Name(), call_callee_name)) {
                        callslot_2 = callslot;
                    }
                }
            }
        }

        if (auto gui_call_ptr = graph_ptr->AddCall(this->GetCallsStock(), callslot_1, callslot_2, false)) {
            gui_call_ptr->SetCapabilities(call_inst.callPtr->GetCapabilities());
#ifdef MEGAMOL_USE_PROFILING
            gui_call_ptr->SetProfilingData(call_inst.callPtr.get(), call_inst.callPtr->GetCallbackCount());
            // printf("setting map for @ %p = %s \n", reinterpret_cast<void*>(cd.core_call.get()),
            //    cd.core_call.get()->GetDescriptiveText().c_str());
            this->call_to_call[call_inst.callPtr.get()] = gui_call_ptr;
#endif
            return true;
        }

        megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Unable to create call: '%s' [%s, %s, line %d]\n",
            call_inst.request.className.c_str(), __FILE__, __FUNCTION__, __LINE__);
    }
    return false;
}


bool megamol::gui::GraphCollection::NotifyRunningGraph_DeleteCall(core::CallInstance_t const& call_inst) {

    if (!this->initialized_syncing) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Graph synchronization not initialized. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    if (auto graph_ptr = this->GetRunningGraph()) {

        for (auto& call_ptr : graph_ptr->Calls()) {
            std::string class_name = call_ptr->ClassName();
            std::string from;
            auto caller_ptr = call_ptr->CallSlotPtr(megamol::gui::CallSlotType::CALLER);
            if (caller_ptr != nullptr) {
                if (caller_ptr->GetParentModule() != nullptr) {
                    from = caller_ptr->GetParentModule()->FullName() + "::" + caller_ptr->Name();
                }
            }
            std::string to;
            auto callee_ptr = call_ptr->CallSlotPtr(megamol::gui::CallSlotType::CALLEE);
            if (callee_ptr != nullptr) {
                if (callee_ptr->GetParentModule() != nullptr) {
                    to = callee_ptr->GetParentModule()->FullName() + "::" + callee_ptr->Name();
                }
            }
            if ((class_name == call_inst.request.className) &&
                gui_utils::CaseInsensitiveStringEqual(from, call_inst.request.from) &&
                gui_utils::CaseInsensitiveStringEqual(to, call_inst.request.to)) {
#ifdef MEGAMOL_USE_PROFILING
                this->call_to_call.erase(call_ptr->GetProfilingParent());
#endif
                bool success = graph_ptr->DeleteCall(call_ptr->UID(), false);
                return success;
            }
        }

#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Unable to find call: '%s' [%s, %s, line %d]\n",
            call_inst.request.className.c_str(), __FILE__, __FUNCTION__, __LINE__);
#endif // GUI_VERBOSE
        /// Error tolerance to ignore redundant changes that have been triggered by the GUI
        return true;
    }
    return false;
}


bool megamol::gui::GraphCollection::NotifyRunningGraph_EnableEntryPoint(core::ModuleInstance_t const& module_inst) {

    if (!this->initialized_syncing) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Graph synchronization not initialized. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    if (auto graph_ptr = this->GetRunningGraph()) {
        if (auto mod_ptr = graph_ptr->GetModule(module_inst.request.id)) {
            return graph_ptr->AddGraphEntry(mod_ptr, graph_ptr->GenerateUniqueGraphEntryName(), false);
        }
    }

    return false;
}


bool megamol::gui::GraphCollection::NotifyRunningGraph_DisableEntryPoint(core::ModuleInstance_t const& module_inst) {

    if (!this->initialized_syncing) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Graph synchronization not initialized. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    if (auto graph_ptr = this->GetRunningGraph()) {
        if (auto mod_ptr = graph_ptr->GetModule(module_inst.request.id)) {
            return graph_ptr->RemoveGraphEntry(mod_ptr, false);
        } else {
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Unable to find module: '%s' [%s, %s, line %d]\n", module_inst.request.id.c_str(), __FILE__,
                __FUNCTION__, __LINE__);
#endif // GUI_VERBOSE
            /// Error tolerance to ignore redundant changes that have been triggered by the GUI
            return true;
        }
    }

    return false;
}


bool megamol::gui::GraphCollection::save_graph_dialog(ImGuiID graph_uid, bool& open_dialog) {

    bool confirmed, aborted;
    bool popup_failed = false;
    std::string project_filename;
    if (open_dialog) {
        if (auto graph_ptr = this->GetGraph(graph_uid)) {
            project_filename = graph_ptr->GetFilename();
        }
    }
    // Default for saving gui state and parameter values
    bool save_all_param_values = true;
    bool save_gui_state = false;
    if (this->gui_file_browser.PopUp_Save("Save Configurator Project", project_filename, open_dialog, {"lua"},
            megamol::core::param::FilePathParam::Flag_File_ToBeCreatedWithRestrExts, save_gui_state,
            save_all_param_values)) {

        std::string gui_state;
        if (save_gui_state) {
            gui_state = this->get_state(graph_uid, project_filename);
        }

        popup_failed = !this->SaveProjectToFile(graph_uid, project_filename, gui_state, save_all_param_values);
    }
    PopUps::Minimal("Failed to Save Project", popup_failed, "See console log output for more information.", "Cancel");

    return !popup_failed;
}


bool megamol::gui::GraphCollection::change_running_graph(ImGuiID graph_uid) {

    if (graph_uid != GUI_INVALID_ID) {
        /// There should always be only one running graph at a time

        // Get currently running graph
        GraphPtr_t last_running_graph = nullptr;
        for (auto& graph_ptr : this->GetGraphs()) {
            if (graph_ptr->IsRunning()) {
                last_running_graph = graph_ptr;
            }
            graph_ptr->SetRunning(false);
        }
        if (last_running_graph != nullptr) {
            if (auto running_graph = this->GetGraph(graph_uid)) {

                // 1] Set new graph running to enable queue
                running_graph->SetRunning(true);

                // 2] Remove all calls and modules from core graph, but keep stopped GUI graph in project untouched
                for (auto& call_ptr : last_running_graph->Calls()) {
                    Graph::QueueData queue_data;
                    auto caller_ptr = call_ptr->CallSlotPtr(megamol::gui::CallSlotType::CALLER);
                    if (caller_ptr != nullptr) {
                        if (caller_ptr->GetParentModule() != nullptr) {
                            queue_data.caller = caller_ptr->GetParentModule()->FullName() + "::" + caller_ptr->Name();
                        }
                    }
                    auto callee_ptr = call_ptr->CallSlotPtr(megamol::gui::CallSlotType::CALLEE);
                    if (callee_ptr != nullptr) {
                        if (callee_ptr->GetParentModule() != nullptr) {
                            queue_data.callee = callee_ptr->GetParentModule()->FullName() + "::" + callee_ptr->Name();
                        }
                    }
                    running_graph->PushSyncQueue(Graph::QueueAction::DELETE_CALL, queue_data);
                }
                for (auto& module_ptr : last_running_graph->Modules()) {
                    Graph::QueueData queue_data;
                    queue_data.name_id = module_ptr->FullName();
                    if (module_ptr->IsGraphEntry()) {
                        running_graph->PushSyncQueue(Graph::QueueAction::REMOVE_GRAPH_ENTRY, queue_data);
                    }
                    if (!running_graph->ModuleExists(module_ptr->FullName())) {
                        // Do not delete module if module with same full name exists in new running graph (prevent double deletion and re-creation of module)
                        running_graph->PushSyncQueue(Graph::QueueAction::DELETE_MODULE, queue_data);
                    }
                    // Reset pointers to core parameters
                    for (auto& param : module_ptr->Parameters()) {
                        param.ResetCoreParamPtr();
                    }
                }

                // 3] Create new modules and calls in core graph, but keep newly running GUI graph in project
                // untouched
                for (auto& module_ptr : running_graph->Modules()) {

                    Graph::QueueData queue_data;
                    queue_data.name_id = module_ptr->FullName();
                    queue_data.class_name = module_ptr->ClassName();
                    running_graph->PushSyncQueue(Graph::QueueAction::ADD_MODULE, queue_data);
                    if (module_ptr->IsGraphEntry()) {
                        running_graph->PushSyncQueue(Graph::QueueAction::CREATE_GRAPH_ENTRY, queue_data);
                    }
                    // Set all parameters in GUI graph dirty in order to propagate current values to new core
                    // modules
                    for (auto& param : module_ptr->Parameters()) {
                        if (param.Type() != ParamType_t::BUTTON) {
                            param.ForceSetValueDirty();
                            param.ForceSetGUIStateDirty();
                        }
                    }
                }
                for (auto& call_ptr : running_graph->Calls()) {
                    Graph::QueueData queue_data;
                    queue_data.class_name = call_ptr->ClassName();
                    auto caller_ptr = call_ptr->CallSlotPtr(megamol::gui::CallSlotType::CALLER);
                    if (caller_ptr != nullptr) {
                        if (caller_ptr->GetParentModule() != nullptr) {
                            queue_data.caller = caller_ptr->GetParentModule()->FullName() + "::" + caller_ptr->Name();
                        }
                    }
                    auto callee_ptr = call_ptr->CallSlotPtr(megamol::gui::CallSlotType::CALLEE);
                    if (callee_ptr != nullptr) {
                        if (callee_ptr->GetParentModule() != nullptr) {
                            queue_data.callee = callee_ptr->GetParentModule()->FullName() + "::" + callee_ptr->Name();
                        }
                    }
                    running_graph->PushSyncQueue(Graph::QueueAction::ADD_CALL, queue_data);
                }
                return true;
            }
        }
    }

    return false;
}
