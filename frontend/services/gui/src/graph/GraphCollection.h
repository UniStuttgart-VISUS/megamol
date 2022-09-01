/*
 * GraphCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_GRAPHCOLLECTION_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_GRAPHCOLLECTION_H_INCLUDED
#pragma once


#include "CommonTypes.h"
#include "Graph.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/MegaMolGraph.h"
#include "mmcore/Module.h"
#include "mmcore/param/AbstractParam.h"
#include "widgets/FileBrowserWidget.h"


namespace megamol {
namespace gui {


// Forward declarations
class GraphCollection;

// Types
typedef std::shared_ptr<Graph> GraphPtr_t;
typedef std::vector<GraphPtr_t> GraphPtrVector_t;


/** ************************************************************************
 * Defines the graph collection
 */
class GraphCollection {
public:
    using lua_func_type = megamol::frontend_resources::common_types::lua_func_type;

    GraphCollection();
    ~GraphCollection() = default;

    bool AddEmptyProject();

    ImGuiID AddGraph();
    bool DeleteGraph(ImGuiID in_graph_uid);
    GraphPtr_t GetGraph(ImGuiID in_graph_uid);
    const GraphPtrVector_t& GetGraphs() {
        return this->graphs;
    }
    GraphPtr_t GetRunningGraph();

    inline const ModuleStockVector_t& GetModulesStock() {
        return this->modules_stock;
    }
    inline const CallStockVector_t& GetCallsStock() {
        return this->calls_stock;
    }
    bool IsCallStockLoaded() const {
        return (!this->calls_stock.empty());
    }
    bool IsModuleStockLoaded() const {
        return (!this->modules_stock.empty());
    }

    void SetLuaFunc(lua_func_type* func);

    // ! Has to be called once before calling SynchronizeGraphs() or NotifyRunningGraph_*()
    bool InitializeGraphSynchronisation(const megamol::core::CoreInstance& core_instance);

    bool SynchronizeGraphs(megamol::core::MegaMolGraph& megamol_graph, megamol::core::CoreInstance& core_instance);

    bool LoadOrAddProjectFromFile(ImGuiID in_graph_uid, const std::string& project_filename);

    bool SaveProjectToFile(ImGuiID in_graph_uid, const std::string& project_filename, const std::string& state_json);

    void Draw(GraphState_t& state);

    void RequestNewRunningGraph(ImGuiID graph_uid) {
        this->change_running_graph(graph_uid);
    }

#ifdef MEGAMOL_USE_PROFILING
    void SetPerformanceManager(frontend_resources::PerformanceManager* perf_manager) {
        this->perf_manager = perf_manager;
    }
    void AppendPerformanceData(const frontend_resources::PerformanceManager::frame_info& fi);
#endif

    bool NotifyRunningGraph_AddModule(core::ModuleInstance_t const& module_inst);
    bool NotifyRunningGraph_DeleteModule(core::ModuleInstance_t const& module_inst);
    bool NotifyRunningGraph_RenameModule(
        std::string const& old_name, std::string const& new_name, core::ModuleInstance_t const& module_inst);
    bool NotifyRunningGraph_AddParameters(
        std::vector<megamol::frontend_resources::ModuleGraphSubscription::ParamSlotPtr> const& param_slots);
    bool NotifyRunningGraph_RemoveParameters(
        std::vector<megamol::frontend_resources::ModuleGraphSubscription::ParamSlotPtr> const& param_slots);
    bool NotifyRunningGraph_ParameterChanged(
        megamol::frontend_resources::ModuleGraphSubscription::ParamSlotPtr const& param_slot,
        std::string const& old_value, std::string const& new_value);
    bool NotifyRunningGraph_AddCall(core::CallInstance_t const& call_inst);
    bool NotifyRunningGraph_DeleteCall(core::CallInstance_t const& call_inst);

private:
    // VARIABLES --------------------------------------------------------------

    GraphPtrVector_t graphs;
    ModuleStockVector_t modules_stock;
    CallStockVector_t calls_stock;
    unsigned int graph_name_uid;

    FileBrowserWidget gui_file_browser;
    ImGuiID gui_graph_delete_uid;

    lua_func_type* input_lua_func = nullptr;

    bool created_running_graph;
    bool initialized_syncing;

    // FUNCTIONS --------------------------------------------------------------

    bool load_module_stock(const megamol::core::CoreInstance& core_instance);
    bool load_call_stock(const megamol::core::CoreInstance& core_instance);

    std::string get_state(ImGuiID graph_id, const std::string& filename);

    bool get_call_stock_data(Call::StockCall& out_call,
        std::shared_ptr<const megamol::core::factories::CallDescription> call_desc, const std::string& plugin_name);
    bool get_module_stock_data(Module::StockModule& out_mod,
        std::shared_ptr<const megamol::core::factories::ModuleDescription> mod_desc, const std::string& plugin_name);

    bool read_project_command_arguments(
        const std::string& line, size_t arg_count, std::vector<std::string>& out_args) const;

    bool project_separate_name_and_namespace(
        const std::string& full_name, std::string& name_space, std::string& name) const;

    inline std::string generate_unique_graph_name() {
        return ("Project_" + std::to_string(++graph_name_uid));
    }

    std::vector<size_t> get_compatible_callee_idxs(const megamol::core::CalleeSlot* callee_slot);
    std::vector<size_t> get_compatible_caller_idxs(const megamol::core::CallerSlot* caller_slot);

    bool load_state_from_file(const std::string& filename, ImGuiID graph_id);

    bool save_graph_dialog(ImGuiID graph_uid, bool& open_dialog);

    bool change_running_graph(ImGuiID graph_uid);

#ifdef MEGAMOL_USE_PROFILING
    std::unordered_map<void*, std::weak_ptr<megamol::gui::Call>> call_to_call;
    std::unordered_map<void*, std::weak_ptr<megamol::gui::Module>> module_to_module;
    frontend_resources::PerformanceManager* perf_manager = nullptr;
#endif
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GRAPHCOLLECTION_H_INCLUDED
