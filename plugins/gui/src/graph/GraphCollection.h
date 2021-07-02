/*
 * GraphCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_GRAPHCOLLECTION_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_GRAPHCOLLECTION_H_INCLUDED
#pragma once


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

        bool LoadModuleStock(const megamol::core::CoreInstance& core_instance);
        bool LoadCallStock(const megamol::core::CoreInstance& core_instance);
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

        /**
         * Load or update project from graph of core instance or directly from megamol graph.
         *
         * @param inout_graph_uid  The graph uid to use. If graph uid is GUI_INVALID_ID a new graph is created.
         * @param megamol_graph    The megamol graph.
         *
         * @return                 True on success, false otherwise.
         */
        bool LoadUpdateProjectFromCore(ImGuiID& inout_graph_uid, megamol::core::MegaMolGraph& megamol_graph);

        bool LoadAddProjectFromFile(ImGuiID in_graph_uid, const std::string& project_filename);

        bool SaveProjectToFile(
            ImGuiID in_graph_uid, const std::string& project_filename, const std::string& state_json);

        void Draw(GraphState_t& state);

        void RequestNewRunningGraph(ImGuiID graph_uid) {
            this->change_running_graph(graph_uid);
        }

    private:
        // VARIABLES --------------------------------------------------------------

        GraphPtrVector_t graphs;
        ModuleStockVector_t modules_stock;
        CallStockVector_t calls_stock;
        unsigned int graph_name_uid;

        FileBrowserWidget gui_file_browser;
        ImGuiID gui_graph_delete_uid;

        // FUNCTIONS --------------------------------------------------------------

        bool add_update_project_from_core(
            ImGuiID in_graph_uid, megamol::core::MegaMolGraph& megamol_graph, bool use_stock);

        std::string get_state(ImGuiID graph_id, const std::string& filename);

        bool get_call_stock_data(
            Call::StockCall& call, std::shared_ptr<const megamol::core::factories::CallDescription> call_desc);
        bool get_module_stock_data(
            Module::StockModule& mod, std::shared_ptr<const megamol::core::factories::ModuleDescription> mod_desc);

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
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GRAPHCOLLECTION_H_INCLUDED
