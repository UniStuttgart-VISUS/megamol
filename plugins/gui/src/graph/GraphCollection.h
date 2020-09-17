/*
 * GraphCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_GRAPHCOLLECTION_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_GRAPHCOLLECTION_H_INCLUDED


#include "GraphCollectionPresentation.h"

#include "Graph.h"

#include "mmcore/CoreInstance.h"
///#include "mmcore/MegaMolGraph.h"
#include "mmcore/Module.h"
#include "mmcore/param/AbstractParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/plugins/AbstractPluginInstance.h"

#include "utility/plugins/PluginManager.h"

/// TEMP
namespace megamol {
namespace core {
class MegaMolGraph;
}
} // namespace megamol
///


namespace megamol {
namespace gui {


// Forward declarations
class GraphCollection;

// Types
typedef std::shared_ptr<GraphCollection> GraphCollectionhPtr_t;
typedef std::shared_ptr<Graph> GraphPtr_t;
typedef std::vector<GraphPtr_t> GraphPtrVector_t;
typedef std::map<megamol::core::param::AbstractParam*, std::shared_ptr<megamol::gui::Parameter>> ParamInterfaceMap_t;


/** ************************************************************************
 * Defines the graph collection.
 */
class GraphCollection {
public:
    // VARIABLES --------------------------------------------------------------

    GraphCollectionPresentation present;

    // FUNCTIONS --------------------------------------------------------------

    GraphCollection(void);
    ~GraphCollection(void);

    ImGuiID AddGraph(void);
    bool DeleteGraph(ImGuiID in_graph_uid);
    bool GetGraph(ImGuiID in_graph_uid, GraphPtr_t& out_graph_ptr);
    const GraphPtrVector_t& GetGraphs(void) { return this->graphs; }

    bool LoadModuleStock(const megamol::core::CoreInstance* core_instance);
    bool LoadCallStock(const megamol::core::CoreInstance* core_instance);
    inline const ModuleStockVector_t& GetModulesStock(void) { return this->modules_stock; }
    inline const CallStockVector_t& GetCallsStock(void) { return this->calls_stock; }

    bool LoadUpdateProjectFromCore(ImGuiID& inout_graph_uid, megamol::core::CoreInstance* core_instance,
        megamol::core::MegaMolGraph* megamol_graph,
        vislib::math::Ternary running_graph = vislib::math::Ternary::TRI_UNKNOWN);

    ImGuiID LoadProjectFromCore(
        megamol::core::CoreInstance* core_instance, megamol::core::MegaMolGraph* megamol_graph) {
        ImGuiID graph_id = GUI_INVALID_ID;
        if (LoadUpdateProjectFromCore(graph_id, core_instance, megamol_graph)) {
            return graph_id;
        }
        return GUI_INVALID_ID;
    }

    bool AddUpdateProjectFromCore(ImGuiID in_graph_uid, megamol::core::CoreInstance* core_instance,
        megamol::core::MegaMolGraph* megamol_graph, bool use_stock);

    ImGuiID LoadAddProjectFromFile(ImGuiID in_graph_uid, const std::string& project_filename);
    bool SaveProjectToFile(ImGuiID in_graph_uid, const std::string& project_filename, bool megamol_graph);

    // Presentation ----------------------------------------------------

    void PresentGUI(GraphState_t& state) { this->present.Present(*this, state); }

private:
    // VARIABLES --------------------------------------------------------------

    GraphPtrVector_t graphs;
    ModuleStockVector_t modules_stock;
    CallStockVector_t calls_stock;
    unsigned int graph_name_uid;

    // FUNCTIONS --------------------------------------------------------------

    bool get_call_stock_data(
        Call::StockCall& call, const std::shared_ptr<const megamol::core::factories::CallDescription> call_desc);
    bool get_module_stock_data(
        Module::StockModule& mod, const std::shared_ptr<const megamol::core::factories::ModuleDescription> mod_desc);

    bool read_project_command_arguments(const std::string& line, size_t arg_count, std::vector<std::string>& out_args);
    ImVec2 project_read_confpos(const std::string& line);
    bool project_separate_name_and_namespace(const std::string& full_name, std::string& name_space, std::string& name);

    bool replace_graph_state(
        const GraphPtr_t& graph_ptr, const std::string& in_json_string, std::string& out_json_string);
    bool replace_parameter_gui_state(
        const GraphPtr_t& graph_ptr, const std::string& in_json_string, std::string& out_json_string);

    inline const std::string generate_unique_graph_name(void) {
        return ("Project_" + std::to_string(++graph_name_uid));
    }

    std::vector<size_t> get_compatible_callee_idxs(const megamol::core::CalleeSlot* callee_slot);
    std::vector<size_t> get_compatible_caller_idxs(const megamol::core::CallerSlot* caller_slot);
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GRAPHCOLLECTION_H_INCLUDED
