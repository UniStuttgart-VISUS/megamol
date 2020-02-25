/*
 * GraphManager.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GraphManager_H_INCLUDED
#define MEGAMOL_GUI_GraphManager_H_INCLUDED


#include "mmcore/CoreInstance.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/plugins/AbstractPluginInstance.h"

#include "utility/plugins/PluginManager.h"

#include "vislib/UTF8Encoder.h"
#include "vislib/sys/Log.h"

#include <map>
#include <vector>

#include "Graph.h"


namespace megamol {
namespace gui {

class GraphManager {
public:
    typedef std::shared_ptr<Graph> GraphPtrType;
    typedef std::vector<GraphPtrType> GraphsType;

    // GraphManager ------------------------------------------------------------------

    GraphManager(void);

    virtual ~GraphManager(void);

    bool AddGraph(std::string name);
    bool DeleteGraph(int graph_uid);
    const GraphManager::GraphsType& GetGraphs(void);
    const GraphPtrType GetGraph(int graph_uid);

    bool UpdateModulesCallsStock(const megamol::core::CoreInstance* core_instance);
    inline const Graph::ModuleStockType& GetModulesStock(void) { return this->modules_stock; }
    inline const Graph::CallStockType& GetCallsStock(void) { return this->calls_stock; }

    bool LoadCurrentCoreProject(std::string name, megamol::core::CoreInstance* core_instance);

    int GetCompatibleCallIndex(Graph::CallSlotPtrType call_slot_1, Graph::CallSlotPtrType call_slot_2);
    int GetCompatibleCallIndex(Graph::CallSlotPtrType call_slot, Graph::StockCallSlot stock_call_slot);

    // Only used for prototype to be able to store current graphs to lua project file.
    bool PROTOTYPE_SaveGraph(int graph_id, std::string project_filename, megamol::core::CoreInstance* cor_iInstance);

private:
    // VARIABLES --------------------------------------------------------------

    GraphManager::GraphsType graphs;

    Graph::ModuleStockType modules_stock;
    Graph::CallStockType calls_stock;

    // FUNCTIONS --------------------------------------------------------------

    bool get_call_stock_data(
        Graph::StockCall& call, const std::shared_ptr<const megamol::core::factories::CallDescription> call_desc);

    bool get_module_stock_data(
        Graph::StockModule& mod, const std::shared_ptr<const megamol::core::factories::ModuleDescription> mod_desc);

    // ------------------------------------------------------------------------
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GraphManager_H_INCLUDED