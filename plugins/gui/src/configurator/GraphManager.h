/*
 * GraphManager.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_GRAPHMANAGER_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_GRAPHMANAGER_H_INCLUDED


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
#include "Stock.h"


namespace megamol {
namespace gui {
namespace configurator {


class GraphManager {
public:
    typedef std::shared_ptr<Graph> GraphPtrType;
    typedef std::vector<GraphPtrType> GraphsType;

    GraphManager(void);

    virtual ~GraphManager(void);

    bool AddGraph(std::string name);
    bool DeleteGraph(int graph_uid);
    const GraphManager::GraphsType& GetGraphs(void);
    const GraphPtrType GetGraph(int graph_uid);

    bool UpdateModulesCallsStock(const megamol::core::CoreInstance* core_instance);
    inline const ModuleStockType& GetModulesStock(void) { return this->modules_stock; }
    inline const CallStockType& GetCallsStock(void) { return this->calls_stock; }

    bool LoadCurrentCoreProject(std::string name, megamol::core::CoreInstance* core_instance);

    int GetCompatibleCallIndex(CallSlotPtrType call_slot_1, CallSlotPtrType call_slot_2);
    int GetCompatibleCallIndex(CallSlotPtrType call_slot, StockCallSlot stock_call_slot);

    // Only used for prototype to be able to store current graphs to lua project file.
    bool PROTOTYPE_SaveGraph(int graph_id, std::string project_filename, megamol::core::CoreInstance* cor_iInstance);

    // GUI Presentation -------------------------------------------------------
    bool Present(float child_width, ImFont* graph_font) {
        return this->present.Present(*this, child_width, graph_font);
    }
    const GraphManager::GraphPtrType GetActiveGraph(void) { return this->present.GetActiveGraph(); }

private:
    // VARIABLES --------------------------------------------------------------

    GraphManager::GraphsType graphs;

    ModuleStockType modules_stock;
    CallStockType calls_stock;

    /**
     * Defines GUI graph present.
     */
    class Presentation {
    public:
        Presentation(void);

        ~Presentation(void);

        bool Present(GraphManager& graph_manager, float child_width, ImFont* graph_font);

        const GraphManager::GraphPtrType GetActiveGraph(void) { return this->active_graph; }

    private:
        GraphManager::GraphPtrType active_graph;

    } present;

    // FUNCTIONS --------------------------------------------------------------

    bool get_call_stock_data(
        StockCall& call, const std::shared_ptr<const megamol::core::factories::CallDescription> call_desc);

    bool get_module_stock_data(
        StockModule& mod, const std::shared_ptr<const megamol::core::factories::ModuleDescription> mod_desc);

    // ------------------------------------------------------------------------
};

} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GRAPHMANAGER_H_INCLUDED