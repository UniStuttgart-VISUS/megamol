/*
 * GraphManager.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GraphManager_H_INCLUDED
#define MEGAMOL_GUI_GraphManager_H_INCLUDED

#include "vislib/sys/Log.h"

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"

#include <map>
#include <vector>

#include "mmcore/CoreInstance.h"
#include "mmcore/Module.h"
#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "utility/plugins/PluginManager.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/TernaryParam.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"

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

    bool UpdateModulesCallsStock(const megamol::core::CoreInstance* core_instance);
    inline Graph::ModuleStockType& GetModulesStock(void) { return this->modules_stock; }
    inline Graph::CallStockType& GetCallsStock(void) { return this->calls_stock; }

    int GetCompatibleCallIndex(Graph::CallSlotPtrType call_slot_1, Graph::CallSlotPtrType call_slot_2);
    int GetCompatibleCallIndex(Graph::CallSlotPtrType call_slot, Graph::StockCallSlot stock_call_slot);

    /**
     * Only used for prototype to be able to store current graphs to lua project file.
     * Later use FileUtils->SaveProjectFile provided in GUI menu.
     */
    bool PROTOTYPE_SaveGraph(int graph_id, std::string project_filename, megamol::core::CoreInstance* cor_iInstance);

private:
    // VARIABLES --------------------------------------------------------------

    GraphManager::GraphsType graphs;

    Graph::ModuleStockType modules_stock;
    Graph::CallStockType calls_stock;

    int generated_uid;

    // FUNCTIONS --------------------------------------------------------------

    bool get_call_stock_data(
        Graph::StockCall& call, const std::shared_ptr<const megamol::core::factories::CallDescription> call_desc);

    bool get_module_stock_data(
        Graph::StockModule& mod, const std::shared_ptr<const megamol::core::factories::ModuleDescription> mod_desc);

    int get_unique_id(void) { return (++this->generated_uid); }

    // ------------------------------------------------------------------------
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GraphManager_H_INCLUDED