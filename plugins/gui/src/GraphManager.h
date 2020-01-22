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

    struct StockParamSlot {
        std::string class_name;
        std::string description;
        Graph::ParamType type;
    };

    struct StockCallSlot {
        std::string name;
        std::string description;
        std::vector<size_t> compatible_call_idxs;
        Graph::CallSlotType type;
    };

    struct StockCall {
        std::string class_name;
        std::string description;
        std::string plugin_name;
        std::vector<std::string> functions;
    };

    struct StockModule {
        std::string class_name;
        std::string description;
        std::string plugin_name;
        bool is_view;
        std::vector<GraphManager::StockParamSlot> param_slots;
        std::map<Graph::CallSlotType, std::vector<GraphManager::StockCallSlot>> call_slots;
    };

    typedef std::vector<GraphManager::StockModule> ModuleStockType;
    typedef std::vector<GraphManager::StockCall> CallStockType;

    struct GraphData {
        Graph::ModuleGraphType modules;
        Graph::CallGraphType calls;
    };

    // GraphManager ------------------------------------------------------------------

    GraphManager(void);

    virtual ~GraphManager(void);

    int AddGraph(void);
    bool DeleteGraph(int GraphManager_index);

    bool UpdateModulesCallsStock(const megamol::core::CoreInstance* core_instance);
    inline const GraphManager::ModuleStockType& GetModulesStock(void) const { return this->modules_stock; }
    inline const GraphManager::CallStockType& GetCallsStock(void) const { return this->calls_stock; }

    int GetCompatibleCallIndex(Graph::CallSlotPtr call_slot_1, Graph::CallSlotPtr call_slot_2);
    int GetCompatibleCallIndex(Graph::CallSlotPtr call_slot, GraphManager::StockCallSlot stock_call_slot);

    /**
     * Only used for prototype to be able to store current GraphManager to lua project file.
     * Later use FileUtils->SaveProjectFile provided in GUI menu.
     */
    bool PROTOTYPE_SaveGraphManager(int GraphManager_index, std::string project_filename, megamol::core::CoreInstance* cor_iInstance);

private:
    // VARIABLES --------------------------------------------------------------

    int uid;

    std::vector<GraphData> graphs;

    GraphManager::ModuleStockType modules_stock;
    GraphManager::CallStockType calls_stock;

    // FUNCTIONS --------------------------------------------------------------

    bool get_call_stock_data(
        GraphManager::StockCall& call, const std::shared_ptr<const megamol::core::factories::CallDescription> call_desc);

    bool get_module_stock_data(
        GraphManager::StockModule& mod, const std::shared_ptr<const megamol::core::factories::ModuleDescription> mod_desc);

    int get_unique_id(void) { return (++this->uid); }

    // ------------------------------------------------------------------------
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GraphManager_H_INCLUDED