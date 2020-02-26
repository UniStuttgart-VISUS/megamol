/*
 * Graph.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_GRAPH_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_GRAPH_H_INCLUDED


#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/TernaryParam.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"

#include "vislib/sys/Log.h"

#include <map>
#include <variant>
#include <vector>

#include "Presentations.h"


namespace megamol {
namespace gui {
namespace graph {

class Graph {
public:

    typedef ParamPresentations ParamGraphType;
    typedef CallSlotPresentations CallSlotGraphType;
    typedef CallPresentations CallGraphType;
    typedef ModulePresentations ModuleGraphType;
    
    typedef std::shared_ptr<CallSlotGraphType> CallSlotGraphPtrType;
    typedef std::shared_ptr<CallGraphType> CallGraphPtrType;
    typedef std::shared_ptr<ModuleGraphType> ModuleGraphPtrType;

    typedef std::vector<ModuleGraphPtrType> ModuleGraphVectorType;
    typedef std::vector<CallGraphPtrType> CallGraphVectorType;

    // GRAPH STOCK DATA STRUCTURE ---------------------------------------------

    struct StockParameter {
        std::string class_name;
        std::string description;
        Parameter::ParamType type;
        std::string value_string;
    };

    struct StockCallSlot {
        std::string name;
        std::string description;
        std::vector<size_t> compatible_call_idxs;
        CallSlot::CallSlotType type;
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
        std::vector<Graph::StockParameter> param_slots;
        std::map<CallSlot::CallSlotType, std::vector<Graph::StockCallSlot>> call_slots;
    };

    typedef std::vector<Graph::StockModule> ModuleStockType;
    typedef std::vector<Graph::StockCall> CallStockType;

    // GRAPH ------------------------------------------------------------------

    Graph(const std::string& graph_name);

    virtual ~Graph(void);

    bool AddModule(const Graph::ModuleStockType& stock_modules, const std::string& module_class_name);
    bool DeleteModule(int module_uid);

    bool AddCall(const Graph::CallStockType& stock_calls, int call_idx, CallSlotGraphPtrType call_slot_1,
        CallSlotGraphPtrType call_slot_2);
    bool DeleteDisconnectedCalls(void);
    bool DeleteCall(int call_uid);

    const const ModuleGraphVectorType& GetGraphModules(void) { return this->modules; }
    const const CallGraphVectorType& GetGraphCalls(void) { return this->calls; }

    inline void SetName(const std::string& graph_name) { this->name = graph_name; }
    inline const std::string& GetName(void) { return this->name; }

    inline bool IsDirty(void) const { return this->dirty_flag; }
    inline void ResetDirty(void) { this->dirty_flag = false; }

    inline int GetUID(void) const { return this->uid; }

    int generate_unique_id(void) { return (++this->generated_uid); }

    struct Gui {
        float slot_radius;
        ImVec2 canvas_position;
        ImVec2 canvas_size;
        ImVec2 canvas_scrolling;
        float canvas_zooming;
        ImVec2 canvas_offset;
        bool show_grid;
        bool show_call_names;
        bool show_slot_names;
        int selected_module_uid;
        int selected_call_uid;
        int hovered_slot_uid;
        CallSlotGraphPtrType selected_slot_ptr;
        int process_selected_slot;
    } gui;

private:
    // VARIABLES --------------------------------------------------------------

    ModuleGraphVectorType modules;
    CallGraphVectorType calls;

    // UIDs are unique within a graph
    const int uid;
    std::string name;
    bool dirty_flag;

    // Global variable for unique id shared/accessible by all graphs.
    static int generated_uid;

    // ------------------------------------------------------------------------
};

} // namespace graph
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GRAPH_H_INCLUDED