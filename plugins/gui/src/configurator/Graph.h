/*
 * Graph.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_GRAPH_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_GRAPH_H_INCLUDED

#include "vislib/sys/Log.h"

#include <map>
#include <vector>

#include "Call.h"
#include "CallSlot.h"
#include "Module.h"
#include "Parameter.h"


namespace megamol {
namespace gui {
namespace configurator {

typedef std::vector<Module::StockModule> ModuleStockVectorType;
typedef std::vector<Call::StockCall> CallStockVectorType;

class Graph {
public:
    typedef std::vector<ModulePtrType> ModuleGraphVectorType;
    typedef std::vector<CallPtrType> CallGraphVectorType;

    Graph(const std::string& graph_name);

    virtual ~Graph(void);

    bool AddModule(const ModuleStockVectorType& stock_modules, const std::string& module_class_name);
    bool DeleteModule(int module_uid);

    bool AddCall(const CallStockVectorType& stock_calls, const std::string& call_class_name,
        CallSlotPtrType call_slot_1, CallSlotPtrType call_slot_2);
    bool DeleteDisconnectedCalls(void);
    bool DeleteCall(int call_uid);

    const ModuleGraphVectorType& GetGraphModules(void) { return this->modules; }
    const CallGraphVectorType& GetGraphCalls(void) { return this->calls; }

    inline void SetName(const std::string& graph_name) { this->name = graph_name; }
    inline std::string& GetName(void) { return this->name; }

    inline bool IsDirty(void) const { return this->dirty_flag; }
    inline void ResetDirty(void) { this->dirty_flag = false; }

    inline int GetUID(void) const { return this->uid; }

    int generate_unique_id(void) { return (++this->generated_uid); }

    // GUI Presentation -------------------------------------------------------
    int GUI_Present(
        float in_child_width, ImFont* in_graph_font, HotKeyArrayType& inout_hotkeys, bool& out_delete_graph) {
        return this->present.Present(*this, in_child_width, in_graph_font, inout_hotkeys, out_delete_graph);
    }
    inline int GUI_GetSelectedCallSlot(void) const { return this->present.GetSelectedCallSlot(); }

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

    /**
     * Defines GUI graph present.
     */
    class Presentation {
    public:
        Presentation(void);

        ~Presentation(void);

        int Present(Graph& inout_graph, float in_child_width, ImFont* in_graph_font, HotKeyArrayType& inout_hotkeys,
            bool& out_delete_graph);

        int GetSelectedCallSlot(void) const { return this->selected_call_slot_uid; }

    private:
        ImFont* font;
        GUIUtils utils;

        ImVec2 canvas_position;
        ImVec2 canvas_size;
        ImVec2 canvas_scrolling;
        float canvas_zooming;
        ImVec2 canvas_offset;

        bool show_grid;
        bool show_call_names;
        bool show_slot_names;
        bool show_module_names;

        int selected_module_uid;
        int selected_call_uid;
        int selected_call_slot_uid;

        bool layout_current_graph;
        float split_width;
        float mouse_wheel;

        bool params_visible;
        bool params_readonly;
        std::string param_name_space;
        Parameter::Presentations param_present;

        void menu(Graph& inout_graph);
        void canvas(Graph& inout_graph, float in_child_width, HotKeyArrayType& inout_hotkeys);
        void parameters(Graph& inout_graph, float in_child_width, HotKeyArrayType& inout_hotkeys);

        void canvas_grid(void);
        void canvas_dragged_call(Graph& inout_graph);

        bool layout_graph(Graph& inout_graph);

    } present;
};

} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GRAPH_H_INCLUDED