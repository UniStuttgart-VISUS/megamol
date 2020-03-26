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
#include "Group.h"

#ifdef GUI_USE_FILESYSTEM
#    include "FileUtils.h"
#endif // GUI_USE_FILESYSTEM


namespace megamol {
namespace gui {
namespace configurator {

typedef std::vector<Module::StockModule> ModuleStockVectorType;
typedef std::vector<Call::StockCall> CallStockVectorType;

typedef std::vector<Group> GroupGraphVectorType;

class Graph {
public:

    Graph(const std::string& graph_name);

    virtual ~Graph(void);

    bool AddModule(const ModuleStockVectorType& stock_modules, const std::string& module_class_name);
    bool DeleteModule(ImGuiID module_uid);

    bool AddCall(const CallStockVectorType& stock_calls, CallSlotPtrType call_slot_1, CallSlotPtrType call_slot_2);
    bool DeleteDisconnectedCalls(void);
    bool DeleteCall(ImGuiID call_uid);

    const ModuleGraphVectorType& GetGraphModules(void) { return this->modules; }
    const CallGraphVectorType& GetGraphCalls(void) { return this->calls; }
    GroupGraphVectorType& GetGraphGroups(void) { return this->groups; }

    inline void SetName(const std::string& graph_name) { this->name = graph_name; }
    inline std::string& GetName(void) { return this->name; }

    inline bool IsDirty(void) const { return this->dirty_flag; }
    inline void ResetDirty(void) { this->dirty_flag = false; }

    inline ImGuiID GetUID(void) const { return this->uid; }

    bool RenameAssignedModuleName(const std::string& module_name);

    // GUI Presentation -------------------------------------------------------

    // Returns uid if graph is the currently active/drawn one.
    ImGuiID GUI_Present(float in_child_width, ImFont* in_graph_font, HotKeyArrayType& inout_hotkeys, bool& out_delete_graph, bool& show_parameter_sidebar) {
        return this->present.Present(*this, in_child_width, in_graph_font, inout_hotkeys, out_delete_graph, show_parameter_sidebar);
    }

    inline ImGuiID GUI_GetSelectedItem(void) const { return this->present.GetSelectedItem(); }
    inline ImGuiID GUI_GetDropCallSlot(void) const { return this->present.GetDropCallSlot(); }

private:
    // VARIABLES --------------------------------------------------------------

    ModuleGraphVectorType modules;
    CallGraphVectorType calls;
    GroupGraphVectorType groups;

    const ImGuiID uid;
    std::string name;
    bool dirty_flag;

    // Global variable for unique id shared/accessible by all graphs.
    static ImGuiID generated_uid;

    unsigned int group_name_uid;

    /**
     * Defines GUI graph present.
     */
    class Presentation {
    public:
        Presentation(void);

        ~Presentation(void);

        ImGuiID Present(Graph& inout_graph, float in_child_width, ImFont* in_graph_font, HotKeyArrayType& inout_hotkeys,
            bool& out_delete_graph, bool& show_parameter_sidebar);

        ImGuiID GetSelectedItem(void) const { return this->state.interact.item_selected_uid; }
        ImGuiID GetDropCallSlot(void) const { return this->state.interact.callslot_dropped_uid; }

        bool GetModuleLabelVisibility(void) const { return this->show_module_names; }
        bool GetCallSlotLabelVisibility(void) const { return this->show_slot_names; }
        bool GetCallLabelVisibility(void) const { return this->show_call_names; }

        bool params_visible;
        bool params_readonly;
        bool params_expert;

        StateType state;

    private:
        ImFont* font;
        GUIUtils utils;

        bool update;
        bool show_grid;
        bool show_call_names;
        bool show_slot_names;
        bool show_module_names;
        bool layout_current_graph;
        float child_split_width;
        bool reset_zooming;
        std::string param_name_space;

        void present_menu(Graph& inout_graph);
        void present_canvas(Graph& inout_graph, float in_child_width);
        void present_parameters(Graph& inout_graph, float in_child_width);

        void present_canvas_grid(void);
        void present_canvas_dragged_call(Graph& inout_graph);

        bool layout_graph(Graph& inout_graph);

    } present;

    // FUNCTIONS --------------------------------------------------------------

    bool add_group(const std::string& group_name);
    bool delete_group(ImGuiID group_uid);
    bool group_exists(const std::string& group_name);

    inline const std::string generate_unique_group_name(void) { return ("Group_" + std::to_string(++group_name_uid)); }
    std::string generate_unique_module_name(const std::string& module_name);
    ImGuiID generate_unique_id(void) { return (++this->generated_uid); }
};

} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GRAPH_H_INCLUDED