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

    ImGuiID AddModule(const ModuleStockVectorType& stock_modules, const std::string& module_class_name);
    bool DeleteModule(ImGuiID module_uid);
    const ModulePtrVectorType& GetGraphModules(void) { return this->modules; }
    const ModulePtrType& GetModule(ImGuiID module_uid);
    
    bool AddCall(const CallStockVectorType& stock_calls, CallSlotPtrType call_slot_1, CallSlotPtrType call_slot_2);
    bool DeleteCall(ImGuiID call_uid);

    ImGuiID AddGroup(void);
    bool DeleteGroup(ImGuiID group_uid);
    bool AddGroupModule(const ModulePtrType& module_ptr);

    inline void SetName(const std::string& graph_name) { this->name = graph_name; }
    inline std::string& GetName(void) { return this->name; }

    inline bool IsDirty(void) const { return this->dirty_flag; }
    inline void ResetDirty(void) { this->dirty_flag = false; }

    inline ImGuiID GetUID(void) const { return this->uid; }

    bool IsMainViewSet(void);
    
    bool UniqueModuleRename(const std::string& module_name);

    // GUI Presentation -------------------------------------------------------

    // Returns uid if graph is the currently active/drawn one.
    void GUI_Present(GraphStateType& state) { this->present.Present(*this, state); }

    inline ImGuiID GUI_GetSelectedGroup(void) const { return this->present.GetSelectedGroup(); }
    inline ImGuiID GUI_GetSelectedCallSlot(void) const { return this->present.GetSelectedCallSlot(); }
    inline ImGuiID GUI_GetDropCallSlot(void) const { return this->present.GetDropCallSlot(); }

    inline bool GUI_SaveGroup(void) const { return this->present.SaveGroup(); }

private:

    // VARIABLES --------------------------------------------------------------

    static ImGuiID generated_uid;
    unsigned int group_name_uid;

    ModulePtrVectorType modules;
    CallPtrVectorType calls;
    GroupGraphVectorType groups;

    const ImGuiID uid;
    std::string name;
    bool dirty_flag;

    /**
     * Defines GUI graph present.
     */
    class Presentation {
    public:
        Presentation(void);

        ~Presentation(void);

        void Present(Graph& inout_graph, GraphStateType& state);

        ImGuiID GetSelectedGroup(void) const { return this->graphstate.interact.group_selected_uid; }
        ImGuiID GetSelectedCallSlot(void) const { return this->graphstate.interact.callslot_selected_uid; }
        ImGuiID GetDropCallSlot(void) const { return this->graphstate.interact.callslot_dropped_uid; }

        bool GetModuleLabelVisibility(void) const { return this->show_module_names; }
        bool GetCallSlotLabelVisibility(void) const { return this->show_slot_names; }
        bool GetCallLabelVisibility(void) const { return this->show_call_names; }

        bool SaveGroup(void) const { return this->graphstate.interact.group_save; }

        void ApplyUpdate(void) { this->update = true; }

        bool params_visible;
        bool params_readonly;
        bool params_expert;

    private:
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

        // State propagated and shared by all graph items.
        GraphItemsStateType graphstate;

        void present_menu(Graph& inout_graph);
        void present_canvas(Graph& inout_graph, float child_width);
        void present_parameters(Graph& inout_graph, float child_width);

        void present_canvas_grid(void);
        void present_canvas_dragged_call(Graph& inout_graph);

        bool layout_graph(Graph& inout_graph);

    } present;

    // FUNCTIONS --------------------------------------------------------------

    const CallPtrVectorType& get_graph_calls(void) { return this->calls; }
    GroupGraphVectorType& get_groups(void) { return this->groups; }

    bool delete_disconnected_calls(void);

    ImGuiID add_group(const std::string& group_name);
    ImGuiID get_group_uid(const std::string& group_name);

    inline const std::string generate_unique_group_name(void) { return ("Group_" + std::to_string(++group_name_uid)); }
    std::string generate_unique_module_name(const std::string& module_name);
    ImGuiID generate_unique_id(void) { return (++this->generated_uid); }
};

} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GRAPH_H_INCLUDED
