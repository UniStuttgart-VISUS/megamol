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
#include "Group.h"
#include "Module.h"
#include "Parameter.h"

#ifdef GUI_USE_FILESYSTEM
#    include "FileUtils.h"
#endif // GUI_USE_FILESYSTEM


namespace megamol {
namespace gui {
namespace configurator {

typedef std::vector<Module::StockModule> ModuleStockVectorType;
typedef std::vector<Call::StockCall> CallStockVectorType;


class Graph {
public:
    const ImGuiID uid;
    std::string name;

    Graph(const std::string& graph_name);

    virtual ~Graph(void);

    ImGuiID AddModule(const ModuleStockVectorType& stock_modules, const std::string& module_class_name);
    bool DeleteModule(ImGuiID module_uid);
    inline const ModulePtrVectorType& GetModules(void) { return this->modules; }
    bool GetModule(ImGuiID module_uid, ModulePtrType& out_module_ptr);

    bool AddCall(const CallStockVectorType& stock_calls, CallSlotPtrType call_slot_1, CallSlotPtrType call_slot_2);
    bool DeleteCall(ImGuiID call_uid);

    ImGuiID AddGroup(const std::string& group_name = "");
    bool DeleteGroup(ImGuiID group_uid);
    ImGuiID AddGroupModule(const std::string& group_name, const ModulePtrType& module_ptr);
    inline const GroupPtrVectorType& GetGroups(void) { return this->groups; }
    bool GetGroup(ImGuiID group_uid, GroupPtrType& out_group_ptr);

    inline bool IsDirty(void) const { return this->dirty_flag; }
    inline void ResetDirty(void) { this->dirty_flag = false; }

    bool IsMainViewSet(void);

    bool UniqueModuleRename(const std::string& module_name);

    // GUI Presentation -------------------------------------------------------

    // Returns uid if graph is the currently active/drawn one.
    inline void GUI_Present(GraphStateType& state) { this->present.Present(*this, state); }

    inline ImGuiID GUI_GetSelectedGroup(void) const { return this->present.GetSelectedGroup(); }
    inline ImGuiID GUI_GetSelectedCallSlot(void) const { return this->present.GetSelectedCallSlot(); }
    inline ImGuiID GUI_GetDropCallSlot(void) const { return this->present.GetDropCallSlot(); }
    inline bool GUI_GetGroupSave(void) { return this->present.GetGroupSave(); }
    inline bool GUI_GetCanvasHoverd(void) const { return this->present.GetCanvasHoverd(); }

    inline void GUI_SetLayoutGraph(void) { this->present.LayoutGraph(); }

private:
    // VARIABLES --------------------------------------------------------------

    static ImGuiID generated_uid;
    unsigned int group_name_uid;

    ModulePtrVectorType modules;
    CallPtrVectorType calls;
    GroupPtrVectorType groups;

    bool dirty_flag;

    /** ************************************************************************
     * Defines GUI graph present.
     */
    class Presentation {
    public:
        Presentation(void);

        ~Presentation(void);

        void Present(Graph& inout_graph, GraphStateType& state);

        ImGuiID GetSelectedGroup(void) const { return this->graph_state.interact.group_selected_uid; }
        ImGuiID GetSelectedCallSlot(void) const { return this->graph_state.interact.callslot_selected_uid; }
        ImGuiID GetDropCallSlot(void) const { return this->graph_state.interact.callslot_dropped_uid; }

        bool GetModuleLabelVisibility(void) const { return this->show_module_names; }
        bool GetCallSlotLabelVisibility(void) const { return this->show_slot_names; }
        bool GetCallLabelVisibility(void) const { return this->show_call_names; }
        bool GetCanvasHoverd(void) const { return this->canvas_hovered; }

        bool GetGroupSave(void) {
            bool retval = this->graph_state.interact.group_save;
            this->graph_state.interact.group_save = false;
            return retval;
        }

        void ForceUpdate(void) { this->update = true; }
        void LayoutGraph(void) { this->layout_current_graph = true; }

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
        ImVec2 multiselect_start_pos;
        ImVec2 multiselect_end_pos;
        bool multiselect_done;
        bool canvas_hovered;
        float current_font_scaling;
        // State propagated and shared by all graph items.
        megamol::gui::GraphItemsStateType graph_state;

        void present_menu(Graph& inout_graph);
        void present_canvas(Graph& inout_graph, float child_width);
        void present_parameters(Graph& inout_graph, float child_width);

        void present_canvas_grid(void);
        void present_canvas_dragged_call(Graph& inout_graph);
        void present_canvas_multiselection(Graph& inout_graph);

        bool layout_graph(Graph& inout_graph);

    } present;

    // FUNCTIONS --------------------------------------------------------------

    bool delete_disconnected_calls(void);
    inline const CallPtrVectorType& get_calls(void) { return this->calls; }
    const std::string generate_unique_group_name(void);
    const std::string generate_unique_module_name(const std::string& name);
    inline ImGuiID generate_unique_id(void) { return (++this->generated_uid); }
};

} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GRAPH_H_INCLUDED
