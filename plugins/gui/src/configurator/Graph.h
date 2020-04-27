/*
 * Graph.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_GRAPH_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_GRAPH_H_INCLUDED


#include "Call.h"
#include "Group.h"
#include "Module.h"


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

    bool AddCall(const CallStockVectorType& stock_calls, CallSlotPtrType callslot_1, CallSlotPtrType callslot_2);
    bool AddCall(const CallStockVectorType& stock_calls, ImGuiID slot_1_uid, ImGuiID slot_2_uid);
    bool DeleteCall(ImGuiID call_uid);

    ImGuiID AddGroup(const std::string& group_name = "");
    bool DeleteGroup(ImGuiID group_uid);
    inline const GroupPtrVectorType& GetGroups(void) { return this->groups; }
    ImGuiID AddGroupModule(const std::string& group_name, const ModulePtrType& module_ptr);
    
    inline bool IsDirty(void) const { return this->dirty_flag; }
    inline void ResetDirty(void) { this->dirty_flag = false; }

    bool IsMainViewSet(void);

    bool UniqueModuleRename(const std::string& module_name);

    const std::string GetFilename(void) const { return this->filename; }
    void SetFilename(const std::string& filename) { this->filename = filename; }

    // GUI Presentation -------------------------------------------------------

    inline void GUI_Present(GraphStateType& state) { this->present.Present(*this, state); }


    void GUI_ResetStatePointers(void) { this->present.ResetStatePointers(); }
    
    bool GUI_StateFromJsonString(const std::string& json_string) {
        return this->present.StateFromJsonString(*this, json_string);
    }
    bool GUI_StateToJSON(nlohmann::json& out_json) { return this->present.StateToJSON(*this, out_json); }

    inline ImGuiID GUI_GetHoveredGroup(void) const { return this->present.GetHoveredGroup(); }
    inline ImGuiID GUI_GetSelectedGroup(void) const { return this->present.GetSelectedGroup(); }
    inline ImGuiID GUI_GetSelectedCallSlot(void) const { return this->present.GetSelectedCallSlot(); }
    inline ImGuiID GUI_GetSelectedInterfaceSlot(void) const { return this->present.GetSelectedInterfaceSlot(); }
    inline ImGuiID GUI_GetDropSlot(void) const { return this->present.GetDropSlot(); }
    inline bool GUI_GetCanvasHoverd(void) const { return this->present.GetCanvasHoverd(); }

    inline void GUI_SetLayoutGraph(void) { this->present.LayoutGraph(); }

private:
    // VARIABLES --------------------------------------------------------------

    unsigned int group_name_uid;

    ModulePtrVectorType modules;
    CallPtrVectorType calls;
    GroupPtrVectorType groups;

    bool dirty_flag;
    std::string filename;

    /** ************************************************************************
     * Defines GUI graph present.
     */
    class Presentation {
    public:
        Presentation(void);

        ~Presentation(void);

        void Present(Graph& inout_graph, GraphStateType& state);

        void ForceUpdate(void) { this->update = true; }
        void ResetStatePointers(void) {
            this->graph_state.interact.callslot_compat_ptr.reset();
            this->graph_state.interact.interfaceslot_compat_ptr.reset();
        }
        
        bool StateFromJsonString(Graph& inout_graph, const std::string& json_string);
        bool StateToJSON(Graph& inout_graph, nlohmann::json& out_json);

        ImGuiID GetHoveredGroup(void) const { return this->graph_state.interact.group_hovered_uid; }
        ImGuiID GetSelectedGroup(void) const { return this->graph_state.interact.group_selected_uid; }
        ImGuiID GetSelectedCallSlot(void) const { return this->graph_state.interact.callslot_selected_uid; }
        ImGuiID GetSelectedInterfaceSlot(void) const { return this->graph_state.interact.interfaceslot_selected_uid; }
        ImGuiID GetDropSlot(void) const { return this->graph_state.interact.slot_dropped_uid; }

        bool GetModuleLabelVisibility(void) const { return this->show_module_names; }
        bool GetCallSlotLabelVisibility(void) const { return this->show_slot_names; }
        bool GetCallLabelVisibility(void) const { return this->show_call_names; }
        bool GetCanvasHoverd(void) const { return this->canvas_hovered; }

        /**
         * Really simple module layouting.
         * Sort modules into differnet layers 'from left to right' following the calls.
         */
        void LayoutGraph(void) { this->layout_current_graph = true; }

        bool params_visible;
        bool params_readonly;
        bool param_expert_mode;

    private:
        GUIUtils utils;
        bool update;
        bool show_grid;
        bool show_call_names;
        bool show_slot_names;
        bool show_module_names;
        bool show_parameter_sidebar;
        bool change_show_parameter_sidebar;
        bool layout_current_graph;
        float parameter_sidebar_width;
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

        void layout_graph(Graph& inout_graph);
        void layout_modules(const ModulePtrVectorType& modules);

    } present;

    // FUNCTIONS --------------------------------------------------------------

    bool delete_disconnected_calls(void);
    inline const CallPtrVectorType& get_calls(void) { return this->calls; }
    bool get_group(ImGuiID group_uid, GroupPtrType& out_group_ptr);
    const std::string generate_unique_group_name(void);
    const std::string generate_unique_module_name(const std::string& name);
};

} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GRAPH_H_INCLUDED
