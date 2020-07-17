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
#include "widgets/HoverToolTip.h"
#include "widgets/RenamePopUp.h"
#include "widgets/SplitterWidget.h"
#include "widgets/StringSearchWidget.h"


namespace megamol {
namespace gui {


// Forward declarations
class Graph;

// Types
typedef std::vector<Module::StockModule> ModuleStockVectorType;
typedef std::vector<Call::StockCall> CallStockVectorType;


/** ************************************************************************
 * Defines GUI graph presentation.
 */
class GraphPresentation {
public:
    friend class Graph;

    // VARIABLES --------------------------------------------------------------

    bool params_visible;
    bool params_readonly;
    bool param_extended_mode;

    // FUNCTIONS --------------------------------------------------------------

    GraphPresentation(void);
    ~GraphPresentation(void);

    void ForceUpdate(void) { this->update = true; }
    void ResetStatePointers(void) {
        this->graph_state.interact.callslot_compat_ptr.reset();
        this->graph_state.interact.interfaceslot_compat_ptr.reset();
    }

    bool StateFromJsonString(Graph& inout_graph, const std::string& json_string);

    ImGuiID GetHoveredGroup(void) const { return this->graph_state.interact.group_hovered_uid; }
    ImGuiID GetSelectedGroup(void) const { return this->graph_state.interact.group_selected_uid; }
    ImGuiID GetSelectedCallSlot(void) const { return this->graph_state.interact.callslot_selected_uid; }
    ImGuiID GetSelectedInterfaceSlot(void) const { return this->graph_state.interact.interfaceslot_selected_uid; }
    ImGuiID GetDropSlot(void) const { return this->graph_state.interact.slot_dropped_uid; }
    bool GetModuleLabelVisibility(void) const { return this->show_module_names; }
    bool GetCallSlotLabelVisibility(void) const { return this->show_slot_names; }
    bool GetCallLabelVisibility(void) const { return this->show_call_names; }
    bool GetCanvasHoverd(void) const { return this->canvas_hovered; }

    void SetLayoutGraph(void) { this->graph_layout = 1; }

private:
    // VARIABLES --------------------------------------------------------------

    bool update;
    bool show_grid;
    bool show_call_names;
    bool show_slot_names;
    bool show_module_names;
    bool show_parameter_sidebar;
    bool change_show_parameter_sidebar;
    unsigned int graph_layout;
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

    // Widgets
    StringSearchWidget search_widget;
    SplitterWidget splitter_widget;
    RenamePopUp rename_popup;
    HoverToolTip tooltip;

    // FUNCTIONS --------------------------------------------------------------

    void Present(Graph& inout_graph, GraphStateType& state);
    bool StateToJSON(Graph& inout_graph, nlohmann::json& out_json);

    void present_menu(Graph& inout_graph);
    void present_canvas(Graph& inout_graph, float child_width);
    void present_parameters(Graph& inout_graph, float child_width);

    void present_canvas_grid(void);
    void present_canvas_dragged_call(Graph& inout_graph);
    void present_canvas_multiselection(Graph& inout_graph);

    void layout_graph(Graph& inout_graph);
    void layout(const ModulePtrVectorType& modules, const GroupPtrVectorType& groups, ImVec2 init_position);

    bool connected_callslot(
        const ModulePtrVectorType& modules, const GroupPtrVectorType& groups, const CallSlotPtrType& callslot_ptr);
    bool connected_interfaceslot(const ModulePtrVectorType& modules, const GroupPtrVectorType& groups,
        const InterfaceSlotPtrType& interfaceslot_ptr);
    bool contains_callslot(const ModulePtrVectorType& modules, ImGuiID callslot_uid);
    bool contains_interfaceslot(const GroupPtrVectorType& groups, ImGuiID interfaceslot_uid);
    bool contains_module(const ModulePtrVectorType& modules, ImGuiID module_uid);
    bool contains_group(const GroupPtrVectorType& groups, ImGuiID group_uid);
};


/** ************************************************************************
 * Defines the graph.
 */

class Graph {
public:
    // VARIABLES --------------------------------------------------------------

    const ImGuiID uid;
    std::string name;
    GraphPresentation present;

    // FUNCTIONS --------------------------------------------------------------

    Graph(const std::string& graph_name);
    ~Graph(void);

    ImGuiID AddModule(const ModuleStockVectorType& stock_modules, const std::string& module_class_name);
    ImGuiID AddEmptyModule(void);
    bool DeleteModule(ImGuiID module_uid);
    inline const ModulePtrVectorType& GetModules(void) { return this->modules; }
    bool GetModule(ImGuiID module_uid, ModulePtrType& out_module_ptr);

    bool AddCall(const CallStockVectorType& stock_calls, ImGuiID slot_1_uid, ImGuiID slot_2_uid);
    bool AddCall(const CallStockVectorType& stock_calls, CallSlotPtrType callslot_1, CallSlotPtrType callslot_2);
    bool AddCall(CallPtrType& call_ptr, CallSlotPtrType callslot_1, CallSlotPtrType callslot_2);

    bool DeleteCall(ImGuiID call_uid);
    inline const CallPtrVectorType& GetCalls(void) { return this->calls; }

    ImGuiID AddGroup(const std::string& group_name = "");
    bool DeleteGroup(ImGuiID group_uid);
    inline const GroupPtrVectorType& GetGroups(void) { return this->groups; }
    bool GetGroup(ImGuiID group_uid, GroupPtrType& out_group_ptr);
    ImGuiID AddGroupModule(const std::string& group_name, const ModulePtrType& module_ptr);

    inline bool IsDirty(void) const { return this->dirty_flag; }
    inline void ResetDirty(void) { this->dirty_flag = false; }
    inline void ForceSetDirty(void) { this->dirty_flag = true; }

    bool IsMainViewSet(void);

    bool UniqueModuleRename(const std::string& module_name);

    const std::string GetFilename(void) const { return this->filename; }
    void SetFilename(const std::string& filename) { this->filename = filename; }

    // Presentation ----------------------------------------------------

    inline void PresentGUI(GraphStateType& state) { this->present.Present(*this, state); }
    bool GUIStateFromJsonString(const std::string& json_string) {
        return this->present.StateFromJsonString(*this, json_string);
    }
    bool GUIStateToJSON(nlohmann::json& out_json) { return this->present.StateToJSON(*this, out_json); }

private:
    // VARIABLES --------------------------------------------------------------

    unsigned int group_name_uid;
    ModulePtrVectorType modules;
    CallPtrVectorType calls;
    GroupPtrVectorType groups;
    bool dirty_flag;
    std::string filename;

    // FUNCTIONS --------------------------------------------------------------

    bool delete_disconnected_calls(void);
    const std::string generate_unique_group_name(void);
    const std::string generate_unique_module_name(const std::string& name);
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GRAPH_H_INCLUDED
