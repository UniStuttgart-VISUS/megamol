/*
 * Graph.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_GRAPH_PRESENTATION_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_GRAPH_PRESENTATION_H_INCLUDED


#include "GUIUtils.h"
#include "widgets/HoverToolTip.h"
#include "widgets/MinimalPopUp.h"
#include "widgets/RenamePopUp.h"
#include "widgets/SplitterWidget.h"
#include "widgets/StringSearchWidget.h"

#include "Call.h"
#include "Group.h"
#include "Module.h"


namespace megamol {
namespace gui {


// Forward declarations
class Graph;


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
    bool IsCanvasHoverd(void) const { return this->canvas_hovered; }

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
    megamol::gui::GraphItemsState_t graph_state;

    // Widgets
    StringSearchWidget search_widget;
    SplitterWidget splitter_widget;
    RenamePopUp rename_popup;
    HoverToolTip tooltip;

    // FUNCTIONS --------------------------------------------------------------

    void Present(Graph& inout_graph, GraphState_t& state);
    bool StateToJSON(Graph& inout_graph, nlohmann::json& out_json);

    void present_menu(Graph& inout_graph);
    void present_canvas(Graph& inout_graph, float child_width);
    void present_parameters(Graph& inout_graph, float child_width);

    void present_canvas_grid(void);
    void present_canvas_dragged_call(Graph& inout_graph);
    void present_canvas_multiselection(Graph& inout_graph);

    void layout_graph(Graph& inout_graph);
    void layout(const ModulePtrVector_t& modules, const GroupPtrVector_t& groups, ImVec2 init_position);

    bool connected_callslot(
        const ModulePtrVector_t& modules, const GroupPtrVector_t& groups, const CallSlotPtr_t& callslot_ptr);
    bool connected_interfaceslot(
        const ModulePtrVector_t& modules, const GroupPtrVector_t& groups, const InterfaceSlotPtr_t& interfaceslot_ptr);
    bool contains_callslot(const ModulePtrVector_t& modules, ImGuiID callslot_uid);
    bool contains_interfaceslot(const GroupPtrVector_t& groups, ImGuiID interfaceslot_uid);
    bool contains_module(const ModulePtrVector_t& modules, ImGuiID module_uid);
    bool contains_group(const GroupPtrVector_t& groups, ImGuiID group_uid);
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GRAPH_PRESENTATION_H_INCLUDED
