/*
 * Graph.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "Call.h"
#include "Group.h"
#include "Module.h"
#include "widgets/HoverToolTip.h"
#include "widgets/PopUps.h"
#include "widgets/SplitterWidget.h"
#include "widgets/StringSearchWidget.h"
#include <list>


namespace megamol::gui {


// Forward declarations
class Graph;

// Types
typedef std::vector<Module::StockModule> ModuleStockVector_t;
typedef std::vector<Call::StockCall> CallStockVector_t;


/** ************************************************************************
 * Defines the graph
 */
class Graph {
public:
    enum QueueAction {
        ADD_MODULE,
        DELETE_MODULE,
        RENAME_MODULE,
        ADD_CALL,
        DELETE_CALL,
        CREATE_GRAPH_ENTRY,
        REMOVE_GRAPH_ENTRY
    };

    struct QueueData {
        std::string name_id;    // Requierd for ADD_MODULE, DELETE_MODUL, RENAME_MODULE
        std::string class_name; // Requierd for ADD_MODULE, ADD_CALL
        std::string rename_id;  // Requierd for RENAME_MODULE
        std::string caller;     // Requierd for ADD_CALL, DELETE_CALL
        std::string callee;     // Requierd for ADD_CALL, DELETE_CALL
    };

    explicit Graph(const std::string& graph_name);
    ~Graph();

    ModulePtr_t AddModule(const ModuleStockVector_t& stock_ms, const std::string& class_name,
        const std::string& module_name, const std::string& group_name);
    ModulePtr_t AddModule(const std::string& class_name, const std::string& module_name, const std::string& group_name,
        const std::string& description, const std::string& plugin_name, bool is_view);
    bool DeleteModule(ImGuiID module_uid, bool use_queue = true);
    inline const ModulePtrVector_t& Modules() {
        return this->modules;
    }
    ModulePtr_t GetModule(ImGuiID module_uid);
    ModulePtr_t GetModule(const std::string& module_fullname);
    bool ModuleExists(const std::string& module_fullname);
    bool UniqueModuleRename(const std::string& module_full_name);

    bool AddCall(const CallStockVector_t& stock_calls, ImGuiID slot_1_uid, ImGuiID slot_2_uid);
    CallPtr_t AddCall(const CallStockVector_t& stock_calls, CallSlotPtr_t callslot_1, CallSlotPtr_t callslot_2,
        bool use_queue = true);
    bool ConnectCall(CallPtr_t& call_ptr, CallSlotPtr_t callslot_1, CallSlotPtr_t callslot_2, bool use_queue = true);
    CallPtr_t GetCall(
        std::string const& call_classname, std::string const& caller_fullname, std::string const& callee_fullname);
    bool DeleteCall(ImGuiID call_uid, bool use_queue = true);
    inline const CallPtrVector_t& Calls() {
        return this->calls;
    }
    bool CallExists(
        std::string const& call_classname, std::string const& caller_fullname, std::string const& callee_fullname);

    ImGuiID AddGroup(const std::string& group_name = "");
    bool DeleteGroup(ImGuiID group_uid);
    inline const GroupPtrVector_t& GetGroups() {
        return this->groups;
    }
    GroupPtr_t GetGroup(ImGuiID group_uid);
    ImGuiID AddGroupModule(const std::string& group_name, const ModulePtr_t& module_ptr, bool use_queue = true);
    ImGuiID AddGroupModule(ImGuiID group_uid, const ModulePtr_t& module_ptr, bool use_queue = true);
    ImGuiID RemoveGroupModule(
        ImGuiID group_uid, ImGuiID module_uid, bool use_queue = true, bool reset_interface = true);

    void Clear();

    inline bool IsDirty() const {
        return this->dirty_flag;
    }
    inline void ResetDirty() {
        this->dirty_flag = false;
    }
    inline void ForceSetDirty() {
        this->dirty_flag = true;
    }

    std::string GetFilename() const;
    void SetFilename(const std::string& filename, bool saved_filename);

    bool ToggleGraphEntry(bool use_queue = true);
    bool AddGraphEntry(const ModulePtr_t& module_ptr, const std::string& name, bool use_queue = true);
    bool RemoveGraphEntry(const ModulePtr_t& module_ptr, bool use_queue = true);

    bool PushSyncQueue(QueueAction in_action, const QueueData& in_data);
    bool PopSyncQueue(QueueAction& out_action, QueueData& out_data);
    QueueData FindQueueEntryByActionName(QueueAction action, const std::string& name);

    inline bool IsRunning() const {
        return this->running;
    }
    inline void SetRunning(bool run) {
        this->running = run;
    }

    std::string GenerateUniqueGraphEntryName();

    bool StateFromJSON(const nlohmann::json& in_json);
    bool StateToJSON(nlohmann::json& inout_json);

    void Draw(GraphState_t& state);

    void DrawGlobalParameterWidgets(megamol::core::utility::PickingBuffer& picking_buffer,
        std::shared_ptr<TransferFunctionEditor> win_tfeditor_ptr);

    inline ImGuiID UID() const {
        return this->uid;
    }

    inline std::string Name() const {
        return this->name;
    }

    void ResetStatePointers() {
        this->gui_graph_state.interact.callslot_compat_ptr.reset();
        this->gui_graph_state.interact.interfaceslot_compat_ptr.reset();
    }

    ImGuiID GetHoveredGroup() const {
        return this->gui_graph_state.interact.group_hovered_uid;
    }
    ImGuiID GetSelectedGroup() const {
        return this->gui_graph_state.interact.group_selected_uid;
    }
    ImGuiID GetSelectedCallSlot() const {
        return this->gui_graph_state.interact.callslot_selected_uid;
    }
    ImGuiID GetSelectedInterfaceSlot() const {
        return this->gui_graph_state.interact.interfaceslot_selected_uid;
    }
    UIDPair_t ConsumeDragAndDropSlots() {
        auto drag_drop_uids = this->gui_graph_state.interact.slot_drag_drop_uids;
        this->gui_graph_state.interact.slot_drag_drop_uids = UIDPair_t(GUI_INVALID_ID, GUI_INVALID_ID);
        return drag_drop_uids;
    }
    bool IsCanvasHovered() const {
        return this->gui_canvas_hovered;
    }
    bool IsModuleHovered() const {
        return (this->gui_graph_state.interact.module_hovered_uid != GUI_INVALID_ID);
    }

    /// TODO get set module/call labels
    /// TODO get/set call coloring

    void SetLayoutGraph(bool layout = true) {
        this->gui_graph_layout = ((layout) ? (1) : (0));
    }

private:
    typedef std::tuple<QueueAction, QueueData> SyncQueueData_t;
    typedef std::list<SyncQueueData_t> SyncQueue_t;

    // VARIABLES --------------------------------------------------------------

    const ImGuiID uid;
    std::string name;

    ModulePtrVector_t modules;
    CallPtrVector_t calls;
    GroupPtrVector_t groups;
    bool dirty_flag;
    std::pair<std::pair<bool, std::string>, std::pair<bool, std::string>>
        filenames; // (1) script path from core | (2) saved file name
    SyncQueue_t sync_queue;
    bool running; // Do not change in Graph class, only via GraphCollection

    megamol::gui::GraphItemsState_t gui_graph_state; /// State propagated and shared by all graph items
    bool gui_update;
    bool gui_show_grid;
    bool gui_show_parameter_sidebar;
    bool gui_change_show_parameter_sidebar;
    float gui_parameter_sidebar_width;
    bool gui_show_profiling_bar;
    bool gui_change_show_profiling_bar;
    float gui_profiling_bar_height;
    unsigned int gui_graph_layout;
    bool gui_reset_zooming;
    bool gui_increment_zooming;
    bool gui_decrement_zooming;
    std::string gui_current_graph_entry_name;
    ImVec2 gui_multiselect_start_pos;
    ImVec2 gui_multiselect_end_pos;
    bool gui_multiselect_done;
    bool gui_canvas_hovered;
    float gui_current_font_scaling;
    StringSearchWidget gui_search_widget;
    SplitterWidget gui_parameters_splitter;
    SplitterWidget gui_profiling_splitter;
    PopUps gui_rename_popup;
    HoverToolTip gui_tooltip;
#ifdef MEGAMOL_USE_PROFILING
    ImageWidget gui_profiling_run_button;
    std::vector<std::pair<std::weak_ptr<gui::Module>, std::weak_ptr<gui::Call>>> profiling_list;
    std::chrono::system_clock::time_point scroll_delta_time;
    bool gui_profiling_graph_link;
#endif // MEGAMOL_USE_PROFILING

    // FUNCTIONS --------------------------------------------------------------

    void draw_menu(GraphState_t& state);
    void draw_canvas(ImVec2 position, ImVec2 size, GraphState_t& state);
    void draw_parameters(ImVec2 position, ImVec2 size);

    void draw_canvas_grid() const;
    void draw_canvas_dragged_call();
    void draw_canvas_multiselection();

    void layout_graph();
    void layout(const ModulePtrVector_t& ms, const GroupPtrVector_t& gs, ImVec2 init_position);

    bool connected_callslot(
        const ModulePtrVector_t& ms, const GroupPtrVector_t& gs, const CallSlotPtr_t& callslot_ptr) const;
    bool connected_interfaceslot(
        const ModulePtrVector_t& ms, const GroupPtrVector_t& gs, const InterfaceSlotPtr_t& interfaceslot_ptr) const;
    bool contains_callslot(const ModulePtrVector_t& ms, ImGuiID callslot_uid) const;
    bool contains_interfaceslot(const GroupPtrVector_t& gs, ImGuiID interfaceslot_uid) const;
    bool contains_module(const ModulePtrVector_t& ms, ImGuiID module_uid) const;
    bool contains_group(const GroupPtrVector_t& gs, ImGuiID group_uid) const;

    std::string generate_unique_group_name() const;
    std::string generate_unique_module_name(const std::string& name) const;

#ifdef MEGAMOL_USE_PROFILING
    void draw_profiling(ImVec2 position, ImVec2 size);
#endif // MEGAMOL_USE_PROFILING
};


} // namespace megamol::gui
