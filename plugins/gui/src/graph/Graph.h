/*
 * Graph.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_GRAPH_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_GRAPH_H_INCLUDED


#include "GUIUtils.h"
#include "widgets/HoverToolTip.h"
#include "widgets/MinimalPopUp.h"
#include "widgets/RenamePopUp.h"
#include "widgets/SplitterWidget.h"
#include "widgets/StringSearchWidget.h"

#include "Call.h"
#include "Group.h"
#include "Module.h"

#include "vislib/math/Ternary.h"

#include <queue>
#include <tuple>


namespace megamol {
namespace gui {


    // Forward declarations
    class Graph;

    // Types
    typedef std::vector<Module::StockModule> ModuleStockVector_t;
    typedef std::vector<Call::StockCall> CallStockVector_t;


    /** ************************************************************************
     * Defines the graph.
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
            std::string name_id = "";    // Requierd for ADD_MODULE, DELETE_MODUL, RENAME_MODULE
            std::string class_name = ""; // Requierd for ADD_MODULE, ADD_CALL
            std::string rename_id = "";  // Requierd for RENAME_MODULE
            std::string caller = "";     // Requierd for ADD_CALL, DELETE_CALL
            std::string callee = "";     // Requierd for ADD_CALL, DELETE_CALL
        };

        Graph(const std::string& graph_name, GraphCoreInterface core_interface);
        ~Graph(void);

        ImGuiID AddModule(const ModuleStockVector_t& stock_modules, const std::string& module_class_name);
        ImGuiID AddEmptyModule(void);
        bool DeleteModule(ImGuiID module_uid, bool force = false);
        inline const ModulePtrVector_t& GetModules(void) {
            return this->modules;
        }
        bool GetModule(ImGuiID module_uid, ModulePtr_t& out_module_ptr);
        bool ModuleExists(const std::string& module_fullname);

        bool AddCall(const CallStockVector_t& stock_calls, ImGuiID slot_1_uid, ImGuiID slot_2_uid);
        bool AddCall(const CallStockVector_t& stock_calls, CallSlotPtr_t callslot_1, CallSlotPtr_t callslot_2);
        bool AddCall(CallPtr_t& call_ptr, CallSlotPtr_t callslot_1, CallSlotPtr_t callslot_2);

        bool DeleteCall(ImGuiID call_uid);
        inline const CallPtrVector_t& GetCalls(void) {
            return this->calls;
        }

        ImGuiID AddGroup(const std::string& group_name = "");
        bool DeleteGroup(ImGuiID group_uid);
        inline const GroupPtrVector_t& GetGroups(void) {
            return this->groups;
        }
        bool GetGroup(ImGuiID group_uid, GroupPtr_t& out_group_ptr);
        ImGuiID AddGroupModule(const std::string& group_name, const ModulePtr_t& module_ptr);

        void Clear(void);

        inline bool IsDirty(void) const {
            return this->dirty_flag;
        }
        inline void ResetDirty(void) {
            this->dirty_flag = false;
        }
        inline void ForceSetDirty(void) {
            this->dirty_flag = true;
        }

        bool UniqueModuleRename(const std::string& module_full_name);

        const std::string GetFilename(void) const {
            return this->filename;
        }
        void SetFilename(const std::string& filename) {
            this->filename = filename;
        }

        bool PushSyncQueue(QueueAction in_action, const QueueData& in_data);
        bool PopSyncQueue(QueueAction& out_action, QueueData& out_data);
        inline void ClearSyncQueue(void) {
            while (!this->sync_queue.empty()) {
                this->sync_queue.pop();
            }
        }

        inline GraphCoreInterface GetCoreInterface(void) {
            return this->graph_core_interface;
        }
        inline bool HasCoreInterface(void) {
            return (this->graph_core_interface != GraphCoreInterface::NO_INTERFACE);
        }

        const std::string GenerateUniqueGraphEntryName(void);

        bool StateFromJSON(const nlohmann::json& in_json);
        bool StateToJSON(nlohmann::json& inout_json);

        void Draw(GraphState_t& state);

    private:
        typedef std::tuple<QueueAction, QueueData> SyncQueueData_t;
        typedef std::queue<SyncQueueData_t> SyncQueue_t;

        // VARIABLES --------------------------------------------------------------

        const ImGuiID uid;
        std::string name;

        ModulePtrVector_t modules;
        CallPtrVector_t calls;
        GroupPtrVector_t groups;
        bool dirty_flag;
        std::string filename;
        SyncQueue_t sync_queue;
        GraphCoreInterface graph_core_interface;

        // GUI --------------------------------------------------------------

        megamol::gui::GraphItemsState_t graph_state; /// State propagated and shared by all graph items.
        bool update;
        bool show_grid;
        bool show_call_label;
        bool show_call_slots_label;
        bool show_slot_label;
        bool show_module_label;
        bool show_parameter_sidebar;
        bool params_visible;
        bool params_readonly;
        bool param_extended_mode;
        bool change_show_parameter_sidebar;
        unsigned int graph_layout;
        float parameter_sidebar_width;
        bool reset_zooming;
        bool increment_zooming;
        bool decrement_zooming;
        std::string param_name_space;
        std::string current_graph_entry_name;
        ImVec2 multiselect_start_pos;
        ImVec2 multiselect_end_pos;
        bool multiselect_done;
        bool canvas_hovered;
        float current_font_scaling;
        StringSearchWidget search_widget;
        SplitterWidget splitter_widget;
        RenamePopUp rename_popup;
        HoverToolTip tooltip;

        // FUNCTIONS --------------------------------------------------------------

        /*

        void ForceUpdate(void) {
            this->update = true;
        }
        void ResetStatePointers(void) {
            this->graph_state.interact.callslot_compat_ptr.reset();
            this->graph_state.interact.interfaceslot_compat_ptr.reset();
        }

        ImGuiID GetHoveredGroup(void) const {
            return this->graph_state.interact.group_hovered_uid;
        }
        ImGuiID GetSelectedGroup(void) const {
            return this->graph_state.interact.group_selected_uid;
        }
        ImGuiID GetSelectedCallSlot(void) const {
            return this->graph_state.interact.callslot_selected_uid;
        }
        ImGuiID GetSelectedInterfaceSlot(void) const {
            return this->graph_state.interact.interfaceslot_selected_uid;
        }
        ImGuiID GetDropSlot(void) const {
            return this->graph_state.interact.slot_dropped_uid;
        }
        bool GetModuleLabelVisibility(void) const {
            return this->show_module_label;
        }
        bool GetSlotLabelVisibility(void) const {
            return this->show_slot_label;
        }
        bool GetCallLabelVisibility(void) const {
            return this->show_call_label;
        }
        bool GetCallSlotLabelVisibility(void) const {
            return this->show_call_slots_label;
        }
        bool IsCanvasHoverd(void) const {
            return this->canvas_hovered;
        }

        void SetLayoutGraph(bool l = true) {
            this->graph_layout = ((l) ? (1) : (0));
        }

        */


        void draw_menu(Graph& inout_graph);
        void draw_canvas(Graph& inout_graph, float child_width);
        void draw_parameters(Graph& inout_graph, float child_width);

        void draw_canvas_grid(void);
        void draw_canvas_dragged_call(Graph& inout_graph);
        void draw_canvas_multiselection(Graph& inout_graph);

        void layout_graph(Graph& inout_graph);
        void layout(const ModulePtrVector_t& modules, const GroupPtrVector_t& groups, ImVec2 init_position);

        bool connected_callslot(
            const ModulePtrVector_t& modules, const GroupPtrVector_t& groups, const CallSlotPtr_t& callslot_ptr);
        bool connected_interfaceslot(const ModulePtrVector_t& modules, const GroupPtrVector_t& groups,
            const InterfaceSlotPtr_t& interfaceslot_ptr);
        bool contains_callslot(const ModulePtrVector_t& modules, ImGuiID callslot_uid);
        bool contains_interfaceslot(const GroupPtrVector_t& groups, ImGuiID interfaceslot_uid);
        bool contains_module(const ModulePtrVector_t& modules, ImGuiID module_uid);
        bool contains_group(const GroupPtrVector_t& groups, ImGuiID group_uid);

        const std::string generate_unique_group_name(void);
        const std::string generate_unique_module_name(const std::string& name);
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GRAPH_H_INCLUDED
