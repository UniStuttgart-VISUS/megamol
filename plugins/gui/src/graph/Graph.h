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

        Graph(const std::string& graph_name);
        ~Graph(void);

        ModulePtr_t AddModule(const ModuleStockVector_t& stock_modules, const std::string& class_name);
        ModulePtr_t AddModule(const std::string& class_name, const std::string& description,
            const std::string& plugin_name, bool is_view);
        bool DeleteModule(ImGuiID module_uid, bool force = false);
        inline const ModulePtrVector_t& Modules(void) {
            return this->modules;
        }
        ModulePtr_t GetModule(ImGuiID module_uid);
        bool ModuleExists(const std::string& module_fullname);

        bool AddCall(const CallStockVector_t& stock_calls, ImGuiID slot_1_uid, ImGuiID slot_2_uid);
        bool AddCall(const CallStockVector_t& stock_calls, CallSlotPtr_t callslot_1, CallSlotPtr_t callslot_2);
        bool AddCall(CallPtr_t& call_ptr, CallSlotPtr_t callslot_1, CallSlotPtr_t callslot_2);

        bool DeleteCall(ImGuiID call_uid);
        inline const CallPtrVector_t& Calls(void) {
            return this->calls;
        }

        ImGuiID AddGroup(const std::string& group_name = "");
        bool DeleteGroup(ImGuiID group_uid);
        inline const GroupPtrVector_t& GetGroups(void) {
            return this->groups;
        }
        GroupPtr_t GetGroup(ImGuiID group_uid);
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

        const std::string GetFilename(void) const;
        void SetFilename(const std::string& filename, bool saved_filename);

        bool PushSyncQueue(QueueAction in_action, const QueueData& in_data);
        bool PopSyncQueue(QueueAction& out_action, QueueData& out_data);
        inline void ClearSyncQueue(void) {
            while (!this->sync_queue.empty()) {
                this->sync_queue.pop();
            }
        }

        inline GraphCoreInterface GetCoreInterface(void) const {
            return this->core_interface;
        }
        inline void SetCoreInterface(GraphCoreInterface graph_core_interface) {
            this->core_interface = graph_core_interface;
        }

        inline bool IsRunning(void) const {
            return this->running;
        }
        inline void SetRunning(bool run) {
            this->running = run;
            if (!this->running) {
                this->core_interface = GraphCoreInterface::NO_INTERFACE;
            }
        }

        const std::string GenerateUniqueGraphEntryName(void);

        bool StateFromJSON(const nlohmann::json& in_json);
        bool StateToJSON(nlohmann::json& inout_json);

        void Draw(GraphState_t& state);

        inline const ImGuiID UID(void) const {
            return this->uid;
        }
        inline const std::string Name(void) const {
            return this->name;
        }

        void ForceUpdate(void) {
            this->gui_update = true;
        }
        void ResetStatePointers(void) {
            this->gui_graph_state.interact.callslot_compat_ptr.reset();
            this->gui_graph_state.interact.interfaceslot_compat_ptr.reset();
        }

        ImGuiID GetHoveredGroup(void) const {
            return this->gui_graph_state.interact.group_hovered_uid;
        }
        ImGuiID GetSelectedGroup(void) const {
            return this->gui_graph_state.interact.group_selected_uid;
        }
        ImGuiID GetSelectedCallSlot(void) const {
            return this->gui_graph_state.interact.callslot_selected_uid;
        }
        ImGuiID GetSelectedInterfaceSlot(void) const {
            return this->gui_graph_state.interact.interfaceslot_selected_uid;
        }
        ImGuiID GetDropSlot(void) const {
            return this->gui_graph_state.interact.slot_dropped_uid;
        }
        bool IsCanvasHoverd(void) const {
            return this->gui_canvas_hovered;
        }

        void SetLayoutGraph(bool layout = true) {
            this->gui_graph_layout = ((layout) ? (1) : (0));
        }

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
        std::pair<std::pair<bool, std::string>, std::pair<bool, std::string>>
            filenames; // (1) script path from core | (2) saved file name
        SyncQueue_t sync_queue;
        GraphCoreInterface core_interface;
        bool running; // Do not change in Graph class, only via GraphCollection

        megamol::gui::GraphItemsState_t gui_graph_state; /// State propagated and shared by all graph items
        bool gui_update;
        bool gui_show_grid;
        bool gui_show_parameter_sidebar;
        bool gui_params_visible;
        bool gui_params_readonly;
        bool gui_change_show_parameter_sidebar;
        unsigned int gui_graph_layout;
        float gui_parameter_sidebar_width;
        bool gui_reset_zooming;
        bool gui_increment_zooming;
        bool gui_decrement_zooming;
        std::string gui_param_name_space;
        std::string gui_current_graph_entry_name;
        ImVec2 gui_multiselect_start_pos;
        ImVec2 gui_multiselect_end_pos;
        bool gui_multiselect_done;
        bool gui_canvas_hovered;
        float gui_current_font_scaling;
        StringSearchWidget gui_search_widget;
        SplitterWidget gui_splitter_widget;
        RenamePopUp gui_rename_popup;
        HoverToolTip gui_tooltip;

        // FUNCTIONS --------------------------------------------------------------

        void draw_menu(GraphState_t& state);
        void draw_canvas(float child_width, GraphState_t& state);
        void draw_parameters(float child_width);

        void draw_canvas_grid(void);
        void draw_canvas_dragged_call(void);
        void draw_canvas_multiselection(void);

        void layout_graph(void);
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
