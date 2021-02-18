/*
 * Graph.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_GRAPH_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_GRAPH_H_INCLUDED


#include "GraphPresentation.h"

#include "Group.h"

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
        // friend class GraphPresentation;

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

        // VARIABLES --------------------------------------------------------------

        const ImGuiID uid;
        std::string name;
        GraphPresentation present;

        // FUNCTIONS --------------------------------------------------------------

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

        // Presentation ----------------------------------------------------

        inline void PresentGUI(GraphState_t& state) {
            this->present.Present(*this, state);
        }

    private:
        typedef std::tuple<QueueAction, QueueData> SyncQueueData_t;
        typedef std::queue<SyncQueueData_t> SyncQueue_t;

        // VARIABLES --------------------------------------------------------------

        ModulePtrVector_t modules;
        CallPtrVector_t calls;
        GroupPtrVector_t groups;
        bool dirty_flag;
        std::string filename;
        SyncQueue_t sync_queue;
        GraphCoreInterface graph_core_interface;

        // FUNCTIONS --------------------------------------------------------------

        const std::string generate_unique_group_name(void);
        const std::string generate_unique_module_name(const std::string& name);
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_GRAPH_H_INCLUDED
