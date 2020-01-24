/*
 * Graph.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_H_INCLUDED

#include "vislib/sys/Log.h"

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"

#include <map>
#include <vector>


namespace megamol {
namespace gui {

class Graph {
public:
    enum ParamType {
        BUTTON,
        BOOL,
        COLOR,
        ENUM,
        FILEPATH,
        FLEXENUM,
        FLOAT,
        INT,
        STRING,
        TERNARY,
        TRANSFERFUNCTION,
        VECTOR2F,
        VECTOR3F,
        VECTOR4F,
        UNKNOWN
    };

    enum CallSlotType { CALLEE, CALLER };

    // GRAPH STOCK DATA STRUCTURE ---------------------------------------------

    struct StockParamSlot {
        std::string class_name;
        std::string description;
        Graph::ParamType type;
        std::string value_string;
    };

    struct StockCallSlot {
        std::string name;
        std::string description;
        std::vector<size_t> compatible_call_idxs;
        Graph::CallSlotType type;
    };

    struct StockCall {
        std::string class_name;
        std::string description;
        std::string plugin_name;
        std::vector<std::string> functions;
    };

    struct StockModule {
        std::string class_name;
        std::string description;
        std::string plugin_name;
        bool is_view;
        std::vector<Graph::StockParamSlot> param_slots;
        std::map<Graph::CallSlotType, std::vector<Graph::StockCallSlot>> call_slots;
    };

    typedef std::vector<Graph::StockModule> ModuleStockType;
    typedef std::vector<Graph::StockCall> CallStockType;

    // GRAPH DATA STRUCTURE ---------------------------------------------------

    // Forward declaration
    class Module;
    class Call;
    class CallSlot;
    class ParamSlot;

    typedef std::shared_ptr<Graph::CallSlot> CallSlotPtrType;
    typedef std::shared_ptr<Graph::Call> CallPtrType;
    typedef std::shared_ptr<Graph::Module> ModulePtrType;

    typedef std::vector<Graph::ModulePtrType> ModuleGraphType;
    typedef std::vector<Graph::CallPtrType> CallGraphType;

    class ParamSlot {
    public:
        ParamSlot(int gui_id);
        ~ParamSlot();

        const int uid;

        std::string class_name;
        std::string description;
        Graph::ParamType type;

        std::string full_name;
        std::string value_string;

    private:
    };

    class CallSlot {
    public:
        CallSlot(int gui_id);
        ~CallSlot();

        const int uid;

        std::string name;
        std::string description;
        std::vector<size_t> compatible_call_idxs; // (Storing only indices of compatible calls for faster comparison.)
        Graph::CallSlotType type;

        struct Gui {
            ImVec2 position;
        } gui;

        bool UpdateGuiPos(void);

        bool CallsConnected(void) const;
        bool ConnectCall(Graph::CallPtrType call);
        bool DisConnectCall(int call_uid, bool called_by_call);
        bool DisConnectCalls(void);
        const std::vector<Graph::CallPtrType> GetConnectedCalls(void);

        bool ParentModuleConnected(void) const;
        bool ConnectParentModule(Graph::ModulePtrType parent_module);
        bool DisConnectParentModule(void);
        const Graph::ModulePtrType GetParentModule(void);

    private:
        Graph::ModulePtrType parent_module;
        std::vector<Graph::CallPtrType> connected_calls;
    };

    class Call {
    public:
        Call(int gui_id);
        ~Call();

        const int uid;

        std::string class_name;
        std::string description;
        std::string plugin_name;
        std::vector<std::string> functions;

        bool IsConnected(void);
        bool ConnectCallSlots(Graph::CallSlotPtrType call_slot_1, Graph::CallSlotPtrType call_slot_2);
        bool DisConnectCallSlots(void);
        const Graph::CallSlotPtrType GetCallSlot(Graph::CallSlotType type);

    private:
        std::map<Graph::CallSlotType, Graph::CallSlotPtrType> connected_call_slots;
    };

    class Module {
    public:
        Module(int gui_id);
        ~Module();

        const int uid;

        std::string class_name;
        std::string description;
        std::string plugin_name;
        bool is_view;

        std::vector<Graph::ParamSlot> param_slots;

        std::string name;
        std::string full_name;
        bool is_view_instance;

        struct Gui {
            std::string class_label;
            std::string name_label;
            ImVec2 position;
            ImVec2 size;
            bool update;
        } gui;

        bool AddCallSlot(Graph::CallSlotPtrType call_slot);
        bool RemoveAllCallSlots(void);
        const std::vector<Graph::CallSlotPtrType> GetCallSlots(Graph::CallSlotType type);
        const std::map<Graph::CallSlotType, std::vector<Graph::CallSlotPtrType>> GetCallSlots(void);

    private:
        std::map<Graph::CallSlotType, std::vector<Graph::CallSlotPtrType>> call_slots;
    };


    // GRAPH ------------------------------------------------------------------

    Graph(int graph_uid, const std::string& graph_name);

    virtual ~Graph(void);

    bool AddModule(Graph::ModuleStockType& stock_modules, const std::string& module_class_name);
    bool DeleteModule(int module_uid);

    bool AddCall(Graph::CallStockType& stock_calls, int call_idx, Graph::CallSlotPtrType call_slot_1,
        Graph::CallSlotPtrType call_slot_2);
    bool DeleteDisconnectedCalls(void);
    bool DeleteCall(int call_uid);

    const Graph::ModuleGraphType& GetGraphModules(void) { return this->modules; }
    const Graph::CallGraphType& GetGraphCalls(void) { return this->calls; }

    inline void SetName(const std::string& graph_name) { this->name = graph_name; }
    inline std::string& GetName(void) { return this->name; }

    inline bool IsDirty(void) const { return this->dirty_flag; }
    inline void ResetDirty(void) { this->dirty_flag = false; }

    inline int GetUID(void) const { return this->uid; }

private:
    // VARIABLES --------------------------------------------------------------

    Graph::ModuleGraphType modules;
    Graph::CallGraphType calls;

    const int uid;
    std::string name;
    bool dirty_flag;

    int generated_uid;

    // FUNCTIONS --------------------------------------------------------------

    int get_unique_id(void) { return (++this->generated_uid); }

    // ------------------------------------------------------------------------
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_H_INCLUDED