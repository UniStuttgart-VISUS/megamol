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

    // GRAPH DATA STRUCTURE ---------------------------------------------------

    // Forward declaration
    class Module;
    class Call;
    class CallSlot;
    class ParamSlot;

    typedef std::shared_ptr<Graph::CallSlot> CallSlotPtr;
    typedef std::shared_ptr<Graph::Call> CallPtr;
    typedef std::shared_ptr<Graph::Module> ModulePtr;

    typedef std::vector<Graph::ModulePtr> ModuleGraphType;
    typedef std::vector<Graph::CallPtr> CallGraphType;

    class ParamSlot {
    public:
        ParamSlot(int gui_id) : gui_uid(gui_id) {}
        ~ParamSlot() {}

        const int gui_uid;

        std::string class_name;
        std::string description;
        Graph::ParamType type;

        std::string full_name;

    private:
    };

    class CallSlot {
    public:
        CallSlot(int gui_id);
        ~CallSlot();

        const int gui_uid;

        std::string name;
        std::string description;
        std::vector<size_t> compatible_call_idxs; // (Storing only indices of compatible calls for faster comparison.)
        Graph::CallSlotType type;

        ImVec2 GetGuiPos(void);

        bool CallsConnected(void) const;
        bool ConnectCall(Graph::CallPtr call);
        bool DisConnectCall(Graph::CallPtr call);
        bool DisConnectCalls(void);
        const std::vector<Graph::CallPtr> GetConnectedCalls(void);

        bool ParentModuleConnected(void) const;
        bool AddParentModule(Graph::ModulePtr parent_module);
        bool RemoveParentModule(void);
        const Graph::ModulePtr GetParentModule(void);

    private:
        Graph::ModulePtr parent_module;
        std::vector<Graph::CallPtr> connected_calls;
    };

    class Call {
    public:
        Call(int gui_id);
        ~Call();

        const int gui_uid;

        std::string class_name;
        std::string description;
        std::string plugin_name;
        std::vector<std::string> functions;

        bool IsConnected(void);
        bool ConnectCallSlot(Graph::CallSlotPtr call_slot);
        bool DisConnectCallSlot(Graph::CallSlotType type);
        bool DisConnectCallSlots(void);
        const Graph::CallSlotPtr GetCallSlot(Graph::CallSlotType type);

    private:
        std::map<Graph::CallSlotType, Graph::CallSlotPtr> connected_call_slots;
    };

    class Module {
    public:
        Module(int gui_id);
        ~Module();

        const int gui_uid;

        std::string class_name;
        std::string description;
        std::string plugin_name;
        bool is_view;

        std::vector<Graph::ParamSlot> param_slots;

        std::string name;
        std::string full_name;
        std::string instance;

        struct Gui {
            ImVec2 position;
            ImVec2 size;
            bool initialized;
        } gui;

        bool AddCallSlot(Graph::CallSlotPtr call_slot);
        bool RemoveCallSlot(Graph::CallSlotPtr call_slot);
        bool RemoveAllCallSlots(Graph::CallSlotType type);
        bool RemoveAllCallSlots(void);
        const std::vector<Graph::CallSlotPtr> GetCallSlots(Graph::CallSlotType type);
        const std::map<Graph::CallSlotType, std::vector<Graph::CallSlotPtr>> GetCallSlots(void);

    private:
        std::map<Graph::CallSlotType, std::vector<Graph::CallSlotPtr>> call_slots;
    };


    // GRAPH ------------------------------------------------------------------

    Graph(void);

    virtual ~Graph(void);

    bool AddModule(const std::string& module_class_name);
    bool DeleteModule(int module_uid);

    bool AddCall(int call_idx, CallSlotPtr call_slot_1, CallSlotPtr call_slot_2);
    bool DeleteCall(int call_uid);

    const ModuleGraphType& GetGraphModules(void);
    const CallGraphType& GetGraphCalls(void);

private:
    // VARIABLES --------------------------------------------------------------

    ModuleGraphType modules;
    CallGraphType calls;

    // ------------------------------------------------------------------------
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_H_INCLUDED