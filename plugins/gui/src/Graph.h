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

#include "mmcore/CoreInstance.h"
#include "mmcore/Module.h"
#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "utility/plugins/PluginManager.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/TernaryParam.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"


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

    // STOCK DATA STRUCTURE ---------------------------------------------------

    struct StockParamSlot {
        std::string class_name;
        std::string description;
        Graph::ParamType type;
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

    typedef std::shared_ptr<Graph::CallSlot> CallSlotPtr;
    typedef std::shared_ptr<Graph::Call> CallPtr;
    typedef std::shared_ptr<Graph::Module> ModulePtr;

    typedef std::vector<ModulePtr> ModuleGraphType;
    typedef std::vector<CallPtr> CallGraphType;

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
        CallSlot(int gui_id) : gui_uid(gui_id) {
            this->parent_module.reset();
            connected_calls.clear();
        }
        ~CallSlot() {}

        const int gui_uid;

        std::string name;
        std::string description;
        std::vector<size_t> compatible_call_idxs; // (Storing only indices of compatible calls for faster comparison.)
        Graph::CallSlotType type;

        // Functions ------------------

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
        Call(int gui_id) : gui_uid(gui_id) {
            this->connected_call_slots.clear();
            this->connected_call_slots.emplace(Graph::CallSlotType::CALLER, nullptr);
            this->connected_call_slots.emplace(Graph::CallSlotType::CALLEE, nullptr);
        }
        ~Call() {}

        const int gui_uid;

        std::string class_name;
        std::string description;
        std::string plugin_name;
        std::vector<std::string> functions;

        // Functions ------------------

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
        Module(int gui_id) : gui_uid(gui_id) {
            this->call_slots.clear();
            this->call_slots.emplace(Graph::CallSlotType::CALLER, std::vector<Graph::CallSlotPtr>());
            this->call_slots.emplace(Graph::CallSlotType::CALLEE, std::vector<Graph::CallSlotPtr>());
        }
        ~Module() {}

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

        // Functions ------------------

        bool AddCallSlot(Graph::CallSlotPtr call_slot);
        bool RemoveCallSlot(Graph::CallSlotPtr call_slot);
        bool RemoveAllCallSlots(Graph::CallSlotType type);
        bool RemoveAllCallSlots(void);
        const std::vector<Graph::CallSlotPtr> GetCallSlots(Graph::CallSlotType type);
        const std::map<Graph::CallSlotType, std::vector<Graph::CallSlotPtr>> GetCallSlots(void);

    private:
        std::map<Graph::CallSlotType, std::vector<Graph::CallSlotPtr>> call_slots;
    };

    struct GraphData {
        ModuleGraphType modules;
        CallGraphType calls;
    };


    // GRAPH ------------------------------------------------------------------

    Graph(void);

    virtual ~Graph(void);

    bool AddModule(const std::string& module_class_name);
    bool DeleteModule(int uid);

    bool AddCall(int call_idx, CallSlotPtr call_slot_1, CallSlotPtr call_slot_2);
    bool DeleteCall(int uid);

    bool UpdateAvailableModulesCallsOnce(const megamol::core::CoreInstance* core_instance);

    inline const ModuleStockType& GetModulesStock(void) const { return this->modules_stock; }
    inline const CallStockType& GetCallsStock(void) const { return this->calls_stock; }

    inline const ModuleGraphType& GetGraphModules(void) { return this->modules_graph; }
    inline const CallGraphType& GetGraphCalls(void) { return this->calls_graph; }

    int GetCompatibleCallIndex(CallSlotPtr call_slot_1, CallSlotPtr call_slot_2);
    int GetCompatibleCallIndex(CallSlotPtr call_slot_1, StockCallSlot stock_call_slot_2);

    /**
     * Only used for prototype to be able to store current graph to lua project file.
     * Later use FileUtils->SaveProjectFile provided in GUI menu.
     */
    bool PROTOTYPE_SaveGraph(std::string project_filename, megamol::core::CoreInstance* cor_iInstance);

private:
    // VARIABLES --------------------------------------------------------------

    ModuleGraphType modules_graph;
    CallGraphType calls_graph;

    ModuleStockType modules_stock;
    CallStockType calls_stock;

    std::vector<GraphData> graphs;

    int uid;

    // FUNCTIONS --------------------------------------------------------------

    bool read_call_data(
        Graph::StockCall& call, const std::shared_ptr<const megamol::core::factories::CallDescription> call_desc);
    bool read_module_data(
        Graph::StockModule& mod, const std::shared_ptr<const megamol::core::factories::ModuleDescription> mod_desc);

    int get_unique_id(void) { return (++this->uid); }

    // ------------------------------------------------------------------------
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_H_INCLUDED