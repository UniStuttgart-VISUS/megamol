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

#include <vector>
#include <map>

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
    class GraphModule;
    class GraphCall;
    class GraphCallSlot;
    class GraphParamSlot;

    typedef std::shared_ptr<Graph::GraphCallSlot> CallSlotPtr;
    typedef std::shared_ptr<Graph::GraphCall> CallPtr;
    typedef std::shared_ptr<Graph::GraphModule> ModulePtr;

    typedef std::vector<ModulePtr> ModuleGraphType;
    typedef std::vector<CallPtr> CallGraphType;

    class GraphParamSlot {
    public:
        GraphParamSlot() {}
        ~GraphParamSlot() {}

        std::string class_name;
        std::string description;
        Graph::ParamType type;

        std::string full_name;

    private:

    };

    class GraphCallSlot {
    public:
        GraphCallSlot() {
            this->parent_module.reset();
            connected_calls.clear();
        }
        ~GraphCallSlot() {}

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

    class GraphCall {
    public:
        GraphCall() {
            this->connected_call_slots.clear();
            this->connected_call_slots.emplace(Graph::CallSlotType::CALLER, nullptr);
            this->connected_call_slots.emplace(Graph::CallSlotType::CALLEE, nullptr);
        }
        ~GraphCall() {}

        std::string class_name;
        std::string description;
        std::string plugin_name;
        std::vector<std::string> functions;

        struct Gui {
            int id;
        } gui;

        // Functions ------------------

        bool IsConnected(void);
        bool ConnectCallSlot(Graph::CallSlotType type, Graph::CallSlotPtr call_slot);
        bool DisConnectCallSlot(Graph::CallSlotType type);
        bool DisConnectCallSlots(void);
        const Graph::CallSlotPtr GetCallSlot(Graph::CallSlotType type);

    private:

        std::map<Graph::CallSlotType, Graph::CallSlotPtr> connected_call_slots;
    };

    class GraphModule {
    public:
        GraphModule() {
            this->call_slots.clear();
            this->call_slots.emplace(Graph::CallSlotType::CALLER, std::vector<Graph::CallSlotPtr>());
            this->call_slots.emplace(Graph::CallSlotType::CALLEE, std::vector<Graph::CallSlotPtr>());
        }
        ~GraphModule() {}

        std::string class_name;
        std::string description;
        std::string plugin_name;
        bool is_view;

        std::vector<Graph::GraphParamSlot> param_slots;

        std::string name;
        std::string full_name;
        std::string instance;

        struct Gui {
            int id;
            ImVec2 position;
            ImVec2 size;
        } gui;

        // Functions ------------------

        bool AddCallSlot(Graph::CallSlotType type, Graph::CallSlotPtr call_slot);
        bool RemoveCallSlot(Graph::CallSlotType type, Graph::CallSlotPtr call_slot);
        bool RemoveAllCallSlot(Graph::CallSlotType type);
        bool RemoveAllCallSlot(void);
        const std::vector<Graph::CallSlotPtr> GetCallSlots(Graph::CallSlotType type);
        const std::map<Graph::CallSlotType, std::vector<Graph::CallSlotPtr>> GetCallSlots(void);

    private:

        std::map<Graph::CallSlotType, std::vector<Graph::CallSlotPtr>> call_slots;
    };

            
    // GRAPH ------------------------------------------------------------------

    Graph(void);

    virtual ~Graph(void);

    bool AddModule(const std::string& module_class_name);
    bool DeleteModule(int gui_id);

    bool AddCall(const std::string& call_class_name, CallSlotPtr caller, CallSlotPtr callee);
    bool DeleteCall(int gui_id);

    bool UpdateAvailableModulesCallsOnce(const megamol::core::CoreInstance* core_instance);

    inline const ModuleStockType& GetAvailableModulesList(void) const { return this->modules_stock; }

    inline const ModuleGraphType& GetGraphModules(void) { return this->modules_graph; }
    inline const CallGraphType& GetGraphCalls(void) { return this->calls_graph; }

    // Selected call slot -------------
    inline bool IsCallSlotSelected(void) const {
        return (this->selected_call_slot != nullptr);
    }

    bool SetSelectedCallSlot(CallSlotPtr call_slot) {
        if (call_slot == nullptr) {
            vislib::sys::Log::DefaultLog.WriteWarn("Pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        }
        this->selected_call_slot = call_slot;
        return true;
    }

    inline CallSlotPtr GetSelectedCallSlot(void) const {
        return this->selected_call_slot;
    }

    inline void ResetSelectedCallSlot(void) {
        this->selected_call_slot = nullptr;
    }
    // --------------------------------

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

    CallSlotPtr selected_call_slot;

    // FUNCTIONS --------------------------------------------------------------

    bool read_call_data(Graph::StockCall& call, const std::shared_ptr<const megamol::core::factories::CallDescription> call_desc); 
    bool read_module_data(Graph::StockModule& mod, const std::shared_ptr<const megamol::core::factories::ModuleDescription> mod_desc);
    
    // ------------------------------------------------------------------------
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_H_INCLUDED