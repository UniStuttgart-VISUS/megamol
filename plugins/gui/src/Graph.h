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

    typedef std::vector<Graph::Module> ModuleListType;
    typedef std::vector<Graph::Call> CallListType;

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


    class ParamSlot {
    public:
        ParamSlot() {}
        ~ParamSlot() {}

        // Initialized on loading
        std::string class_name;
        std::string description;
        Graph::ParamType type;

        // Initilized after/on creation
        std::string full_name;

    private:

    };

    class CallSlot {
    public:
        CallSlot() {
            this->parent_module.reset();
            connected_calls.clear();
        }
        ~CallSlot() {}

        // Initialized on loading 
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

        // Initilized after/on creation
        Graph::ModulePtr parent_module;
        std::vector<Graph::CallPtr> connected_calls;
    };

    class Call {
    public:
        Call() {
            this->connected_call_slots.clear();
            this->connected_call_slots.emplace(Graph::CallSlotType::CALLER, nullptr);
            this->connected_call_slots.emplace(Graph::CallSlotType::CALLEE, nullptr);
        }
        ~Call() {}

        // Initialized on loading 
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

        // Initilized after/on creation
        std::map<Graph::CallSlotType, Graph::CallSlotPtr> connected_call_slots;
    };

    class Module {
    public:
        Module() {
            this->call_slots.clear();
            this->call_slots.emplace(Graph::CallSlotType::CALLER, std::vector<Graph::CallSlotPtr>());
            this->call_slots.emplace(Graph::CallSlotType::CALLEE, std::vector<Graph::CallSlotPtr>());
        }
        ~Module() {}

        // Initialized on loading 
        std::string class_name;
        std::string description;
        std::string plugin_name;
        std::vector<Graph::ParamSlot> param_slots;
        bool is_view;

        // Initilized after/on creation
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

        // Initialized on loading 
        std::map<Graph::CallSlotType, std::vector<Graph::CallSlotPtr>> call_slots;
    };
            
    // GRAPH ------------------------------------------------------------------

    Graph(void);

    virtual ~Graph(void);

    bool AddModule(const std::string& module_class_name);
    bool DeleteModule(int gui_id);

    bool AddCall(const std::string& call_class_name);
    bool DeleteCall(int gui_id);

    bool UpdateAvailableModulesCallsOnce(const megamol::core::CoreInstance* core_instance);

    inline const ModuleListType& GetAvailableModulesList(void) const { return this->modules_stock; }

    inline ModuleGraphType& GetGraphModules(void) { return this->modules_graph; }
    inline CallGraphType& GetGraphCalls(void) { return this->calls_graph; }

    inline bool IsCallSlotSelected(void) const {
        return (this->selected_call_slot != nullptr);
    }

    bool SetSelectedCallSlot(const std::string& module_full_name, const std::string& slot_name);

    inline CallSlotPtr GetSelectedCallSlot(void) {
        return this->selected_call_slot;
    }

    inline void ResetSelectedCallSlot(void) {
        this->selected_call_slot = nullptr;
    }

    /**
     * Only used for prototype to be able to store current graph to lua project file.
     * Later use FileUtils->SaveProjectFile provided in GUI menu.
     */
    bool PROTOTYPE_SaveGraph(std::string project_filename, megamol::core::CoreInstance* cor_iInstance);

private:

    // VARIABLES --------------------------------------------------------------

    ModuleGraphType modules_graph;
    CallGraphType calls_graph;

    ModuleListType modules_stock;
    CallListType calls_stock;

    CallSlotPtr selected_call_slot;

    // FUNCTIONS --------------------------------------------------------------

    bool read_call_data(Graph::Call& call, const std::shared_ptr<const megamol::core::factories::CallDescription> call_desc); 
    bool read_module_data(Graph::Module& mod, const std::shared_ptr<const megamol::core::factories::ModuleDescription> mod_desc);
    
    // ------------------------------------------------------------------------
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_H_INCLUDED