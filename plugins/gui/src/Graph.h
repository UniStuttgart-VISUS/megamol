/*
 * Graph.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_H_INCLUDED


#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/TernaryParam.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"

#include "vislib/sys/Log.h"

#include <iostream>
#include <map>
#include <variant>
#include <vector>

#include "GUIUtils.h"


namespace megamol {
namespace gui {

class Graph {
public:
    enum ParamType {
        BOOL,
        BUTTON,
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

    // ParamSlot --------------------------------
    class ParamSlot {
    public:
        ParamSlot(int uid, ParamType type);
        ~ParamSlot() {}

        const int uid;
        const Graph::ParamType type;

        std::string class_name;
        std::string description;

        std::string full_name;

        // Get ----------------------------------
        std::string GetValueString(void);

        template <typename T> const T& GetValue(void) const {
            try {
                return std::get<T>(this->value);
            } catch (std::bad_variant_access&) {
            }
        }

        template <typename T> const T& GetMinValue(void) const {
            try {
                return std::get<T>(this->minval);
            } catch (std::bad_variant_access&) {
            }
        }

        template <typename T> const T& GetMaxValue(void) const {
            try {
                return std::get<T>(this->maxval);
            } catch (std::bad_variant_access&) {
            }
        }

        template <typename T> const T& GetStorage(void) const {
            try {
                return std::get<T>(this->storage);
            } catch (std::bad_variant_access&) {
            }
        }

        // SET ----------------------------------
        template <typename T> void SetValue(T val) {
            if (std::holds_alternative<T>(this->value)) {
                this->value = val;
            } else {
                vislib::sys::Log::DefaultLog.WriteError(
                    "Bad variant access. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            }
        }

        template <typename T> void SetMinValue(T min) {
            if (std::holds_alternative<T>(this->minval)) {
                this->minval = min;
            } else {
                vislib::sys::Log::DefaultLog.WriteError(
                    "Bad variant access. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            }
        }

        template <typename T> void SetMaxValue(T max) {
            if (std::holds_alternative<T>(this->maxval)) {
                this->maxval = max;
            } else {
                vislib::sys::Log::DefaultLog.WriteError(
                    "Bad variant access. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            }
        }

        template <typename T> void SetStorage(T store) {
            if (std::holds_alternative<T>(this->storage)) {
                this->storage = store;
            } else {
                vislib::sys::Log::DefaultLog.WriteError(
                    "Bad variant access. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            }
        }

    private:
        std::variant<std::monostate, // default (unused/unavailable)
            float,                   // FLOAT
            int,                     // INT
            glm::vec2,               // VECTOR_2f
            glm::vec3,               // VECTOR_3f
            glm::vec4                // VECTOR_4f
            >
            minval;

        std::variant<std::monostate, // default (unused/unavailable)
            float,                   // FLOAT
            int,                     // INT
            glm::vec2,               // VECTOR_2f
            glm::vec3,               // VECTOR_3f
            glm::vec4                // VECTOR_4f
            >
            maxval;

        std::variant<std::monostate,                       // default (unused/unavailable)
            megamol::core::view::KeyCode,                  // BUTTON
            vislib::Map<int, vislib::TString>,             // ENUM
            megamol::core::param::FlexEnumParam::Storage_t // FLEXENUM
            >
            storage;

        std::variant<std::monostate,                     // default  BUTTON
            bool,                                        // BOOL
            megamol::core::param::ColorParam::ColorType, // COLOR
            float,                                       // FLOAT
            int,                                         // INT      ENUM
            std::string,                                 // STRING   TRANSFERFUNCTION    FILEPATH    FLEXENUM
            vislib::math::Ternary,                       // TERNARY
            glm::vec2,                                   // VECTOR2F
            glm::vec3,                                   // VECTOR3F
            glm::vec4                                    // VECTOR4F
            >
            value;
    };

    // CallSlot ---------------------------------
    class CallSlot {
    public:
        CallSlot(int uid);
        ~CallSlot();

        const int uid;

        std::string name;
        std::string description;
        std::vector<size_t> compatible_call_idxs; // (Storing only indices of compatible calls for faster comparison.)
        Graph::CallSlotType type;

        struct Gui {
            ImVec2 position;
        } gui;

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

    // Call -------------------------------------
    class Call {
    public:
        Call(int uid);
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

    // Module -----------------------------------
    class Module {
    public:
        Module(int uid);
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
        } gui;

        bool AddCallSlot(Graph::CallSlotPtrType call_slot);
        bool RemoveAllCallSlots(void);
        const std::vector<Graph::CallSlotPtrType> GetCallSlots(Graph::CallSlotType type);
        const std::map<Graph::CallSlotType, std::vector<Graph::CallSlotPtrType>> GetCallSlots(void);

    private:
        std::map<Graph::CallSlotType, std::vector<Graph::CallSlotPtrType>> call_slots;
    };


    // GRAPH ------------------------------------------------------------------

    Graph(const std::string& graph_name);

    virtual ~Graph(void);

    bool AddModule(const Graph::ModuleStockType& stock_modules, const std::string& module_class_name);
    bool DeleteModule(int module_uid);

    bool AddCall(const Graph::CallStockType& stock_calls, int call_idx, Graph::CallSlotPtrType call_slot_1,
        Graph::CallSlotPtrType call_slot_2);
    bool DeleteDisconnectedCalls(void);
    bool DeleteCall(int call_uid);

    const const Graph::ModuleGraphType& GetGraphModules(void) { return this->modules; }
    const const Graph::CallGraphType& GetGraphCalls(void) { return this->calls; }

    inline void SetName(const std::string& graph_name) { this->name = graph_name; }
    inline const std::string& GetName(void) { return this->name; }

    inline bool IsDirty(void) const { return this->dirty_flag; }
    inline void ResetDirty(void) { this->dirty_flag = false; }

    inline int GetUID(void) const { return this->uid; }

    int generate_unique_id(void) {
        ++this->generated_uid;
        std::cout << "UID: " << this->generated_uid << std::endl;
        return this->generated_uid;
    }

    struct Gui {
        float slot_radius;
        ImVec2 canvas_position;
        ImVec2 canvas_size;
        ImVec2 canvas_scrolling;
        float canvas_zooming;
        ImVec2 canvas_offset;
        bool show_grid;
        bool show_call_names;
        bool show_slot_names;
        int selected_module_uid;
        int selected_call_uid;
        int hovered_slot_uid;
        Graph::CallSlotPtrType selected_slot_ptr;
        int process_selected_slot;
    } gui;

private:
    // VARIABLES --------------------------------------------------------------

    Graph::ModuleGraphType modules;
    Graph::CallGraphType calls;

    // UIDs are unique within a graph
    const int uid;
    std::string name;
    bool dirty_flag;S

    // Global variable for unique id shared/accessible by all graphs.
    static int generated_uid;

    // ------------------------------------------------------------------------
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_H_INCLUDED