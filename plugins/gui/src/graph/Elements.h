/*
 * Elements.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_ELEMENTS_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_ELEMENTS_H_INCLUDED


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

#include <map>
#include <variant>
#include <vector>

#include "Presentations.h"


namespace megamol {
namespace gui {
namespace graph {

// GRAPH DATA STRUCTURE ---------------------------------------------------

// Forward declaration
class Parameter;
class CallSlot;
class Call;
class Module;

// Pointer types to classes
typedef std::shared_ptr<Parameter> ParamPtrType;
typedef std::shared_ptr<CallSlot> CallSlotPtrType;
typedef std::shared_ptr<Call> CallPtrType;
typedef std::shared_ptr<Module> ModulePtrType;


/**
 * Defines parameter data structure.
 */
class Parameter {
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

    Parameter(int uid, ParamType type);
    ~Parameter() {}

    const int uid;
    const ParamType type;

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
    ParamPresentations present;

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


/**
 * Defines call slot data structure.
 */
class CallSlot {
public:
    enum CallSlotType { CALLEE, CALLER };

    CallSlot(int uid);
    ~CallSlot();

    const int uid;

    std::string name;
    std::string description;
    std::vector<size_t> compatible_call_idxs; // (Storing only indices of compatible calls for faster comparison.)
    CallSlotType type;

    bool CallsConnected(void) const;
    bool ConnectCall(CallPtrType call);
    bool DisConnectCall(int call_uid, bool called_by_call);
    bool DisConnectCalls(void);
    const std::vector<CallPtrType> GetConnectedCalls(void);

    bool ParentModuleConnected(void) const;
    bool ConnectParentModule(ModulePtrType parent_module);
    bool DisConnectParentModule(void);
    const ModulePtrType GetParentModule(void);

    CallSlotPresentations present;

private:
    ModulePtrType parent_module;
    std::vector<CallPtrType> connected_calls;
};


/**
 * Defines call data structure.
 */
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
    bool ConnectCallSlots(CallSlotPtrType call_slot_1, CallSlotPtrType call_slot_2);
    bool DisConnectCallSlots(void);
    const CallSlotPtrType GetCallSlot(CallSlot::CallSlotType type);

    CallPresentations present;

private:
    std::map<CallSlot::CallSlotType, CallSlotPtrType> connected_call_slots;
};


/**
 * Defines module data structure.
 */
class Module {
public:
    Module(int uid);
    ~Module();

    const int uid;

    std::string class_name;
    std::string description;
    std::string plugin_name;
    bool is_view;

    std::vector<Parameter> param_slots;

    std::string name;
    std::string full_name;
    bool is_view_instance;

    bool AddCallSlot(CallSlotPtrType call_slot);
    bool RemoveAllCallSlots(void);
    const std::vector<CallSlotPtrType> GetCallSlots(CallSlot::CallSlotType type);
    const std::map<CallSlot::CallSlotType, std::vector<CallSlotPtrType>> GetCallSlots(void);

    ModulePresentations present;

private:
    std::map<CallSlot::CallSlotType, std::vector<CallSlotPtrType>> call_slots;
};

} // namespace graph
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_ELEMENTS_H_INCLUDED