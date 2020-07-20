/*
 * Parameter.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_PARAMETER_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_PARAMETER_H_INCLUDED


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

#include <variant>

#include "FileUtils.h"
#include "GUIUtils.h"
#include "TransferFunctionEditor.h"

// Used for platform independent clipboard (ImGui so far only provides windows implementation)
#ifdef GUI_USE_GLFW
#    include "GLFW/glfw3.h"
#endif


namespace megamol {
namespace gui {
namespace configurator {

// Forward declaration
class Call;
class CallSlot;
class Module;
class Parameter;

// Pointer types to classes
typedef std::shared_ptr<Parameter> ParamPtrType;
typedef std::shared_ptr<Call> CallPtrType;
typedef std::shared_ptr<CallSlot> CallSlotPtrType;
typedef std::shared_ptr<Module> ModulePtrType;


/**
 * Defines parameter data structure for graph.
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

    typedef std::map<int, std::string> EnumStorageType;

    typedef std::variant<std::monostate,             // default  BUTTON
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
        ValueType;

    typedef std::variant<std::monostate, // default (unused/unavailable)
        float,                           // FLOAT
        int,                             // INT
        glm::vec2,                       // VECTOR_2f
        glm::vec3,                       // VECTOR_3f
        glm::vec4                        // VECTOR_4f
        >
        MinType;

    typedef std::variant<std::monostate, // default (unused/unavailable)
        float,                           // FLOAT
        int,                             // INT
        glm::vec2,                       // VECTOR_2f
        glm::vec3,                       // VECTOR_3f
        glm::vec4                        // VECTOR_4f
        >
        MaxType;

    typedef std::variant<std::monostate,               // default (unused/unavailable)
        megamol::core::view::KeyCode,                  // BUTTON
        EnumStorageType,                               // ENUM
        megamol::core::param::FlexEnumParam::Storage_t // FLEXENUM
        >
        StroageType;

    struct StockParameter {
        std::string full_name;
        std::string description;
        Parameter::ParamType type;
        std::string default_value;
        MinType minval;
        MaxType maxval;
        StroageType storage;
    };

    Parameter(ImGuiID uid, ParamType type, StroageType store, MinType min, MaxType max);
    ~Parameter() {}

    const ImGuiID uid;
    const ParamType type;

    // Init when adding parameter from stock
    std::string full_name;
    std::string description;

    // Get ----------------------------------
    std::string GetName(void) {
        std::string name = this->full_name;
        auto idx = this->full_name.rfind(':');
        if (idx != std::string::npos) {
            name = name.substr(idx + 1);
        }
        return name;
    }
    std::string GetNameSpace(void) {
        std::string name_space = "";
        auto idx = this->full_name.rfind(':');
        if (idx != std::string::npos) {
            name_space = this->full_name.substr(0, idx - 1);
            name_space.erase(std::remove(name_space.begin(), name_space.end(), ':'), name_space.end());
        }
        return name_space;
    }

    std::string GetValueString(void);

    ValueType& GetValue(void) { return this->value; }

    template <typename T> const T& GetMinValue(void) const { return std::get<T>(this->minval); }

    template <typename T> const T& GetMaxValue(void) const { return std::get<T>(this->maxval); }

    template <typename T> const T& GetStorage(void) const { return std::get<T>(this->storage); }

    bool DefaultValueMismatch(void) { return this->default_value_mismatch; }

    // SET ----------------------------------
    bool SetValueString(const std::string& val_str, bool set_default_val = false);

    template <typename T> void SetValue(T val, bool set_default_val = false) {
        if (std::holds_alternative<T>(this->value)) {
            this->value = val;
            if (set_default_val) {
                this->default_value = val;
                this->default_value_mismatch = false;
            } else {
                try {
                    this->default_value_mismatch = (std::get<T>(this->default_value) != val);
                } catch (...) {
                }
            }
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

    // GUI Presentation -------------------------------------------------------

    inline bool GUI_Present(void) { return this->present.Present(*this); }

    inline void GUI_SetLabelVisibility(bool visible) { this->present.visible = visible; }
    inline void GUI_SetReadOnly(bool readonly) { this->present.read_only = readonly; }
    inline void GUI_SetExpert(bool expert) { this->present.expert = expert; }

    inline float GUI_GetHeight(void) { return this->present.GetHeight(*this); }

private:
    // VARIABLES --------------------------------------------------------------

    MinType minval;
    MaxType maxval;
    StroageType storage;
    ValueType value;

    ValueType default_value;
    bool default_value_mismatch;

    /** ************************************************************************
     * Defines GUI parameter presentation.
     */
    class Presentation {
    public:
        enum Presentations : size_t { DEFAULT = 0, _COUNT_ = 1 };

        Presentation(void);

        ~Presentation(void);

        bool Present(Parameter& inout_param);

        float GetHeight(Parameter& inout_param);

        bool read_only;
        bool visible;
        bool expert;

    private:
        Presentations presentations;
        std::string help;
        megamol::gui::GUIUtils utils;
        megamol::gui::FileUtils file_utils;
        bool show_tf_editor;
        megamol::gui::TransferFunctionEditor tf_editor;
        std::variant<std::monostate, std::string, int, float, glm::vec2, glm::vec3, glm::vec4> widget_store;
        const std::string float_format;
        float height;

        void present_prefix(void);
        void present_value_DEFAULT(Parameter& inout_param);
        // void present_value_PIN_VALUE_TO_MOUSE(Parameter& param);
        void present_postfix(Parameter& inout_param);
        bool presentation_button(void);
        void transfer_function_edit(Parameter& inout_param);

    } present;
};


} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_PARAMETER_H_INCLUDED
