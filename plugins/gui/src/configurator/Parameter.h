/*
 * Parameter.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_PARAMETER_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_PARAMETER_H_INCLUDED


#include "FileUtils.h"
#include "GUIUtils.h"
#include "TransferFunctionEditor.h"

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

// Used for platform independent clipboard (ImGui so far only provides windows implementation)
#ifdef GUI_USE_GLFW
#    include "GLFW/glfw3.h"
#endif


namespace megamol {
namespace gui {
namespace configurator {

// Forward declarations
class Parameter;
class Call;
class CallSlot;
class Module;
typedef std::shared_ptr<Call> CallPtrType;
typedef std::shared_ptr<CallSlot> CallSlotPtrType;
typedef std::shared_ptr<Module> ModulePtrType;

// Types
typedef std::shared_ptr<Parameter> ParamPtrType;


/** ************************************************************************
 * Defines GUI parameter presentation.
 */
class ParameterPresentation : public megamol::core::param::AbstractParamPresentation {
public:

    enum WidgetScope { GLOBAL, LOCAL };

    ParameterPresentation(ParamType type);

    ~ParameterPresentation(void);

    bool Present(Parameter& inout_param, WidgetScope scope);

    float GetHeight(Parameter& inout_param);

    bool expert;

private:
    std::string help;
    std::string description;
    megamol::gui::GUIUtils utils;
    megamol::gui::FileUtils file_utils;
    bool show_tf_editor;
    megamol::gui::TransferFunctionEditor tf_editor;
    std::variant<std::monostate, std::string, int, float, glm::vec2, glm::vec3, glm::vec4> widget_store;
    const std::string float_format;
    float height;
    UINT set_focus;

    bool present_parameter(Parameter& inout_parameter, WidgetScope scope);

    // Local widgets
    bool widget_button(const std::string& label, const megamol::core::view::KeyCode& keycode);
    bool widget_bool(const std::string& label, bool& value);
    bool widget_string(const std::string& label, std::string& value);
    bool widget_color(const std::string& label, glm::vec4& value);
    bool widget_enum(const std::string& label, int& value, EnumStorageType storage);
    bool widget_flexenum(const std::string& label, std::string& value, megamol::core::param::FlexEnumParam::Storage_t storage);
    bool widget_filepath(const std::string& label, std::string& value);
    bool widget_ternary(const std::string& label, vislib::math::Ternary& value);
    bool widget_int(const std::string& label, int& value, int min, int max);
    bool widget_float(const std::string& label, float& value, float min, float max);
    bool widget_vector2f(const std::string& label, glm::vec2& value, glm::vec2 min, glm::vec2 max);
    bool widget_vector3f(const std::string& label, glm::vec3& value, glm::vec3 min, glm::vec3 max);
    bool widget_vector4f(const std::string& label, glm::vec4& value, glm::vec4 min, glm::vec4 max);
    // Local and global widgets
    bool widget_pinvaluetomouse(const std::string& label, const std::string& value, WidgetScope scope);
    bool widget_transfer_function_editor(Parameter& inout_parameter, WidgetScope scope);
};


/** ************************************************************************
 * Defines parameter data structure for graph.
 */
class Parameter {
public:

    typedef std::variant<std::monostate,             // default  BUTTON
        bool,                                        // BOOL
        float,                                       // FLOAT
        int,                                         // INT      ENUM
        std::string,                                 // STRING   TRANSFERFUNCTION    FILEPATH    FLEXENUM
        vislib::math::Ternary,                       // TERNARY
        glm::vec2,                                   // VECTOR2F
        glm::vec3,                                   // VECTOR3F
        glm::vec4                                    // VECTOR4F, COLOR
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
        ParamType type;
        std::string default_value;
        MinType minval;
        MaxType maxval;
        StroageType storage;
        bool gui_visibility;
        bool gui_read_only;
        PresentType gui_presentation;
    };

    Parameter(ImGuiID uid, ParamType type, StroageType store, MinType min, MaxType max);
    ~Parameter(void);

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
            if (this->type == ParamType::FLEXENUM) {
                auto storage = this->GetStorage<megamol::core::param::FlexEnumParam::Storage_t>();
                storage.insert(std::get<std::string>(this->value));
                this->SetStorage(storage);
            }

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

    inline bool GUI_Present(ParameterPresentation::WidgetScope scope) { return this->present.Present(*this, scope); }

    inline void GUI_SetVisibility(bool visible) { this->present.SetGUIVisible(visible); }
    inline void GUI_SetReadOnly(bool readonly) { this->present.SetGUIReadOnly(readonly); }
    inline void GUI_SetPresentation(PresentType presentation) { this->present.SetGUIPresentation(presentation); }
    inline void GUI_SetExpert(bool expert) { this->present.expert = expert; }

    inline bool GUI_GetVisibility(void) { return this->present.IsGUIVisible(); }
    inline bool GUI_GetReadOnly(void) { return this->present.IsGUIReadOnly(); }
    inline PresentType GUI_GetPresentation(void) { return this->present.GetGUIPresentation(); }

    inline float GUI_GetHeight(void) { return this->present.GetHeight(*this); }

private:
    // VARIABLES --------------------------------------------------------------

    MinType minval;
    MaxType maxval;
    StroageType storage;
    ValueType value;

    ValueType default_value;
    bool default_value_mismatch;

    ParameterPresentation present;
};

} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_PARAMETER_H_INCLUDED
