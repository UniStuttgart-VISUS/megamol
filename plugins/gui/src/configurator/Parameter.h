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
    /*
     * Globally scoped widgets (widget parts) are always called each frame.
     * Locally scoped widgets (widget parts) are only called if respective parameter appears in GUI.
     */
    enum WidgetScope { GLOBAL, LOCAL };

    // VARIABLES --------------------------------------------------------------
    
    bool expert;
        
    // FUCNTIONS --------------------------------------------------------------
    
    ParameterPresentation(ParamType type);
    ~ParameterPresentation(void);

    void SetTransferFunctionEditorHash(size_t hash) { this->tf_editor_hash = hash; }
    inline void ConnectExternalTransferFunctionEditor(std::shared_ptr<megamol::gui::TransferFunctionEditor> tfe_ptr) {
        this->tf_editor_ptr = tfe_ptr;
        this->external_tf_editor = true;
    }

private:
    // VARIABLES --------------------------------------------------------------
    
    std::string help;
    std::string description;
    megamol::gui::GUIUtils utils;
    megamol::gui::FileUtils file_utils;
    std::variant<std::monostate, std::string, int, float, glm::vec2, glm::vec3, glm::vec4> widget_store;
    const std::string float_format;
    float height;
    UINT set_focus;

    std::shared_ptr<megamol::gui::TransferFunctionEditor> tf_editor_ptr;
    bool external_tf_editor;
    bool show_tf_editor;
    size_t tf_editor_hash;

    // FUNCTIONS --------------------------------------------------------------
    
    friend void Parameter::GUI_Present(ParameterPresentation::WidgetScope scope) ; 
    bool Present(Parameter& inout_param, WidgetScope scope);
    
    friend float Parameter::GUI_GetHeight(void);
    float GetHeight(Parameter& inout_param);
        
    bool present_parameter(Parameter& inout_parameter, WidgetScope scope);

    bool widget_button(WidgetScope scope, const std::string& labelel, const megamol::core::view::KeyCode& keycode);
    bool widget_bool(WidgetScope scope, const std::string& labelel, bool& value);
    bool widget_string(WidgetScope scope, const std::string& labelel, std::string& value);
    bool widget_color(WidgetScope scope, const std::string& labelel, glm::vec4& value);
    bool widget_enum(WidgetScope scope, const std::string& labelel, int& value, EnumStorageType storage);
    bool widget_flexenum(WidgetScope scope, const std::string& label, std::string& value,
        megamol::core::param::FlexEnumParam::Storage_t storage);
    bool widget_filepath(WidgetScope scope, const std::string& labelel, std::string& value);
    bool widget_ternary(WidgetScope scope, const std::string& labelel, vislib::math::Ternary& value);
    bool widget_int(WidgetScope scope, const std::string& labelel, int& value, int min, int max);
    bool widget_float(WidgetScope scope, const std::string& labelel, float& value, float min, float max);
    bool widget_vector2f(WidgetScope scope, const std::string& labelel, glm::vec2& value, glm::vec2 min, glm::vec2 max);
    bool widget_vector3f(WidgetScope scope, const std::string& labelel, glm::vec3& value, glm::vec3 min, glm::vec3 max);
    bool widget_vector4f(WidgetScope scope, const std::string& labelel, glm::vec4& value, glm::vec4 min, glm::vec4 max);
    bool widget_pinvaluetomouse(WidgetScope scope, const std::string& label, const std::string& value);
    bool widget_transfer_function_editor(WidgetScope scope, Parameter& inout_parameter);
};


/** ************************************************************************
 * Defines parameter data structure for graph.
 */
class Parameter {
public:
    typedef std::variant<std::monostate, // default  BUTTON
        bool,                            // BOOL
        float,                           // FLOAT
        int,                             // INT      ENUM
        std::string,                     // STRING   TRANSFERFUNCTION    FILEPATH    FLEXENUM
        vislib::math::Ternary,           // TERNARY
        glm::vec2,                       // VECTOR2F
        glm::vec3,                       // VECTOR3F
        glm::vec4                        // VECTOR4F, COLOR
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

    // VARIABLES --------------------------------------------------------------
    
    const ImGuiID uid;
    const ParamType type;
    ParameterPresentation present;
    
    // Init when adding parameter from stock
    std::string full_name;
    std::string description;

    // FUNCTIONS --------------------------------------------------------------
    
    Parameter(ImGuiID uid, ParamType type, StroageType store, MinType min, MaxType max);
    ~Parameter(void);

    bool IsDirty(void) { return this->dirty; }
    void ResetDirty(void) { this->dirty = false; }
    void ForceSetDirty(void) { this->dirty = true; }

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

    const size_t GetTransferFunctionHash(void) const { return this->string_hash; }

    // SET ----------------------------------
    
    bool SetValueString(const std::string& val_str, bool set_default_val = false);

    template <typename T> void SetValue(T val, bool set_default_val = false) {
        if (std::holds_alternative<T>(this->value)) {
            // Set value
            if (std::get<T>(this->value) != val) {
                this->value = val;
                this->dirty = true;
            }
            // Check for new flex enum entry
            if (this->type == ParamType::FLEXENUM) {
                auto storage = this->GetStorage<megamol::core::param::FlexEnumParam::Storage_t>();
                storage.insert(std::get<std::string>(this->value));
                this->SetStorage(storage);
            }
            // Calculate hash for parameters using string
            if constexpr (std::is_same_v<T, std::string>) {
                this->string_hash = std::hash<std::string>()(val);
            }
            // Check default value
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

    // Presentation ----------------------------------------------------

    inline bool GUI_Present(ParameterPresentation::WidgetScope scope) { return this->present.Present(*this, scope); }
    inline float GUI_GetHeight(void) { return this->present.GetHeight(*this); }

private:
    // VARIABLES --------------------------------------------------------------

    MinType minval;
    MaxType maxval;
    StroageType storage;
    ValueType value;
    size_t string_hash;
    ValueType default_value;
    bool default_value_mismatch;
    bool dirty;
};


// Static interface functions for core and GUI parameters //////////////////////////////////////////

static bool ReadCoreParameter(
    megamol::core::param::ParamSlot& in_param_slot, megamol::gui::configurator::Parameter::StockParameter& out_param) {

    auto parameter_ptr = in_param_slot.Parameter();
    if (parameter_ptr.IsNull()) {
        return false;
    }

    out_param.full_name = std::string(in_param_slot.Name().PeekBuffer());
    out_param.description = std::string(in_param_slot.Description().PeekBuffer());
    out_param.gui_visibility = parameter_ptr->IsGUIVisible();
    out_param.gui_read_only = parameter_ptr->IsGUIReadOnly();
    auto core_param_presentation = static_cast<size_t>(parameter_ptr->GetGUIPresentation());
    out_param.gui_presentation = static_cast<PresentType>(core_param_presentation);

    if (auto* p_ptr = in_param_slot.Param<core::param::ButtonParam>()) {
        out_param.type = ParamType::BUTTON;
        out_param.storage = p_ptr->GetKeyCode();
    } else if (auto* p_ptr = in_param_slot.Param<core::param::BoolParam>()) {
        out_param.type = ParamType::BOOL;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::ColorParam>()) {
        out_param.type = ParamType::COLOR;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::EnumParam>()) {
        out_param.type = ParamType::ENUM;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
        EnumStorageType map;
        auto psd_map = p_ptr->getMap();
        auto iter = psd_map.GetConstIterator();
        while (iter.HasNext()) {
            auto pair = iter.Next();
            map.emplace(pair.Key(), std::string(pair.Value().PeekBuffer()));
        }
        out_param.storage = map;
    } else if (auto* p_ptr = in_param_slot.Param<core::param::FilePathParam>()) {
        out_param.type = ParamType::FILEPATH;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::FlexEnumParam>()) {
        out_param.type = ParamType::FLEXENUM;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
        out_param.storage = p_ptr->getStorage();
    } else if (auto* p_ptr = in_param_slot.Param<core::param::FloatParam>()) {
        out_param.type = ParamType::FLOAT;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
        out_param.minval = p_ptr->MinValue();
        out_param.maxval = p_ptr->MaxValue();
    } else if (auto* p_ptr = in_param_slot.Param<core::param::IntParam>()) {
        out_param.type = ParamType::INT;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
        out_param.minval = p_ptr->MinValue();
        out_param.maxval = p_ptr->MaxValue();
    } else if (auto* p_ptr = in_param_slot.Param<core::param::StringParam>()) {
        out_param.type = ParamType::STRING;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::TernaryParam>()) {
        out_param.type = ParamType::TERNARY;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::TransferFunctionParam>()) {
        out_param.type = ParamType::TRANSFERFUNCTION;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::Vector2fParam>()) {
        out_param.type = ParamType::VECTOR2F;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
        auto min = p_ptr->MinValue();
        out_param.minval = glm::vec2(min.X(), min.Y());
        auto max = p_ptr->MaxValue();
        out_param.maxval = glm::vec2(max.X(), max.Y());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::Vector3fParam>()) {
        out_param.type = ParamType::VECTOR3F;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
        auto min = p_ptr->MinValue();
        out_param.minval = glm::vec3(min.X(), min.Y(), min.Z());
        auto max = p_ptr->MaxValue();
        out_param.maxval = glm::vec3(max.X(), max.Y(), max.Z());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::Vector4fParam>()) {
        out_param.type = ParamType::VECTOR4F;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
        auto min = p_ptr->MinValue();
        out_param.minval = glm::vec4(min.X(), min.Y(), min.Z(), min.W());
        auto max = p_ptr->MaxValue();
        out_param.maxval = glm::vec4(max.X(), max.Y(), max.Z(), max.W());
    } else {
        vislib::sys::Log::DefaultLog.WriteError("Found unknown parameter type. Please extend parameter types "
                                                "for the configurator. [%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        out_param.type = ParamType::UNKNOWN;
        return false;
    }

    return true;
}


static bool ReadCoreParameter(megamol::core::param::ParamSlot& in_param_slot,
    megamol::gui::configurator::Parameter& out_param, const std::string& module_full_name) {

    bool type_error = false;

    auto parameter_ptr = in_param_slot.Parameter();
    if (parameter_ptr.IsNull()) {
        return false;
    }

    out_param.full_name = module_full_name + "::" + std::string(in_param_slot.Name().PeekBuffer());
    out_param.GUI_SetVisibility(parameter_ptr->IsGUIVisible());
    out_param.GUI_SetReadOnly(parameter_ptr->IsGUIReadOnly());
    out_param.GUI_SetPresentation(parameter_ptr->GetGUIPresentation());

    if (auto* p_ptr = in_param_slot.Param<core::param::ButtonParam>()) {
        if (out_param.type == ParamType::BUTTON) {
            out_param.SetStorage(p_ptr->GetKeyCode());
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_slot.Param<core::param::BoolParam>()) {
        if (out_param.type == ParamType::BOOL) {
            out_param.SetValue(p_ptr->Value());
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_slot.Param<core::param::ColorParam>()) {
        if (out_param.type == ParamType::COLOR) {
            auto value = p_ptr->Value();
            out_param.SetValue(glm::vec4(value[0], value[1], value[2], value[3]));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_slot.Param<core::param::EnumParam>()) {
        if (out_param.type == ParamType::ENUM) {
            out_param.SetValue(p_ptr->Value());
            EnumStorageType map;
            auto param_map = p_ptr->getMap();
            auto iter = param_map.GetConstIterator();
            while (iter.HasNext()) {
                auto pair = iter.Next();
                map.emplace(pair.Key(), std::string(pair.Value().PeekBuffer()));
            }
            out_param.SetStorage(map);
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_slot.Param<core::param::FilePathParam>()) {
        if (out_param.type == ParamType::FILEPATH) {
            out_param.SetValue(std::string(p_ptr->Value().PeekBuffer()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_slot.Param<core::param::FlexEnumParam>()) {
        if (out_param.type == ParamType::FLEXENUM) {
            out_param.SetValue(p_ptr->Value());
            out_param.SetStorage(p_ptr->getStorage());
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_slot.Param<core::param::FloatParam>()) {
        if (out_param.type == ParamType::FLOAT) {
            out_param.SetValue(p_ptr->Value());
            out_param.SetMinValue(p_ptr->MinValue());
            out_param.SetMaxValue(p_ptr->MaxValue());
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_slot.Param<core::param::IntParam>()) {
        if (out_param.type == ParamType::INT) {
            out_param.SetValue(p_ptr->Value());
            out_param.SetMinValue(p_ptr->MinValue());
            out_param.SetMaxValue(p_ptr->MaxValue());
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_slot.Param<core::param::StringParam>()) {
        if (out_param.type == ParamType::STRING) {
            out_param.SetValue(std::string(p_ptr->Value().PeekBuffer()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_slot.Param<core::param::TernaryParam>()) {
        if (out_param.type == ParamType::TERNARY) {
            out_param.SetValue(p_ptr->Value());
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_slot.Param<core::param::TransferFunctionParam>()) {
        if (out_param.type == ParamType::TRANSFERFUNCTION) {
            out_param.SetValue(p_ptr->Value());
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_slot.Param<core::param::Vector2fParam>()) {
        if (out_param.type == ParamType::VECTOR2F) {
            auto val = p_ptr->Value();
            out_param.SetValue(glm::vec2(val.X(), val.Y()));
            auto min = p_ptr->MinValue();
            out_param.SetMinValue(glm::vec2(min.X(), min.Y()));
            auto max = p_ptr->MaxValue();
            out_param.SetMaxValue(glm::vec2(max.X(), max.Y()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_slot.Param<core::param::Vector3fParam>()) {
        if (out_param.type == ParamType::VECTOR3F) {
            auto val = p_ptr->Value();
            out_param.SetValue(glm::vec3(val.X(), val.Y(), val.Z()));
            auto min = p_ptr->MinValue();
            out_param.SetMinValue(glm::vec3(min.X(), min.Y(), min.Z()));
            auto max = p_ptr->MaxValue();
            out_param.SetMaxValue(glm::vec3(max.X(), max.Y(), max.Z()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_slot.Param<core::param::Vector4fParam>()) {
        if (out_param.type == ParamType::VECTOR4F) {
            auto val = p_ptr->Value();
            out_param.SetValue(glm::vec4(val.X(), val.Y(), val.Z(), val.W()));
            auto min = p_ptr->MinValue();
            out_param.SetMinValue(glm::vec4(min.X(), min.Y(), min.Z(), min.W()));
            auto max = p_ptr->MaxValue();
            out_param.SetMaxValue(glm::vec4(max.X(), max.Y(), max.Z(), max.W()));
        } else {
            type_error = true;
        }
    } else {
        vislib::sys::Log::DefaultLog.WriteError(
            "Found unknown parameter type. Please extend parameter types for the configurator. "
            "[%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    if (type_error) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Mismatch of parameter types. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


static bool ReadCoreParameter(megamol::core::param::ParamSlot& in_param_slot,
    std::shared_ptr<megamol::gui::configurator::Parameter>& out_param, const std::string& module_full_name) {

    auto parameter_ptr = in_param_slot.Parameter();
    if (parameter_ptr.IsNull()) {
        return false;
    }

    out_param.reset();

    if (auto* p_ptr = in_param_slot.template Param<core::param::BoolParam>()) {
        out_param = std::make_shared<configurator::Parameter>(
            megamol::gui::GenerateUniqueID(), ParamType::BOOL, std::monostate(), std::monostate(), std::monostate());
        out_param->SetValue(p_ptr->Value());
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::ButtonParam>()) {
        out_param = std::make_shared<configurator::Parameter>(megamol::gui::GenerateUniqueID(), ParamType::BUTTON,
            p_ptr->GetKeyCode(), std::monostate(), std::monostate());
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::ColorParam>()) {
        out_param = std::make_shared<configurator::Parameter>(
            megamol::gui::GenerateUniqueID(), ParamType::COLOR, std::monostate(), std::monostate(), std::monostate());
        auto value = p_ptr->Value();
        out_param->SetValue(glm::vec4(value[0], value[1], value[2], value[3]));
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::TransferFunctionParam>()) {
        out_param = std::make_shared<configurator::Parameter>(megamol::gui::GenerateUniqueID(),
            ParamType::TRANSFERFUNCTION, std::monostate(), std::monostate(), std::monostate());
        out_param->SetValue(p_ptr->Value());
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::EnumParam>()) {
        EnumStorageType map;
        auto param_map = p_ptr->getMap();
        auto iter = param_map.GetConstIterator();
        while (iter.HasNext()) {
            auto pair = iter.Next();
            map.emplace(pair.Key(), std::string(pair.Value().PeekBuffer()));
        }
        out_param = std::make_shared<configurator::Parameter>(
            megamol::gui::GenerateUniqueID(), ParamType::ENUM, map, std::monostate(), std::monostate());
        out_param->SetValue(p_ptr->Value());
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::FlexEnumParam>()) {
        out_param = std::make_shared<configurator::Parameter>(megamol::gui::GenerateUniqueID(), ParamType::FLEXENUM,
            p_ptr->getStorage(), std::monostate(), std::monostate());
        out_param->SetValue(p_ptr->Value());
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::FloatParam>()) {
        out_param = std::make_shared<configurator::Parameter>(
            megamol::gui::GenerateUniqueID(), ParamType::FLOAT, std::monostate(), p_ptr->MinValue(), p_ptr->MaxValue());
        out_param->SetValue(p_ptr->Value());
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::IntParam>()) {
        out_param = std::make_shared<configurator::Parameter>(
            megamol::gui::GenerateUniqueID(), ParamType::INT, std::monostate(), p_ptr->MinValue(), p_ptr->MaxValue());
        out_param->SetValue(p_ptr->Value());
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::Vector2fParam>()) {
        auto min = p_ptr->MinValue();
        auto max = p_ptr->MaxValue();
        auto val = p_ptr->Value();
        out_param = std::make_shared<configurator::Parameter>(megamol::gui::GenerateUniqueID(), ParamType::VECTOR2F,
            std::monostate(), glm::vec2(min.X(), min.Y()), glm::vec2(max.X(), max.Y()));
        out_param->SetValue(glm::vec2(val.X(), val.Y()));
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::Vector3fParam>()) {
        auto min = p_ptr->MinValue();
        auto max = p_ptr->MaxValue();
        auto val = p_ptr->Value();
        out_param = std::make_shared<configurator::Parameter>(megamol::gui::GenerateUniqueID(), ParamType::VECTOR3F,
            std::monostate(), glm::vec3(min.X(), min.Y(), min.Z()), glm::vec3(max.X(), max.Y(), max.Z()));
        out_param->SetValue(glm::vec3(val.X(), val.Y(), val.Z()));
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::Vector4fParam>()) {
        auto min = p_ptr->MinValue();
        auto max = p_ptr->MaxValue();
        auto val = p_ptr->Value();
        out_param = std::make_shared<configurator::Parameter>(megamol::gui::GenerateUniqueID(), ParamType::VECTOR4F,
            std::monostate(), glm::vec4(min.X(), min.Y(), min.Z(), min.W()),
            glm::vec4(max.X(), max.Y(), max.Z(), max.W()));
        out_param->SetValue(glm::vec4(val.X(), val.Y(), val.Z(), val.W()));
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::TernaryParam>()) {
        out_param = std::make_shared<configurator::Parameter>(
            megamol::gui::GenerateUniqueID(), ParamType::TERNARY, std::monostate(), std::monostate(), std::monostate());
        out_param->SetValue(p_ptr->Value());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::StringParam>()) {
        out_param = std::make_shared<configurator::Parameter>(
            megamol::gui::GenerateUniqueID(), ParamType::STRING, std::monostate(), std::monostate(), std::monostate());
        out_param->SetValue(std::string(p_ptr->Value().PeekBuffer()));
    } else if (auto* p_ptr = in_param_slot.Param<core::param::FilePathParam>()) {
        out_param = std::make_shared<configurator::Parameter>(megamol::gui::GenerateUniqueID(), ParamType::FILEPATH,
            std::monostate(), std::monostate(), std::monostate());
        out_param->SetValue(std::string(p_ptr->Value().PeekBuffer()));
    } else {
        vislib::sys::Log::DefaultLog.WriteError(
            "Found unknown parameter type. Please extend parameter types for the configurator. "
            "[%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    out_param->full_name = module_full_name + "::" + std::string(in_param_slot.Name().PeekBuffer());
    out_param->description = std::string(in_param_slot.Description().PeekBuffer());
    out_param->GUI_SetVisibility(parameter_ptr->IsGUIVisible());
    out_param->GUI_SetReadOnly(parameter_ptr->IsGUIReadOnly());
    out_param->GUI_SetPresentation(parameter_ptr->GetGUIPresentation());

    return true;
}


static bool WriteCoreParameter(
    megamol::gui::configurator::Parameter& in_param, megamol::core::param::ParamSlot& out_param_slot) {
    bool type_error = false;

    auto parameter_ptr = out_param_slot.Parameter();
    if (parameter_ptr.IsNull()) {
        return false;
    }

    parameter_ptr->SetGUIVisible(in_param.GUI_IsVisible());
    parameter_ptr->SetGUIReadOnly(in_param.GUI_IsReadOnly());
    parameter_ptr->SetGUIPresentation(in_param.GUI_GetPresentation());

    if (auto* p_ptr = out_param_slot.Param<core::param::ButtonParam>()) {
        if (in_param.type == ParamType::BUTTON) {
            p_ptr->setDirty();
            // KeyCode can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_slot.Param<core::param::BoolParam>()) {
        if (in_param.type == ParamType::BOOL) {
            p_ptr->SetValue(std::get<bool>(in_param.GetValue()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_slot.Param<core::param::ColorParam>()) {
        if (in_param.type == ParamType::COLOR) {
            auto value = std::get<glm::vec4>(in_param.GetValue());
            p_ptr->SetValue(core::param::ColorParam::ColorType{value[0], value[1], value[2], value[3]});
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_slot.Param<core::param::EnumParam>()) {
        if (in_param.type == ParamType::ENUM) {
            p_ptr->SetValue(std::get<int>(in_param.GetValue()));
            // Map can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_slot.Param<core::param::FilePathParam>()) {
        if (in_param.type == ParamType::FILEPATH) {
            p_ptr->SetValue(vislib::StringA(std::get<std::string>(in_param.GetValue()).c_str()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_slot.Param<core::param::FlexEnumParam>()) {
        if (in_param.type == ParamType::FLEXENUM) {
            p_ptr->SetValue(std::get<std::string>(in_param.GetValue()));
            // Storage can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_slot.Param<core::param::FloatParam>()) {
        if (in_param.type == ParamType::FLOAT) {
            p_ptr->SetValue(std::get<float>(in_param.GetValue()));
            // Min and Max can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_slot.Param<core::param::IntParam>()) {
        if (in_param.type == ParamType::INT) {
            p_ptr->SetValue(std::get<int>(in_param.GetValue()));
            // Min and Max can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_slot.Param<core::param::StringParam>()) {
        if (in_param.type == ParamType::STRING) {
            p_ptr->SetValue(vislib::StringA(std::get<std::string>(in_param.GetValue()).c_str()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_slot.Param<core::param::TernaryParam>()) {
        if (in_param.type == ParamType::TERNARY) {
            p_ptr->SetValue(std::get<vislib::math::Ternary>(in_param.GetValue()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_slot.Param<core::param::TransferFunctionParam>()) {
        if (in_param.type == ParamType::TRANSFERFUNCTION) {
            p_ptr->SetValue(std::get<std::string>(in_param.GetValue()).c_str());
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_slot.Param<core::param::Vector2fParam>()) {
        if (in_param.type == ParamType::VECTOR2F) {
            auto value = std::get<glm::vec2>(in_param.GetValue());
            p_ptr->SetValue(vislib::math::Vector<float, 2>(value[0], value[1]));
            // Min and Max can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_slot.Param<core::param::Vector3fParam>()) {
        if (in_param.type == ParamType::VECTOR3F) {
            auto value = std::get<glm::vec3>(in_param.GetValue());
            p_ptr->SetValue(vislib::math::Vector<float, 3>(value[0], value[1], value[2]));
            // Min and Max can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_slot.Param<core::param::Vector4fParam>()) {
        if (in_param.type == ParamType::VECTOR4F) {
            auto value = std::get<glm::vec4>(in_param.GetValue());
            p_ptr->SetValue(vislib::math::Vector<float, 4>(value[0], value[1], value[2], value[3]));
            // Min and Max can not be changed
        } else {
            type_error = true;
        }
    } else {
        vislib::sys::Log::DefaultLog.WriteError(
            "Found unknown parameter type. Please extend parameter types for the configurator. "
            "[%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    if (type_error) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Mismatch of parameter types. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_PARAMETER_H_INCLUDED
