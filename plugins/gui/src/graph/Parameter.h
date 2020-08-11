/*
 * Parameter.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_PARAMETER_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_PARAMETER_H_INCLUDED


#include "ParameterPresentation.h"

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


namespace megamol {
namespace gui {


// Forward declarations
class Parameter;
class Call;
class CallSlot;
class Module;
typedef std::shared_ptr<Call> CallPtr_t;
typedef std::shared_ptr<CallSlot> CallSlotPtr_t;
typedef std::shared_ptr<Module> ModulePtr_t;

// Types
typedef std::shared_ptr<Parameter> ParamPtr_t;
typedef std::vector<Parameter> ParamVector_t;


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
        Value_t;

    typedef std::variant<std::monostate, // default (unused/unavailable)
        float,                           // FLOAT
        int,                             // INT
        glm::vec2,                       // VECTOR_2f
        glm::vec3,                       // VECTOR_3f
        glm::vec4                        // VECTOR_4f
        >
        Min_t;

    typedef std::variant<std::monostate, // default (unused/unavailable)
        float,                           // FLOAT
        int,                             // INT
        glm::vec2,                       // VECTOR_2f
        glm::vec3,                       // VECTOR_3f
        glm::vec4                        // VECTOR_4f
        >
        Max_t;

    typedef std::variant<std::monostate,               // default (unused/unavailable)
        megamol::core::view::KeyCode,                  // BUTTON
        EnumStorage_t,                                 // ENUM
        megamol::core::param::FlexEnumParam::Storage_t // FLEXENUM
        >
        Stroage_t;

    struct StockParameter {
        std::string full_name;
        std::string description;
        Param_t type;
        std::string default_value;
        Min_t minval;
        Max_t maxval;
        Stroage_t storage;
        bool gui_visibility;
        bool gui_read_only;
        Present_t gui_presentation;
    };

    // VARIABLES --------------------------------------------------------------

    const ImGuiID uid;
    const Param_t type;
    ParameterPresentation present;

    // Init when adding parameter from stock
    std::string full_name;
    std::string description;

    vislib::SmartPtr<megamol::core::param::AbstractParam> core_param_ptr;

    // FUNCTIONS --------------------------------------------------------------

    Parameter(ImGuiID uid, Param_t type, Stroage_t store, Min_t minval, Max_t maxval);
    ~Parameter(void);

    bool IsValueDirty(void) { return this->value_dirty; }
    void ResetValueDirty(void) { this->value_dirty = false; }
    void ForceSetValueDirty(void) { this->value_dirty = true; }

    static bool ReadNewCoreParameterToStockParameter(
        megamol::core::param::ParamSlot& in_param_slot, megamol::gui::Parameter::StockParameter& out_param);

    static bool ReadNewCoreParameterToNewParameter(megamol::core::param::ParamSlot& in_param_slot,
        std::shared_ptr<megamol::gui::Parameter>& out_param, bool set_default_val, bool set_dirty,
        bool save_core_param_pointer);

    static bool ReadCoreParameterToParameter(vislib::SmartPtr<megamol::core::param::AbstractParam>& in_param_ptr,
        megamol::gui::Parameter& out_param, bool set_default_val, bool set_dirty);

    static bool ReadNewCoreParameterToExistingParameter(megamol::core::param::ParamSlot& in_param_slot,
        megamol::gui::Parameter& out_param, bool set_default_val, bool set_dirty, bool save_core_param_pointer);

    static bool WriteCoreParameterGUIState(
        megamol::gui::Parameter& in_param, vislib::SmartPtr<megamol::core::param::AbstractParam>& out_param_ptr);

    static bool WriteCoreParameterValue(
        megamol::gui::Parameter& in_param, vislib::SmartPtr<megamol::core::param::AbstractParam>& out_param_ptr);

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

    Value_t& GetValue(void) { return this->value; }

    template <typename T> const T& GetMinValue(void) const { return std::get<T>(this->minval); }

    template <typename T> const T& GetMaxValue(void) const { return std::get<T>(this->maxval); }

    template <typename T> const T& GetStorage(void) const { return std::get<T>(this->storage); }

    bool DefaultValueMismatch(void) { return this->default_value_mismatch; }

    size_t GetTransferFunctionHash(void) const { return this->tf_string_hash; }

    // SET ----------------------------------

    bool SetValueString(const std::string& val_str, bool set_default_val = false, bool set_dirty = true);

    template <typename T> void SetValue(T val, bool set_default_val = false, bool set_dirty = true) {
        if (std::holds_alternative<T>(this->value)) {

            // Set value
            if (std::get<T>(this->value) != val) {
                this->value = val;
                if (set_dirty) {
                    this->value_dirty = true;
                }

                // Check for new flex enum entry
                if (this->type == Param_t::FLEXENUM) {
                    auto storage = this->GetStorage<megamol::core::param::FlexEnumParam::Storage_t>();
                    storage.insert(std::get<std::string>(this->value));
                    this->SetStorage(storage);
                } else if (this->type == Param_t::TRANSFERFUNCTION) {
                    if constexpr (std::is_same_v<T, std::string>) {
                        int texture_width, texture_height;
                        std::vector<float> texture_data;
                        if (megamol::core::param::TransferFunctionParam::GetTextureData(
                                val, texture_data, texture_width, texture_height)) {
                            this->present.LoadTransferFunctionTexture(texture_data, texture_width, texture_height);
                        }
                        this->tf_string_hash = std::hash<std::string>()(val);
                    }
                }
            }

            // Check default value
            if (set_default_val) {
                this->value_dirty = false;
                this->default_value = val;
                this->default_value_mismatch = false;
            } else {
                try {
                    this->default_value_mismatch = (std::get<T>(this->default_value) != val);
                } catch (...) {
                }
            }
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Bad variant access. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        }
    }

    template <typename T> void SetMinValue(T minval) {
        if (std::holds_alternative<T>(this->minval)) {
            this->minval = minval;
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Bad variant access. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        }
    }

    template <typename T> void SetMaxValue(T maxval) {
        if (std::holds_alternative<T>(this->maxval)) {
            this->maxval = maxval;
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Bad variant access. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        }
    }

    template <typename T> void SetStorage(T store) {
        if (std::holds_alternative<T>(this->storage)) {
            this->storage = store;
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Bad variant access. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        }
    }

    // Presentation ----------------------------------------------------

    inline bool PresentGUI(ParameterPresentation::WidgetScope scope) { return this->present.Present(*this, scope); }

private:
    // VARIABLES --------------------------------------------------------------

    Min_t minval;
    Max_t maxval;
    Stroage_t storage;
    Value_t value;
    size_t tf_string_hash;
    Value_t default_value;
    bool default_value_mismatch;
    bool value_dirty;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_PARAMETER_H_INCLUDED
