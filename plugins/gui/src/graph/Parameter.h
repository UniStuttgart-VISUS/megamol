/*
 * Parameter.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_PARAMETER_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_PARAMETER_H_INCLUDED
#pragma once


#include <variant>
#include "mmcore/param/FlexEnumParam.h"
#include "widgets/FileBrowserWidget.h"
#include "widgets/HoverToolTip.h"
#include "widgets/ParameterOrbitalWidget.h"
#include "windows/TransferFunctionEditor.h"


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
    typedef std::vector<Parameter> ParamVector_t;


    /** ************************************************************************
     * Defines parameter data structure for graph
     */
    class Parameter : public megamol::core::param::AbstractParamPresentation {
    public:
        /*
         * Globally scoped widgets (widget parts) are always called each frame.
         * Locally scoped widgets (widget parts) are only called if respective parameter appears in GUI.
         */
        enum WidgetScope { GLOBAL, LOCAL };

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
            Storage_t;

        struct StockParameter {
            std::string param_name;
            std::string description;
            ParamType_t type;
            std::string default_value;
            Min_t minval;
            Max_t maxval;
            Storage_t storage;
            bool gui_visibility;
            bool gui_read_only;
            Present_t gui_presentation;
        };

        // STATIC ---------------------

        static bool ReadNewCoreParameterToStockParameter(
            megamol::core::param::ParamSlot& in_param_slot, megamol::gui::Parameter::StockParameter& out_param);

        static bool ReadNewCoreParameterToNewParameter(megamol::core::param::ParamSlot& in_param_slot,
            std::shared_ptr<megamol::gui::Parameter>& out_param, bool set_default_val, bool set_dirty,
            bool save_core_param_pointer, const std::string& parent_module_name);

        static bool ReadCoreParameterToParameter(vislib::SmartPtr<megamol::core::param::AbstractParam> in_param_ptr,
            megamol::gui::Parameter& out_param, bool set_default_val, bool set_dirty);

        static bool ReadNewCoreParameterToExistingParameter(megamol::core::param::ParamSlot& in_param_slot,
            megamol::gui::Parameter& out_param, bool set_default_val, bool set_dirty, bool save_core_param_pointer);

        static bool WriteCoreParameterGUIState(
            megamol::gui::Parameter& in_param, vislib::SmartPtr<megamol::core::param::AbstractParam> out_param_ptr);

        static bool WriteCoreParameterValue(
            megamol::gui::Parameter& in_param, vislib::SmartPtr<megamol::core::param::AbstractParam> out_param_ptr);

        // ----------------------------

        Parameter(ImGuiID uid, ParamType_t type, Storage_t store, Min_t minval, Max_t maxval,
            const std::string& param_name, const std::string& description);

        ~Parameter() override;

        bool Draw(WidgetScope scope);

        bool IsValueDirty() const {
            return this->value_dirty;
        }
        void ResetValueDirty() {
            this->value_dirty = false;
        }
        void ForceSetValueDirty() {
            this->value_dirty = true;
        }

        bool IsGUIStateDirty() const {
            return this->gui_state_dirty;
        }
        void ResetGUIStateDirty() {
            this->gui_state_dirty = false;
        }
        void ForceSetGUIStateDirty() {
            this->gui_state_dirty = true;
        }

        void TransferFunctionEditor_SetHash(size_t hash) {
            this->tf_editor_hash = hash;
        }
        inline void TransferFunctionEditor_ConnectExternal(
            std::shared_ptr<megamol::gui::TransferFunctionEditor> tfe_ptr, bool use_external_editor) {
            this->tf_editor_external_ptr = tfe_ptr;
            if (use_external_editor && (this->tf_editor_external_ptr != nullptr)) {
                this->tf_use_external_editor = true;
            }
        }
        void TransferFunction_LoadTexture(
            std::vector<float>& in_texture_data, int& in_texture_width, int& in_texture_height);

        // GET ----------------------------------------------------------------

        inline ImGuiID UID() const {
            return this->uid;
        }
        // <param_name>
        inline std::string Name() const {
            std::string name = this->param_name;
            auto idx = this->param_name.rfind(':');
            if (idx != std::string::npos) {
                name = name.substr(idx + 1, std::string::npos);
            }
            return name;
        }
        // <param_namespace>
        inline std::string NameSpace() const {
            std::string name_space;
            auto idx = this->param_name.find_first_of(':');
            if (idx != std::string::npos) {
                name_space = this->param_name.substr(0, idx);
            }
            return name_space;
        }
        // ::<module_group>::<module_name> + :: + <param_namespace>::<param_name>
        inline std::string FullNameProject() const {
            return std::string(this->parent_module_name + "::" + this->param_name);
        }
        // :: + ::<module_group>::<module_name> + :: + <param_namespace>::<param_name>
        inline std::string FullNameCore() const {
            return std::string("::" + this->parent_module_name + "::" + this->param_name);
        }

        std::string GetValueString() const;

        Value_t& GetValue() {
            return this->value;
        }

        template<typename T>
        const T& GetMinValue() const {
            return std::get<T>(this->minval);
        }

        template<typename T>
        const T& GetMaxValue() const {
            return std::get<T>(this->maxval);
        }

        template<typename T>
        const T& GetStorage() const {
            return std::get<T>(this->storage);
        }

        inline bool DefaultValueMismatch() {
            return this->default_value_mismatch;
        }
        inline size_t GetTransferFunctionHash() const {
            return this->tf_string_hash;
        }
        inline const ParamType_t Type() const {
            return this->type;
        }
        inline const std::string FloatFormat() const {
            return this->gui_float_format;
        }
        inline const bool IsExtended() const {
            return this->gui_extended;
        }
        inline vislib::SmartPtr<megamol::core::param::AbstractParam> CoreParamPtr() const {
            return this->core_param_ptr;
        }
        inline void ResetCoreParamPtr() {
            this->core_param_ptr = nullptr;
        }

        // SET ----------------------------------------------------------------

        inline void SetParentModuleName(const std::string& name) {
            this->parent_module_name = name;
        }
        inline void SetDescription(const std::string& desc) {
            this->description = desc;
        }

        bool SetValueString(const std::string& val_str, bool set_default_val = false, bool set_dirty = true);

        template<typename T>
        void SetValue(T val, bool set_default_val = false, bool set_dirty = true) {
            if (std::holds_alternative<T>(this->value)) {

                // Set value
                if (std::get<T>(this->value) != val) {
                    this->value = val;
                    if (set_dirty) {
                        this->value_dirty = true;
                    }

                    if (this->type == ParamType_t::FLEXENUM) {
                        // Flex Enum
                        auto flex_storage = this->GetStorage<megamol::core::param::FlexEnumParam::Storage_t>();
                        flex_storage.insert(std::get<std::string>(this->value));
                        this->SetStorage(flex_storage);
                    } else if (this->type == ParamType_t::TRANSFERFUNCTION) {
                        // Transfer Function
                        if constexpr (std::is_same_v<T, std::string>) {
                            int texture_width, texture_height;
                            std::vector<float> texture_data;
                            if (megamol::core::param::TransferFunctionParam::GetTextureData(
                                    val, texture_data, texture_width, texture_height)) {
                                this->TransferFunction_LoadTexture(texture_data, texture_width, texture_height);
                            }
                            this->tf_string_hash = std::hash<std::string>()(val);
                        }
                    } else if (this->type == ParamType_t::FILEPATH) {
                        // Push log message to GUI pop-up for not existing files
                        /* TODO Disabled until suitable FilePathParam flags are available
                        auto file = std::get<std::string>(this->value);
                        if (!megamol::core::utility::FileUtils::FileExists(file)) {
                            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                                "%sFile Parameter%sFile not found: '%s' > Parameter '%s'\n\n",
                                LOGMESSAGE_GUI_POPUP_START_TAG, LOGMESSAGE_GUI_POPUP_END_TAG,
                                this->FullNameProject().c_str(), file.c_str());
                        }
                        */
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
                    } catch (...) {}
                }
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Bad variant access. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            }
        }

        template<typename T>
        void SetMinValue(T minv) {
            if (std::holds_alternative<T>(this->minval)) {
                this->minval = minv;
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Bad variant access. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            }
        }

        template<typename T>
        void SetMaxValue(T maxv) {
            if (std::holds_alternative<T>(this->maxval)) {
                this->maxval = maxv;
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Bad variant access. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            }
        }

        template<typename T>
        void SetStorage(T store) {
            if (std::holds_alternative<T>(this->storage)) {
                this->storage = store;
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Bad variant access. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            }
        }

        inline void SetExtended(bool extended) {
            this->gui_extended = extended;
        }

    private:
        // VARIABLES --------------------------------------------------------------

        const ImGuiID uid;
        const ParamType_t type;
        std::string param_name;         /// <param_namespace>::<param_name>
        std::string parent_module_name; /// ::<module_group>::<module_name>
        std::string description;

        vislib::SmartPtr<megamol::core::param::AbstractParam> core_param_ptr;

        Min_t minval;
        Max_t maxval;
        Storage_t storage;
        Value_t value;

        Value_t default_value;
        bool default_value_mismatch;
        bool value_dirty;

        bool gui_extended;
        const std::string gui_float_format;
        std::string gui_help;
        std::string gui_tooltip_text;
        std::variant<std::monostate, std::string, int, float, glm::vec2, glm::vec3, glm::vec4> gui_widget_store;
        unsigned int gui_set_focus;
        bool gui_state_dirty;
        bool gui_show_minmax;
        FileBrowserWidget gui_file_browser;
        HoverToolTip gui_tooltip;
        ImageWidget gui_image_widget;
        ParameterOrbitalWidget gui_rotation_widget;

        size_t tf_string_hash;
        std::shared_ptr<megamol::gui::TransferFunctionEditor> tf_editor_external_ptr;
        megamol::gui::TransferFunctionEditor tf_editor_inplace;
        bool tf_use_external_editor;
        bool tf_show_editor;
        size_t tf_editor_hash;

        // FUNCTIONS ----------------------------------------------------------

        bool draw_parameter(WidgetScope scope);

        bool widget_button(WidgetScope scope, const std::string& label, const megamol::core::view::KeyCode& keycode);
        bool widget_bool(WidgetScope scope, const std::string& label, bool& val);
        bool widget_string(WidgetScope scope, const std::string& label, std::string& val);
        bool widget_color(WidgetScope scope, const std::string& label, glm::vec4& val);
        bool widget_enum(WidgetScope scope, const std::string& label, int& val, EnumStorage_t store);
        bool widget_flexenum(WidgetScope scope, const std::string& label, std::string& val,
            megamol::core::param::FlexEnumParam::Storage_t store);
        bool widget_filepath(WidgetScope scope, const std::string& label, std::string& val);
        bool widget_ternary(WidgetScope scope, const std::string& label, vislib::math::Ternary& val);
        bool widget_int(WidgetScope scope, const std::string& label, int& val, int minv, int maxv);
        bool widget_float(WidgetScope scope, const std::string& label, float& val, float minv, float maxv);
        bool widget_vector2f(
            WidgetScope scope, const std::string& label, glm::vec2& val, glm::vec2 minv, glm::vec2 maxv);
        bool widget_vector3f(
            WidgetScope scope, const std::string& label, glm::vec3& val, glm::vec3 minv, glm::vec3 maxv);
        bool widget_vector4f(
            WidgetScope scope, const std::string& label, glm::vec4& val, glm::vec4 minv, glm::vec4 maxv);
        bool widget_pinvaluetomouse(WidgetScope scope, const std::string& label, const std::string& val);
        bool widget_transfer_function_editor(WidgetScope scope);
        bool widget_knob(WidgetScope scope, const std::string& label, float& val, float minv, float maxv);
        bool widget_rotation_axes(
            WidgetScope scope, const std::string& label, glm::vec4& val, glm::vec4 minv, glm::vec4 maxv);
        bool widget_rotation_direction(
            WidgetScope scope, const std::string& label, glm::vec3& val, glm::vec3 minv, glm::vec3 maxv);
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_PARAMETER_H_INCLUDED
