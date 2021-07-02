/*
 * Parameter.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "Parameter.h"
#include "imgui_stdlib.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/TernaryParam.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"
#include "widgets/ButtonWidgets.h"
#include "widgets/ImageWidget_gl.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::Parameter::Parameter(ImGuiID uid, ParamType_t type, Storage_t store, Min_t minv, Max_t maxv,
    const std::string& param_name, const std::string& description)
        : megamol::core::param::AbstractParamPresentation()
        , uid(uid)
        , type(type)
        , param_name(param_name)
        , parent_module_name()
        , description(description)
        , core_param_ptr(nullptr)
        , minval(minv)
        , maxval(maxv)
        , storage(store)
        , value()
        , default_value()
        , default_value_mismatch(false)
        , value_dirty(false)
        , gui_extended(false)
        , gui_float_format("%.7f")
        , gui_help()
        , gui_tooltip_text()
        , gui_widget_store()
        , gui_set_focus(0)
        , gui_state_dirty(false)
        , gui_show_minmax(false)
        , gui_file_browser()
        , gui_tooltip()
        , gui_image_widget()
        , gui_rotation_widget()
        , tf_string_hash(0)
        , tf_editor_external_ptr(nullptr)
        , tf_editor_inplace(std::string("inplace_tfeditor_parameter_" + std::to_string(uid)), false)
        , tf_use_external_editor(false)
        , tf_show_editor(false)
        , tf_editor_hash(0) {

    this->InitPresentation(type);

    // Initialize variant types which should/can not be changed afterwards.
    // Default ctor of variants initializes std::monostate.
    switch (this->type) {
    case (ParamType_t::BOOL): {
        this->value = bool(false);
    } break;
    case (ParamType_t::BUTTON): {
        // nothing to do ...
    } break;
    case (ParamType_t::COLOR): {
        this->value = glm::vec4();
    } break;
    case (ParamType_t::ENUM): {
        this->value = int(0);
    } break;
    case (ParamType_t::FILEPATH): {
        this->value = std::string();
    } break;
    case (ParamType_t::FLEXENUM): {
        this->value = std::string();
    } break;
    case (ParamType_t::FLOAT): {
        this->value = float(0.0f);
    } break;
    case (ParamType_t::INT): {
        this->value = int();
    } break;
    case (ParamType_t::STRING): {
        this->value = std::string();
    } break;
    case (ParamType_t::TERNARY): {
        this->value = vislib::math::Ternary();
    } break;
    case (ParamType_t::TRANSFERFUNCTION): {
        this->value = std::string();
    } break;
    case (ParamType_t::VECTOR2F): {
        this->value = glm::vec2();
    } break;
    case (ParamType_t::VECTOR3F): {
        this->value = glm::vec3();
    } break;
    case (ParamType_t::VECTOR4F): {
        this->value = glm::vec4();
    } break;
    default:
        break;
    }

    this->default_value = this->value;
}


megamol::gui::Parameter::~Parameter() {

    if (this->tf_editor_external_ptr != nullptr) {
        this->tf_editor_external_ptr->SetConnectedParameter(nullptr, "");
        this->tf_editor_external_ptr = nullptr;
    }
}


std::string megamol::gui::Parameter::GetValueString() const {

    std::string value_string("UNKNOWN PARAMETER TYPE");

    auto visitor = [this, &value_string](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, bool>) {
            auto parameter = megamol::core::param::BoolParam(arg);
            value_string = std::string(parameter.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, float>) {
            auto parameter = megamol::core::param::FloatParam(arg);
            value_string = std::string(parameter.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, int>) {
            switch (this->type) {
            case (ParamType_t::INT): {
                auto parameter = megamol::core::param::IntParam(arg);
                value_string = std::string(parameter.ValueString().PeekBuffer());
            } break;
            case (ParamType_t::ENUM): {
                auto parameter = megamol::core::param::EnumParam(arg);
                // Initialization of enum storage required
                auto map = this->GetStorage<EnumStorage_t>();
                for (auto& pair : map) {
                    parameter.SetTypePair(pair.first, pair.second.c_str());
                }
                value_string = std::string(parameter.ValueString().PeekBuffer());
            } break;
            default:
                break;
            }
        } else if constexpr (std::is_same_v<T, std::string>) {
            value_string = arg;
        } else if constexpr (std::is_same_v<T, vislib::math::Ternary>) {
            auto parameter = megamol::core::param::TernaryParam(arg);
            value_string = std::string(parameter.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, glm::vec2>) {
            auto parameter = megamol::core::param::Vector2fParam(vislib::math::Vector<float, 2>(arg.x, arg.y));
            value_string = std::string(parameter.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, glm::vec3>) {
            auto parameter = megamol::core::param::Vector3fParam(vislib::math::Vector<float, 3>(arg.x, arg.y, arg.z));
            value_string = std::string(parameter.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, glm::vec4>) {
            switch (this->type) {
            case (ParamType_t::COLOR): {
                auto parameter = megamol::core::param::ColorParam(arg[0], arg[1], arg[2], arg[3]);
                value_string = std::string(parameter.ValueString().PeekBuffer());
            } break;

            case (ParamType_t::VECTOR4F): {
                auto parameter =
                    megamol::core::param::Vector4fParam(vislib::math::Vector<float, 4>(arg.x, arg.y, arg.z, arg.w));
                value_string = std::string(parameter.ValueString().PeekBuffer());
            } break;
            default:
                break;
            }
        } else if constexpr (std::is_same_v<T, std::monostate>) {
            switch (this->type) {
            case (ParamType_t::BUTTON): {
                auto parameter = megamol::core::param::ButtonParam();
                value_string = std::string(parameter.ValueString().PeekBuffer());
                break;
            }
            default:
                break;
            }
        }
    };
    std::visit(visitor, this->value);
    return value_string;
}


bool megamol::gui::Parameter::SetValueString(const std::string& val_str, bool set_default_val, bool set_dirty) {

    bool retval = false;
    vislib::TString val_tstr(val_str.c_str());

    switch (this->type) {
    case (ParamType_t::BOOL): {
        megamol::core::param::BoolParam parameter(false);
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val, set_dirty);
    } break;
    case (ParamType_t::BUTTON): {
        retval = true;
    } break;
    case (ParamType_t::COLOR): {
        megamol::core::param::ColorParam parameter(val_tstr);
        retval = parameter.ParseValue(val_tstr);
        auto val = parameter.Value();
        this->SetValue(glm::vec4(val[0], val[1], val[2], val[3]), set_default_val, set_dirty);
    } break;
    case (ParamType_t::ENUM): {
        megamol::core::param::EnumParam parameter(0);
        // Initialization of enum storage required
        auto map = this->GetStorage<EnumStorage_t>();
        for (auto& pair : map) {
            parameter.SetTypePair(pair.first, pair.second.c_str());
        }
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val, set_dirty);
    } break;
    case (ParamType_t::FILEPATH): {
        megamol::core::param::FilePathParam parameter(val_tstr.PeekBuffer());
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(std::string(parameter.Value().PeekBuffer()), set_default_val, set_dirty);
    } break;
    case (ParamType_t::FLEXENUM): {
        megamol::core::param::FlexEnumParam parameter(val_str);
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val, set_dirty);
    } break;
    case (ParamType_t::FLOAT): {
        megamol::core::param::FloatParam parameter(0.0f);
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val, set_dirty);
    } break;
    case (ParamType_t::INT): {
        megamol::core::param::IntParam parameter(0);
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val, set_dirty);
    } break;
    case (ParamType_t::STRING): {
        megamol::core::param::StringParam parameter(val_tstr.PeekBuffer());
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(std::string(parameter.Value().PeekBuffer()), set_default_val, set_dirty);
    } break;
    case (ParamType_t::TERNARY): {
        megamol::core::param::TernaryParam parameter(vislib::math::Ternary::TRI_UNKNOWN);
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val, set_dirty);
    } break;
    case (ParamType_t::TRANSFERFUNCTION): {
        megamol::core::param::TransferFunctionParam parameter;
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val, set_dirty);
    } break;
    case (ParamType_t::VECTOR2F): {
        megamol::core::param::Vector2fParam parameter(vislib::math::Vector<float, 2>(0.0f, 0.0f));
        retval = parameter.ParseValue(val_tstr);
        auto val = parameter.Value();
        this->SetValue(glm::vec2(val.X(), val.Y()), set_default_val, set_dirty);
    } break;
    case (ParamType_t::VECTOR3F): {
        megamol::core::param::Vector3fParam parameter(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
        retval = parameter.ParseValue(val_tstr);
        auto val = parameter.Value();
        this->SetValue(glm::vec3(val.X(), val.Y(), val.Z()), set_default_val, set_dirty);
    } break;
    case (ParamType_t::VECTOR4F): {
        megamol::core::param::Vector4fParam parameter(vislib::math::Vector<float, 4>(0.0f, 0.0f, 0.0f, 0.0f));
        retval = parameter.ParseValue(val_tstr);
        auto val = parameter.Value();
        this->SetValue(glm::vec4(val.X(), val.Y(), val.Z(), val.W()), set_default_val, set_dirty);
    } break;
    default:
        break;
    }

    return retval;
}


bool megamol::gui::Parameter::ReadNewCoreParameterToStockParameter(
    megamol::core::param::ParamSlot& in_param_slot, megamol::gui::Parameter::StockParameter& out_param) {

    auto parameter_ptr = in_param_slot.Parameter();
    if (parameter_ptr.IsNull()) {
        return false;
    }

    out_param.param_name = std::string(in_param_slot.Name().PeekBuffer());
    out_param.description = std::string(in_param_slot.Description().PeekBuffer());
    out_param.gui_visibility = parameter_ptr->IsGUIVisible();
    out_param.gui_read_only = parameter_ptr->IsGUIReadOnly();
    out_param.gui_presentation = static_cast<Present_t>(parameter_ptr->GetGUIPresentation());

    if (auto* p_ptr = in_param_slot.Param<core::param::ButtonParam>()) {
        out_param.type = ParamType_t::BUTTON;
        out_param.storage = p_ptr->GetKeyCode();
    } else if (auto* p_ptr = in_param_slot.Param<core::param::BoolParam>()) {
        out_param.type = ParamType_t::BOOL;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::ColorParam>()) {
        out_param.type = ParamType_t::COLOR;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::EnumParam>()) {
        out_param.type = ParamType_t::ENUM;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
        EnumStorage_t map;
        auto psd_map = p_ptr->getMap();
        auto iter = psd_map.GetConstIterator();
        while (iter.HasNext()) {
            auto pair = iter.Next();
            map.emplace(pair.Key(), std::string(pair.Value().PeekBuffer()));
        }
        out_param.storage = map;
    } else if (auto* p_ptr = in_param_slot.Param<core::param::FilePathParam>()) {
        out_param.type = ParamType_t::FILEPATH;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::FlexEnumParam>()) {
        out_param.type = ParamType_t::FLEXENUM;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
        out_param.storage = p_ptr->getStorage();
    } else if (auto* p_ptr = in_param_slot.Param<core::param::FloatParam>()) {
        out_param.type = ParamType_t::FLOAT;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
        out_param.minval = p_ptr->MinValue();
        out_param.maxval = p_ptr->MaxValue();
    } else if (auto* p_ptr = in_param_slot.Param<core::param::IntParam>()) {
        out_param.type = ParamType_t::INT;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
        out_param.minval = p_ptr->MinValue();
        out_param.maxval = p_ptr->MaxValue();
    } else if (auto* p_ptr = in_param_slot.Param<core::param::StringParam>()) {
        out_param.type = ParamType_t::STRING;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::TernaryParam>()) {
        out_param.type = ParamType_t::TERNARY;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::TransferFunctionParam>()) {
        out_param.type = ParamType_t::TRANSFERFUNCTION;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::Vector2fParam>()) {
        out_param.type = ParamType_t::VECTOR2F;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
        auto minv = p_ptr->MinValue();
        out_param.minval = glm::vec2(minv.X(), minv.Y());
        auto maxv = p_ptr->MaxValue();
        out_param.maxval = glm::vec2(maxv.X(), maxv.Y());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::Vector3fParam>()) {
        out_param.type = ParamType_t::VECTOR3F;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
        auto minv = p_ptr->MinValue();
        out_param.minval = glm::vec3(minv.X(), minv.Y(), minv.Z());
        auto maxv = p_ptr->MaxValue();
        out_param.maxval = glm::vec3(maxv.X(), maxv.Y(), maxv.Z());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::Vector4fParam>()) {
        out_param.type = ParamType_t::VECTOR4F;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
        auto minv = p_ptr->MinValue();
        out_param.minval = glm::vec4(minv.X(), minv.Y(), minv.Z(), minv.W());
        auto maxv = p_ptr->MaxValue();
        out_param.maxval = glm::vec4(maxv.X(), maxv.Y(), maxv.Z(), maxv.W());
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Found unknown parameter type. Please extend parameter types "
            "for the configurator. [%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        out_param.type = ParamType_t::UNKNOWN;
        return false;
    }

    return true;
}


bool megamol::gui::Parameter::ReadNewCoreParameterToNewParameter(megamol::core::param::ParamSlot& in_param_slot,
    std::shared_ptr<megamol::gui::Parameter>& out_param, bool set_default_val, bool set_dirty,
    bool save_core_param_pointer, const std::string& parent_module_name) {

    auto parameter_ptr = in_param_slot.Parameter();
    if (parameter_ptr.IsNull()) {
        return false;
    }

    out_param.reset();
    auto param_name = std::string(in_param_slot.Name().PeekBuffer());
    auto param_fullname = std::string(in_param_slot.FullName().PeekBuffer());
    auto description = std::string(in_param_slot.Description().PeekBuffer());

    if (auto* p_ptr = in_param_slot.template Param<core::param::BoolParam>()) {
        out_param = std::make_shared<Parameter>(megamol::gui::GenerateUniqueID(), ParamType_t::BOOL, std::monostate(),
            std::monostate(), std::monostate(), param_name, description);
        out_param->SetValue(p_ptr->Value(), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::ButtonParam>()) {
        out_param = std::make_shared<Parameter>(megamol::gui::GenerateUniqueID(), ParamType_t::BUTTON,
            p_ptr->GetKeyCode(), std::monostate(), std::monostate(), param_name, description);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::ColorParam>()) {
        out_param = std::make_shared<Parameter>(megamol::gui::GenerateUniqueID(), ParamType_t::COLOR, std::monostate(),
            std::monostate(), std::monostate(), param_name, description);
        auto value = p_ptr->Value();
        out_param->SetValue(glm::vec4(value[0], value[1], value[2], value[3]), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::TransferFunctionParam>()) {
        out_param = std::make_shared<Parameter>(megamol::gui::GenerateUniqueID(), ParamType_t::TRANSFERFUNCTION,
            std::monostate(), std::monostate(), std::monostate(), param_name, description);
        out_param->SetValue(p_ptr->Value(), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::EnumParam>()) {
        EnumStorage_t map;
        auto param_map = p_ptr->getMap();
        auto iter = param_map.GetConstIterator();
        while (iter.HasNext()) {
            auto pair = iter.Next();
            map.emplace(pair.Key(), std::string(pair.Value().PeekBuffer()));
        }
        out_param = std::make_shared<Parameter>(megamol::gui::GenerateUniqueID(), ParamType_t::ENUM, map,
            std::monostate(), std::monostate(), param_name, description);
        out_param->SetValue(p_ptr->Value(), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::FlexEnumParam>()) {
        out_param = std::make_shared<Parameter>(megamol::gui::GenerateUniqueID(), ParamType_t::FLEXENUM,
            p_ptr->getStorage(), std::monostate(), std::monostate(), param_name, description);
        out_param->SetValue(p_ptr->Value(), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::FloatParam>()) {
        out_param = std::make_shared<Parameter>(megamol::gui::GenerateUniqueID(), ParamType_t::FLOAT, std::monostate(),
            p_ptr->MinValue(), p_ptr->MaxValue(), param_name, description);
        out_param->SetValue(p_ptr->Value(), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::IntParam>()) {
        out_param = std::make_shared<Parameter>(megamol::gui::GenerateUniqueID(), ParamType_t::INT, std::monostate(),
            p_ptr->MinValue(), p_ptr->MaxValue(), param_name, description);
        out_param->SetValue(p_ptr->Value(), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::Vector2fParam>()) {
        auto minv = p_ptr->MinValue();
        auto maxv = p_ptr->MaxValue();
        auto val = p_ptr->Value();
        out_param = std::make_shared<Parameter>(megamol::gui::GenerateUniqueID(), ParamType_t::VECTOR2F,
            std::monostate(), glm::vec2(minv.X(), minv.Y()), glm::vec2(maxv.X(), maxv.Y()), param_name, description);
        out_param->SetValue(glm::vec2(val.X(), val.Y()), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::Vector3fParam>()) {
        auto minv = p_ptr->MinValue();
        auto maxv = p_ptr->MaxValue();
        auto val = p_ptr->Value();
        out_param = std::make_shared<Parameter>(megamol::gui::GenerateUniqueID(), ParamType_t::VECTOR3F,
            std::monostate(), glm::vec3(minv.X(), minv.Y(), minv.Z()), glm::vec3(maxv.X(), maxv.Y(), maxv.Z()),
            param_name, description);
        out_param->SetValue(glm::vec3(val.X(), val.Y(), val.Z()), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::Vector4fParam>()) {
        auto minv = p_ptr->MinValue();
        auto maxv = p_ptr->MaxValue();
        auto val = p_ptr->Value();
        out_param = std::make_shared<Parameter>(megamol::gui::GenerateUniqueID(), ParamType_t::VECTOR4F,
            std::monostate(), glm::vec4(minv.X(), minv.Y(), minv.Z(), minv.W()),
            glm::vec4(maxv.X(), maxv.Y(), maxv.Z(), maxv.W()), param_name, description);
        out_param->SetValue(glm::vec4(val.X(), val.Y(), val.Z(), val.W()), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::TernaryParam>()) {
        out_param = std::make_shared<Parameter>(megamol::gui::GenerateUniqueID(), ParamType_t::TERNARY,
            std::monostate(), std::monostate(), std::monostate(), param_name, description);
        out_param->SetValue(p_ptr->Value(), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.Param<core::param::StringParam>()) {
        out_param = std::make_shared<Parameter>(megamol::gui::GenerateUniqueID(), ParamType_t::STRING, std::monostate(),
            std::monostate(), std::monostate(), param_name, description);
        out_param->SetValue(std::string(p_ptr->Value().PeekBuffer()), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.Param<core::param::FilePathParam>()) {
        out_param = std::make_shared<Parameter>(megamol::gui::GenerateUniqueID(), ParamType_t::FILEPATH,
            std::monostate(), std::monostate(), std::monostate(), param_name, description);
        out_param->SetValue(std::string(p_ptr->Value().PeekBuffer()), set_default_val, set_dirty);
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Found unknown parameter type. Please extend parameter types for the configurator. "
            "[%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    out_param->SetParentModuleName(parent_module_name);

    out_param->SetGUIVisible(parameter_ptr->IsGUIVisible());
    out_param->SetGUIReadOnly(parameter_ptr->IsGUIReadOnly());
    out_param->SetGUIPresentation(parameter_ptr->GetGUIPresentation());
    if (save_core_param_pointer) {
        out_param->core_param_ptr = parameter_ptr;
    }

    return true;
}


bool megamol::gui::Parameter::ReadCoreParameterToParameter(
    vislib::SmartPtr<megamol::core::param::AbstractParam> in_param_ptr, megamol::gui::Parameter& out_param,
    bool set_default_val, bool set_dirty) {

    out_param.SetGUIVisible(in_param_ptr->IsGUIVisible());
    out_param.SetGUIReadOnly(in_param_ptr->IsGUIReadOnly());
    out_param.SetGUIPresentation(in_param_ptr->GetGUIPresentation());

    // Do not read param value from core param if gui param has already updated value
    if (out_param.IsValueDirty())
        return false;

    bool type_error = false;

    if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::ButtonParam>()) {
        if (out_param.type == ParamType_t::BUTTON) {
            out_param.SetStorage(p_ptr->GetKeyCode());
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::BoolParam>()) {
        if (out_param.type == ParamType_t::BOOL) {
            out_param.SetValue(p_ptr->Value(), set_default_val, set_dirty);
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::ColorParam>()) {
        if (out_param.type == ParamType_t::COLOR) {
            auto value = p_ptr->Value();
            out_param.SetValue(glm::vec4(value[0], value[1], value[2], value[3]), set_default_val, set_dirty);
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::EnumParam>()) {
        if (out_param.type == ParamType_t::ENUM) {
            out_param.SetValue(p_ptr->Value(), set_default_val, set_dirty);
            EnumStorage_t map;
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
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::FilePathParam>()) {
        if (out_param.type == ParamType_t::FILEPATH) {
            out_param.SetValue(std::string(p_ptr->Value().PeekBuffer()), set_default_val, set_dirty);
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::FlexEnumParam>()) {
        if (out_param.type == ParamType_t::FLEXENUM) {
            out_param.SetValue(p_ptr->Value(), set_default_val, set_dirty);
            out_param.SetStorage(p_ptr->getStorage());
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::FloatParam>()) {
        if (out_param.type == ParamType_t::FLOAT) {
            out_param.SetValue(p_ptr->Value(), set_default_val, set_dirty);
            out_param.SetMinValue(p_ptr->MinValue());
            out_param.SetMaxValue(p_ptr->MaxValue());
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::IntParam>()) {
        if (out_param.type == ParamType_t::INT) {
            out_param.SetValue(p_ptr->Value(), set_default_val, set_dirty);
            out_param.SetMinValue(p_ptr->MinValue());
            out_param.SetMaxValue(p_ptr->MaxValue());
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::StringParam>()) {
        if (out_param.type == ParamType_t::STRING) {
            out_param.SetValue(std::string(p_ptr->Value().PeekBuffer()), set_default_val, set_dirty);
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::TernaryParam>()) {
        if (out_param.type == ParamType_t::TERNARY) {
            out_param.SetValue(p_ptr->Value(), set_default_val, set_dirty);
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::TransferFunctionParam>()) {
        if (out_param.type == ParamType_t::TRANSFERFUNCTION) {
            out_param.SetValue(p_ptr->Value(), set_default_val, set_dirty);
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::Vector2fParam>()) {
        if (out_param.type == ParamType_t::VECTOR2F) {
            auto val = p_ptr->Value();
            out_param.SetValue(glm::vec2(val.X(), val.Y()), set_default_val, set_dirty);
            auto minv = p_ptr->MinValue();
            out_param.SetMinValue(glm::vec2(minv.X(), minv.Y()));
            auto maxv = p_ptr->MaxValue();
            out_param.SetMaxValue(glm::vec2(maxv.X(), maxv.Y()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::Vector3fParam>()) {
        if (out_param.type == ParamType_t::VECTOR3F) {
            auto val = p_ptr->Value();
            out_param.SetValue(glm::vec3(val.X(), val.Y(), val.Z()), set_default_val, set_dirty);
            auto minv = p_ptr->MinValue();
            out_param.SetMinValue(glm::vec3(minv.X(), minv.Y(), minv.Z()));
            auto maxv = p_ptr->MaxValue();
            out_param.SetMaxValue(glm::vec3(maxv.X(), maxv.Y(), maxv.Z()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::Vector4fParam>()) {
        if (out_param.type == ParamType_t::VECTOR4F) {
            auto val = p_ptr->Value();
            out_param.SetValue(glm::vec4(val.X(), val.Y(), val.Z(), val.W()), set_default_val, set_dirty);
            auto minv = p_ptr->MinValue();
            out_param.SetMinValue(glm::vec4(minv.X(), minv.Y(), minv.Z(), minv.W()));
            auto maxv = p_ptr->MaxValue();
            out_param.SetMaxValue(glm::vec4(maxv.X(), maxv.Y(), maxv.Z(), maxv.W()));
        } else {
            type_error = true;
        }
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Found unknown parameter type. Please extend parameter types for the configurator. "
            "[%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    if (type_error) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Mismatch of parameter types. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


bool megamol::gui::Parameter::ReadNewCoreParameterToExistingParameter(megamol::core::param::ParamSlot& in_param_slot,
    megamol::gui::Parameter& out_param, bool set_default_val, bool set_dirty, bool save_core_param_pointer) {

    auto parameter_ptr = in_param_slot.Parameter();
    if (parameter_ptr.IsNull()) {
        return false;
    }
    out_param.SetDescription(std::string(in_param_slot.Description().PeekBuffer()));
    if (save_core_param_pointer) {
        out_param.core_param_ptr = parameter_ptr;
    }
    return megamol::gui::Parameter::ReadCoreParameterToParameter(parameter_ptr, out_param, set_default_val, set_dirty);
}


bool megamol::gui::Parameter::WriteCoreParameterGUIState(
    megamol::gui::Parameter& in_param, vislib::SmartPtr<megamol::core::param::AbstractParam> out_param_ptr) {

    out_param_ptr->SetGUIVisible(in_param.IsGUIVisible());
    out_param_ptr->SetGUIReadOnly(in_param.IsGUIReadOnly());
    out_param_ptr->SetGUIPresentation(in_param.GetGUIPresentation());

    return true;
}


bool megamol::gui::Parameter::WriteCoreParameterValue(
    megamol::gui::Parameter& in_param, vislib::SmartPtr<megamol::core::param::AbstractParam> out_param_ptr) {
    bool type_error = false;

    if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::ButtonParam>()) {
        if (in_param.type == ParamType_t::BUTTON) {
            p_ptr->setDirty();
            // KeyCode can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::BoolParam>()) {
        if (in_param.type == ParamType_t::BOOL) {
            p_ptr->SetValue(std::get<bool>(in_param.GetValue()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::ColorParam>()) {
        if (in_param.type == ParamType_t::COLOR) {
            auto value = std::get<glm::vec4>(in_param.GetValue());
            p_ptr->SetValue(core::param::ColorParam::ColorType{value[0], value[1], value[2], value[3]});
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::EnumParam>()) {
        if (in_param.type == ParamType_t::ENUM) {
            p_ptr->SetValue(std::get<int>(in_param.GetValue()));
            // Map can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::FilePathParam>()) {
        if (in_param.type == ParamType_t::FILEPATH) {
            p_ptr->SetValue(vislib::StringA(std::get<std::string>(in_param.GetValue()).c_str()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::FlexEnumParam>()) {
        if (in_param.type == ParamType_t::FLEXENUM) {
            p_ptr->SetValue(std::get<std::string>(in_param.GetValue()));
            // Storage can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::FloatParam>()) {
        if (in_param.type == ParamType_t::FLOAT) {
            p_ptr->SetValue(std::get<float>(in_param.GetValue()));
            // Min and Max can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::IntParam>()) {
        if (in_param.type == ParamType_t::INT) {
            p_ptr->SetValue(std::get<int>(in_param.GetValue()));
            // Min and Max can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::StringParam>()) {
        if (in_param.type == ParamType_t::STRING) {
            p_ptr->SetValue(vislib::StringA(std::get<std::string>(in_param.GetValue()).c_str()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::TernaryParam>()) {
        if (in_param.type == ParamType_t::TERNARY) {
            p_ptr->SetValue(std::get<vislib::math::Ternary>(in_param.GetValue()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::TransferFunctionParam>()) {
        if (in_param.type == ParamType_t::TRANSFERFUNCTION) {
            p_ptr->SetValue(std::get<std::string>(in_param.GetValue()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::Vector2fParam>()) {
        if (in_param.type == ParamType_t::VECTOR2F) {
            auto value = std::get<glm::vec2>(in_param.GetValue());
            p_ptr->SetValue(vislib::math::Vector<float, 2>(value[0], value[1]));
            // Min and Max can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::Vector3fParam>()) {
        if (in_param.type == ParamType_t::VECTOR3F) {
            auto value = std::get<glm::vec3>(in_param.GetValue());
            p_ptr->SetValue(vislib::math::Vector<float, 3>(value[0], value[1], value[2]));
            // Min and Max can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::Vector4fParam>()) {
        if (in_param.type == ParamType_t::VECTOR4F) {
            auto value = std::get<glm::vec4>(in_param.GetValue());
            p_ptr->SetValue(vislib::math::Vector<float, 4>(value[0], value[1], value[2], value[3]));
            // Min and Max can not be changed
        } else {
            type_error = true;
        }
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Found unknown parameter type. Please extend parameter types for the configurator. "
            "[%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    if (type_error) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Mismatch of parameter types. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


bool megamol::gui::Parameter::Draw(megamol::gui::Parameter::WidgetScope scope) {

    bool retval = false;

    if (ImGui::GetCurrentContext() == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    try {
        ImGui::PushID(static_cast<int>(this->uid));

        this->gui_help = "";
        this->gui_tooltip_text = this->description;

        switch (scope) {
        case (WidgetScope::LOCAL): {
            if (this->IsGUIVisible() || this->gui_extended) {

                ImGui::BeginGroup();
                if (this->gui_extended) {
                    /// PREFIX ---------------------------------------------

                    // Visibility
                    if (ImGui::RadioButton("###visible", this->IsGUIVisible())) {
                        this->SetGUIVisible(!this->IsGUIVisible());
                        this->ForceSetGUIStateDirty();
                    }
                    this->gui_tooltip.ToolTip("Visibility (Basic Mode)");

                    ImGui::SameLine();

                    // Read-only option
                    bool read_only = this->IsGUIReadOnly();
                    if (ImGui::Checkbox("###readonly", &read_only)) {
                        this->SetGUIReadOnly(read_only);
                        this->ForceSetGUIStateDirty();
                    }
                    this->gui_tooltip.ToolTip("Read-Only");

                    ImGui::SameLine();

                    // Presentation
                    ButtonWidgets::OptionButton(
                        "param_present_button", "", (this->GetGUIPresentation() != Present_t::Basic), false);
                    if (ImGui::BeginPopupContextItem("param_present_button_context", ImGuiPopupFlags_MouseButtonLeft)) {
                        for (auto& present_name_pair : this->GetPresentationNameMap()) {
                            if (this->IsPresentationCompatible(present_name_pair.first)) {
                                if (ImGui::MenuItem(present_name_pair.second.c_str(), nullptr,
                                        (present_name_pair.first == this->GetGUIPresentation()))) {
                                    this->SetGUIPresentation(present_name_pair.first);
                                    this->ForceSetGUIStateDirty();
                                }
                            }
                        }
                        ImGui::EndPopup();
                    }
                    this->gui_tooltip.ToolTip("Presentation");

                    ImGui::SameLine();

                    // Lua
                    ButtonWidgets::LuaButton("param_lua_button", (*this), this->FullNameProject());
                    this->gui_tooltip.ToolTip("Copy lua command to clipboard.");

                    ImGui::SameLine();
                }

                /// PARAMETER VALUE WIDGET ---------------------------------
                /// DEBUG ImGui::TextUnformatted(this->FullName().c_str());
                if (this->draw_parameter(scope)) {
                    retval = true;
                }

                ImGui::SameLine();

                /// POSTFIX ------------------------------------------------
                if (!ImGui::IsItemActive()) {
                    this->gui_tooltip.ToolTip(this->gui_tooltip_text, ImGui::GetItemID(), 1.0f, 4.0f);
                }
                this->gui_tooltip.Marker(this->gui_help);

                ImGui::EndGroup();
            }
        } break;
        case (WidgetScope::GLOBAL): {

            if (this->draw_parameter(scope)) {
                retval = true;
            }

        } break;
        default:
            break;
        }

        ImGui::PopID();

    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return retval;
}


void megamol::gui::Parameter::TransferFunction_LoadTexture(
    std::vector<float>& in_texture_data, int& in_texture_width, int& in_texture_height) {

    gui_image_widget.LoadTextureFromData(in_texture_width, in_texture_height, in_texture_data.data());
}


bool megamol::gui::Parameter::draw_parameter(megamol::gui::Parameter::WidgetScope scope) {

    bool retval = false;
    bool error = true;
    std::string param_label = this->Name();

    // Implementation of presentation with parameter type mapping defined in
    // AbstractParamPresentation::InitPresentation().
    auto visitor = [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;

        // LOCAL -----------------------------------------------------------
        if (scope == megamol::gui::Parameter::WidgetScope::LOCAL) {
            // Set general proportional item width
            float widget_width = ImGui::GetContentRegionAvail().x * 0.65f;
            ImGui::PushItemWidth(widget_width);
            // Set read only
            gui_utils::PushReadOnly(this->IsGUIReadOnly());
        }

        switch (this->GetGUIPresentation()) {
            // BASIC ///////////////////////////////////////////////////
        case (Present_t::Basic): {
            // BOOL ------------------------------------------------
            if constexpr (std::is_same_v<T, bool>) {
                auto val = arg;
                if (this->widget_bool(scope, param_label, val)) {
                    this->SetValue(val);
                    retval = true;
                }
                error = false;
            }
            // FLOAT -----------------------------------------------
            else if constexpr (std::is_same_v<T, float>) {
                auto val = arg;
                if (this->widget_float(scope, param_label, val, this->GetMinValue<T>(), this->GetMaxValue<T>())) {
                    this->SetValue(val);
                    retval = true;
                }
                error = false;
            } else if constexpr (std::is_same_v<T, int>) {
                switch (this->type) {
                    // INT ---------------------------------------------
                case (ParamType_t::INT): {
                    auto val = arg;
                    if (this->widget_int(scope, param_label, val, this->GetMinValue<T>(), this->GetMaxValue<T>())) {
                        this->SetValue(val);
                        retval = true;
                    }
                    error = false;
                } break;
                    // ENUM --------------------------------------------
                case (ParamType_t::ENUM): {
                    auto val = arg;
                    if (this->widget_enum(scope, param_label, val, this->GetStorage<EnumStorage_t>())) {
                        this->SetValue(val);
                        retval = true;
                    }
                    error = false;
                } break;
                default:
                    break;
                }
            } else if constexpr (std::is_same_v<T, std::string>) {
                switch (this->type) {
                    // STRING ------------------------------------------
                case (ParamType_t::STRING): {
                    auto val = arg;
                    if (this->widget_string(scope, param_label, val)) {
                        this->SetValue(val);
                        retval = true;
                    }
                    error = false;
                } break;
                    // TRANSFER FUNCTION -------------------------------
                case (ParamType_t::TRANSFERFUNCTION): {
                    auto val = arg;
                    if (this->widget_string(scope, param_label, val)) {
                        this->SetValue(val);
                        retval = true;
                    }
                    error = false;
                } break;
                    // FILE PATH ---------------------------------------
                case (ParamType_t::FILEPATH): {
                    auto val = arg;
                    if (this->widget_string(scope, param_label, val)) {
                        this->SetValue(val);
                        retval = true;
                    }
                    error = false;
                } break;
                    // FLEX ENUM ---------------------------------------
                case (ParamType_t::FLEXENUM): {
                    auto val = arg;
                    if (this->widget_flexenum(scope, param_label, val,
                            this->GetStorage<megamol::core::param::FlexEnumParam::Storage_t>())) {
                        this->SetValue(val);
                        retval = true;
                    }
                    error = false;
                } break;
                default:
                    break;
                }
            }
            // TERNARY ---------------------------------------------
            else if constexpr (std::is_same_v<T, vislib::math::Ternary>) {
                auto val = arg;
                if (this->widget_ternary(scope, param_label, val)) {
                    this->SetValue(val);
                    retval = true;
                }
                error = false;
            }
            // VECTOR 2 --------------------------------------------
            else if constexpr (std::is_same_v<T, glm::vec2>) {
                auto val = arg;
                if (this->widget_vector2f(scope, param_label, val, this->GetMinValue<T>(), this->GetMaxValue<T>())) {
                    this->SetValue(val);
                    retval = true;
                }
                error = false;
            }
            // VECTOR 3 --------------------------------------------
            else if constexpr (std::is_same_v<T, glm::vec3>) {
                auto val = arg;
                if (this->widget_vector3f(scope, param_label, val, this->GetMinValue<T>(), this->GetMaxValue<T>())) {
                    this->SetValue(val);
                    retval = true;
                }
                error = false;
            } else if constexpr (std::is_same_v<T, glm::vec4>) {
                switch (this->type) {
                    // VECTOR 4 ----------------------------------------
                case (ParamType_t::VECTOR4F): {
                    auto val = arg;
                    if (this->widget_vector4f(
                            scope, param_label, val, this->GetMinValue<T>(), this->GetMaxValue<T>())) {
                        this->SetValue(val);
                        retval = true;
                    }
                    error = false;
                } break;
                    // COLOR -------------------------------------------
                case (ParamType_t::COLOR): {
                    auto val = arg;
                    if (this->widget_vector4f(scope, param_label, val, glm::vec4(0.0f), glm::vec4(1.0f))) {
                        this->SetValue(val);
                        retval = true;
                    }
                    error = false;
                } break;
                default:
                    break;
                }
            } else if constexpr (std::is_same_v<T, std::monostate>) {
                switch (this->type) {
                    // BUTTON ------------------------------------------
                case (ParamType_t::BUTTON): {
                    if (this->widget_button(scope, param_label, this->GetStorage<megamol::core::view::KeyCode>())) {
                        this->ForceSetValueDirty();
                        retval = true;
                    }
                    error = false;
                } break;
                default:
                    break;
                }
            }
        } break;
            // STRING //////////////////////////////////////////////////
        case (Present_t::String): {
            auto val = this->GetValueString();
            if (this->widget_string(scope, param_label, val)) {
                this->SetValueString(val);
                retval = true;
            }
            error = false;
        } break;
            // COLOR ///////////////////////////////////////////////////
        case (Present_t::Color): {
            if constexpr (std::is_same_v<T, glm::vec4>) {
                switch (this->type) {
                    // VECTOR 4 ----------------------------------------
                case (ParamType_t::VECTOR4F): {
                    auto val = arg;
                    if (this->widget_color(scope, param_label, val)) {
                        this->SetValue(val);
                        retval = true;
                    }
                    error = false;
                } break;
                    // COLOR -------------------------------------------
                case (ParamType_t::COLOR): {
                    auto val = arg;
                    if (this->widget_color(scope, param_label, val)) {
                        this->SetValue(val);
                        retval = true;
                    }
                    error = false;
                } break;
                default:
                    break;
                }
            }
        } break;
            // FILE PATH ///////////////////////////////////////////////
        case (Present_t::FilePath): {
            if constexpr (std::is_same_v<T, std::string>) {
                switch (this->type) {
                    // FILE PATH ---------------------------------------
                case (ParamType_t::FILEPATH): {
                    auto val = arg;
                    if (this->widget_filepath(scope, param_label, val)) {
                        this->SetValue(val);
                        retval = true;
                    }
                    error = false;
                } break;
                default:
                    break;
                }
            }
        } break;
            // TRANSFER FUNCTION ///////////////////////////////////////
        case (Present_t::TransferFunction): {
            if constexpr (std::is_same_v<T, std::string>) {
                switch (this->type) {
                    // TRANSFER FUNCTION -------------------------------
                case (ParamType_t::TRANSFERFUNCTION): {
                    auto val = arg;
                    if (this->widget_transfer_function_editor(scope)) {
                        retval = true;
                    }
                    error = false;
                } break;
                default:
                    break;
                }
            }
        } break;
            // PIN VALUE TO MOUSE //////////////////////////////////////
        case (Present_t::PinMouse): {
            bool compatible_type = false;
            // FLOAT -----------------------------------------------
            if constexpr (std::is_same_v<T, float>) {
                compatible_type = true;
            } else if constexpr (std::is_same_v<T, int>) {
                switch (this->type) {
                    // INT ---------------------------------------------
                case (ParamType_t::INT): {
                    compatible_type = true;
                } break;
                default:
                    break;
                }
            }
            // VECTOR 2 --------------------------------------------
            else if constexpr (std::is_same_v<T, glm::vec2>) {
                compatible_type = true;
            }
            // VECTOR 3 --------------------------------------------
            else if constexpr (std::is_same_v<T, glm::vec3>) {
                compatible_type = true;
            } else if constexpr (std::is_same_v<T, glm::vec4>) {
                switch (this->type) {
                    // VECTOR 4 ----------------------------------------
                case (ParamType_t::VECTOR4F): {
                    compatible_type = true;
                } break;
                default:
                    break;
                }
            }
            if (compatible_type) {
                this->widget_pinvaluetomouse(scope, param_label, this->GetValueString());
                error = false;
            }
        } break;
            // KNOB //////////////////////////////////////////////////
        case (Present_t::Knob): {
            // FLOAT -----------------------------------------------
            if constexpr (std::is_same_v<T, float>) {
                auto val = arg;
                if (this->widget_knob(scope, param_label, val, this->GetMinValue<T>(), this->GetMaxValue<T>())) {
                    this->SetValue(val);
                    retval = true;
                }
                error = false;
            }
        } break;
            // SLIDER ////////////////////////////////////////////////
            // DRAG //////////////////////////////////////////////////
        case (Present_t::Slider):
        case (Present_t::Drag): {
            // FLOAT -----------------------------------------------
            if constexpr (std::is_same_v<T, float>) {
                auto val = arg;
                if (this->widget_float(scope, param_label, val, this->GetMinValue<T>(), this->GetMaxValue<T>())) {
                    this->SetValue(val);
                    retval = true;
                }
                error = false;
            } else if constexpr (std::is_same_v<T, int>) {
                switch (this->type) {
                    // INT ---------------------------------------------
                case (ParamType_t::INT): {
                    auto val = arg;
                    if (this->widget_int(scope, param_label, val, this->GetMinValue<T>(), this->GetMaxValue<T>())) {
                        this->SetValue(val);
                        retval = true;
                    }
                    error = false;
                } break;
                default:
                    break;
                }
            }
            // VECTOR 2 --------------------------------------------
            else if constexpr (std::is_same_v<T, glm::vec2>) {
                auto val = arg;
                if (this->widget_vector2f(scope, param_label, val, this->GetMinValue<T>(), this->GetMaxValue<T>())) {
                    this->SetValue(val);
                    retval = true;
                }
                error = false;
            }
            // VECTOR 3 --------------------------------------------
            else if constexpr (std::is_same_v<T, glm::vec3>) {
                auto val = arg;
                if (this->widget_vector3f(scope, param_label, val, this->GetMinValue<T>(), this->GetMaxValue<T>())) {
                    this->SetValue(val);
                    retval = true;
                }
                error = false;
            } else if constexpr (std::is_same_v<T, glm::vec4>) {
                switch (this->type) {
                    // VECTOR 4 ----------------------------------------
                case (ParamType_t::VECTOR4F): {
                    auto val = arg;
                    if (this->widget_vector4f(
                            scope, param_label, val, this->GetMinValue<T>(), this->GetMaxValue<T>())) {
                        this->SetValue(val);
                        retval = true;
                    }
                    error = false;
                } break;
                default:
                    break;
                }
            }
        } break;
            // 3D ROTATION //////////////////////////////////////////////////
        case (Present_t::Rotation): {
            // FLOAT -----------------------------------------------
            if constexpr (std::is_same_v<T, glm::vec4>) {
                switch (this->type) {
                    // VECTOR 4 ----------------------------------------
                case (ParamType_t::VECTOR4F): {
                    auto val = arg;
                    if (this->widget_rotation_axes(
                            scope, param_label, val, this->GetMinValue<T>(), this->GetMaxValue<T>())) {
                        this->SetValue(val);
                        retval = true;
                    }
                    error = false;
                } break;
                default:
                    break;
                }
            }
        } break;
            // 3D DIRECTION //////////////////////////////////////////////////
        case (Present_t::Direction): {
            // FLOAT -----------------------------------------------
            if constexpr (std::is_same_v<T, glm::vec3>) {
                switch (this->type) {
                    // VECTOR 3 ----------------------------------------
                case (ParamType_t::VECTOR3F): {
                    auto val = arg;
                    if (this->widget_rotation_direction(
                            scope, param_label, val, this->GetMinValue<T>(), this->GetMaxValue<T>())) {
                        this->SetValue(val);
                        retval = true;
                    }
                    error = false;
                } break;
                default:
                    break;
                }
            }
        } break;
            // CHECKBOX //////////////////////////////////////////////////
        case (Present_t::Checkbox): {
            // BOOL ------------------------------------------------
            if constexpr (std::is_same_v<T, bool>) {
                auto val = arg;
                if (this->widget_bool(scope, param_label, val)) {
                    this->SetValue(val);
                    retval = true;
                }
                error = false;
            }
        } break;
        default:
            break;
        }

        // LOCAL -----------------------------------------------------------
        if (scope == megamol::gui::Parameter::WidgetScope::LOCAL) {
            // Reset read only
            gui_utils::PopReadOnly(this->IsGUIReadOnly());
            // Reset item width
            ImGui::PopItemWidth();
        }
    };

    std::visit(visitor, this->GetValue());

    if (error) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No widget presentation '%s' available for '%s' . [%s, %s, line %d]\n",
            this->GetPresentationName(this->GetGUIPresentation()).c_str(),
            megamol::core::param::AbstractParamPresentation::GetTypeName(this->type).c_str(), __FILE__, __FUNCTION__,
            __LINE__);
    }

    return retval;
}


bool megamol::gui::Parameter::widget_button(
    megamol::gui::Parameter::WidgetScope scope, const std::string& label, const megamol::core::view::KeyCode& keycode) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == megamol::gui::Parameter::WidgetScope::LOCAL) {
        std::string button_hotkey = keycode.ToString();
        std::string hotkey;
        std::string edit_label = label;

        bool hotkey_in_tooltip = false;
        bool hotkey_in_label = true;

        // Add hotkey to hover tooltip
        if (hotkey_in_tooltip) {
            if (!button_hotkey.empty())
                hotkey = "\n Hotkey: " + button_hotkey;
            this->gui_tooltip_text += hotkey;
        }
        // Add hotkey to param label
        if (hotkey_in_label) {
            if (!button_hotkey.empty())
                hotkey = " [" + button_hotkey + "]";
            edit_label += hotkey;
        }

        retval = ImGui::Button(edit_label.c_str());
    }
    return retval;
}


bool megamol::gui::Parameter::widget_bool(
    megamol::gui::Parameter::WidgetScope scope, const std::string& label, bool& val) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == megamol::gui::Parameter::WidgetScope::LOCAL) {
        auto p = this->GetGUIPresentation();

        if (p == Present_t::Checkbox) {
            retval = ImGui::Checkbox(label.c_str(), &val);
        } else { // Present_t::Basic
            retval = megamol::gui::ButtonWidgets::ToggleButton(label, val);
        }
    }
    return retval;
}


bool megamol::gui::Parameter::widget_string(
    megamol::gui::Parameter::WidgetScope scope, const std::string& label, std::string& val) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == megamol::gui::Parameter::WidgetScope::LOCAL) {
        ImGui::BeginGroup();
        /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
        if (!std::holds_alternative<std::string>(this->gui_widget_store)) {
            std::string utf8Str = val;
            gui_utils::Utf8Encode(utf8Str);
            this->gui_widget_store = utf8Str;
        }
        std::string hidden_label = "###" + label;

        // Determine multi line count of string
        int multiline_cnt = static_cast<int>(std::count(std::get<std::string>(this->gui_widget_store).begin(),
            std::get<std::string>(this->gui_widget_store).end(), '\n'));
        multiline_cnt = std::min(static_cast<int>(GUI_MAX_MULITLINE), multiline_cnt);
        ImVec2 multiline_size = ImVec2(ImGui::CalcItemWidth(),
            ImGui::GetFrameHeightWithSpacing() + (ImGui::GetFontSize() * static_cast<float>(multiline_cnt)));
        ImGui::InputTextMultiline(hidden_label.c_str(), &std::get<std::string>(this->gui_widget_store), multiline_size,
            ImGuiInputTextFlags_CtrlEnterForNewLine);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            std::string utf8Str = std::get<std::string>(this->gui_widget_store);
            gui_utils::Utf8Decode(utf8Str);
            val = utf8Str;
            retval = true;
        } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
            std::string utf8Str = val;
            gui_utils::Utf8Encode(utf8Str);
            this->gui_widget_store = utf8Str;
        }
        ImGui::SameLine();

        ImGui::TextUnformatted(label.c_str());
        ImGui::EndGroup();

        this->gui_help = "[Ctrl + Enter] for new line.\nPress [Return] to confirm changes.";
    }
    return retval;
}


bool megamol::gui::Parameter::widget_color(
    megamol::gui::Parameter::WidgetScope scope, const std::string& label, glm::vec4& val) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == megamol::gui::Parameter::WidgetScope::LOCAL) {
        auto color_flags = ImGuiColorEditFlags_AlphaPreview; // | ImGuiColorEditFlags_Float;
        retval = ImGui::ColorEdit4(label.c_str(), glm::value_ptr(val), color_flags);

        this->gui_help = "[Left Click] on the colored square to open a color picker.\n"
                         "[CTRL + Left Click] on individual component to input value.\n"
                         "[Right Click] on the individual color widget to show options.";
    }
    return retval;
}


bool megamol::gui::Parameter::widget_enum(
    megamol::gui::Parameter::WidgetScope scope, const std::string& label, int& val, EnumStorage_t store) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == megamol::gui::Parameter::WidgetScope::LOCAL) {
        /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
        std::string utf8Str = store[val];
        gui_utils::Utf8Encode(utf8Str);
        auto combo_flags = ImGuiComboFlags_HeightRegular;
        if (ImGui::BeginCombo(label.c_str(), utf8Str.c_str(), combo_flags)) {
            for (auto& pair : store) {
                bool isSelected = (pair.first == val);
                utf8Str = pair.second;
                gui_utils::Utf8Encode(utf8Str);
                if (ImGui::Selectable(utf8Str.c_str(), isSelected)) {
                    val = pair.first;
                    retval = true;
                }
            }
            ImGui::EndCombo();
        }
    }
    return retval;
}


bool megamol::gui::Parameter::widget_flexenum(megamol::gui::Parameter::WidgetScope scope, const std::string& label,
    std::string& val, megamol::core::param::FlexEnumParam::Storage_t store) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == megamol::gui::Parameter::WidgetScope::LOCAL) {
        /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
        if (!std::holds_alternative<std::string>(this->gui_widget_store)) {
            this->gui_widget_store = std::string();
        }
        std::string utf8Str = val;
        gui_utils::Utf8Encode(utf8Str);
        auto combo_flags = ImGuiComboFlags_HeightRegular;
        if (ImGui::BeginCombo(label.c_str(), utf8Str.c_str(), combo_flags)) {
            bool one_present = false;
            for (auto& valueOption : store) {
                bool isSelected = (valueOption == val);
                utf8Str = valueOption;
                gui_utils::Utf8Encode(utf8Str);
                if (ImGui::Selectable(utf8Str.c_str(), isSelected)) {
                    gui_utils::Utf8Decode(utf8Str);
                    val = utf8Str;
                    retval = true;
                }
                if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                }
                one_present = true;
            }
            if (one_present) {
                ImGui::Separator();
            }
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted("Add");
            ImGui::SameLine();
            /// Keyboard focus needs to be set in/until second frame
            if (this->gui_set_focus < 2) {
                ImGui::SetKeyboardFocusHere();
                this->gui_set_focus++;
            }
            ImGui::InputText(
                "###flex_enum_text_edit", &std::get<std::string>(this->gui_widget_store), ImGuiInputTextFlags_None);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                if (!std::get<std::string>(this->gui_widget_store).empty()) {
                    gui_utils::Utf8Decode(std::get<std::string>(this->gui_widget_store));
                    val = std::get<std::string>(this->gui_widget_store);
                    retval = true;
                    std::get<std::string>(this->gui_widget_store) = std::string();
                }
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndCombo();
        } else {
            this->gui_set_focus = 0;
        }
        this->gui_help = "Only selected value will be saved to project file";
    }
    return retval;
}


bool megamol::gui::Parameter::widget_filepath(
    megamol::gui::Parameter::WidgetScope scope, const std::string& label, std::string& val) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == megamol::gui::Parameter::WidgetScope::LOCAL) {
        ImGui::BeginGroup();
        /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
        if (!std::holds_alternative<std::string>(this->gui_widget_store)) {
            std::string utf8Str = val;
            gui_utils::Utf8Encode(utf8Str);
            this->gui_widget_store = utf8Str;
        }
        ImGuiStyle& style = ImGui::GetStyle();

        float widget_width = ImGui::CalcItemWidth() - (ImGui::GetFrameHeightWithSpacing() + style.ItemSpacing.x);
        float text_width = ImGui::CalcTextSize(std::get<std::string>(this->gui_widget_store).c_str()).x +
                           (2.0f * style.ItemInnerSpacing.x);
        widget_width = std::max(widget_width, text_width);

        ImGui::PushItemWidth(widget_width);
        /// TODO Provide path to file name of parent graph/project
        bool button_edit =
            this->gui_file_browser.Button_Select({}, std::get<std::string>(this->gui_widget_store), false, "");
        ImGui::SameLine();
        ImGui::InputText(label.c_str(), &std::get<std::string>(this->gui_widget_store), ImGuiInputTextFlags_None);
        if (button_edit || ImGui::IsItemDeactivatedAfterEdit()) {
            gui_utils::Utf8Decode(std::get<std::string>(this->gui_widget_store));
            val = std::get<std::string>(this->gui_widget_store);
            retval = true;
        } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
            std::string utf8Str = val;
            gui_utils::Utf8Encode(utf8Str);
            this->gui_widget_store = utf8Str;
        }
        ImGui::PopItemWidth();
        ImGui::EndGroup();
    }
    return retval;
}


bool megamol::gui::Parameter::widget_ternary(
    megamol::gui::Parameter::WidgetScope scope, const std::string& label, vislib::math::Ternary& val) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == megamol::gui::Parameter::WidgetScope::LOCAL) {
        ImGui::BeginGroup();
        if (ImGui::RadioButton("True", val.IsTrue())) {
            val = vislib::math::Ternary::TRI_TRUE;
            retval = true;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("False", val.IsFalse())) {
            val = vislib::math::Ternary::TRI_FALSE;
            retval = true;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Unknown", val.IsUnknown())) {
            val = vislib::math::Ternary::TRI_UNKNOWN;
            retval = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        ImGui::TextUnformatted(label.c_str());
        ImGui::EndGroup();
    }
    return retval;
}


bool megamol::gui::Parameter::widget_int(
    megamol::gui::Parameter::WidgetScope scope, const std::string& label, int& val, int minv, int maxv) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == megamol::gui::Parameter::WidgetScope::LOCAL) {
        if (!std::holds_alternative<int>(this->gui_widget_store)) {
            this->gui_widget_store = val;
        }
        auto p = this->GetGUIPresentation();

        // Min Max Values
        ImGui::BeginGroup();
        if (ImGui::ArrowButton("###_min_max", ((this->gui_show_minmax) ? (ImGuiDir_Down) : (ImGuiDir_Up)))) {
            this->gui_show_minmax = !this->gui_show_minmax;
        }
        this->gui_tooltip.ToolTip("Min/Max Values");
        ImGui::SameLine();

        // Relative step size
        int min_step_size = 1;
        int max_step_size = 10;
        if ((minv > INT_MIN) && (maxv < INT_MAX)) {
            min_step_size = static_cast<int>(static_cast<float>(maxv - minv) * 0.003f); // 0.3%
            max_step_size = static_cast<int>(static_cast<float>(maxv - minv) * 0.03f);  // 3%
        }

        // Value
        if (p == Present_t::Slider) {
            const int offset = 2;
            auto slider_min = (minv > INT_MIN) ? (minv) : ((val == 0) ? (-offset) : (val - (offset * val)));
            auto slider_max = (maxv < INT_MAX) ? (maxv) : ((val == 0) ? (offset) : (val + (offset * val)));
            ImGui::SliderInt(label.c_str(), &std::get<int>(this->gui_widget_store), slider_min, slider_max);
            this->gui_help = "[Ctrl + Click] to turn slider into an input box.";
        } else if (p == Present_t::Drag) {
            ImGui::DragInt(
                label.c_str(), &std::get<int>(this->gui_widget_store), static_cast<float>(min_step_size), minv, maxv);
            this->gui_help = "[Ctrl + Click] to turn slider into an input box.";
        } else { // Present_t::Basic
            ImGui::InputInt(label.c_str(), &std::get<int>(this->gui_widget_store), min_step_size, max_step_size,
                ImGuiInputTextFlags_None);
        }
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            this->gui_widget_store = std::max(minv, std::min(std::get<int>(this->gui_widget_store), maxv));
            val = std::get<int>(this->gui_widget_store);
            retval = true;
        } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
            this->gui_widget_store = val;
        }
        if (this->gui_show_minmax) {
            gui_utils::PushReadOnly();
            auto min_value = minv;
            ImGui::InputInt("Min Value", &min_value, min_step_size, max_step_size, ImGuiInputTextFlags_None);
            auto max_value = maxv;
            ImGui::InputInt("Max Value", &max_value, min_step_size, max_step_size, ImGuiInputTextFlags_None);
            gui_utils::PopReadOnly();
        }
        ImGui::EndGroup();
    }
    return retval;
}


bool megamol::gui::Parameter::widget_float(
    megamol::gui::Parameter::WidgetScope scope, const std::string& label, float& val, float minv, float maxv) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == megamol::gui::Parameter::WidgetScope::LOCAL) {
        if (!std::holds_alternative<float>(this->gui_widget_store)) {
            this->gui_widget_store = val;
        }

        auto p = this->GetGUIPresentation();
        ImGui::BeginGroup();

        // Min Max Option
        if ((p == Present_t::Basic) || (p == Present_t::Slider) || (p == Present_t::Drag)) {
            if (ImGui::ArrowButton("###_min_max", ((this->gui_show_minmax) ? (ImGuiDir_Down) : (ImGuiDir_Up)))) {
                this->gui_show_minmax = !this->gui_show_minmax;
            }
            this->gui_tooltip.ToolTip("Min/Max Values");
            ImGui::SameLine();
        }

        // Relative step size
        float min_step_size = 1.0f;
        float max_step_size = 10.0f;
        if ((minv > -FLT_MAX) && (maxv < FLT_MAX)) {
            min_step_size = (maxv - minv) * 0.003f; // 0.3%
            max_step_size = (maxv - minv) * 0.03f;  // 3%
        }

        // Value
        if (p == Present_t::Slider) {
            const float offset = 2.0f;
            auto slider_min = (minv > -FLT_MAX) ? (minv) : ((val == 0.0f) ? (-offset) : (val - (offset * val)));
            auto slider_max = (maxv < FLT_MAX) ? (maxv) : ((val == 0.0f) ? (offset) : (val + (offset * val)));
            ImGui::SliderFloat(label.c_str(), &std::get<float>(this->gui_widget_store), slider_min, slider_max,
                this->gui_float_format.c_str());
            this->gui_help = "[Ctrl + Click] to turn slider into an input box.";
        } else if (p == Present_t::Drag) {
            ImGui::DragFloat(label.c_str(), &std::get<float>(this->gui_widget_store), min_step_size, minv, maxv);
            this->gui_help = "[Ctrl + Click] to turn slider into an input box.";
        } else { // Present_t::Basic
            ImGui::InputFloat(label.c_str(), &std::get<float>(this->gui_widget_store), min_step_size, max_step_size,
                this->gui_float_format.c_str(), ImGuiInputTextFlags_None);
        }
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            this->gui_widget_store = std::max(minv, std::min(std::get<float>(this->gui_widget_store), maxv));
            val = std::get<float>(this->gui_widget_store);
            retval = true;
        } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
            this->gui_widget_store = val;
        }

        // Min Max Values
        if ((p == Present_t::Basic) || (p == Present_t::Slider) || (p == Present_t::Drag)) {
            if (this->gui_show_minmax) {
                gui_utils::PushReadOnly();
                auto min_value = minv;
                ImGui::InputFloat("Min Value", &min_value, min_step_size, max_step_size, this->gui_float_format.c_str(),
                    ImGuiInputTextFlags_None);
                auto max_value = maxv;
                ImGui::InputFloat("Max Value", &max_value, min_step_size, max_step_size, this->gui_float_format.c_str(),
                    ImGuiInputTextFlags_None);
                gui_utils::PopReadOnly();
            }
        }
        ImGui::EndGroup();
    }
    return retval;
}


bool megamol::gui::Parameter::widget_vector2f(megamol::gui::Parameter::WidgetScope scope, const std::string& label,
    glm::vec2& val, glm::vec2 minv, glm::vec2 maxv) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == megamol::gui::Parameter::WidgetScope::LOCAL) {
        if (!std::holds_alternative<glm::vec2>(this->gui_widget_store)) {
            this->gui_widget_store = val;
        }

        auto p = this->GetGUIPresentation();
        ImGui::BeginGroup();

        // Min Max Option
        if ((p == Present_t::Basic) || (p == Present_t::Slider) || (p == Present_t::Drag)) {
            if (ImGui::ArrowButton("###_min_max", ((this->gui_show_minmax) ? (ImGuiDir_Down) : (ImGuiDir_Up)))) {
                this->gui_show_minmax = !this->gui_show_minmax;
            }
            this->gui_tooltip.ToolTip("Min/Max Values");
            ImGui::SameLine();
        }
        float vec_min = std::max(minv.x, minv.y);
        float vec_max = std::min(maxv.x, maxv.y);

        // Value
        if (p == Present_t::Slider) {
            const float offset = 2.0f;
            float value_min = std::min(val.x, val.y);
            float value_max = std::max(val.x, val.y);
            auto slider_min =
                std::max(vec_min, ((value_min == 0.0f) ? (-offset) : (value_min - (offset * fabsf(value_min)))));
            auto slider_max =
                std::min(vec_max, ((value_max == 0.0f) ? (offset) : (value_max + (offset * fabsf(value_max)))));
            ImGui::SliderFloat2(label.c_str(), glm::value_ptr(std::get<glm::vec2>(this->gui_widget_store)), slider_min,
                slider_max, this->gui_float_format.c_str());
            this->gui_help = "[Ctrl + Click] to turn slider into an input box.";
        } else if (p == Present_t::Drag) {
            // Relative step size
            float min_step_size = 1.0f;
            if ((vec_min > -FLT_MAX) && (vec_max < FLT_MAX)) {
                min_step_size = (vec_max - vec_min) * 0.003f; // 0.3%
            }
            ImGui::DragFloat2(label.c_str(), glm::value_ptr(std::get<glm::vec2>(this->gui_widget_store)), min_step_size,
                vec_min, vec_max);
            this->gui_help = "[Ctrl + Click] to turn slider into an input box.";
        } else { // Present_t::Basic
            ImGui::InputFloat2(label.c_str(), glm::value_ptr(std::get<glm::vec2>(this->gui_widget_store)),
                this->gui_float_format.c_str(), ImGuiInputTextFlags_None);
        }
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            auto x = std::max(minv.x, std::min(std::get<glm::vec2>(this->gui_widget_store).x, maxv.x));
            auto y = std::max(minv.y, std::min(std::get<glm::vec2>(this->gui_widget_store).y, maxv.y));
            this->gui_widget_store = glm::vec2(x, y);
            val = std::get<glm::vec2>(this->gui_widget_store);
            retval = true;
        } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
            this->gui_widget_store = val;
        }

        // Min Max Values
        if ((p == Present_t::Basic) || (p == Present_t::Slider) || (p == Present_t::Drag)) {
            if (this->gui_show_minmax) {
                gui_utils::PushReadOnly();
                auto min_value = minv;
                ImGui::InputFloat2(
                    "Min Value", glm::value_ptr(min_value), this->gui_float_format.c_str(), ImGuiInputTextFlags_None);
                auto max_value = maxv;
                ImGui::InputFloat2(
                    "Max Value", glm::value_ptr(max_value), this->gui_float_format.c_str(), ImGuiInputTextFlags_None);
                gui_utils::PopReadOnly();
            }
        }
        ImGui::EndGroup();
    }
    return retval;
}


bool megamol::gui::Parameter::widget_vector3f(megamol::gui::Parameter::WidgetScope scope, const std::string& label,
    glm::vec3& val, glm::vec3 minv, glm::vec3 maxv) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == megamol::gui::Parameter::WidgetScope::LOCAL) {
        if (!std::holds_alternative<glm::vec3>(this->gui_widget_store)) {
            this->gui_widget_store = val;
        }

        auto p = this->GetGUIPresentation();
        ImGui::BeginGroup();

        // Min Max Option
        if ((p == Present_t::Basic) || (p == Present_t::Slider) || (p == Present_t::Drag)) {
            if (ImGui::ArrowButton("###_min_max", ((this->gui_show_minmax) ? (ImGuiDir_Down) : (ImGuiDir_Up)))) {
                this->gui_show_minmax = !this->gui_show_minmax;
            }
            this->gui_tooltip.ToolTip("Min/Max Values");
            ImGui::SameLine();
        }

        float vec_min = std::max(minv.x, std::max(minv.y, minv.z));
        float vec_max = std::min(maxv.x, std::min(maxv.y, maxv.z));

        // Value
        if (p == Present_t::Slider) {
            const float offset = 2.0f;
            float value_min = std::min(val.x, std::min(val.y, val.z));
            float value_max = std::max(val.x, std::max(val.y, val.z));
            auto slider_min =
                std::max(vec_min, ((value_min == 0.0f) ? (-offset) : (value_min - (offset * fabsf(value_min)))));
            auto slider_max =
                std::min(vec_max, ((value_max == 0.0f) ? (offset) : (value_max + (offset * fabsf(value_max)))));
            ImGui::SliderFloat3(label.c_str(), glm::value_ptr(std::get<glm::vec3>(this->gui_widget_store)), slider_min,
                slider_max, this->gui_float_format.c_str());
            this->gui_help = "[Ctrl + Click] to turn slider into an input box.";
        } else if (p == Present_t::Drag) {
            // Relative step size
            float min_step_size = 1.0f;
            if ((vec_min > -FLT_MAX) && (vec_max < FLT_MAX)) {
                min_step_size = (vec_max - vec_min) * 0.003f; // 0.3%
            }
            ImGui::DragFloat3(label.c_str(), glm::value_ptr(std::get<glm::vec3>(this->gui_widget_store)), min_step_size,
                vec_min, vec_max);
            this->gui_help = "[Ctrl + Click] to turn slider into an input box.";
        } else { // Present_t::Basic
            ImGui::InputFloat3(label.c_str(), glm::value_ptr(std::get<glm::vec3>(this->gui_widget_store)),
                this->gui_float_format.c_str(), ImGuiInputTextFlags_None);
        }
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            auto x = std::max(minv.x, std::min(std::get<glm::vec3>(this->gui_widget_store).x, maxv.x));
            auto y = std::max(minv.y, std::min(std::get<glm::vec3>(this->gui_widget_store).y, maxv.y));
            auto z = std::max(minv.z, std::min(std::get<glm::vec3>(this->gui_widget_store).z, maxv.z));
            this->gui_widget_store = glm::vec3(x, y, z);
            val = std::get<glm::vec3>(this->gui_widget_store);
            retval = true;
        } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
            this->gui_widget_store = val;
        }

        // Min Max Values
        if ((p == Present_t::Basic) || (p == Present_t::Slider) || (p == Present_t::Drag)) {
            if (this->gui_show_minmax) {
                gui_utils::PushReadOnly();
                auto min_value = minv;
                ImGui::InputFloat3(
                    "Min Value", glm::value_ptr(min_value), this->gui_float_format.c_str(), ImGuiInputTextFlags_None);
                auto max_value = maxv;
                ImGui::InputFloat3(
                    "Max Value", glm::value_ptr(max_value), this->gui_float_format.c_str(), ImGuiInputTextFlags_None);
                gui_utils::PopReadOnly();
            }
        }
        ImGui::EndGroup();
    }
    return retval;
}


bool megamol::gui::Parameter::widget_vector4f(megamol::gui::Parameter::WidgetScope scope, const std::string& label,
    glm::vec4& val, glm::vec4 minv, glm::vec4 maxv) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == megamol::gui::Parameter::WidgetScope::LOCAL) {
        if (!std::holds_alternative<glm::vec4>(this->gui_widget_store)) {
            this->gui_widget_store = val;
        }

        auto p = this->GetGUIPresentation();
        ImGui::BeginGroup();

        // Min Max Option
        if ((p == Present_t::Basic) || (p == Present_t::Slider) || (p == Present_t::Drag)) {
            if (ImGui::ArrowButton("###_min_max", ((this->gui_show_minmax) ? (ImGuiDir_Down) : (ImGuiDir_Up)))) {
                this->gui_show_minmax = !this->gui_show_minmax;
            }
            this->gui_tooltip.ToolTip("Min/Max Values");
            ImGui::SameLine();
        }
        float vec_min = std::max(minv.x, std::max(minv.y, std::max(minv.z, minv.w)));
        float vec_max = std::min(maxv.x, std::min(maxv.y, std::min(maxv.z, maxv.w)));

        // Value
        if (p == Present_t::Slider) {
            const float offset = 2.0f;
            float value_min = std::min(val.x, std::min(val.y, std::min(val.z, val.w)));
            float value_max = std::max(val.x, std::max(val.y, std::max(val.z, val.w)));
            auto slider_min =
                std::max(vec_min, ((value_min == 0.0f) ? (-offset) : (value_min - (offset * fabsf(value_min)))));
            auto slider_max =
                std::min(vec_max, ((value_max == 0.0f) ? (offset) : (value_max + (offset * fabsf(value_max)))));
            ImGui::SliderFloat4(label.c_str(), glm::value_ptr(std::get<glm::vec4>(this->gui_widget_store)), slider_min,
                slider_max, this->gui_float_format.c_str());
            this->gui_help = "[Ctrl + Click] to turn slider into an input box.";
        } else if (p == Present_t::Drag) {
            // Relative step size
            float min_step_size = 1.0f;
            if ((vec_min > -FLT_MAX) && (vec_max < FLT_MAX)) {
                min_step_size = (vec_max - vec_min) * 0.003f; // 0.3%
            }
            ImGui::DragFloat4(label.c_str(), glm::value_ptr(std::get<glm::vec4>(this->gui_widget_store)), min_step_size,
                vec_min, vec_max);
            this->gui_help = "[Ctrl + Click] to turn slider into an input box.";
        } else { // Present_t::Basic
            ImGui::InputFloat4(label.c_str(), glm::value_ptr(std::get<glm::vec4>(this->gui_widget_store)),
                this->gui_float_format.c_str(), ImGuiInputTextFlags_None);
        }
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            auto x = std::max(minv.x, std::min(std::get<glm::vec4>(this->gui_widget_store).x, maxv.x));
            auto y = std::max(minv.y, std::min(std::get<glm::vec4>(this->gui_widget_store).y, maxv.y));
            auto z = std::max(minv.z, std::min(std::get<glm::vec4>(this->gui_widget_store).z, maxv.z));
            auto w = std::max(minv.w, std::min(std::get<glm::vec4>(this->gui_widget_store).w, maxv.w));
            this->gui_widget_store = glm::vec4(x, y, z, w);
            val = std::get<glm::vec4>(this->gui_widget_store);
            retval = true;
        } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
            this->gui_widget_store = val;
        }

        // Min Max Values
        if ((p == Present_t::Basic) || (p == Present_t::Slider) || (p == Present_t::Drag)) {
            if (this->gui_show_minmax) {
                gui_utils::PushReadOnly();
                auto min_value = minv;
                ImGui::InputFloat4(
                    "Min Value", glm::value_ptr(min_value), this->gui_float_format.c_str(), ImGuiInputTextFlags_None);
                auto max_value = maxv;
                ImGui::InputFloat4(
                    "Max Value", glm::value_ptr(max_value), this->gui_float_format.c_str(), ImGuiInputTextFlags_None);
                gui_utils::PopReadOnly();
            }
        }
        ImGui::EndGroup();
    }
    return retval;
}


bool megamol::gui::Parameter::widget_pinvaluetomouse(
    megamol::gui::Parameter::WidgetScope scope, const std::string& label, const std::string& val) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == megamol::gui::Parameter::WidgetScope::LOCAL) {

        ImGui::TextDisabled(label.c_str());
    }
    // GLOBAL -----------------------------------------------------------
    else if (scope == megamol::gui::Parameter::WidgetScope::GLOBAL) {

        auto hoverFlags = ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_AllowWhenDisabled |
                          ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem;
        // Show only if mouse is outside any gui window
        if (!ImGui::IsWindowHovered(hoverFlags)) {
            ImGui::BeginTooltip();
            ImGui::TextDisabled(label.c_str());
            ImGui::SameLine();
            ImGui::TextUnformatted(val.c_str());
            ImGui::EndTooltip();
        }
    }

    return retval;
}


bool megamol::gui::Parameter::widget_transfer_function_editor(megamol::gui::Parameter::WidgetScope scope) {

    bool retval = false;
    bool param_externally_connected = false;
    bool update_editor = false;
    auto val = std::get<std::string>(this->GetValue());
    std::string label = this->Name();

    ImGuiStyle& style = ImGui::GetStyle();

    if (this->tf_use_external_editor) {
        if (this->tf_editor_external_ptr != nullptr) {
            param_externally_connected = this->tf_editor_external_ptr->IsParameterConnected();
        }
    }

    // LOCAL -----------------------------------------------------------
    if (scope == megamol::gui::Parameter::WidgetScope::LOCAL) {
        ImGui::BeginGroup();

        // Reduced display of value and editor state.
        if (val.empty()) {
            ImGui::TextDisabled("{    (empty)    }");
            ImGui::SameLine();
        } else {
            // Draw texture
            if (this->gui_image_widget.IsLoaded()) {
                this->gui_image_widget.Widget(ImVec2(ImGui::CalcItemWidth(), ImGui::GetFrameHeight()));
                ImGui::SameLine(ImGui::CalcItemWidth() + style.ItemInnerSpacing.x);
            } else {
                ImGui::TextUnformatted("{ ............. }");
                ImGui::SameLine();
            }
        }

        // Label
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label.c_str(), ImGui::FindRenderedTextEnd(label.c_str()));

        // Toggle inplace and external editor, if available
        gui_utils::PushReadOnly((this->tf_editor_external_ptr == nullptr));
        if (ImGui::RadioButton("External Editor", this->tf_use_external_editor)) {
            this->tf_use_external_editor = true;
            this->tf_show_editor = false;
        }
        gui_utils::PopReadOnly((this->tf_editor_external_ptr == nullptr));
        if (this->tf_use_external_editor && (this->tf_editor_external_ptr != nullptr)) {
            ImGui::SameLine();
            gui_utils::PushReadOnly(param_externally_connected);
            if (ImGui::Button("Connect")) {
                this->tf_editor_external_ptr->SetConnectedParameter(this, this->FullNameProject());
                this->tf_editor_external_ptr->Config().show = true;
                retval = true;
            }
            gui_utils::PopReadOnly(param_externally_connected);
            ImGui::SameLine();
            bool readonly = this->tf_editor_external_ptr->Config().show;
            gui_utils::PushReadOnly(readonly);
            if (ImGui::Button("Open")) {
                this->tf_editor_external_ptr->Config().show = true;
            }
            gui_utils::PopReadOnly(readonly);
        }

        if (ImGui::RadioButton("Inplace Editor", !this->tf_use_external_editor)) {
            this->tf_use_external_editor = false;
            if (this->tf_editor_external_ptr != nullptr) {
                this->tf_editor_external_ptr->SetConnectedParameter(nullptr, "");
            }
        }
        if (!this->tf_use_external_editor) {
            ImGui::SameLine();
            if (megamol::gui::ButtonWidgets::ToggleButton(
                    ((this->tf_show_editor) ? ("Hide") : ("Show")), this->tf_show_editor)) {
                if (this->tf_show_editor) {
                    update_editor = true;
                    retval = true;
                }
            }
        }

        // Copy
        if (ImGui::Button("Copy")) {
            ImGui::SetClipboardText(val.c_str());
        }
        ImGui::SameLine();

        // Paste
        if (ImGui::Button("Paste")) {
            this->SetValue(std::string(ImGui::GetClipboardText()));
            val = std::get<std::string>(this->GetValue());
            if (this->tf_use_external_editor) {
                if (this->tf_editor_external_ptr != nullptr) {
                    this->tf_editor_external_ptr->SetTransferFunction(val, true, true);
                }
            } else {
                this->tf_editor_inplace.SetTransferFunction(val, false, true);
            }
        }
        ImGui::Separator();

        if (!this->tf_use_external_editor) { // Internal Editor

            if (this->tf_editor_hash != this->GetTransferFunctionHash()) {
                update_editor = true;
            }
            // Propagate the transfer function to the editor.
            if (update_editor) {
                this->tf_editor_inplace.SetTransferFunction(val, false, true);
            }
            // Draw inplace transfer function editor
            if (this->tf_show_editor) {
                if (this->tf_editor_inplace.Draw()) {
                    std::string val_str;
                    if (this->tf_editor_inplace.GetTransferFunction(val_str)) {
                        this->SetValue(val_str);
                        retval = true;
                    }
                }
            }

            this->tf_editor_hash = this->GetTransferFunctionHash();
        }

        /// ImGui::Separator();
        ImGui::EndGroup();
    }
    // GLOBAL -----------------------------------------------------------
    else if (scope == megamol::gui::Parameter::WidgetScope::GLOBAL) {

        if (this->tf_use_external_editor) {
            // Check for changed parameter value which should be forced to the editor once.
            if (param_externally_connected) {
                if (this->tf_editor_hash != this->GetTransferFunctionHash()) {
                    update_editor = true;
                }
            }
            // Propagate the transfer function to the editor.
            if (param_externally_connected && update_editor) {
                this->tf_editor_external_ptr->SetTransferFunction(val, true, false);
                retval = true;
            }
            this->tf_editor_hash = this->GetTransferFunctionHash();
        }
    }

    return retval;
}


bool megamol::gui::Parameter::widget_knob(
    megamol::gui::Parameter::WidgetScope scope, const std::string& label, float& val, float minv, float maxv) {
    bool retval = false;

    ImGuiStyle& style = ImGui::GetStyle();

    // LOCAL -----------------------------------------------------------
    if (scope == megamol::gui::Parameter::WidgetScope::LOCAL) {

        // Draw knob
        const float knob_size = ImGui::GetTextLineHeightWithSpacing() + ImGui::GetFrameHeightWithSpacing();
        if (ButtonWidgets::KnobButton("param_knob", knob_size, val, minv, maxv)) {
            retval = true;
        }

        ImGui::SameLine();

        // Draw Value
        std::string value_label;
        float left_widget_x_offset = knob_size + style.ItemInnerSpacing.x;
        ImVec2 pos = ImGui::GetCursorPos();
        ImGui::PushItemWidth(ImGui::CalcItemWidth() - left_widget_x_offset);

        if (this->widget_float(scope, label, val, minv, maxv)) {
            retval = true;
        }
        ImGui::PopItemWidth();

        // Draw min max
        ImGui::SetCursorPos(pos + ImVec2(0.0f, ImGui::GetFrameHeightWithSpacing()));
        if (minv > -FLT_MAX) {
            value_label = "Min: " + this->gui_float_format;
            ImGui::Text(value_label.c_str(), minv);
        } else {
            ImGui::TextUnformatted("Min: -inf");
        }
        ImGui::SameLine();
        if (maxv < FLT_MAX) {
            value_label = "Max: " + this->gui_float_format;
            ImGui::Text(value_label.c_str(), maxv);
        } else {
            ImGui::TextUnformatted("Max: inf");
        }
    }
    // GLOBAL -----------------------------------------------------------
    else if (scope == megamol::gui::Parameter::WidgetScope::GLOBAL) {

        // no global implementation ...
    }

    return retval;
}


bool megamol::gui::Parameter::widget_rotation_axes(megamol::gui::Parameter::WidgetScope scope, const std::string& label,
    glm::vec4& val, glm::vec4 minv, glm::vec4 maxv) {

    bool retval = false;
    // LOCAL -----------------------------------------------------------
    if (scope == megamol::gui::Parameter::WidgetScope::LOCAL) {

        auto x_cursor_pos = ImGui::GetCursorPosX();
        retval = this->widget_vector4f(scope, label, val, minv, maxv);
        ImGui::SetCursorPosX(x_cursor_pos);
        retval |= this->gui_rotation_widget.gizmo3D_rotation_axes(val);
        // ImGui::SameLine();
        // ImGui::TextUnformatted(label.c_str());
    }
    // GLOBAL -----------------------------------------------------------
    else if (scope == megamol::gui::Parameter::WidgetScope::GLOBAL) {

        // no global implementation ...
    }

    return retval;
}


bool megamol::gui::Parameter::widget_rotation_direction(megamol::gui::Parameter::WidgetScope scope,
    const std::string& label, glm::vec3& val, glm::vec3 minv, glm::vec3 maxv) {

    bool retval = false;
    // LOCAL -----------------------------------------------------------
    if (scope == megamol::gui::Parameter::WidgetScope::LOCAL) {

        auto x_cursor_pos = ImGui::GetCursorPosX();
        retval = this->widget_vector3f(scope, label, val, minv, maxv);
        ImGui::SetCursorPosX(x_cursor_pos);
        retval |= this->gui_rotation_widget.gizmo3D_rotation_direction(val);
        // ImGui::SameLine();
        // ImGui::TextUnformatted(label.c_str());

    }
    // GLOBAL -----------------------------------------------------------
    else if (scope == megamol::gui::Parameter::WidgetScope::GLOBAL) {

        // no global implementation ...
    }

    return retval;
}
