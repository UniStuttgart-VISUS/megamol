/*
 * Parameter.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Parameter.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::Parameter::Parameter(ImGuiID uid, Param_t type, Stroage_t store, Min_t minval, Max_t maxval)
    : uid(uid)
    , type(type)
    , present(type)
    , full_name()
    , description()
    , core_param_ptr(nullptr)
    , minval(minval)
    , maxval(maxval)
    , storage(store)
    , value()
    , tf_string_hash(0)
    , default_value()
    , default_value_mismatch(false)
    , value_dirty(false) {

    // Initialize variant types which should/can not be changed afterwards.
    // Default ctor of variants initializes std::monostate.
    switch (this->type) {
    case (Param_t::BOOL): {
        this->value = bool(false);
    } break;
    case (Param_t::BUTTON): {
        // nothing to do ...
    } break;
    case (Param_t::COLOR): {
        this->value = glm::vec4();
    } break;
    case (Param_t::ENUM): {
        this->value = int(0);
    } break;
    case (Param_t::FILEPATH): {
        this->value = std::string();
    } break;
    case (Param_t::FLEXENUM): {
        this->value = std::string();
    } break;
    case (Param_t::FLOAT): {
        this->value = float(0.0f);
    } break;
    case (Param_t::INT): {
        this->value = int();
    } break;
    case (Param_t::STRING): {
        this->value = std::string();
    } break;
    case (Param_t::TERNARY): {
        this->value = vislib::math::Ternary();
    } break;
    case (Param_t::TRANSFERFUNCTION): {
        this->value = std::string();
    } break;
    case (Param_t::VECTOR2F): {
        this->value = glm::vec2();
    } break;
    case (Param_t::VECTOR3F): {
        this->value = glm::vec3();
    } break;
    case (Param_t::VECTOR4F): {
        this->value = glm::vec4();
    } break;
    default:
        break;
    }

    this->default_value = this->value;
}


megamol::gui::Parameter::~Parameter(void) {}


std::string megamol::gui::Parameter::GetValueString(void) {

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
            case (Param_t::INT): {
                auto parameter = megamol::core::param::IntParam(arg);
                value_string = std::string(parameter.ValueString().PeekBuffer());
            } break;
            case (Param_t::ENUM): {
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
            switch (this->type) {
            case (Param_t::STRING): {
                auto parameter = megamol::core::param::StringParam(arg.c_str());
                value_string = std::string(parameter.ValueString().PeekBuffer());
            } break;
            case (Param_t::TRANSFERFUNCTION): {
                auto parameter = megamol::core::param::TransferFunctionParam(arg);
                value_string = std::string(parameter.ValueString().PeekBuffer());
            } break;
            case (Param_t::FILEPATH): {
                auto parameter = megamol::core::param::FilePathParam(arg.c_str());
                value_string = std::string(parameter.ValueString().PeekBuffer());
            } break;
            case (Param_t::FLEXENUM): {
                auto parameter = megamol::core::param::FlexEnumParam(arg.c_str());
                value_string = std::string(parameter.ValueString().PeekBuffer());
            } break;
            default:
                break;
            }
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
            case (Param_t::COLOR): {
                auto parameter = megamol::core::param::ColorParam(arg[0], arg[1], arg[2], arg[3]);
                value_string = std::string(parameter.ValueString().PeekBuffer());
            } break;

            case (Param_t::VECTOR4F): {
                auto parameter =
                    megamol::core::param::Vector4fParam(vislib::math::Vector<float, 4>(arg.x, arg.y, arg.z, arg.w));
                value_string = std::string(parameter.ValueString().PeekBuffer());
            } break;
            default:
                break;
            }
        } else if constexpr (std::is_same_v<T, std::monostate>) {
            switch (this->type) {
            case (Param_t::BUTTON): {
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
    case (Param_t::BOOL): {
        megamol::core::param::BoolParam parameter(false);
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val, set_dirty);
    } break;
    case (Param_t::BUTTON): {
        retval = true;
    } break;
    case (Param_t::COLOR): {
        megamol::core::param::ColorParam parameter(val_tstr);
        retval = parameter.ParseValue(val_tstr);
        auto value = parameter.Value();
        this->SetValue(glm::vec4(value[0], value[1], value[2], value[3]), set_default_val, set_dirty);
    } break;
    case (Param_t::ENUM): {
        megamol::core::param::EnumParam parameter(0);
        // Initialization of enum storage required
        auto map = this->GetStorage<EnumStorage_t>();
        for (auto& pair : map) {
            parameter.SetTypePair(pair.first, pair.second.c_str());
        }
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val, set_dirty);
    } break;
    case (Param_t::FILEPATH): {
        megamol::core::param::FilePathParam parameter(val_tstr.PeekBuffer());
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(std::string(parameter.Value().PeekBuffer()), set_default_val, set_dirty);
    } break;
    case (Param_t::FLEXENUM): {
        megamol::core::param::FlexEnumParam parameter(val_str);
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val, set_dirty);
    } break;
    case (Param_t::FLOAT): {
        megamol::core::param::FloatParam parameter(0.0f);
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val, set_dirty);
    } break;
    case (Param_t::INT): {
        megamol::core::param::IntParam parameter(0);
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val, set_dirty);
    } break;
    case (Param_t::STRING): {
        megamol::core::param::StringParam parameter(val_tstr.PeekBuffer());
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(std::string(parameter.Value().PeekBuffer()), set_default_val, set_dirty);
    } break;
    case (Param_t::TERNARY): {
        megamol::core::param::TernaryParam parameter(vislib::math::Ternary::TRI_UNKNOWN);
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val, set_dirty);
    } break;
    case (Param_t::TRANSFERFUNCTION): {
        megamol::core::param::TransferFunctionParam parameter;
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val, set_dirty);
    } break;
    case (Param_t::VECTOR2F): {
        megamol::core::param::Vector2fParam parameter(vislib::math::Vector<float, 2>(0.0f, 0.0f));
        retval = parameter.ParseValue(val_tstr);
        auto val = parameter.Value();
        this->SetValue(glm::vec2(val.X(), val.Y()), set_default_val, set_dirty);
    } break;
    case (Param_t::VECTOR3F): {
        megamol::core::param::Vector3fParam parameter(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
        retval = parameter.ParseValue(val_tstr);
        auto val = parameter.Value();
        this->SetValue(glm::vec3(val.X(), val.Y(), val.Z()), set_default_val, set_dirty);
    } break;
    case (Param_t::VECTOR4F): {
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

    out_param.full_name = std::string(in_param_slot.Name().PeekBuffer());
    out_param.description = std::string(in_param_slot.Description().PeekBuffer());
    out_param.gui_visibility = parameter_ptr->IsGUIVisible();
    out_param.gui_read_only = parameter_ptr->IsGUIReadOnly();
    auto core_param_presentation = static_cast<size_t>(parameter_ptr->GetGUIPresentation());
    out_param.gui_presentation = static_cast<Present_t>(core_param_presentation);

    if (auto* p_ptr = in_param_slot.Param<core::param::ButtonParam>()) {
        out_param.type = Param_t::BUTTON;
        out_param.storage = p_ptr->GetKeyCode();
    } else if (auto* p_ptr = in_param_slot.Param<core::param::BoolParam>()) {
        out_param.type = Param_t::BOOL;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::ColorParam>()) {
        out_param.type = Param_t::COLOR;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::EnumParam>()) {
        out_param.type = Param_t::ENUM;
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
        out_param.type = Param_t::FILEPATH;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::FlexEnumParam>()) {
        out_param.type = Param_t::FLEXENUM;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
        out_param.storage = p_ptr->getStorage();
    } else if (auto* p_ptr = in_param_slot.Param<core::param::FloatParam>()) {
        out_param.type = Param_t::FLOAT;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
        out_param.minval = p_ptr->MinValue();
        out_param.maxval = p_ptr->MaxValue();
    } else if (auto* p_ptr = in_param_slot.Param<core::param::IntParam>()) {
        out_param.type = Param_t::INT;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
        out_param.minval = p_ptr->MinValue();
        out_param.maxval = p_ptr->MaxValue();
    } else if (auto* p_ptr = in_param_slot.Param<core::param::StringParam>()) {
        out_param.type = Param_t::STRING;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::TernaryParam>()) {
        out_param.type = Param_t::TERNARY;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::TransferFunctionParam>()) {
        out_param.type = Param_t::TRANSFERFUNCTION;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::Vector2fParam>()) {
        out_param.type = Param_t::VECTOR2F;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
        auto minval = p_ptr->MinValue();
        out_param.minval = glm::vec2(minval.X(), minval.Y());
        auto maxval = p_ptr->MaxValue();
        out_param.maxval = glm::vec2(maxval.X(), maxval.Y());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::Vector3fParam>()) {
        out_param.type = Param_t::VECTOR3F;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
        auto minval = p_ptr->MinValue();
        out_param.minval = glm::vec3(minval.X(), minval.Y(), minval.Z());
        auto maxval = p_ptr->MaxValue();
        out_param.maxval = glm::vec3(maxval.X(), maxval.Y(), maxval.Z());
    } else if (auto* p_ptr = in_param_slot.Param<core::param::Vector4fParam>()) {
        out_param.type = Param_t::VECTOR4F;
        out_param.default_value = std::string(p_ptr->ValueString().PeekBuffer());
        auto minval = p_ptr->MinValue();
        out_param.minval = glm::vec4(minval.X(), minval.Y(), minval.Z(), minval.W());
        auto maxval = p_ptr->MaxValue();
        out_param.maxval = glm::vec4(maxval.X(), maxval.Y(), maxval.Z(), maxval.W());
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Found unknown parameter type. Please extend parameter types "
            "for the configurator. [%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        out_param.type = Param_t::UNKNOWN;
        return false;
    }

    return true;
}


bool megamol::gui::Parameter::ReadNewCoreParameterToNewParameter(megamol::core::param::ParamSlot& in_param_slot,
    std::shared_ptr<megamol::gui::Parameter>& out_param, bool set_default_val, bool set_dirty,
    bool save_core_param_pointer) {

    auto parameter_ptr = in_param_slot.Parameter();
    if (parameter_ptr.IsNull()) {
        return false;
    }

    out_param.reset();

    if (auto* p_ptr = in_param_slot.template Param<core::param::BoolParam>()) {
        out_param = std::make_shared<Parameter>(
            megamol::gui::GenerateUniqueID(), Param_t::BOOL, std::monostate(), std::monostate(), std::monostate());
        out_param->SetValue(p_ptr->Value(), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::ButtonParam>()) {
        out_param = std::make_shared<Parameter>(
            megamol::gui::GenerateUniqueID(), Param_t::BUTTON, p_ptr->GetKeyCode(), std::monostate(), std::monostate());
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::ColorParam>()) {
        out_param = std::make_shared<Parameter>(
            megamol::gui::GenerateUniqueID(), Param_t::COLOR, std::monostate(), std::monostate(), std::monostate());
        auto value = p_ptr->Value();
        out_param->SetValue(glm::vec4(value[0], value[1], value[2], value[3]), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::TransferFunctionParam>()) {
        out_param = std::make_shared<Parameter>(megamol::gui::GenerateUniqueID(), Param_t::TRANSFERFUNCTION,
            std::monostate(), std::monostate(), std::monostate());
        out_param->SetValue(p_ptr->Value(), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::EnumParam>()) {
        EnumStorage_t map;
        auto param_map = p_ptr->getMap();
        auto iter = param_map.GetConstIterator();
        while (iter.HasNext()) {
            auto pair = iter.Next();
            map.emplace(pair.Key(), std::string(pair.Value().PeekBuffer()));
        }
        out_param = std::make_shared<Parameter>(
            megamol::gui::GenerateUniqueID(), Param_t::ENUM, map, std::monostate(), std::monostate());
        out_param->SetValue(p_ptr->Value(), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::FlexEnumParam>()) {
        out_param = std::make_shared<Parameter>(megamol::gui::GenerateUniqueID(), Param_t::FLEXENUM,
            p_ptr->getStorage(), std::monostate(), std::monostate());
        out_param->SetValue(p_ptr->Value(), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::FloatParam>()) {
        out_param = std::make_shared<Parameter>(
            megamol::gui::GenerateUniqueID(), Param_t::FLOAT, std::monostate(), p_ptr->MinValue(), p_ptr->MaxValue());
        out_param->SetValue(p_ptr->Value(), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::IntParam>()) {
        out_param = std::make_shared<Parameter>(
            megamol::gui::GenerateUniqueID(), Param_t::INT, std::monostate(), p_ptr->MinValue(), p_ptr->MaxValue());
        out_param->SetValue(p_ptr->Value(), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::Vector2fParam>()) {
        auto minval = p_ptr->MinValue();
        auto maxval = p_ptr->MaxValue();
        auto val = p_ptr->Value();
        out_param = std::make_shared<Parameter>(megamol::gui::GenerateUniqueID(), Param_t::VECTOR2F, std::monostate(),
            glm::vec2(minval.X(), minval.Y()), glm::vec2(maxval.X(), maxval.Y()));
        out_param->SetValue(glm::vec2(val.X(), val.Y()), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::Vector3fParam>()) {
        auto minval = p_ptr->MinValue();
        auto maxval = p_ptr->MaxValue();
        auto val = p_ptr->Value();
        out_param = std::make_shared<Parameter>(megamol::gui::GenerateUniqueID(), Param_t::VECTOR3F, std::monostate(),
            glm::vec3(minval.X(), minval.Y(), minval.Z()), glm::vec3(maxval.X(), maxval.Y(), maxval.Z()));
        out_param->SetValue(glm::vec3(val.X(), val.Y(), val.Z()), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::Vector4fParam>()) {
        auto minval = p_ptr->MinValue();
        auto maxval = p_ptr->MaxValue();
        auto val = p_ptr->Value();
        out_param = std::make_shared<Parameter>(megamol::gui::GenerateUniqueID(), Param_t::VECTOR4F, std::monostate(),
            glm::vec4(minval.X(), minval.Y(), minval.Z(), minval.W()),
            glm::vec4(maxval.X(), maxval.Y(), maxval.Z(), maxval.W()));
        out_param->SetValue(glm::vec4(val.X(), val.Y(), val.Z(), val.W()), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.template Param<core::param::TernaryParam>()) {
        out_param = std::make_shared<Parameter>(
            megamol::gui::GenerateUniqueID(), Param_t::TERNARY, std::monostate(), std::monostate(), std::monostate());
        out_param->SetValue(p_ptr->Value(), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.Param<core::param::StringParam>()) {
        out_param = std::make_shared<Parameter>(
            megamol::gui::GenerateUniqueID(), Param_t::STRING, std::monostate(), std::monostate(), std::monostate());
        out_param->SetValue(std::string(p_ptr->Value().PeekBuffer()), set_default_val, set_dirty);
    } else if (auto* p_ptr = in_param_slot.Param<core::param::FilePathParam>()) {
        out_param = std::make_shared<Parameter>(
            megamol::gui::GenerateUniqueID(), Param_t::FILEPATH, std::monostate(), std::monostate(), std::monostate());
        out_param->SetValue(std::string(p_ptr->Value().PeekBuffer()), set_default_val, set_dirty);
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Found unknown parameter type. Please extend parameter types for the configurator. "
            "[%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    out_param->full_name = std::string(in_param_slot.Name().PeekBuffer());
    out_param->description = std::string(in_param_slot.Description().PeekBuffer());
    out_param->present.SetGUIVisible(parameter_ptr->IsGUIVisible());
    out_param->present.SetGUIReadOnly(parameter_ptr->IsGUIReadOnly());
    out_param->present.SetGUIPresentation(parameter_ptr->GetGUIPresentation());
    if (save_core_param_pointer) {
        out_param->core_param_ptr = parameter_ptr;
    }

    return true;
}


bool megamol::gui::Parameter::ReadCoreParameterToParameter(
    vislib::SmartPtr<megamol::core::param::AbstractParam>& in_param_ptr, megamol::gui::Parameter& out_param,
    bool set_default_val, bool set_dirty) {

    out_param.present.SetGUIVisible(in_param_ptr->IsGUIVisible());
    out_param.present.SetGUIReadOnly(in_param_ptr->IsGUIReadOnly());
    out_param.present.SetGUIPresentation(in_param_ptr->GetGUIPresentation());

    bool type_error = false;

    if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::ButtonParam>()) {
        if (out_param.type == Param_t::BUTTON) {
            out_param.SetStorage(p_ptr->GetKeyCode());
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::BoolParam>()) {
        if (out_param.type == Param_t::BOOL) {
            out_param.SetValue(p_ptr->Value(), set_default_val, set_dirty);
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::ColorParam>()) {
        if (out_param.type == Param_t::COLOR) {
            auto value = p_ptr->Value();
            out_param.SetValue(glm::vec4(value[0], value[1], value[2], value[3]), set_default_val, set_dirty);
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::EnumParam>()) {
        if (out_param.type == Param_t::ENUM) {
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
        if (out_param.type == Param_t::FILEPATH) {
            out_param.SetValue(std::string(p_ptr->Value().PeekBuffer()), set_default_val, set_dirty);
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::FlexEnumParam>()) {
        if (out_param.type == Param_t::FLEXENUM) {
            out_param.SetValue(p_ptr->Value(), set_default_val, set_dirty);
            out_param.SetStorage(p_ptr->getStorage());
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::FloatParam>()) {
        if (out_param.type == Param_t::FLOAT) {
            out_param.SetValue(p_ptr->Value(), set_default_val, set_dirty);
            out_param.SetMinValue(p_ptr->MinValue());
            out_param.SetMaxValue(p_ptr->MaxValue());
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::IntParam>()) {
        if (out_param.type == Param_t::INT) {
            out_param.SetValue(p_ptr->Value(), set_default_val, set_dirty);
            out_param.SetMinValue(p_ptr->MinValue());
            out_param.SetMaxValue(p_ptr->MaxValue());
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::StringParam>()) {
        if (out_param.type == Param_t::STRING) {
            out_param.SetValue(std::string(p_ptr->Value().PeekBuffer()), set_default_val, set_dirty);
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::TernaryParam>()) {
        if (out_param.type == Param_t::TERNARY) {
            out_param.SetValue(p_ptr->Value(), set_default_val, set_dirty);
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::TransferFunctionParam>()) {
        if (out_param.type == Param_t::TRANSFERFUNCTION) {
            out_param.SetValue(p_ptr->Value(), set_default_val, set_dirty);
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::Vector2fParam>()) {
        if (out_param.type == Param_t::VECTOR2F) {
            auto val = p_ptr->Value();
            out_param.SetValue(glm::vec2(val.X(), val.Y()), set_default_val, set_dirty);
            auto minval = p_ptr->MinValue();
            out_param.SetMinValue(glm::vec2(minval.X(), minval.Y()));
            auto maxval = p_ptr->MaxValue();
            out_param.SetMaxValue(glm::vec2(maxval.X(), maxval.Y()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::Vector3fParam>()) {
        if (out_param.type == Param_t::VECTOR3F) {
            auto val = p_ptr->Value();
            out_param.SetValue(glm::vec3(val.X(), val.Y(), val.Z()), set_default_val, set_dirty);
            auto minval = p_ptr->MinValue();
            out_param.SetMinValue(glm::vec3(minval.X(), minval.Y(), minval.Z()));
            auto maxval = p_ptr->MaxValue();
            out_param.SetMaxValue(glm::vec3(maxval.X(), maxval.Y(), maxval.Z()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = in_param_ptr.DynamicCast<core::param::Vector4fParam>()) {
        if (out_param.type == Param_t::VECTOR4F) {
            auto val = p_ptr->Value();
            out_param.SetValue(glm::vec4(val.X(), val.Y(), val.Z(), val.W()), set_default_val, set_dirty);
            auto minval = p_ptr->MinValue();
            out_param.SetMinValue(glm::vec4(minval.X(), minval.Y(), minval.Z(), minval.W()));
            auto maxval = p_ptr->MaxValue();
            out_param.SetMaxValue(glm::vec4(maxval.X(), maxval.Y(), maxval.Z(), maxval.W()));
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

    out_param.full_name = std::string(in_param_slot.Name().PeekBuffer());
    out_param.description = std::string(in_param_slot.Description().PeekBuffer());
    if (save_core_param_pointer) {
        out_param.core_param_ptr = parameter_ptr;
    }

    return megamol::gui::Parameter::ReadCoreParameterToParameter(parameter_ptr, out_param, set_default_val, set_dirty);
}


bool megamol::gui::Parameter::WriteCoreParameterGUIState(
    megamol::gui::Parameter& in_param, vislib::SmartPtr<megamol::core::param::AbstractParam>& out_param_ptr) {

    out_param_ptr->SetGUIVisible(in_param.present.IsGUIVisible());
    out_param_ptr->SetGUIReadOnly(in_param.present.IsGUIReadOnly());
    out_param_ptr->SetGUIPresentation(in_param.present.GetGUIPresentation());

    return true;
}


bool megamol::gui::Parameter::WriteCoreParameterValue(
    megamol::gui::Parameter& in_param, vislib::SmartPtr<megamol::core::param::AbstractParam>& out_param_ptr) {
    bool type_error = false;

    if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::ButtonParam>()) {
        if (in_param.type == Param_t::BUTTON) {
            p_ptr->setDirty();
            // KeyCode can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::BoolParam>()) {
        if (in_param.type == Param_t::BOOL) {
            p_ptr->SetValue(std::get<bool>(in_param.GetValue()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::ColorParam>()) {
        if (in_param.type == Param_t::COLOR) {
            auto value = std::get<glm::vec4>(in_param.GetValue());
            p_ptr->SetValue(core::param::ColorParam::ColorType{value[0], value[1], value[2], value[3]});
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::EnumParam>()) {
        if (in_param.type == Param_t::ENUM) {
            p_ptr->SetValue(std::get<int>(in_param.GetValue()));
            // Map can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::FilePathParam>()) {
        if (in_param.type == Param_t::FILEPATH) {
            p_ptr->SetValue(vislib::StringA(std::get<std::string>(in_param.GetValue()).c_str()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::FlexEnumParam>()) {
        if (in_param.type == Param_t::FLEXENUM) {
            p_ptr->SetValue(std::get<std::string>(in_param.GetValue()));
            // Storage can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::FloatParam>()) {
        if (in_param.type == Param_t::FLOAT) {
            p_ptr->SetValue(std::get<float>(in_param.GetValue()));
            // Min and Max can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::IntParam>()) {
        if (in_param.type == Param_t::INT) {
            p_ptr->SetValue(std::get<int>(in_param.GetValue()));
            // Min and Max can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::StringParam>()) {
        if (in_param.type == Param_t::STRING) {
            p_ptr->SetValue(vislib::StringA(std::get<std::string>(in_param.GetValue()).c_str()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::TernaryParam>()) {
        if (in_param.type == Param_t::TERNARY) {
            p_ptr->SetValue(std::get<vislib::math::Ternary>(in_param.GetValue()));
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::TransferFunctionParam>()) {
        if (in_param.type == Param_t::TRANSFERFUNCTION) {
            p_ptr->SetValue(std::get<std::string>(in_param.GetValue()).c_str());
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::Vector2fParam>()) {
        if (in_param.type == Param_t::VECTOR2F) {
            auto value = std::get<glm::vec2>(in_param.GetValue());
            p_ptr->SetValue(vislib::math::Vector<float, 2>(value[0], value[1]));
            // Min and Max can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::Vector3fParam>()) {
        if (in_param.type == Param_t::VECTOR3F) {
            auto value = std::get<glm::vec3>(in_param.GetValue());
            p_ptr->SetValue(vislib::math::Vector<float, 3>(value[0], value[1], value[2]));
            // Min and Max can not be changed
        } else {
            type_error = true;
        }
    } else if (auto* p_ptr = out_param_ptr.DynamicCast<core::param::Vector4fParam>()) {
        if (in_param.type == Param_t::VECTOR4F) {
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
