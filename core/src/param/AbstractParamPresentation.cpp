/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/param/AbstractParamPresentation.h"

#include "mmcore/utility/JSONHelper.h"

using namespace megamol::core::param;


std::string AbstractParamPresentation::GetTypeName(ParamType type) {
    switch (type) {
    case (ParamType::BOOL):
        return "BoolParam";
    case (ParamType::BUTTON):
        return "ButtonParam";
    case (ParamType::COLOR):
        return "ColorParam";
    case (ParamType::ENUM):
        return "EnumParam";
    case (ParamType::FILEPATH):
        return "FilePathParam";
    case (ParamType::FLEXENUM):
        return "FlexEnumParam";
    case (ParamType::FLOAT):
        return "FloatParam";
    case (ParamType::INT):
        return "IntParam";
    case (ParamType::STRING):
        return "StringParam";
    case (ParamType::TERNARY):
        return "TernaryParam";
    case (ParamType::TRANSFERFUNCTION):
        return "TransferFunctionParam";
    case (ParamType::VECTOR2F):
        return "Vector2fParam";
    case (ParamType::VECTOR3F):
        return "Vector3fParam";
    case (ParamType::VECTOR4F):
        return "Vector4fParam";
    case (ParamType::GROUP_ANIMATION):
        return "AnimationGroup";
    case (ParamType::GROUP_3D_CUBE):
        return "3DCubeGroup";
    default:
        return "UNKNOWN";
    }
}


AbstractParamPresentation::AbstractParamPresentation()
        : visible(true)
        , read_only(false)
        , highlight(false)
        , presentation(AbstractParamPresentation::Presentation::Basic)
        , compatible(Presentation::Basic)
        , initialised(false)
        , presentation_name_map() {

    this->presentation_name_map.clear();
    this->presentation_name_map.emplace(Presentation::Basic, "Basic");
    this->presentation_name_map.emplace(Presentation::String, "String");
    this->presentation_name_map.emplace(Presentation::Color, "Color");
    this->presentation_name_map.emplace(Presentation::FilePath, "File Path");
    this->presentation_name_map.emplace(Presentation::TransferFunction, "Transfer Function");
    this->presentation_name_map.emplace(Presentation::Knob, "Knob");
    this->presentation_name_map.emplace(Presentation::Slider, "Slider");
    this->presentation_name_map.emplace(Presentation::Drag, "Drag");
    this->presentation_name_map.emplace(Presentation::Direction, "Direction");
    this->presentation_name_map.emplace(Presentation::Rotation, "Rotation");
    this->presentation_name_map.emplace(Presentation::PinMouse, "Pin to Mouse");
    this->presentation_name_map.emplace(Presentation::Group_Animation, "Animation");
    this->presentation_name_map.emplace(Presentation::Group_3D_Cube, "3D Cube");
    this->presentation_name_map.emplace(Presentation::Checkbox, "Checkbox");
}


void AbstractParamPresentation::InitPresentation(AbstractParamPresentation::ParamType param_type) {
    if (this->initialised) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "Parameter presentation should only be initilised once. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
        return;
    }

    this->initialised = true;

    this->SetGUIVisible(visible);
    this->SetGUIReadOnly(read_only);
    this->SetHighlight(highlight);

    // Initialize presentations depending on parameter type
    switch (param_type) {
    case (ParamType::BOOL): {
        this->compatible = Presentation::Basic | Presentation::String | Presentation::Checkbox;
        this->SetGUIPresentation(Presentation::Basic);
    } break;
    case (ParamType::BUTTON): {
        this->compatible = Presentation::Basic | Presentation::String;
        this->SetGUIPresentation(Presentation::Basic);
    } break;
    case (ParamType::COLOR): {
        this->compatible = Presentation::Basic | Presentation::String | Presentation::Color;
        this->SetGUIPresentation(Presentation::Color);
    } break;
    case (ParamType::ENUM): {
        this->compatible = Presentation::Basic | Presentation::String;
        this->SetGUIPresentation(Presentation::Basic);
    } break;
    case (ParamType::FILEPATH): {
        this->compatible = Presentation::Basic | Presentation::String | Presentation::FilePath;
        this->SetGUIPresentation(Presentation::FilePath);
    } break;
    case (ParamType::FLEXENUM): {
        this->compatible = Presentation::Basic | Presentation::String;
        this->SetGUIPresentation(Presentation::Basic);
    } break;
    case (ParamType::FLOAT): {
        this->compatible =
            Presentation::Basic | Presentation::String | Presentation::Knob | Presentation::Slider | Presentation::Drag;
        this->SetGUIPresentation(Presentation::Basic);
    } break;
    case (ParamType::INT): {
        this->compatible = Presentation::Basic | Presentation::String | Presentation::Slider | Presentation::Drag;
        this->SetGUIPresentation(Presentation::Basic);
    } break;
    case (ParamType::STRING): {
        this->compatible = Presentation::Basic | Presentation::String;
        this->SetGUIPresentation(Presentation::Basic);
    } break;
    case (ParamType::TERNARY): {
        this->compatible = Presentation::Basic | Presentation::String;
        this->SetGUIPresentation(Presentation::Basic);
    } break;
    case (ParamType::TRANSFERFUNCTION): {
        this->compatible = Presentation::Basic | Presentation::String | Presentation::TransferFunction;
        this->SetGUIPresentation(Presentation::TransferFunction);
    } break;
    case (ParamType::VECTOR2F): {
        this->compatible = Presentation::Basic | Presentation::String | Presentation::Slider | Presentation::Drag;
        this->SetGUIPresentation(Presentation::Basic);
    } break;
    case (ParamType::VECTOR3F): {
        this->compatible = Presentation::Basic | Presentation::String | Presentation::Direction | Presentation::Slider |
                           Presentation::Drag;
        this->SetGUIPresentation(Presentation::Basic);
    } break;
    case (ParamType::VECTOR4F): {
        this->compatible = Presentation::Basic | Presentation::String | Presentation::Color | Presentation::Rotation |
                           Presentation::Slider | Presentation::Drag;
        this->SetGUIPresentation(Presentation::Basic);
    } break;
    case (ParamType::GROUP_ANIMATION): {
        this->compatible = Presentation::Basic | Presentation::Group_Animation;
        this->SetGUIPresentation(Presentation::Basic);
    } break;
    case (ParamType::GROUP_3D_CUBE): {
        this->compatible = Presentation::Basic | Presentation::Group_3D_Cube;
        this->SetGUIPresentation(Presentation::Basic);
    } break;
    default:
        break;
    }
}


void AbstractParamPresentation::SetGUIPresentation(AbstractParamPresentation::Presentation present) {
    if (this->IsPresentationCompatible(present)) {
        this->presentation = present;
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "Incompatible parameter presentation. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
}


bool AbstractParamPresentation::StateFromJSON(const nlohmann::json& in_json, const std::string& param_fullname) {

    try {
        if (!in_json.is_object()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Invalid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        bool found_parameters = false;
        bool valid = true;
        for (auto& header_item : in_json.items()) {
            if (header_item.key() == GUI_JSON_TAG_GUISTATE_PARAMETERS) {
                found_parameters = true;
                for (auto& config_item : header_item.value().items()) {
                    const std::string& json_param_name = config_item.key();
                    if (json_param_name == param_fullname) {
                        const auto& gui_state = config_item.value();

                        valid = true;
                        bool gui_visibility = true;
                        valid &= megamol::core::utility::get_json_value<bool>(
                            gui_state, {"gui_visibility"}, &gui_visibility);

                        bool gui_read_only = false;
                        valid &=
                            megamol::core::utility::get_json_value<bool>(gui_state, {"gui_read-only"}, &gui_read_only);

                        bool gui_highlight = false;
                        valid &=
                            megamol::core::utility::get_json_value<bool>(gui_state, {"gui_highlight"}, &gui_highlight);

                        int presentation_mode = 0;
                        valid &= megamol::core::utility::get_json_value<int>(
                            gui_state, {"gui_presentation_mode"}, &presentation_mode);
                        auto gui_presentation_mode = static_cast<Presentation>(presentation_mode);

                        if (valid) {
                            this->SetGUIVisible(gui_visibility);
                            this->SetGUIReadOnly(gui_read_only);
                            this->SetHighlight(gui_highlight);
                            this->SetGUIPresentation(gui_presentation_mode);
                            return true;
                        }
                    }
                }
            }
        }
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON Error - Unable to read state from JSON. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return false;
}


bool AbstractParamPresentation::StateToJSON(nlohmann::json& inout_json, const std::string& param_fullname) {

    try {
        // Append to given json
        inout_json[GUI_JSON_TAG_GUISTATE_PARAMETERS][param_fullname]["gui_visibility"] = this->IsGUIVisible();
        inout_json[GUI_JSON_TAG_GUISTATE_PARAMETERS][param_fullname]["gui_read-only"] = this->IsGUIReadOnly();
        inout_json[GUI_JSON_TAG_GUISTATE_PARAMETERS][param_fullname]["gui_highlight"] = this->IsHighlight();
        inout_json[GUI_JSON_TAG_GUISTATE_PARAMETERS][param_fullname]["gui_presentation_mode"] =
            static_cast<int>(this->GetGUIPresentation());
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON Error - Unable to write state to JSON. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}
