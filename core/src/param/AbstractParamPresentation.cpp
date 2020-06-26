/*
 * AbstractParamPresentation.cpp
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/AbstractParamPresentation.h"

using namespace megamol::core::param;


const std::string AbstractParamPresentation::GetTypeName(ParamType type){
    switch (type) {
    case(BOOL)              : return "BoolParam";
    case(BUTTON)            : return "ButtonParam";
    case(COLOR)             : return "ColorParam";
    case(ENUM)              : return "EnumParam";
    case(FILEPATH)          : return "FilePathParam";
    case(FLEXENUM)          : return "FlexEnumParam";
    case(FLOAT)             : return "FloatParam";
    case(INT)               : return "IntParam";
    case(STRING)            : return "StringParam";
    case(TERNARY)           : return "TernaryParam";
    case(TRANSFERFUNCTION)  : return "TransferFunctionParam";
    case(VECTOR2F)          : return "Vector2fParam";
    case(VECTOR3F)          : return "Vector3fParam";
    case(VECTOR4F)          : return "Vector4fParam";
    case(GROUP_ANIMATION)   : return "AnimationGroup";
    default                 : return "UNKNOWN";
    }
}
     

AbstractParamPresentation::AbstractParamPresentation(void)
    : visible(true)
    , read_only(false)
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
    this->presentation_name_map.emplace(Presentation::PinValueToMouse, "Pin Value To Mouse");
    this->presentation_name_map.emplace(Presentation::Group_Animation, "Animation");
}


bool AbstractParamPresentation::InitPresentation(AbstractParamPresentation::ParamType param_type) {
    if (!this->initialised) {
        this->initialised = true;

        this->SetGUIVisible(visible);
        this->SetGUIReadOnly(read_only);

        // Initialize presentations depending on parameter type
        switch (param_type) {
        case (ParamType::BOOL): {
            this->compatible = Presentation::Basic | Presentation::String;
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
            this->compatible = Presentation::Basic | Presentation::String | Presentation::PinValueToMouse;
            this->SetGUIPresentation(Presentation::Basic);
        } break;
        case (ParamType::INT): {
            this->compatible = Presentation::Basic | Presentation::String | Presentation::PinValueToMouse;
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
            this->compatible = Presentation::Basic | Presentation::String | Presentation::PinValueToMouse;
            this->SetGUIPresentation(Presentation::Basic);
        } break;
        case (ParamType::VECTOR3F): {
            this->compatible = Presentation::Basic | Presentation::String | Presentation::PinValueToMouse;
            this->SetGUIPresentation(Presentation::Basic);
        } break;
        case (ParamType::VECTOR4F): {
            this->compatible = Presentation::Basic | Presentation::String | Presentation::PinValueToMouse | Presentation::Color;
            this->SetGUIPresentation(Presentation::Basic);
        } break;
        case (ParamType::GROUP_ANIMATION): {
            this->compatible = Presentation::Group_Animation;
            this->SetGUIPresentation(Presentation::Group_Animation);
        } break;
        default:
            break;
        }
        return true;
    }
    vislib::sys::Log::DefaultLog.WriteWarn(
        "Parameter presentation should only be initilised once. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}


void AbstractParamPresentation::SetGUIPresentation(AbstractParamPresentation::Presentation present) {
    if (this->IsPresentationCompatible(present)) {
        this->presentation = present;
    }
    else {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Incompatible parameter presentation. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
}
