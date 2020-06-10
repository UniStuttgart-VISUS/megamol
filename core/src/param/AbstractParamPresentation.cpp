/*
 * AbstractParamPresentation.cpp
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/AbstractParamPresentation.h"

using namespace megamol::core::param;


AbstractParamPresentation::AbstractParamPresentation(void)
    : visible(true)
    , read_only(false)
    , presentation(Presentation::Basic)
    , compatible(Presentation::Basic)
    , initialised(false) {
}


bool AbstractParamPresentation::InitPresentation(ParamType param_type) {
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
        default:
            break;
        }
        return true;
    }
    vislib::sys::Log::DefaultLog.WriteWarn(
        "Parameter presentation should only be initilised once. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}


void AbstractParamPresentation::SetGUIPresentation(AbstractParamPresentation::Presentation presentation) {
    if (this->IsPresentationCompatible(presentation)) {
        this->presentation = presentation;
    }
    else {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Incompatible parameter presentation. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
}
