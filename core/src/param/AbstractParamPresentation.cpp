/*
 * AbstractParamPresentation.cpp
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/AbstractParamPresentation.h"

using namespace megamol::core::param;


const std::string AbstractParamPresentation::GetTypeName(ParamType type) {
    switch (type) {
    case(BOOL): return "BoolParam";
    case(BUTTON): return "ButtonParam";
    case(COLOR): return "ColorParam";
    case(ENUM): return "EnumParam";
    case(FILEPATH): return "FilePathParam";
    case(FLEXENUM): return "FlexEnumParam";
    case(FLOAT): return "FloatParam";
    case(INT): return "IntParam";
    case(STRING): return "StringParam";
    case(TERNARY): return "TernaryParam";
    case(TRANSFERFUNCTION): return "TransferFunctionParam";
    case(VECTOR2F): return "Vector2fParam";
    case(VECTOR3F): return "Vector3fParam";
    case(VECTOR4F): return "Vector4fParam";
    case(GROUP_ANIMATION): return "AnimationGroup";
    default: return "UNKNOWN";
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
    this->presentation_name_map.emplace(Presentation::Knob, "Knob");
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
            this->compatible = Presentation::Basic | Presentation::String | Presentation::Knob | Presentation::PinValueToMouse;
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
            this->compatible = Presentation::Basic | Presentation::Group_Animation;
            this->SetGUIPresentation(Presentation::Basic);
        } break;
        default:
            break;
        }
        return true;
    }
    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
        "Parameter presentation should only be initilised once. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}


void AbstractParamPresentation::SetGUIPresentation(AbstractParamPresentation::Presentation present) {
    if (this->IsPresentationCompatible(present)) {
        this->presentation = present;
    }
    else {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "Incompatible parameter presentation. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
}


bool AbstractParamPresentation::ParameterGUIStateFromJSONString(const std::string& in_json_string, const std::string& param_fullname) {

    bool retval = false;

    try {
        if (in_json_string.empty()) {
            return false;
        }

        bool found_parameters = false;
        bool valid = true;
        nlohmann::json json;
        json = nlohmann::json::parse(in_json_string);
        if (!json.is_object()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "State is no valid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        for (auto& header_item : json.items()) {
            if (header_item.key() == GUI_JSON_TAG_GUISTATE_PARAMETERS) {

                found_parameters = true;
                for (auto& config_item : header_item.value().items()) {
                    std::string json_param_name = config_item.key();
                    if (json_param_name == param_fullname) {

                        auto gui_state = config_item.value();
                        valid = true;

                        // gui_visibility
                        bool gui_visibility;
                        if (gui_state.at("gui_visibility").is_boolean()) {
                            gui_state.at("gui_visibility").get_to(gui_visibility);
                        }
                        else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "JSON state: Failed to read 'gui_visibility' as boolean. [%s, %s, line %d]\n", __FILE__,
                                __FUNCTION__, __LINE__);
                            valid = false;
                        }

                        // gui_read-only
                        bool gui_read_only;
                        if (gui_state.at("gui_read-only").is_boolean()) {
                            gui_state.at("gui_read-only").get_to(gui_read_only);
                        }
                        else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "JSON state: Failed to read 'gui_read-only' as boolean. [%s, %s, line %d]\n", __FILE__,
                                __FUNCTION__, __LINE__);
                            valid = false;
                        }

                        // gui_presentation_mode
                        Presentation gui_presentation_mode;
                        if (gui_state.at("gui_presentation_mode").is_number_integer()) {
                            gui_presentation_mode =
                                static_cast<Presentation>(gui_state.at("gui_presentation_mode").get<int>());
                        }
                        else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "JSON state: Failed to read 'gui_presentation_mode' as integer. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                            valid = false;
                        }

                        if (valid) {
                            this->SetGUIVisible(gui_visibility);
                            this->SetGUIReadOnly(gui_read_only);
                            this->SetGUIPresentation(gui_presentation_mode);
                            retval = true;
                        }
                    }
                }
            }
        }

        if (retval) {
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("[AbstractParamPresentation] Read parameter state from JSON string.");
#endif // GUI_VERBOSE
        }
        else {
            /// megamol::core::utility::log::Log::DefaultLog.WriteWarn("Could not find parameter gui state in JSON for '%s' [%s, %s, line
            /// %d]\n", param_fullname.c_str(), __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
    }
    catch (nlohmann::json::type_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::invalid_iterator& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::out_of_range& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::other_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unknown Error - Unable to parse JSON string. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return retval;
}


bool AbstractParamPresentation::ParameterGUIStateToJSON(nlohmann::json& inout_json, const std::string& param_fullname) {

    try {
        // Append to given json

        inout_json[GUI_JSON_TAG_GUISTATE_PARAMETERS][param_fullname]["gui_visibility"] =
            this->IsGUIVisible();
        inout_json[GUI_JSON_TAG_GUISTATE_PARAMETERS][param_fullname]["gui_read-only"] =
            this->IsGUIReadOnly();
        inout_json[GUI_JSON_TAG_GUISTATE_PARAMETERS][param_fullname]["gui_presentation_mode"] =
            static_cast<int>(this->GetGUIPresentation());

#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[AbstractParamPresentation] Wrote parameter state to JSON.");
#endif // GUI_VERBOSE

    }
    catch (nlohmann::json::type_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::invalid_iterator& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::out_of_range& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (nlohmann::json::other_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    }
    catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unknown Error - Unable to write JSON of state. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}
