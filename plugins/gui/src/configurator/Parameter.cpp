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
using namespace megamol::gui::configurator;


// PARAMETER PRESENTATION ####################################################

megamol::gui::configurator::ParameterPresentation::ParameterPresentation(ParamType type)
    : megamol::core::param::AbstractParamPresentation()
    , expert(false)
    , help()
    , description()
    , utils()
    , file_utils()
    , widget_store()
    , float_format("%.7f")
    , height(0.0f)
    , set_focus(0)     
    , tf_editor_ptr(nullptr)
    , external_tf_editor(false)
    , show_tf_editor(false)    
    , tf_editor_hash(0) {
        
    this->InitPresentation(type);
}


megamol::gui::configurator::ParameterPresentation::~ParameterPresentation(void) {}


bool megamol::gui::configurator::ParameterPresentation::Present(
    megamol::gui::configurator::Parameter& inout_parameter, WidgetScope scope) {
        
    bool retval = false;

    if (ImGui::GetCurrentContext() == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    try {
        ImGui::PushID(inout_parameter.uid);
        
        this->help = "";
        this->description = inout_parameter.description;
                        
        switch (scope) {
        case(ParameterPresentation::WidgetScope::LOCAL) : {
            if (this->IsGUIVisible() || this->expert) {
                
                ImGui::BeginGroup();
                if (this->expert) {
                    /// PREFIX ---------------------------------------------
                    
                    // Visibility
                    if (ImGui::RadioButton("###visible", this->IsGUIVisible())) {
                        this->SetGUIVisible(!this->IsGUIVisible());
                        retval = true;
                    }
                    this->utils.HoverToolTip("Visibility", ImGui::GetItemID(), 0.5f);

                    ImGui::SameLine();

                    // Read-only option
                    bool read_only = this->IsGUIReadOnly();
                    if (ImGui::Checkbox("###readonly", &read_only)) {
                        this->SetGUIReadOnly(read_only);
                         retval = true;
                    }
                    this->utils.HoverToolTip("Read-Only", ImGui::GetItemID(), 0.5f);

                    ImGui::SameLine();

                    // Presentation
                    this->utils.PointCircleButton("", (this->GetGUIPresentation() != PresentType::Basic));
                    if (ImGui::BeginPopupContextItem("param_present_button_context", 0)) {
                        for (auto& present_name_pair : this->GetPresentationNameMap()) {
                            if (this->IsPresentationCompatible(present_name_pair.first)) {
                                if (ImGui::MenuItem(present_name_pair.second.c_str(), nullptr, (present_name_pair.first == this->GetGUIPresentation()))) {
                                    this->SetGUIPresentation(present_name_pair.first);
                                    retval = true;
                                }
                            }
                        }
                        ImGui::EndPopup();
                    }
                    this->utils.HoverToolTip("Presentation", ImGui::GetItemID(), 0.5f);
                    
                    ImGui::SameLine();
                }

                /// PARAMETER VALUE WIDGET ---------------------------------  
                if (this->present_parameter(inout_parameter, scope)) {
                    retval = true;
                }
                    
                ImGui::SameLine();
                
                /// POSTFIX ------------------------------------------------
                this->utils.HoverToolTip(this->description, ImGui::GetItemID(), 0.5f);
                this->utils.HelpMarkerToolTip(this->help);

                ImGui::EndGroup();
            }
        } break;
        case (ParameterPresentation::WidgetScope::GLOBAL) : {
            
            if (this->present_parameter(inout_parameter, scope)) {
                retval = true;
            }
            
        } break;
        default: break;
        }
        
        ImGui::PopID();
                        
    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return retval;
}


float megamol::gui::configurator::ParameterPresentation::GetHeight(Parameter& inout_parameter) {


    float height = (ImGui::GetFrameHeightWithSpacing() * (1.15f));
    if (inout_parameter.type == ParamType::TRANSFERFUNCTION) {
        if (this->show_tf_editor) {
            height = (ImGui::GetFrameHeightWithSpacing() * (10.0f) + (150.0f + 30.0f));
        } else {
            height = (ImGui::GetFrameHeightWithSpacing() * (1.5f));
        }
    }
    return height;
}


bool megamol::gui::configurator::ParameterPresentation::present_parameter(
    megamol::gui::configurator::Parameter& inout_parameter, WidgetScope scope) {

    bool retval = false;
    bool error = true;
    std::string param_label = inout_parameter.GetName();
    
    // Implementation of presentation and parameter type mapping defined in AbstractParamPresentation::InitPresentation() to widget.
    auto visitor = [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        
        // LOCAL #######################################################
        if (scope == WidgetScope::LOCAL) {
            // Set general proportional item width
            float widget_width = ImGui::GetContentRegionAvail().x * 0.6f;
            ImGui::PushItemWidth(widget_width);
            // Set read only
            if (this->IsGUIReadOnly()) {
                GUIUtils::ReadOnlyWigetStyle(true);
            }
                
            switch (this->GetGUIPresentation()) {
            // BASIC ///////////////////////////////////////////////////
            case(PresentType::Basic) : {
                // BOOL ------------------------------------------------
                if constexpr (std::is_same_v<T, bool>) {
                    if (this->widget_bool(param_label, arg)) { 
                        inout_parameter.SetValue(arg);
                        retval = true;
                    }
                    error = false;
                } 
                // FLOAT -----------------------------------------------      
                else if constexpr (std::is_same_v<T, float>) {
                    if (this->widget_float(param_label, arg, inout_parameter.GetMinValue<T>(), inout_parameter.GetMaxValue<T>())) { 
                        inout_parameter.SetValue(arg);
                        retval = true;
                    }
                    error = false;
                } else if constexpr (std::is_same_v<T, int>) {
                    switch (inout_parameter.type) {
                    // INT ---------------------------------------------
                    case (ParamType::INT): {
                        if (this->widget_int(param_label, arg, inout_parameter.GetMinValue<T>(), inout_parameter.GetMaxValue<T>())) { 
                            inout_parameter.SetValue(arg);
                            retval = true;
                        }
                        error = false;
                    } break;
                    // ENUM --------------------------------------------
                    case (ParamType::ENUM): {
                        if (this->widget_enum(param_label, arg, inout_parameter.GetStorage<EnumStorageType>())) { 
                            inout_parameter.SetValue(arg);
                            retval = true;
                        }
                        error = false;    
                    } break;
                    default: break;
                    }
                } else if constexpr (std::is_same_v<T, std::string>) {
                    switch (inout_parameter.type) {
                    // STRING ------------------------------------------
                    case (ParamType::STRING): {
                        if (this->widget_string(param_label, arg)) { 
                            inout_parameter.SetValue(arg);
                            retval = true;
                        }
                        error = false;
                    } break;
                    // TRANSFER FUNCTION -------------------------------
                    case (ParamType::TRANSFERFUNCTION): {
                        if (this->widget_string(param_label, arg)) { 
                            inout_parameter.SetValue(arg);
                            retval = true;
                        }
                        error = false;
                    } break;
                    // FILE PATH ---------------------------------------
                    case (ParamType::FILEPATH): {
                        if (this->widget_string(param_label, arg)) { 
                            inout_parameter.SetValue(arg);
                            retval = true;
                        }
                        error = false;
                    } break;
                    // FLEX ENUM ---------------------------------------
                    case (ParamType::FLEXENUM): {
                        if (this->widget_flexenum(param_label, arg, inout_parameter.GetStorage<megamol::core::param::FlexEnumParam::Storage_t>())) { 
                            inout_parameter.SetValue(arg);
                            retval = true;
                        }
                        error = false;
                    } break;
                    default: break;
                    }
                } 
                // TERNARY ---------------------------------------------
                else if constexpr (std::is_same_v<T, vislib::math::Ternary>) {
                    if (this->widget_ternary(param_label, arg)) { 
                        inout_parameter.SetValue(arg); 
                            retval = true;
                        }
                    error = false;
                } 
                // VECTOR 2 --------------------------------------------
                else if constexpr (std::is_same_v<T, glm::vec2>) {
                    if (this->widget_vector2f(param_label, arg, inout_parameter.GetMinValue<T>(), inout_parameter.GetMaxValue<T>())) { 
                        inout_parameter.SetValue(arg);
                            retval = true;
                        }
                    error = false;
                } 
                // VECTOR 3 --------------------------------------------
                else if constexpr (std::is_same_v<T, glm::vec3>) {
                    if (this->widget_vector3f(param_label, arg, inout_parameter.GetMinValue<T>(), inout_parameter.GetMaxValue<T>())) { 
                        inout_parameter.SetValue(arg);   
                            retval = true;
                        }          
                    error = false;
                } 
                else if constexpr (std::is_same_v<T, glm::vec4>) {
                    switch (inout_parameter.type) {
                    // VECTOR 4 ----------------------------------------
                    case (ParamType::VECTOR4F): {
                        if (this->widget_vector4f(param_label, arg, inout_parameter.GetMinValue<T>(), inout_parameter.GetMaxValue<T>())) { 
                            inout_parameter.SetValue(arg);
                            retval = true;
                        }
                        error = false;                    
                    } break;
                    // COLOR -------------------------------------------
                    case (ParamType::COLOR): {
                        if (this->widget_vector4f(param_label, arg, glm::vec4(0.0f), glm::vec4(1.0f))) { 
                            inout_parameter.SetValue(arg);
                            retval = true;
                        }
                        error = false;                    
                    } break;
                    default: break;                
                    }
                } else if constexpr (std::is_same_v<T, std::monostate>) {
                    switch (inout_parameter.type) {
                    // BUTTON ------------------------------------------
                    case (ParamType::BUTTON): {
                        if (this->widget_button(param_label, inout_parameter.GetStorage<megamol::core::view::KeyCode>())) {
                            retval = true;
                        }
                        error = false;
                    } break;
                    default: break;
                    }
                }
            } break;
            // STRING ////////////////////////////////////////////////// 
            case(PresentType::String) : {
                std::string value_string = inout_parameter.GetValueString();
                if (this->widget_string(param_label, value_string)) { 
                    inout_parameter.SetValueString(value_string);
                    retval = true;
                }
                error = false;
            } break;
            // COLOR ///////////////////////////////////////////////////  
            case(PresentType::Color) : {
                if constexpr (std::is_same_v<T, glm::vec4>) {
                    switch (inout_parameter.type) {
                    // VECTOR 4 ----------------------------------------
                    case (ParamType::VECTOR4F): {
                        if (this->widget_color(param_label, arg)) { 
                            inout_parameter.SetValue(arg);
                            retval = true;
                        }
                        error = false;                    
                    } break;
                    // COLOR -------------------------------------------
                    case (ParamType::COLOR): {
                        if (this->widget_color(param_label, arg)) { 
                            inout_parameter.SetValue(arg);
                            retval = true;
                        }
                        error = false;                    
                    } break;
                    default: break;                
                    }
                }         
            } break;
            // FILE PATH ///////////////////////////////////////////////    
            case(PresentType::FilePath) : {
                if constexpr (std::is_same_v<T, std::string>) {
                    switch (inout_parameter.type) {
                    // FILE PATH ---------------------------------------
                    case (ParamType::FILEPATH): {
                        if (this->widget_filepath(param_label, arg)) { 
                            inout_parameter.SetValue(arg);
                            retval = true;
                        }
                        error = false;
                    } break;
                    default: break;
                    }
                } 
            } break;
            // TRANSFER FUNCTION ///////////////////////////////////////
            case(PresentType::TransferFunction) : {
                if constexpr (std::is_same_v<T, std::string>) {
                    switch (inout_parameter.type) {
                    // TRANSFER FUNCTION -------------------------------
                    case (ParamType::TRANSFERFUNCTION): {
                        if (this->widget_transfer_function_editor(inout_parameter, WidgetScope::LOCAL)) {
                            retval = true;
                        }
                        error = false;
                    } break;
                    default: break;
                    }
                }         
            } break;
            // PIN VALUE TO MOUSE //////////////////////////////////////
            case(PresentType::PinValueToMouse) : {
                bool compatible_type = false;
                // FLOAT -----------------------------------------------        
                if constexpr (std::is_same_v<T, float>) {
                    compatible_type = true;
                } else if constexpr (std::is_same_v<T, int>) {
                    switch (inout_parameter.type) {
                    // INT ---------------------------------------------
                    case (ParamType::INT): {
                        compatible_type = true;
                    } break;
                    default: break;
                    }
                }
                // VECTOR 2 --------------------------------------------
                else if constexpr (std::is_same_v<T, glm::vec2>) {
                    compatible_type = true;
                } 
                // VECTOR 3 --------------------------------------------
                else if constexpr (std::is_same_v<T, glm::vec3>) {
                    compatible_type = true;
                } 
                else if constexpr (std::is_same_v<T, glm::vec4>) {
                    switch (inout_parameter.type) {
                    // VECTOR 4 ----------------------------------------
                    case (ParamType::VECTOR4F): {
                        compatible_type = true;                
                    } break;
                    default: break;                
                    }
                }
                if (compatible_type) {
                    this->widget_pinvaluetomouse(param_label, inout_parameter.GetValueString(), WidgetScope::LOCAL);
                    error = false;
                }
            } break; 
            default : break;
            }
            
            // Reset read only
            if (this->IsGUIReadOnly()) {
                GUIUtils::ReadOnlyWigetStyle(false);
            }
            // Reset item width
            ImGui::PopItemWidth(); 
        }
        // GLOBAL ######################################################
        else if (scope == WidgetScope::GLOBAL) {

            switch (this->GetGUIPresentation()) {
            // TRANSFER FUNCTION ///////////////////////////////////////
            case(PresentType::TransferFunction) : {
                if constexpr (std::is_same_v<T, std::string>) {
                    switch (inout_parameter.type) {
                    // TRANSFER FUNCTION -------------------------------
                    case (ParamType::TRANSFERFUNCTION): {
                        if (this->widget_transfer_function_editor(inout_parameter, WidgetScope::GLOBAL)) {
                            retval = true;
                        }
                        error = false;
                    } break;
                    default: break;
                    }
                }         
            } break;
            // PIN VALUE TO MOUSE //////////////////////////////////////
            case(PresentType::PinValueToMouse) : {
                bool compatible_type = false;
                // FLOAT -----------------------------------------------        
                if constexpr (std::is_same_v<T, float>) {
                    compatible_type = true;
                } else if constexpr (std::is_same_v<T, int>) {
                    switch (inout_parameter.type) {
                    // INT ---------------------------------------------
                    case (ParamType::INT): {
                        compatible_type = true;
                    } break;
                    default: break;
                    }
                }
                // VECTOR 2 --------------------------------------------
                else if constexpr (std::is_same_v<T, glm::vec2>) {
                    compatible_type = true;
                } 
                // VECTOR 3 --------------------------------------------
                else if constexpr (std::is_same_v<T, glm::vec3>) {
                    compatible_type = true;
                } 
                else if constexpr (std::is_same_v<T, glm::vec4>) {
                    switch (inout_parameter.type) {
                    // VECTOR 4 ----------------------------------------
                    case (ParamType::VECTOR4F): {
                        compatible_type = true;                
                    } break;
                    default: break;                
                    }
                }
                if (compatible_type) {
                    this->widget_pinvaluetomouse(param_label, inout_parameter.GetValueString(), WidgetScope::GLOBAL);
                    error = false;
                }
            } break; 
            default : {
                error = false;
            } break;
            }
        }
    };

    std::visit(visitor, inout_parameter.GetValue());
    
    if (error) {
        vislib::sys::Log::DefaultLog.WriteError("No widget presentation '%s' available for '%s' . [%s, %s, line %d]\n",
        this->GetPresentationName(this->GetGUIPresentation()).c_str(), 
        megamol::core::param::AbstractParamPresentation::GetTypeName(inout_parameter.type).c_str(), 
        __FILE__, __FUNCTION__, __LINE__); 
    }
        
    return retval;
}


bool megamol::gui::configurator::ParameterPresentation::widget_button(const std::string& label, const megamol::core::view::KeyCode& keycode) {
    bool retval = false;
    
    std::string button_hotkey = keycode.ToString();
    std::string hotkey = "";
    if (!button_hotkey.empty()) hotkey = "\n Hotkey: " + button_hotkey;
    this->description += hotkey;
    
    //if (!button_hotkey.empty()) hotkey = " (" + button_hotkey + ")";
    //std::string edit_label = label + hotkey;
    //retval = ImGui::Button(edit_label.c_str()); 
    
    retval = ImGui::Button(label.c_str()); 
    
    return retval;
} 
    

bool megamol::gui::configurator::ParameterPresentation::widget_bool(const std::string& label, bool& value) {
    bool retval = false;
    
    retval = ImGui::Checkbox(label.c_str(), &value);
    
    return retval;
}


bool megamol::gui::configurator::ParameterPresentation::widget_string(const std::string& label, std::string& value) {
    bool retval = false;
    
    ImGui::BeginGroup();
    /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
    if (!std::holds_alternative<std::string>(this->widget_store)) {
        std::string utf8Str = value;
        GUIUtils::Utf8Encode(utf8Str);
        this->widget_store = utf8Str;
    }
    std::string hidden_label = "###" + label;
        
    // Determine multi line count of string
    int multiline_cnt = static_cast<int>(std::count(std::get<std::string>(this->widget_store).begin(),
        std::get<std::string>(this->widget_store).end(), '\n'));
    multiline_cnt = std::min(static_cast<int>(GUI_MAX_MULITLINE), multiline_cnt);
    ///if (multiline_cnt == 0) {
    ///    ImGui::InputText(hidden_label.c_str(), &std::get<std::string>(this->widget_store), ImGuiInputTextFlags_CtrlEnterForNewLine);
    ///}
    ///else {
        ImVec2 multiline_size = ImVec2(ImGui::CalcItemWidth(), ImGui::GetFrameHeightWithSpacing() + (ImGui::GetFontSize() * static_cast<float>(multiline_cnt)));
        ImGui::InputTextMultiline(hidden_label.c_str(), &std::get<std::string>(this->widget_store), multiline_size, ImGuiInputTextFlags_CtrlEnterForNewLine);
    ///}
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        std::string utf8Str = std::get<std::string>(this->widget_store);
        GUIUtils::Utf8Decode(utf8Str);
        value = utf8Str;
        retval = true;
    } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
        std::string utf8Str = value;
        GUIUtils::Utf8Encode(utf8Str);
        this->widget_store = utf8Str;
    }
    ImGui::SameLine();
    
    ImGui::TextUnformatted(label.c_str());
    ImGui::EndGroup();
    
    this->help = "[Ctrl + Enter] for new line.\nPress [Return] to confirm changes.";
    
    return retval;
}
      

bool megamol::gui::configurator::ParameterPresentation::widget_color(const std::string& label, glm::vec4& value) {
    bool retval = false;
    
    auto color_flags = ImGuiColorEditFlags_AlphaPreview; // | ImGuiColorEditFlags_Float;
    retval = ImGui::ColorEdit4(label.c_str(), glm::value_ptr(value), color_flags);
    
    this->help = "[Click] on the colored square to open a color picker.\n"
                 "[CTRL+Click] on individual component to input value.\n"
                 "[Right-Click] on the individual color widget to show options.";
    
    return retval;
}


bool megamol::gui::configurator::ParameterPresentation::widget_enum(const std::string& label, int& value, EnumStorageType storage) {
    bool retval = false;
    
    /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
    std::string utf8Str = storage[value];
    GUIUtils::Utf8Encode(utf8Str);
    if (ImGui::BeginCombo(label.c_str(), utf8Str.c_str())) {
        for (auto& pair : storage) {
            bool isSelected = (pair.first == value);
            utf8Str = pair.second;
            GUIUtils::Utf8Encode(utf8Str);
            if (ImGui::Selectable(utf8Str.c_str(), isSelected)) {
                value = pair.first;
                retval = true;
            }
        }
        ImGui::EndCombo();
    }
    
    return retval;
}


bool megamol::gui::configurator::ParameterPresentation::widget_flexenum(const std::string& label, std::string& value, megamol::core::param::FlexEnumParam::Storage_t storage) {
    bool retval = false;
    
    /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
    if (!std::holds_alternative<std::string>(this->widget_store)) {
        this->widget_store = std::string();
    }
    std::string utf8Str = value;
    GUIUtils::Utf8Encode(utf8Str);
    if (ImGui::BeginCombo(label.c_str(), utf8Str.c_str())) {
        bool one_present = false;
        for (auto& valueOption : storage) {
            bool isSelected = (valueOption == value);
            utf8Str = valueOption;
            GUIUtils::Utf8Encode(utf8Str);
            if (ImGui::Selectable(utf8Str.c_str(), isSelected)) {
                GUIUtils::Utf8Decode(utf8Str);
                value = utf8Str;
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
        /// Keyboard focus needs to be set in/untill second frame
        if (this->set_focus < 2) {
            ImGui::SetKeyboardFocusHere();
            this->set_focus++;
        }
        ImGui::InputText(
            "###flex_enum_text_edit", &std::get<std::string>(this->widget_store), ImGuiInputTextFlags_None);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            if (!std::get<std::string>(this->widget_store).empty()) {
                GUIUtils::Utf8Decode(std::get<std::string>(this->widget_store));
                value = std::get<std::string>(this->widget_store);
                retval = true;
                std::get<std::string>(this->widget_store) = std::string();
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndCombo();
    } else {
        this->set_focus = 0;
    }
    this->help = "Only selected value will be saved to project file";
        
    return retval;
}


bool megamol::gui::configurator::ParameterPresentation::widget_filepath(const std::string& label, std::string& value) {
    bool retval = false;
    
    ImGui::BeginGroup();
    /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
    if (!std::holds_alternative<std::string>(this->widget_store)) {
        std::string utf8Str = value;
        GUIUtils::Utf8Encode(utf8Str);
        this->widget_store = utf8Str;
    }
    ImGuiStyle& style = ImGui::GetStyle();
    float widget_width = ImGui::CalcItemWidth() - (ImGui::GetFrameHeightWithSpacing() + style.ItemSpacing.x);
    ImGui::PushItemWidth(widget_width);
    bool button_edit = this->file_utils.FileBrowserButton(std::get<std::string>(this->widget_store));
    ImGui::SameLine();
    ImGui::InputText(
        label.c_str(), &std::get<std::string>(this->widget_store), ImGuiInputTextFlags_None);
    if (button_edit || ImGui::IsItemDeactivatedAfterEdit()) {
        GUIUtils::Utf8Decode(std::get<std::string>(this->widget_store));
        value = std::get<std::string>(this->widget_store);
        retval = true;
    } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
        std::string utf8Str = value;
        GUIUtils::Utf8Encode(utf8Str);
        this->widget_store = utf8Str;
    }
    ImGui::PopItemWidth();
    ImGui::EndGroup();
                    
    return retval;
}


bool megamol::gui::configurator::ParameterPresentation::widget_ternary(const std::string& label, vislib::math::Ternary& value) {
    bool retval = false;
    
    ImGui::BeginGroup();
    if (ImGui::RadioButton("True", value.IsTrue())) {
        value = vislib::math::Ternary::TRI_TRUE;
        retval = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("False", value.IsFalse())) {
        value = vislib::math::Ternary::TRI_FALSE;
        retval = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Unknown", value.IsUnknown())) {
        value = vislib::math::Ternary::TRI_UNKNOWN;
        retval = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    ImGui::TextUnformatted(label.c_str());
    ImGui::EndGroup();
                
    return retval;
}


bool megamol::gui::configurator::ParameterPresentation::widget_int(const std::string& label, int& value, int min, int max) {
    bool retval = false;
    
    if (!std::holds_alternative<int>(this->widget_store)) {
        this->widget_store = value;
    }
    ImGui::InputInt(label.c_str(), &std::get<int>(this->widget_store), 1, 10, ImGuiInputTextFlags_None);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        this->widget_store = std::max(min, std::min(std::get<int>(this->widget_store), max));
        value = std::get<int>(this->widget_store);
        retval = true;
    } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
        this->widget_store = value;
    }
        
    return retval;
}


bool megamol::gui::configurator::ParameterPresentation::widget_float(const std::string& label, float& value, float min, float max) {
    bool retval = false;

    if (!std::holds_alternative<float>(this->widget_store)) {
        this->widget_store = value;
    }
    ImGui::InputFloat(label.c_str(), &std::get<float>(this->widget_store), 1.0f, 10.0f, this->float_format.c_str(), ImGuiInputTextFlags_None);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        this->widget_store = std::max(min, std::min(std::get<float>(this->widget_store), max));
        value = std::get<float>(this->widget_store);
        retval = true;
    } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
        this->widget_store = value;
    }
                    
    return retval;
}


bool megamol::gui::configurator::ParameterPresentation::widget_vector2f(const std::string& label, glm::vec2& value, glm::vec2 min, glm::vec2 max) {
    bool retval = false;
    
    if (!std::holds_alternative<glm::vec2>(this->widget_store)) {
        this->widget_store = value;
    }
    ImGui::InputFloat2(label.c_str(), glm::value_ptr(std::get<glm::vec2>(this->widget_store)),
        this->float_format.c_str(), ImGuiInputTextFlags_None);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        auto x = std::max(min.x, std::min(std::get<glm::vec2>(this->widget_store).x, max.x));
        auto y = std::max(min.y, std::min(std::get<glm::vec2>(this->widget_store).y, max.y));
        this->widget_store = glm::vec2(x, y);
        value = std::get<glm::vec2>(this->widget_store);
        retval = true;
    } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
        this->widget_store = value;
    }
        
    return retval;
}


bool megamol::gui::configurator::ParameterPresentation::widget_vector3f(const std::string& label, glm::vec3& value, glm::vec3 min, glm::vec3 max) {
    bool retval = false;
    
    if (!std::holds_alternative<glm::vec3>(this->widget_store)) {
        this->widget_store = value;
    }
    ImGui::InputFloat3(label.c_str(), glm::value_ptr(std::get<glm::vec3>(this->widget_store)),
        this->float_format.c_str(), ImGuiInputTextFlags_None);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        auto x = std::max(min.x, std::min(std::get<glm::vec3>(this->widget_store).x, max.x));
        auto y = std::max(min.y, std::min(std::get<glm::vec3>(this->widget_store).y, max.y));
        auto z = std::max(min.z, std::min(std::get<glm::vec3>(this->widget_store).z, max.z));
        this->widget_store = glm::vec3(x, y, z);
        value = std::get<glm::vec3>(this->widget_store);
        retval = true;
    } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
        this->widget_store = value;
    }
                
    return retval;
}


bool megamol::gui::configurator::ParameterPresentation::widget_vector4f(const std::string& label, glm::vec4& value, glm::vec4 min, glm::vec4 max) {
    bool retval = false;
    
    if (!std::holds_alternative<glm::vec4>(this->widget_store)) {
        this->widget_store = value;
    }
    ImGui::InputFloat4(label.c_str(), glm::value_ptr(std::get<glm::vec4>(this->widget_store)),
        this->float_format.c_str(), ImGuiInputTextFlags_None);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        auto x = std::max(min.x, std::min(std::get<glm::vec4>(this->widget_store).x, max.x));
        auto y = std::max(min.y, std::min(std::get<glm::vec4>(this->widget_store).y, max.y));
        auto z = std::max(min.z, std::min(std::get<glm::vec4>(this->widget_store).z, max.z));
        auto w = std::max(min.w, std::min(std::get<glm::vec4>(this->widget_store).w, max.w));
        this->widget_store = glm::vec4(x, y, z, w);
        value = std::get<glm::vec4>(this->widget_store);
        retval = true;
    } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
        this->widget_store = value;
    }
                
    return retval;
}


bool megamol::gui::configurator::ParameterPresentation::widget_pinvaluetomouse(const std::string& label, const std::string& value, WidgetScope scope) {
    bool retval = false;
    
    // LOCAL -----------------------------------------------------------    
    if (scope == ParameterPresentation::WidgetScope::LOCAL) {
        
        ImGui::TextDisabled(label.c_str());
    }
    // GLOBAL -----------------------------------------------------------    
    else if (scope == ParameterPresentation::WidgetScope::GLOBAL) {
        
        auto hoverFlags = ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_AllowWhenDisabled |
                          ImGuiHoveredFlags_AllowWhenBlockedByPopup |
                          ImGuiHoveredFlags_AllowWhenBlockedByActiveItem;
        // Show only if mouse is outside any gui window
        if (!ImGui::IsWindowHovered(hoverFlags)) {
            ImGui::BeginTooltip();
            ImGui::TextDisabled(label.c_str());
            ImGui::SameLine();
            ImGui::TextUnformatted(value.c_str());
            ImGui::EndTooltip();
        }
    }
    
    return retval;
}


bool megamol::gui::configurator::ParameterPresentation::widget_transfer_function_editor(
    megamol::gui::configurator::Parameter& inout_parameter, WidgetScope scope) {

    bool retval = false;
    bool local = (scope == ParameterPresentation::WidgetScope::LOCAL);
    //bool global = (scope == ParameterPresentation::WidgetScope::GLOBAL);
    bool isActive = false;
    bool updateEditor = false;
    auto value = std::get<std::string>(inout_parameter.GetValue());
    std::string label = inout_parameter.full_name;
            
    ImGuiStyle& style = ImGui::GetStyle();
            
    if (this->external_tf_editor) {
        if (this->tf_editor_ptr == nullptr) {
            vislib::sys::Log::DefaultLog.WriteWarn(
                "Pointer to external transfer function editor is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        isActive = (this->tf_editor_ptr->GetConnectedParameter() != nullptr);
    } else {
        if (this->tf_editor_ptr == nullptr) {
            this->tf_editor_ptr = std::make_shared<megamol::gui::TransferFunctionEditor>();
        }
    }
    
    if (local) {
        ImGui::BeginGroup();
        
        if (this->external_tf_editor) {
            // Reduced display of value and editor state.
            if (value.empty()) {
                ImGui::TextDisabled("{    (empty)    }");
                ImGui::SameLine();
            } else {
                // Draw texture
                if (this->tf_editor_ptr->GetHorizontalTexture() != 0) {
                    ImGui::Image(reinterpret_cast<ImTextureID>(this->tf_editor_ptr->GetHorizontalTexture()),
                        ImVec2(ImGui::CalcItemWidth(), ImGui::GetFrameHeight()), ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f),
                        ImVec4(1.0f, 1.0f, 1.0f, 1.0f), style.Colors[ImGuiCol_Border]);
                    ImGui::SameLine(ImGui::CalcItemWidth() + style.ItemInnerSpacing.x);
                } else {
                    ImGui::TextUnformatted("{ ............. }");
                    ImGui::SameLine();
                }
            }
            
            // Label
            ImGui::AlignTextToFramePadding();
            ImGui::TextEx(label.c_str(), ImGui::FindRenderedTextEnd(label.c_str()));

            ImGui::Indent();
        }
        
        // Copy
        if (ImGui::Button("Copy")) {
    #ifdef GUI_USE_GLFW
            auto glfw_win = ::glfwGetCurrentContext();
            ::glfwSetClipboardString(glfw_win, value.c_str());
    #elif _WIN32
            ImGui::SetClipboardText(value.c_str());
    #else // LINUX
            vislib::sys::Log::DefaultLog.WriteWarn(
                "No clipboard use provided. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            vislib::sys::Log::DefaultLog.WriteInfo("[Configurator] Transfer Function JSON String:\n%s", value.c_str());
    #endif
        }
        ImGui::SameLine();

        // Paste
        if (ImGui::Button("Paste")) {
    #ifdef GUI_USE_GLFW
            auto glfw_win = ::glfwGetCurrentContext();
            inout_parameter.SetValue(std::string(::glfwGetClipboardString(glfw_win)));
    #elif _WIN32
            inout_parameter.SetValue(std::string(ImGui::GetClipboardText()));
    #else // LINUX
            vislib::sys::Log::DefaultLog.WriteWarn(
                "No clipboard use provided. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    #endif
            updateEditor = true;
        }
        ImGui::SameLine();

        if (this->external_tf_editor) {
            // Editor
            ImGui::PushID("Edit_");
            ImGui::PushStyleColor(ImGuiCol_Button, style.Colors[isActive ? ImGuiCol_ButtonHovered : ImGuiCol_Button]);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, style.Colors[isActive ? ImGuiCol_Button : ImGuiCol_ButtonHovered]);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, style.Colors[ImGuiCol_ButtonActive]);
            if (ImGui::Button("Edit")) {
                updateEditor = true;
                isActive = true;
                // TODO
                this->tf_editor_ptr->SetConnectedParameter(std::make_shared<Parameter>(inout_parameter));
                
                retval = true;
            }
            ImGui::PopStyleColor(3);
            ImGui::PopID();

            ImGui::Unindent();   
        }
        else {
            // Editor
            if (ImGui::Checkbox("Editor ", &this->show_tf_editor)) {
                // Set once
                if (this->show_tf_editor) {
                    updateEditor = true;
                }
            }
            ImGui::SameLine();

            // Indicate unset transfer function state
            if (value.empty()) {
                ImGui::TextDisabled(" { empty } ");
            }
            ImGui::SameLine();
            
            // Label
            ImGui::TextUnformatted(label.c_str(), ImGui::FindRenderedTextEnd(label.c_str()));
            
            // Propagate the transfer function to the editor.
            if (updateEditor) {
                this->tf_editor_ptr->SetTransferFunction(value, false);
            }
            // Draw transfer function editor
            if (this->show_tf_editor) {
                if (this->tf_editor_ptr->Draw(false)) {
                    std::string value;
                    if (this->tf_editor_ptr->GetTransferFunction(value)) {
                        inout_parameter.SetValue(value);
                        retval = true;
                    }
                }
                ImGui::Separator();
            }
        }

        ImGui::EndGroup();
    }

    // LOCAL and GLOBAL
    if (this->external_tf_editor) {
            
        // Check for changed parameter value which should be forced to the editor once.
        if (isActive) {
            if (this->tf_editor_hash != inout_parameter.GetStringHash()) { 
                updateEditor = true; 
            }
        }
        // Propagate the transfer function to the editor.
        if (isActive && updateEditor) {
            this->tf_editor_ptr->SetTransferFunction(value, true);
            retval = true;
        }
        this->tf_editor_hash = inout_parameter.GetStringHash();     
    }

    return retval;
}


// PARAMETER ##################################################################

megamol::gui::configurator::Parameter::Parameter(
    ImGuiID uid, ParamType type, StroageType store, MinType min, MaxType max)
    : uid(uid)
    , type(type)
    , full_name()
    , description()
    , minval(min)
    , maxval(max)
    , storage(store)
    , value()
    , string_hash(0)
    , default_value()
    , default_value_mismatch(false)
    , present(type) {

    // Initialize variant types which should/can not be changed afterwards.
    // Default ctor of variants initializes std::monostate.
    switch (this->type) {
    case (ParamType::BOOL): {
        this->value = bool(false);
    } break;
    case (ParamType::BUTTON): {
        /// set_default_value_mismatch = true;
    } break;
    case (ParamType::COLOR): {
        this->value = glm::vec4();
    } break;
    case (ParamType::ENUM): {
        this->value = int(0);
    } break;
    case (ParamType::FILEPATH): {
        this->value = std::string();
    } break;
    case (ParamType::FLEXENUM): {
        this->value = std::string();
    } break;
    case (ParamType::FLOAT): {
        this->value = float(0.0f);
    } break;
    case (ParamType::INT): {
        this->value = int();
    } break;
    case (ParamType::STRING): {
        this->value = std::string();
    } break;
    case (ParamType::TERNARY): {
        this->value = vislib::math::Ternary();
    } break;
    case (ParamType::TRANSFERFUNCTION): {
        this->value = std::string();
    } break;
    case (ParamType::VECTOR2F): {
        this->value = glm::vec2();
    } break;
    case (ParamType::VECTOR3F): {
        this->value = glm::vec3();
    } break;
    case (ParamType::VECTOR4F): {
        this->value = glm::vec4();
    } break;
    default:
        break;
    }

    this->default_value = this->value;
}


megamol::gui::configurator::Parameter::~Parameter(void) {}


std::string megamol::gui::configurator::Parameter::GetValueString(void) {
    std::string value_string = "UNKNOWN PARAMETER TYPE";
    auto visitor = [this, &value_string](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, bool>) {
            auto parameter = megamol::core::param::BoolParam(arg);
            value_string = std::string(parameter.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, megamol::core::param::ColorParam::ColorType>) {
            auto parameter = megamol::core::param::ColorParam(arg[0], arg[1], arg[2], arg[3]);
            value_string = std::string(parameter.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, float>) {
            auto parameter = megamol::core::param::FloatParam(arg);
            value_string = std::string(parameter.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, int>) {
            switch (this->type) {
            case (ParamType::INT): {
                auto parameter = megamol::core::param::IntParam(arg);
                value_string = std::string(parameter.ValueString().PeekBuffer());
            } break;
            case (ParamType::ENUM): {
                auto parameter = megamol::core::param::EnumParam(arg);
                // Initialization of enum storage required
                auto map = this->GetStorage<EnumStorageType>();
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
            case (ParamType::STRING): {
                auto parameter = megamol::core::param::StringParam(arg.c_str());
                value_string = std::string(parameter.ValueString().PeekBuffer());
            } break;
            case (ParamType::TRANSFERFUNCTION): {
                auto parameter = megamol::core::param::TransferFunctionParam(arg);
                value_string = std::string(parameter.ValueString().PeekBuffer());
            } break;
            case (ParamType::FILEPATH): {
                auto parameter = megamol::core::param::FilePathParam(arg.c_str());
                value_string = std::string(parameter.ValueString().PeekBuffer());
            } break;
            case (ParamType::FLEXENUM): {
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
            auto parameter =
                megamol::core::param::Vector4fParam(vislib::math::Vector<float, 4>(arg.x, arg.y, arg.z, arg.w));
            value_string = std::string(parameter.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, std::monostate>) {
            switch (this->type) {
            case (ParamType::BUTTON): {
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


bool megamol::gui::configurator::Parameter::SetValueString(const std::string& val_str, bool set_default_val) {

    bool retval = false;
    vislib::TString val_tstr(val_str.c_str());

    switch (this->type) {
    case (ParamType::BOOL): {
        megamol::core::param::BoolParam parameter(false);
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val);
    } break;
    case (ParamType::BUTTON): {
        retval = true;
    } break;
    case (ParamType::COLOR): {
        megamol::core::param::ColorParam parameter(val_tstr);
        retval = parameter.ParseValue(val_tstr);
        auto value = parameter.Value();
        this->SetValue(glm::vec4(value[0], value[1], value[2], value[3]), set_default_val);
    } break;
    case (ParamType::ENUM): {
        megamol::core::param::EnumParam parameter(0);
        // Initialization of enum storage required
        auto map = this->GetStorage<EnumStorageType>();
        for (auto& pair : map) {
            parameter.SetTypePair(pair.first, pair.second.c_str());
        }
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val);
    } break;
    case (ParamType::FILEPATH): {
        megamol::core::param::FilePathParam parameter(val_tstr.PeekBuffer());
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(std::string(parameter.Value().PeekBuffer()), set_default_val);
    } break;
    case (ParamType::FLEXENUM): {
        megamol::core::param::FlexEnumParam parameter(val_str);
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val);
    } break;
    case (ParamType::FLOAT): {
        megamol::core::param::FloatParam parameter(0.0f);
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val);
    } break;
    case (ParamType::INT): {
        megamol::core::param::IntParam parameter(0);
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val);
    } break;
    case (ParamType::STRING): {
        megamol::core::param::StringParam parameter(val_tstr.PeekBuffer());
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(std::string(parameter.Value().PeekBuffer()), set_default_val);
    } break;
    case (ParamType::TERNARY): {
        megamol::core::param::TernaryParam parameter(vislib::math::Ternary::TRI_UNKNOWN);
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val);
    } break;
    case (ParamType::TRANSFERFUNCTION): {
        megamol::core::param::TransferFunctionParam parameter;
        retval = parameter.ParseValue(val_tstr);
        this->SetValue(parameter.Value(), set_default_val);
    } break;
    case (ParamType::VECTOR2F): {
        megamol::core::param::Vector2fParam parameter(vislib::math::Vector<float, 2>(0.0f, 0.0f));
        retval = parameter.ParseValue(val_tstr);
        auto val = parameter.Value();
        this->SetValue(glm::vec2(val.X(), val.Y()), set_default_val);
    } break;
    case (ParamType::VECTOR3F): {
        megamol::core::param::Vector3fParam parameter(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
        retval = parameter.ParseValue(val_tstr);
        auto val = parameter.Value();
        this->SetValue(glm::vec3(val.X(), val.Y(), val.Z()), set_default_val);
    } break;
    case (ParamType::VECTOR4F): {
        megamol::core::param::Vector4fParam parameter(vislib::math::Vector<float, 4>(0.0f, 0.0f, 0.0f, 0.0f));
        retval = parameter.ParseValue(val_tstr);
        auto val = parameter.Value();
        this->SetValue(glm::vec4(val.X(), val.Y(), val.Z(), val.W()), set_default_val);
    } break;
    default:
        break;
    }

    return retval;
}
