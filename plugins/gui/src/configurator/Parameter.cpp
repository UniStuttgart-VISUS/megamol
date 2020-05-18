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
    , default_value()
    , default_value_mismatch(false)
    , present() {

    // Initialize variant types which should/can not be changed afterwards.
    // Default ctor of variants initializes std::monostate.
    switch (this->type) {
    case (Parameter::ParamType::BOOL): {
        this->value = bool(false);
    } break;
    case (Parameter::ParamType::BUTTON): {
        // set_default_value_mismatch = true;
    } break;
    case (Parameter::ParamType::COLOR): {
        this->value = megamol::core::param::ColorParam::ColorType();
    } break;
    case (Parameter::ParamType::ENUM): {
        this->value = int(0);
    } break;
    case (Parameter::ParamType::FILEPATH): {
        this->value = std::string();
    } break;
    case (Parameter::ParamType::FLEXENUM): {
        this->value = std::string();
    } break;
    case (Parameter::ParamType::FLOAT): {
        this->value = float(0.0f);
    } break;
    case (Parameter::ParamType::INT): {
        this->value = int();
    } break;
    case (Parameter::ParamType::STRING): {
        this->value = std::string();
    } break;
    case (Parameter::ParamType::TERNARY): {
        this->value = vislib::math::Ternary();
    } break;
    case (Parameter::ParamType::TRANSFERFUNCTION): {
        this->value = std::string();
    } break;
    case (Parameter::ParamType::VECTOR2F): {
        this->value = glm::vec2();
    } break;
    case (Parameter::ParamType::VECTOR3F): {
        this->value = glm::vec3();
    } break;
    case (Parameter::ParamType::VECTOR4F): {
        this->value = glm::vec4();
    } break;
    default:
        break;
    }

    this->default_value = this->value;
}


std::string megamol::gui::configurator::Parameter::GetValueString(void) {
    std::string value_string = "UNKNOWN PARAMETER TYPE";
    auto visitor = [this, &value_string](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, bool>) {
            auto param = megamol::core::param::BoolParam(arg);
            value_string = std::string(param.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, megamol::core::param::ColorParam::ColorType>) {
            auto param = megamol::core::param::ColorParam(arg);
            value_string = std::string(param.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, float>) {
            auto param = megamol::core::param::FloatParam(arg);
            value_string = std::string(param.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, int>) {
            switch (this->type) {
            case (Parameter::ParamType::INT): {
                auto param = megamol::core::param::IntParam(arg);
                value_string = std::string(param.ValueString().PeekBuffer());
            } break;
            case (Parameter::ParamType::ENUM): {
                auto param = megamol::core::param::EnumParam(arg);
                // Initialization of enum storage required
                auto map = this->GetStorage<EnumStorageType>();
                for (auto& pair : map) {
                    param.SetTypePair(pair.first, pair.second.c_str());
                }
                value_string = std::string(param.ValueString().PeekBuffer());
            } break;
            default:
                break;
            }
        } else if constexpr (std::is_same_v<T, std::string>) {
            switch (this->type) {
            case (Parameter::ParamType::STRING): {
                auto param = megamol::core::param::StringParam(arg.c_str());
                value_string = std::string(param.ValueString().PeekBuffer());
            } break;
            case (Parameter::ParamType::TRANSFERFUNCTION): {
                auto param = megamol::core::param::TransferFunctionParam(arg);
                value_string = std::string(param.ValueString().PeekBuffer());
            } break;
            case (Parameter::ParamType::FILEPATH): {
                auto param = megamol::core::param::FilePathParam(arg.c_str());
                value_string = std::string(param.ValueString().PeekBuffer());
            } break;
            case (Parameter::ParamType::FLEXENUM): {
                auto param = megamol::core::param::FlexEnumParam(arg.c_str());
                value_string = std::string(param.ValueString().PeekBuffer());
            } break;
            default:
                break;
            }
        } else if constexpr (std::is_same_v<T, vislib::math::Ternary>) {
            auto param = megamol::core::param::TernaryParam(arg);
            value_string = std::string(param.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, glm::vec2>) {
            auto param = megamol::core::param::Vector2fParam(vislib::math::Vector<float, 2>(arg.x, arg.y));
            value_string = std::string(param.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, glm::vec3>) {
            auto param = megamol::core::param::Vector3fParam(vislib::math::Vector<float, 3>(arg.x, arg.y, arg.z));
            value_string = std::string(param.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, glm::vec4>) {
            auto param =
                megamol::core::param::Vector4fParam(vislib::math::Vector<float, 4>(arg.x, arg.y, arg.z, arg.w));
            value_string = std::string(param.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, std::monostate>) {
            switch (this->type) {
            case (Parameter::ParamType::BUTTON): {
                auto param = megamol::core::param::ButtonParam();
                value_string = std::string(param.ValueString().PeekBuffer());
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
    case (Parameter::ParamType::BOOL): {
        megamol::core::param::BoolParam param(false);
        retval = param.ParseValue(val_tstr);
        this->SetValue(param.Value(), set_default_val);
    } break;
    case (Parameter::ParamType::BUTTON): {
        retval = true;
    } break;
    case (Parameter::ParamType::COLOR): {
        megamol::core::param::ColorParam param(val_tstr);
        retval = param.ParseValue(val_tstr);
        this->SetValue(param.Value(), set_default_val);
    } break;
    case (Parameter::ParamType::ENUM): {
        megamol::core::param::EnumParam param(0);
        // Initialization of enum storage required
        auto map = this->GetStorage<EnumStorageType>();
        for (auto& pair : map) {
            param.SetTypePair(pair.first, pair.second.c_str());
        }
        retval = param.ParseValue(val_tstr);
        this->SetValue(param.Value(), set_default_val);
    } break;
    case (Parameter::ParamType::FILEPATH): {
        megamol::core::param::FilePathParam param(val_tstr.PeekBuffer());
        retval = param.ParseValue(val_tstr);
        this->SetValue(std::string(param.Value().PeekBuffer()), set_default_val);
    } break;
    case (Parameter::ParamType::FLEXENUM): {
        megamol::core::param::FlexEnumParam param(val_str);
        retval = param.ParseValue(val_tstr);
        this->SetValue(param.Value(), set_default_val);
    } break;
    case (Parameter::ParamType::FLOAT): {
        megamol::core::param::FloatParam param(0.0f);
        retval = param.ParseValue(val_tstr);
        this->SetValue(param.Value(), set_default_val);
    } break;
    case (Parameter::ParamType::INT): {
        megamol::core::param::IntParam param(0);
        retval = param.ParseValue(val_tstr);
        this->SetValue(param.Value(), set_default_val);
    } break;
    case (Parameter::ParamType::STRING): {
        megamol::core::param::StringParam param(val_tstr.PeekBuffer());
        retval = param.ParseValue(val_tstr);
        this->SetValue(std::string(param.Value().PeekBuffer()), set_default_val);
    } break;
    case (Parameter::ParamType::TERNARY): {
        megamol::core::param::TernaryParam param(vislib::math::Ternary::TRI_UNKNOWN);
        retval = param.ParseValue(val_tstr);
        this->SetValue(param.Value(), set_default_val);
    } break;
    case (Parameter::ParamType::TRANSFERFUNCTION): {
        megamol::core::param::TransferFunctionParam param;
        retval = param.ParseValue(val_tstr);
        this->SetValue(param.Value(), set_default_val);
    } break;
    case (Parameter::ParamType::VECTOR2F): {
        megamol::core::param::Vector2fParam param(vislib::math::Vector<float, 2>(0.0f, 0.0f));
        retval = param.ParseValue(val_tstr);
        auto val = param.Value();
        this->SetValue(glm::vec2(val.X(), val.Y()), set_default_val);
    } break;
    case (Parameter::ParamType::VECTOR3F): {
        megamol::core::param::Vector3fParam param(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
        retval = param.ParseValue(val_tstr);
        auto val = param.Value();
        this->SetValue(glm::vec3(val.X(), val.Y(), val.Z()), set_default_val);
    } break;
    case (Parameter::ParamType::VECTOR4F): {
        megamol::core::param::Vector4fParam param(vislib::math::Vector<float, 4>(0.0f, 0.0f, 0.0f, 0.0f));
        retval = param.ParseValue(val_tstr);
        auto val = param.Value();
        this->SetValue(glm::vec4(val.X(), val.Y(), val.Z(), val.W()), set_default_val);
    } break;
    default:
        break;
    }

    return retval;
}


// PARAMETER PRESENTATION ####################################################


megamol::gui::configurator::Parameter::Presentation::Presentation(void)
    : read_only(false)
    , visible(true)
    , expert(false)
    , presentations(Presentations::DEFAULT)
    , help()
    , utils()
    , file_utils()
    , show_tf_editor(false)
    , tf_editor()
    , widget_store()
    , float_format("%.7f")
    , height(0.0f) {}


megamol::gui::configurator::Parameter::Presentation::~Presentation(void) {}


bool megamol::gui::configurator::Parameter::Presentation::Present(megamol::gui::configurator::Parameter& inout_param) {

    if (ImGui::GetCurrentContext() == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    try {
        // (Show all parameters in expert mode)
        if (this->visible || this->expert) {
            ImGui::BeginGroup();
            ImGui::PushID(inout_param.uid);

            if (this->expert) {
                this->present_prefix();
                ImGui::SameLine();
            }

            switch (this->presentations) {
            case (Presentations::DEFAULT): {
                this->present_value_DEFAULT(inout_param);
            } break;
            // case (Presentations::PIN_VALUE_TO_MOUSE): {
            //     this->present_value_DEFAULT(inout_param);
            //     ImGui::PopID();
            //     this->present_value_PIN_VALUE_TO_MOUSE(inout_param);
            //     ImGui::PushID(inout_param.uid);
            // } break;
            default:
                break;
            }

            ImGui::SameLine();
            this->present_postfix(inout_param);

            ImGui::PopID();
            ImGui::EndGroup();
        }
    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


float megamol::gui::configurator::Parameter::Presentation::GetHeight(Parameter& inout_param) {

    float height = (ImGui::GetFrameHeightWithSpacing() * 1.15f);
    if (inout_param.type == Parameter::ParamType::TRANSFERFUNCTION) {
        if (this->show_tf_editor) {
            height = (ImGui::GetFrameHeightWithSpacing() * 18.5f);
        } else {
            height = (ImGui::GetFrameHeightWithSpacing() * 1.5f);
        }
    }
    return height;
}


bool megamol::gui::configurator::Parameter::Presentation::presentation_button(void) {

    bool retval = false;

    this->utils.PointCircleButton();
    if (ImGui::BeginPopupContextItem("param_present_button_context", 0)) { // 0 = left mouse button
        for (int i = 0; i < static_cast<int>(Presentations::_COUNT_); i++) {
            std::string presentation_str;
            switch (static_cast<Presentations>(i)) {
            case (Presentations::DEFAULT):
                presentation_str = "Default";
                break;
            // case (Presentations::PIN_VALUE_TO_MOUSE):
            //     presentation_str = "Pin Value to Mouse";
            //     break;
            default:
                break;
            }
            if (presentation_str.empty()) break;
            auto presentation_i = static_cast<Presentations>(i);
            if (ImGui::MenuItem(presentation_str.c_str(), nullptr, (presentation_i == this->presentations))) {
                this->presentations = presentation_i;
                retval = true;
            }
        }
        ImGui::EndPopup();
    }

    return retval;
}


void megamol::gui::configurator::Parameter::Presentation::present_prefix(void) {

    // Visibility
    if (ImGui::RadioButton("###visible", !this->visible)) {
        this->visible = !this->visible;
    }
    this->utils.HoverToolTip("Visibility", ImGui::GetItemID(), 0.5f);

    ImGui::SameLine();

    // Read-only option
    ImGui::Checkbox("###readonly", &this->read_only);
    this->utils.HoverToolTip("Read-Only", ImGui::GetItemID(), 0.5f);

    ImGui::SameLine();

    // Presentation
    this->presentation_button();

    this->utils.HoverToolTip("Presentation", ImGui::GetItemID(), 0.5f);
}


void megamol::gui::configurator::Parameter::Presentation::present_value_DEFAULT(
    megamol::gui::configurator::Parameter& inout_param) {

    this->help.clear();

    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.65f); // set general proportional item width

    if (this->read_only) {
        GUIUtils::ReadOnlyWigetStyle(true);
    }

    std::string param_label = inout_param.GetName();

    auto visitor = [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, bool>) {
            if (ImGui::Checkbox(param_label.c_str(), &arg)) {
                inout_param.SetValue(arg);
            }
        } else if constexpr (std::is_same_v<T, megamol::core::param::ColorParam::ColorType>) {
            auto color_flags = ImGuiColorEditFlags_AlphaPreview; // | ImGuiColorEditFlags_Float;
            if (ImGui::ColorEdit4(param_label.c_str(), (float*)arg.data(), color_flags)) {
                inout_param.SetValue(arg);
            }
            this->help = "[Click] on the colored square to open a color picker.\n"
                         "[CTRL+Click] on individual component to input value.\n"
                         "[Right-Click] on the individual color widget to show options.";
        } else if constexpr (std::is_same_v<T, float>) {
            if (!std::holds_alternative<T>(this->widget_store)) {
                this->widget_store = arg;
            }
            ImGui::InputFloat(param_label.c_str(), &std::get<float>(this->widget_store), 1.0f, 10.0f,
                this->float_format.c_str(), ImGuiInputTextFlags_None);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                this->widget_store = std::max(inout_param.GetMinValue<float>(),
                    std::min(std::get<float>(this->widget_store), inout_param.GetMaxValue<float>()));
                inout_param.SetValue(std::get<float>(this->widget_store));
            } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                this->widget_store = arg;
            }
        } else if constexpr (std::is_same_v<T, int>) {
            switch (inout_param.type) {
            case (Parameter::ParamType::INT): {
                if (!std::holds_alternative<T>(this->widget_store)) {
                    this->widget_store = arg;
                }
                ImGui::InputInt(
                    param_label.c_str(), &std::get<int>(this->widget_store), 1, 10, ImGuiInputTextFlags_None);
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    this->widget_store = std::max(inout_param.GetMinValue<int>(),
                        std::min(std::get<int>(this->widget_store), inout_param.GetMaxValue<int>()));
                    inout_param.SetValue(std::get<int>(this->widget_store));
                } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                    this->widget_store = arg;
                }
            } break;
            case (Parameter::ParamType::ENUM): {
                /// XXX: no UTF8 fanciness required here?
                auto map = inout_param.GetStorage<EnumStorageType>();
                if (ImGui::BeginCombo(param_label.c_str(), map[arg].c_str())) {
                    for (auto& pair : map) {
                        bool isSelected = (pair.first == arg);
                        if (ImGui::Selectable(pair.second.c_str(), isSelected)) {
                            inout_param.SetValue(pair.first);
                        }
                        if (isSelected) {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }
            } break;
            default:
                break;
            }
        } else if constexpr (std::is_same_v<T, std::string>) {
            switch (inout_param.type) {
            case (Parameter::ParamType::STRING): {
                if (!std::holds_alternative<T>(this->widget_store)) {
                    /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
                    std::string utf8Str = arg;
                    GUIUtils::Utf8Encode(utf8Str);
                    this->widget_store = utf8Str;
                }
                // Determine multi line count of string
                int lcnt = static_cast<int>(std::count(std::get<std::string>(this->widget_store).begin(),
                    std::get<std::string>(this->widget_store).end(), '\n'));
                lcnt = std::min(static_cast<int>(GUI_MAX_MULITLINE), lcnt);
                ImVec2 ml_dim = ImVec2(ImGui::CalcItemWidth(),
                    ImGui::GetFrameHeight() + (ImGui::GetFontSize() * static_cast<float>(lcnt)));
                std::string hidden_label = "###" + param_label;
                ImGui::InputTextMultiline(hidden_label.c_str(), &std::get<std::string>(this->widget_store), ml_dim,
                    ImGuiInputTextFlags_CtrlEnterForNewLine);
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    std::string utf8Str = std::get<std::string>(this->widget_store);
                    GUIUtils::Utf8Decode(utf8Str);
                    inout_param.SetValue(utf8Str);
                } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                    std::string utf8Str = arg;
                    GUIUtils::Utf8Encode(utf8Str);
                    this->widget_store = utf8Str;
                }
                ImGui::SameLine();
                ImGui::TextUnformatted(param_label.c_str());
                this->help = "[Ctrl + Enter] for new line.\nPress [Return] to confirm changes.";
            } break;
            case (Parameter::ParamType::TRANSFERFUNCTION): {
                this->transfer_function_edit(inout_param);
            } break;
            case (Parameter::ParamType::FILEPATH): {
                if (!std::holds_alternative<T>(this->widget_store)) {
                    /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
                    std::string utf8Str = arg;
                    GUIUtils::Utf8Encode(utf8Str);
                    this->widget_store = utf8Str;
                }
                ImGuiStyle& style = ImGui::GetStyle();
                ImGui::PushItemWidth(
                    ImGui::GetContentRegionAvail().x * 0.65f - ImGui::GetFrameHeight() - style.ItemSpacing.x);
                bool button_edit = this->file_utils.FileBrowserButton(std::get<std::string>(this->widget_store));
                ImGui::SameLine();
                ImGui::InputText(
                    param_label.c_str(), &std::get<std::string>(this->widget_store), ImGuiInputTextFlags_None);
                if (button_edit || ImGui::IsItemDeactivatedAfterEdit()) {
                    GUIUtils::Utf8Decode(std::get<std::string>(this->widget_store));
                    inout_param.SetValue(std::get<std::string>(this->widget_store));
                } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                    std::string utf8Str = arg;
                    GUIUtils::Utf8Encode(utf8Str);
                    this->widget_store = utf8Str;
                }
                ImGui::PopItemWidth();
            } break;
            case (Parameter::ParamType::FLEXENUM): {
                /// XXX: no UTF8 fanciness required here?
                if (ImGui::BeginCombo(param_label.c_str(), arg.c_str())) {
                    for (auto valueOption : inout_param.GetStorage<megamol::core::param::FlexEnumParam::Storage_t>()) {
                        bool isSelected = (valueOption == arg);
                        if (ImGui::Selectable(valueOption.c_str(), isSelected)) {
                            inout_param.SetValue(valueOption);
                        }
                        if (isSelected) {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }
            } break;
            default:
                break;
            }
        } else if constexpr (std::is_same_v<T, vislib::math::Ternary>) {
            if (ImGui::RadioButton("True", arg.IsTrue())) {
                inout_param.SetValue(vislib::math::Ternary::TRI_TRUE);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("False", arg.IsFalse())) {
                inout_param.SetValue(vislib::math::Ternary::TRI_FALSE);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Unknown", arg.IsUnknown())) {
                inout_param.SetValue(vislib::math::Ternary::TRI_UNKNOWN);
            }
            ImGui::SameLine();
            ImGui::TextDisabled("|");
            ImGui::SameLine();
            ImGui::TextUnformatted(param_label.c_str());
        } else if constexpr (std::is_same_v<T, glm::vec2>) {
            if (!std::holds_alternative<T>(this->widget_store)) {
                this->widget_store = arg;
            }
            ImGui::InputFloat2(param_label.c_str(), glm::value_ptr(std::get<glm::vec2>(this->widget_store)),
                this->float_format.c_str(), ImGuiInputTextFlags_None);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                auto max = inout_param.GetMaxValue<glm::vec2>();
                auto min = inout_param.GetMinValue<glm::vec2>();
                auto x = std::max(min.x, std::min(std::get<glm::vec2>(this->widget_store).x, max.x));
                auto y = std::max(min.y, std::min(std::get<glm::vec2>(this->widget_store).y, max.y));
                this->widget_store = glm::vec2(x, y);
                inout_param.SetValue(std::get<glm::vec2>(this->widget_store));
            } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                this->widget_store = arg;
            }
        } else if constexpr (std::is_same_v<T, glm::vec3>) {
            if (!std::holds_alternative<T>(this->widget_store)) {
                this->widget_store = arg;
            }
            ImGui::InputFloat3(param_label.c_str(), glm::value_ptr(std::get<glm::vec3>(this->widget_store)),
                this->float_format.c_str(), ImGuiInputTextFlags_None);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                auto max = inout_param.GetMaxValue<glm::vec3>();
                auto min = inout_param.GetMinValue<glm::vec3>();
                auto x = std::max(min.x, std::min(std::get<glm::vec3>(this->widget_store).x, max.x));
                auto y = std::max(min.y, std::min(std::get<glm::vec3>(this->widget_store).y, max.y));
                auto z = std::max(min.z, std::min(std::get<glm::vec3>(this->widget_store).z, max.z));
                this->widget_store = glm::vec3(x, y, z);
                inout_param.SetValue(std::get<glm::vec3>(this->widget_store));
            } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                this->widget_store = arg;
            }
        } else if constexpr (std::is_same_v<T, glm::vec4>) {
            if (!std::holds_alternative<T>(this->widget_store)) {
                this->widget_store = arg;
            }
            ImGui::InputFloat4(param_label.c_str(), glm::value_ptr(std::get<glm::vec4>(this->widget_store)),
                this->float_format.c_str(), ImGuiInputTextFlags_None);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                auto max = inout_param.GetMaxValue<glm::vec4>();
                auto min = inout_param.GetMinValue<glm::vec4>();
                auto x = std::max(min.x, std::min(std::get<glm::vec4>(this->widget_store).x, max.x));
                auto y = std::max(min.y, std::min(std::get<glm::vec4>(this->widget_store).y, max.y));
                auto z = std::max(min.z, std::min(std::get<glm::vec4>(this->widget_store).z, max.z));
                auto w = std::max(min.w, std::min(std::get<glm::vec4>(this->widget_store).w, max.w));
                this->widget_store = glm::vec4(x, y, z, w);
                inout_param.SetValue(std::get<glm::vec4>(this->widget_store));
            } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                this->widget_store = arg;
            }
        } else if constexpr (std::is_same_v<T, std::monostate>) {
            switch (inout_param.type) {
            case (Parameter::ParamType::BUTTON): {
                std::string hotkey = "";
                auto keycode = inout_param.GetStorage<megamol::core::view::KeyCode>();
                std::string button_hotkey = keycode.ToString();
                if (!button_hotkey.empty()) {
                    hotkey = " (" + button_hotkey + ")";
                }
                param_label += hotkey;
                if (ImGui::Button(param_label.c_str())) {
                    // inout_param.setDirty();
                }
            } break;
            default:
                break;
            }
        }
    };

    std::visit(visitor, inout_param.GetValue());

    if (this->read_only) {
        GUIUtils::ReadOnlyWigetStyle(false);
    }

    ImGui::PopItemWidth();
}

/*
void megamol::gui::configurator::Parameter::Presentation::present_value_PIN_VALUE_TO_MOUSE(
    megamol::gui::configurator::Parameter& inout_param) {

    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    std::string param_label = inout_param.GetName();

    ImGui::BeginTooltip();

    auto visitor = [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, bool>) {

        } else if constexpr (std::is_same_v<T, megamol::core::param::ColorParam::ColorType>) {

        } else if constexpr (std::is_same_v<T, float>) {
            ImGui::TextDisabled(inout_param.GetValueString().c_str());
        } else if constexpr (std::is_same_v<T, int>) {
            switch (inout_param.type) {
            case (Parameter::ParamType::INT): {
                if (!std::holds_alternative<T>(this->widget_store)) {
                    this->widget_store = arg;
                }
                ImGui::InputInt(
                    param_label.c_str(), &std::get<int>(this->widget_store), ImGuiInputTextFlags_ReadOnly);

            } break;
            case (Parameter::ParamType::ENUM): {

            } break;
            default:
                break;
            }
        } else if constexpr (std::is_same_v<T, std::string>) {
            switch (inout_param.type) {
            case (Parameter::ParamType::STRING): {

            } break;
            case (Parameter::ParamType::TRANSFERFUNCTION): {

            } break;
            case (Parameter::ParamType::FILEPATH): {

            } break;
            case (Parameter::ParamType::FLEXENUM): {

            } break;
            default:
                break;
            }
        } else if constexpr (std::is_same_v<T, vislib::math::Ternary>) {

        } else if constexpr (std::is_same_v<T, glm::vec2>) {

        } else if constexpr (std::is_same_v<T, glm::vec3>) {

        } else if constexpr (std::is_same_v<T, glm::vec4>) {

        } else if constexpr (std::is_same_v<T, std::monostate>) {
            switch (inout_param.type) {
            case (Parameter::ParamType::BUTTON): {

            } break;
            default:
                break;
            }
        }
    };

    std::visit(visitor, inout_param.GetValue());

    ImGui::EndTooltip();
}
*/

void megamol::gui::configurator::Parameter::Presentation::present_postfix(
    megamol::gui::configurator::Parameter& inout_param) {

    this->utils.HoverToolTip(inout_param.description, ImGui::GetItemID(), 0.5f);
    this->utils.HelpMarkerToolTip(this->help);
}


void megamol::gui::configurator::Parameter::Presentation::transfer_function_edit(
    megamol::gui::configurator::Parameter& inout_param) {

    if ((inout_param.type != Parameter::ParamType::TRANSFERFUNCTION) ||
        (!std::holds_alternative<std::string>(inout_param.GetValue()))) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Transfer Function Editor is called for incompatible parameter type. [%s, %s, line %d]\n", __FILE__,
            __FUNCTION__, __LINE__);
        return;
    }
    auto value = std::get<std::string>(inout_param.GetValue());

    ImGui::BeginGroup();

    // Reduced display of value and editor state.
    if (value.empty()) {
        ImGui::TextDisabled("{    (empty)    }");
    } else {
        /// XXX: A gradient texture would be nice here (sharing some editor code?)
        ImGui::TextUnformatted("{ ............. }");
    }
    ImGui::SameLine();

    bool updateEditor = false;

    // Edit transfer function.
    if (ImGui::Checkbox("Editor", &this->show_tf_editor)) {
        // Set once
        if (this->show_tf_editor) {
            updateEditor = true;
        }
    }
    ImGui::SameLine();

    // Copy transfer function.
    if (ImGui::Button("Copy")) {
#ifdef GUI_USE_GLFW
        auto glfw_win = ::glfwGetCurrentContext();
        ::glfwSetClipboardString(glfw_win, value.c_str());
#elif _WIN32
        ImGui::SetClipboardText(value.c_str());
#else // LINUX
        vislib::sys::Log::DefaultLog.WriteWarn(
            "No clipboard use provided. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        vislib::sys::Log::DefaultLog.WriteInfo("Transfer Function JSON String:\n%s", value.c_str());
#endif
    }
    ImGui::SameLine();

    //  Paste transfer function.
    if (ImGui::Button("Paste")) {
#ifdef GUI_USE_GLFW
        auto glfw_win = ::glfwGetCurrentContext();
        inout_param.SetValue(std::string(::glfwGetClipboardString(glfw_win)));
#elif _WIN32
        inout_param.SetValue(std::string(ImGui::GetClipboardText()));
#else // LINUX
        vislib::sys::Log::DefaultLog.WriteWarn(
            "No clipboard use provided. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
#endif
        updateEditor = true;
    }

    ImGui::SameLine();

    std::string label = inout_param.full_name;
    ImGui::TextUnformatted(label.c_str(), ImGui::FindRenderedTextEnd(label.c_str()));

    ImGui::EndGroup();

    // Propagate the transfer function to the editor.
    if (updateEditor) {
        this->tf_editor.SetTransferFunction(value, false);
    }

    // Draw transfer function editor
    if (this->show_tf_editor) {
        if (this->tf_editor.Draw(false)) {
            std::string value;
            if (this->tf_editor.GetTransferFunction(value)) {
                inout_param.SetValue(value);
            }
        }
        ImGui::Separator();
    }
}
