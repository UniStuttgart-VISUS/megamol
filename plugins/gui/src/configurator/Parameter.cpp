/*
 * Parameter.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Parameter.h"


#define GUI_MAX_MULITLINE 7


using namespace megamol;
using namespace megamol::gui::configurator;


megamol::gui::configurator::Parameter::Parameter(int uid, ParamType type, StroageType store, MinType min, MaxType max)
    : uid(uid), type(type), minval(min), maxval(max), storage(store), value(), present() {

    // Initialize variant types which should/can not be changed afterwards.
    // Default ctor of variants initializes std::monostate.
    switch (this->type) {
    case (Parameter::ParamType::BOOL): {
        this->value = bool(false);
    } break;
    case (Parameter::ParamType::BUTTON): {
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
            }
            case (Parameter::ParamType::ENUM): {
                auto param = megamol::core::param::EnumParam(arg);
                value_string = std::string(param.ValueString().PeekBuffer());
            }
            default:
                break;
            }
        } else if constexpr (std::is_same_v<T, std::string>) {
            switch (this->type) {
            case (Parameter::ParamType::STRING): {
                auto param = megamol::core::param::StringParam(arg.c_str());
                value_string = std::string(param.ValueString().PeekBuffer());
                break;
            }
            case (Parameter::ParamType::TRANSFERFUNCTION): {
                auto param = megamol::core::param::TransferFunctionParam(arg);
                value_string = std::string(param.ValueString().PeekBuffer());
                break;
            }
            case (Parameter::ParamType::FILEPATH): {
                auto param = megamol::core::param::FilePathParam(arg.c_str());
                value_string = std::string(param.ValueString().PeekBuffer());
                break;
            }
            case (Parameter::ParamType::FLEXENUM): {
                auto param = megamol::core::param::FlexEnumParam(arg.c_str());
                value_string = std::string(param.ValueString().PeekBuffer());
                break;
            }
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


bool megamol::gui::configurator::Parameter::SetValueString(std::string val_str) {

    bool retval = false;
    vislib::TString val_tstr(val_str.c_str());

    switch (this->type) {
    case (Parameter::ParamType::BOOL): {
        megamol::core::param::BoolParam param(false);
        retval = param.ParseValue(val_tstr);
        this->SetValue(param.Value());
    } break;
    case (Parameter::ParamType::BUTTON): {
        retval = true;
    } break;
    case (Parameter::ParamType::COLOR): {
        megamol::core::param::ColorParam param(val_tstr);
        retval = param.ParseValue(val_tstr);
        this->SetValue(param.Value());
    } break;
    case (Parameter::ParamType::ENUM): {
        megamol::core::param::EnumParam param(0);
        retval = param.ParseValue(val_tstr);
        this->SetValue(param.Value());
    } break;
    case (Parameter::ParamType::FILEPATH): {
        megamol::core::param::FilePathParam param(val_tstr.PeekBuffer());
        retval = param.ParseValue(val_tstr);
        this->SetValue(std::string(param.Value().PeekBuffer()));
    } break;
    case (Parameter::ParamType::FLEXENUM): {
        megamol::core::param::FlexEnumParam param(val_str);
        retval = param.ParseValue(val_tstr);
        this->SetValue(param.Value());
    } break;
    case (Parameter::ParamType::FLOAT): {
        megamol::core::param::FloatParam param(0.0f);
        retval = param.ParseValue(val_tstr);
        this->SetValue(param.Value());
    } break;
    case (Parameter::ParamType::INT): {
        megamol::core::param::IntParam param(0);
        retval = param.ParseValue(val_tstr);
        this->SetValue(param.Value());
    } break;
    case (Parameter::ParamType::STRING): {
        megamol::core::param::StringParam param(val_tstr.PeekBuffer());
        retval = param.ParseValue(val_tstr);
        this->SetValue(std::string(param.Value().PeekBuffer()));
    } break;
    case (Parameter::ParamType::TERNARY): {
        megamol::core::param::TernaryParam param(vislib::math::Ternary::TRI_UNKNOWN);
        retval = param.ParseValue(val_tstr);
        this->SetValue(param.Value());
    } break;
    case (Parameter::ParamType::TRANSFERFUNCTION): {
        megamol::core::param::TransferFunctionParam param;
        retval = param.ParseValue(val_tstr);
        this->SetValue(param.Value());
    } break;
    case (Parameter::ParamType::VECTOR2F): {
        megamol::core::param::Vector2fParam param(vislib::math::Vector<float, 2>(0.0f, 0.0f));
        retval = param.ParseValue(val_tstr);
        auto val = param.Value();
        this->SetValue(glm::vec2(val.X(), val.Y()));
    } break;
    case (Parameter::ParamType::VECTOR3F): {
        megamol::core::param::Vector3fParam param(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
        retval = param.ParseValue(val_tstr);
        auto val = param.Value();
        this->SetValue(glm::vec3(val.X(), val.Y(), val.Z()));
    } break;
    case (Parameter::ParamType::VECTOR4F): {
        megamol::core::param::Vector4fParam param(vislib::math::Vector<float, 4>(0.0f, 0.0f, 0.0f, 0.0f));
        retval = param.ParseValue(val_tstr);
        auto val = param.Value();
        this->SetValue(glm::vec4(val.X(), val.Y(), val.Z(), val.W()));
    } break;
    default:
        break;
    }

    return retval;
}


// PARAMETER PRESENTATION ####################################################

megamol::gui::configurator::Parameter::Presentation::Presentation(void)
    : presentations(Parameter::Presentations::SIMPLE)
    , read_only(false)
    , visible(true)
    , help()
    , utils()
    , tf_editor()
    , show_tf_editor(false)
    , widget_store() {}


megamol::gui::configurator::Parameter::Presentation::~Presentation(void) {}


bool megamol::gui::configurator::Parameter::Presentation::Present(megamol::gui::configurator::Parameter& param) {

    try {

        if (ImGui::GetCurrentContext() == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        if (this->visible) {
            ImGui::BeginGroup();
            ImGui::PushID(param.uid);

            switch (this->presentations) {
            case (Parameter::Presentations::DEFAULT): {
                this->present_prefix(param);
                ImGui::SameLine();
                this->present_value(param);
                ImGui::SameLine();
                this->present_postfix(param);
                break;
            }
            case (Parameter::Presentations::SIMPLE): {
                this->present_value(param);
                ImGui::SameLine();
                this->present_postfix(param);
                break;
            }
            default:
                break;
            }

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


bool megamol::gui::configurator::Parameter::Presentation::PresentationButton(
    megamol::gui::configurator::Parameter::Presentations& inout_present, std::string label) {
    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    bool retval = false;

    float height = ImGui::GetFrameHeight();
    float half_height = height / 2.0f;
    ImVec2 position = ImGui::GetCursorScreenPos();

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_FrameBg]));
    ImGui::BeginChild("special_button_background", ImVec2(height, height), false,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    float thickness = height / 5.0f;
    ImVec2 center = position + ImVec2(half_height, half_height);

    ImU32 color_front = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonActive]);

    draw_list->AddCircleFilled(center, thickness, color_front, 12);
    draw_list->AddCircle(center, 2.0f * thickness, color_front, 12, (thickness / 2.0f));
    ImGui::EndChild();
    ImGui::PopStyleColor();

    ImGui::SetCursorScreenPos(position);
    ImVec2 rect = ImVec2(height, height);
    ImGui::InvisibleButton("special_button", rect);
    if (ImGui::BeginPopupContextItem("special_button_context", 0)) { // 0 = left mouse button
        for (int i = 0; i < static_cast<int>(Parameter::Presentations::_COUNT_); i++) {
            std::string presentation_str;
            switch (static_cast<Parameter::Presentations>(i)) {
            case (Parameter::Presentations::DEFAULT):
                presentation_str = "DEFAULT";
                break;
            case (Parameter::Presentations::SIMPLE):
                presentation_str = "SIMPLE";
                break;
            defalt:
                break;
            }
            if (presentation_str.empty()) break;
            auto presentation_i = static_cast<Parameter::Presentations>(i);
            if (ImGui::MenuItem(presentation_str.c_str(), nullptr, false)) { //(presentation_i == inout_present))) {
                inout_present = presentation_i;
                retval = true;
            }
        }
        ImGui::EndPopup();
    }

    if (!label.empty()) {
        ImGui::SameLine();
        position = ImGui::GetCursorScreenPos();
        position.y += (ImGui::GetFrameHeight() / 2.0f - ImGui::GetFontSize() / 2.0f);
        ImGui::SetCursorScreenPos(position);
        ImGui::Text(label.c_str());
    }

    return retval;
}


void megamol::gui::configurator::Parameter::Presentation::present_prefix(megamol::gui::configurator::Parameter& param) {

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
    this->PresentationButton(this->presentations);

    this->utils.HoverToolTip("Presentation", ImGui::GetItemID(), 0.5f);
}


void megamol::gui::configurator::Parameter::Presentation::present_value(megamol::gui::configurator::Parameter& param) {

    this->help.clear();

    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.65f); // set general proportional item width

    if (this->read_only) {
        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }

    std::string param_label = param.GetName();
    std::string float_format = "%.7f";

    auto visitor = [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, bool>) {
            if (ImGui::Checkbox(param_label.c_str(), &arg)) {
                param.SetValue(arg);
            }
        } else if constexpr (std::is_same_v<T, megamol::core::param::ColorParam::ColorType>) {
            auto color_flags = ImGuiColorEditFlags_AlphaPreview; // | ImGuiColorEditFlags_Float;
            if (ImGui::ColorEdit4(param_label.c_str(), (float*)arg.data(), color_flags)) {
                param.SetValue(arg);
            }
            this->help = "[Click] on the colored square to open a color picker.\n"
                         "[CTRL+Click] on individual component to input value.\n"
                         "[Right-Click] on the individual color widget to show options.";
        } else if constexpr (std::is_same_v<T, float>) {
            if (!std::holds_alternative<T>(this->widget_store)) {
                this->widget_store = arg;
            }
            ImGui::InputFloat(param_label.c_str(), &std::get<float>(this->widget_store), 1.0f, 10.0f,
                float_format.c_str(), ImGuiInputTextFlags_None);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                this->widget_store = std::max(param.GetMinValue<float>(),
                    std::min(std::get<float>(this->widget_store), param.GetMaxValue<float>()));
                param.SetValue(std::get<float>(this->widget_store));
            } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                this->widget_store = arg;
            }
        } else if constexpr (std::is_same_v<T, int>) {
            switch (param.type) {
            case (Parameter::ParamType::INT): {
                if (!std::holds_alternative<T>(this->widget_store)) {
                    this->widget_store = arg;
                }
                ImGui::InputInt(
                    param_label.c_str(), &std::get<int>(this->widget_store), 1, 10, ImGuiInputTextFlags_None);
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    this->widget_store = std::max(param.GetMinValue<int>(),
                        std::min(std::get<int>(this->widget_store), param.GetMaxValue<int>()));
                    param.SetValue(std::get<int>(this->widget_store));
                } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                    this->widget_store = arg;
                }
            } break;
            case (Parameter::ParamType::ENUM): {
                /// XXX: no UTF8 fanciness required here?
                auto map = param.GetStorage<EnumStorageType>();
                if (ImGui::BeginCombo(param_label.c_str(), map[arg].c_str())) {
                    for (auto& pair : map) {
                        bool isSelected = (pair.first == arg);
                        if (ImGui::Selectable(pair.second.c_str(), isSelected)) {
                            param.SetValue(pair.first);
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
            switch (param.type) {
            case (Parameter::ParamType::STRING): {
                if (!std::holds_alternative<T>(this->widget_store)) {
                    /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
                    std::string utf8Str = arg;
                    this->utils.Utf8Encode(utf8Str);
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
                    this->utils.Utf8Decode(utf8Str);
                    param.SetValue(utf8Str);
                } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                    std::string utf8Str = arg;
                    this->utils.Utf8Encode(utf8Str);
                    this->widget_store = utf8Str;
                }
                ImGui::SameLine();
                ImGui::Text(param_label.c_str());
                this->help = "[Ctrl + Enter] for new line.\nPress [Return] to confirm changes.";
            } break;
            case (Parameter::ParamType::TRANSFERFUNCTION): {
                this->transfer_function_edit(param);
            } break;
            case (Parameter::ParamType::FILEPATH): {
                if (!std::holds_alternative<T>(this->widget_store)) {
                    /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
                    std::string utf8Str = arg;
                    this->utils.Utf8Encode(utf8Str);
                    this->widget_store = utf8Str;
                }
                ImGui::InputText(
                    param_label.c_str(), &std::get<std::string>(this->widget_store), ImGuiInputTextFlags_None);
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    this->utils.Utf8Decode(std::get<std::string>(this->widget_store));
                    param.SetValue(std::get<std::string>(this->widget_store));
                } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                    std::string utf8Str = arg;
                    this->utils.Utf8Encode(utf8Str);
                    this->widget_store = utf8Str;
                }
            } break;
            case (Parameter::ParamType::FLEXENUM): {
                /// XXX: no UTF8 fanciness required here?
                if (ImGui::BeginCombo(param_label.c_str(), arg.c_str())) {
                    for (auto valueOption : param.GetStorage<megamol::core::param::FlexEnumParam::Storage_t>()) {
                        bool isSelected = (valueOption == arg);
                        if (ImGui::Selectable(valueOption.c_str(), isSelected)) {
                            param.SetValue(valueOption);
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
                param.SetValue(vislib::math::Ternary::TRI_TRUE);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("False", arg.IsFalse())) {
                param.SetValue(vislib::math::Ternary::TRI_FALSE);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Unknown", arg.IsUnknown())) {
                param.SetValue(vislib::math::Ternary::TRI_UNKNOWN);
            }
            ImGui::SameLine();
            ImGui::TextDisabled("|");
            ImGui::SameLine();
            ImGui::Text(param_label.c_str());
        } else if constexpr (std::is_same_v<T, glm::vec2>) {
            if (!std::holds_alternative<T>(this->widget_store)) {
                this->widget_store = arg;
            }
            ImGui::InputFloat2(param_label.c_str(), glm::value_ptr(std::get<glm::vec2>(this->widget_store)),
                float_format.c_str(), ImGuiInputTextFlags_None);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                auto max = param.GetMaxValue<glm::vec2>();
                auto min = param.GetMinValue<glm::vec2>();
                auto x = std::max(min.x, std::min(std::get<glm::vec2>(this->widget_store).x, max.x));
                auto y = std::max(min.y, std::min(std::get<glm::vec2>(this->widget_store).y, max.y));
                this->widget_store = glm::vec2(x, y);
                param.SetValue(std::get<glm::vec2>(this->widget_store));
            } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                this->widget_store = arg;
            }
        } else if constexpr (std::is_same_v<T, glm::vec3>) {
            if (!std::holds_alternative<T>(this->widget_store)) {
                this->widget_store = arg;
            }
            ImGui::InputFloat3(param_label.c_str(), glm::value_ptr(std::get<glm::vec3>(this->widget_store)),
                float_format.c_str(), ImGuiInputTextFlags_None);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                auto max = param.GetMaxValue<glm::vec3>();
                auto min = param.GetMinValue<glm::vec3>();
                auto x = std::max(min.x, std::min(std::get<glm::vec3>(this->widget_store).x, max.x));
                auto y = std::max(min.y, std::min(std::get<glm::vec3>(this->widget_store).y, max.y));
                auto z = std::max(min.z, std::min(std::get<glm::vec3>(this->widget_store).z, max.z));
                this->widget_store = glm::vec3(x, y, z);
                param.SetValue(std::get<glm::vec3>(this->widget_store));
            } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                this->widget_store = arg;
            }
        } else if constexpr (std::is_same_v<T, glm::vec4>) {
            if (!std::holds_alternative<T>(this->widget_store)) {
                this->widget_store = arg;
            }
            ImGui::InputFloat4(param_label.c_str(), glm::value_ptr(std::get<glm::vec4>(this->widget_store)),
                float_format.c_str(), ImGuiInputTextFlags_None);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                auto max = param.GetMaxValue<glm::vec4>();
                auto min = param.GetMinValue<glm::vec4>();
                auto x = std::max(min.x, std::min(std::get<glm::vec4>(this->widget_store).x, max.x));
                auto y = std::max(min.y, std::min(std::get<glm::vec4>(this->widget_store).y, max.y));
                auto z = std::max(min.z, std::min(std::get<glm::vec4>(this->widget_store).z, max.z));
                auto w = std::max(min.w, std::min(std::get<glm::vec4>(this->widget_store).w, max.w));
                this->widget_store = glm::vec4(x, y, z, w);
                param.SetValue(std::get<glm::vec4>(this->widget_store));
            } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                this->widget_store = arg;
            }
        } else if constexpr (std::is_same_v<T, std::monostate>) {
            switch (param.type) {
            case (Parameter::ParamType::BUTTON): {
                std::string hotkey = "";
                auto keycode = param.GetStorage<megamol::core::view::KeyCode>();
                std::string button_hotkey = keycode.ToString();
                if (!button_hotkey.empty()) {
                    hotkey = " (" + button_hotkey + ")";
                }
                param_label += hotkey;
                if (ImGui::Button(param_label.c_str())) {
                    // param.setDirty();
                }
            } break;
            default:
                break;
            }
        }
    };

    std::visit(visitor, param.GetValue());

    if (this->read_only) {
        ImGui::PopItemFlag();
        ImGui::PopStyleVar();
    }

    ImGui::PopItemWidth();
}


void megamol::gui::configurator::Parameter::Presentation::present_postfix(
    megamol::gui::configurator::Parameter& param) {

    this->utils.HoverToolTip(param.description, ImGui::GetItemID(), 0.5f);
    this->utils.HelpMarkerToolTip(this->help);
}


void megamol::gui::configurator::Parameter::Presentation::transfer_function_edit(
    megamol::gui::configurator::Parameter& param) {

    if (!std::holds_alternative<std::string>(param.GetValue())) {
        return;
    }
    auto value = std::get<std::string>(param.GetValue());

    ImGuiStyle& style = ImGui::GetStyle();

    ImGui::BeginGroup();

    // Reduced display of value and editor state.
    if (value.empty()) {
        ImGui::TextDisabled("{    (empty)    }");
    } else {
        /// XXX: A gradient texture would be nice here (sharing some editor code?)
        ImGui::Text("{ ............. }");
    }
    ImGui::SameLine();

    bool updateEditor = false;

    // Edit transfer function.
    if (ImGui::Checkbox("Editor", &this->show_tf_editor)) {
        // Set once
        if (this->show_tf_editor) {
            updateEditor = true;
            /// XXX this->tf_editor.SetActiveParameter(&param);
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
        param.SetValue(std::string(::glfwGetClipboardString(glfw_win)));
#elif _WIN32
        param.SetValue(std::string(ImGui::GetClipboardText()));
#else // LINUX
        vislib::sys::Log::DefaultLog.WriteWarn(
            "No clipboard use provided. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
#endif
        updateEditor = true;
    }

    ImGui::SameLine();

    std::string label = param.full_name;
    ImGui::Text(label.c_str(), ImGui::FindRenderedTextEnd(label.c_str()));

    ImGui::EndGroup();

    // Propagate the transfer function to the editor.
    if (updateEditor) {
        this->tf_editor.SetTransferFunction(value);
    }

    // Draw transfer function editor
    if (this->show_tf_editor) {
        if (this->tf_editor.DrawTransferFunctionEditor(false)) {
            std::string value;
            if (this->tf_editor.GetTransferFunction(value)) {
                param.SetValue(value);
            }
        }
    }

    ImGui::Separator();
}
