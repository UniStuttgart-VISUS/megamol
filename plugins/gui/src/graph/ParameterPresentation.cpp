/*
 * Parameter.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ParameterPresentation.h"

#include "Parameter.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::ParameterPresentation::ParameterPresentation(Param_t type)
    : megamol::core::param::AbstractParamPresentation()
    , extended(false)
    , float_format("%.7f")
    , help()
    , description()
    , widget_store()
    , set_focus(0)
    , guistate_dirty(false)
    , tf_editor_external_ptr(nullptr)
    , tf_editor_internal()
    , use_external_tf_editor(false)
    , show_tf_editor(false)
    , tf_editor_hash(0)
    , file_browser()
    , tooltip()
    , image_widget()
    , rotation_widget() {

    this->InitPresentation(type);
}


megamol::gui::ParameterPresentation::~ParameterPresentation(void) {}


bool megamol::gui::ParameterPresentation::Present(megamol::gui::Parameter& inout_parameter, WidgetScope scope) {

    bool retval = false;

    if (ImGui::GetCurrentContext() == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    try {
        ImGui::PushID(inout_parameter.uid);

        this->help = "";
        this->description = inout_parameter.description;

        switch (scope) {
        case (ParameterPresentation::WidgetScope::LOCAL): {
            if (this->IsGUIVisible() || this->extended) {

                ImGui::BeginGroup();
                if (this->extended) {
                    /// PREFIX ---------------------------------------------

                    // Visibility
                    if (ImGui::RadioButton("###visible", this->IsGUIVisible())) {
                        this->SetGUIVisible(!this->IsGUIVisible());
                        this->ForceSetGUIStateDirty();
                    }
                    this->tooltip.ToolTip("Visibility", ImGui::GetItemID(), 0.5f);

                    ImGui::SameLine();

                    // Read-only option
                    bool read_only = this->IsGUIReadOnly();
                    if (ImGui::Checkbox("###readonly", &read_only)) {
                        this->SetGUIReadOnly(read_only);
                        this->ForceSetGUIStateDirty();
                    }
                    this->tooltip.ToolTip("Read-Only", ImGui::GetItemID(), 0.5f);

                    ImGui::SameLine();

                    // Presentation
                    ParameterPresentation::OptionButton(
                        "param_present_button", "", (this->GetGUIPresentation() != Present_t::Basic));
                    if (ImGui::BeginPopupContextItem("param_present_button_context", 0)) {
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
                    this->tooltip.ToolTip("Presentation", ImGui::GetItemID(), 0.5f);

                    ImGui::SameLine();
                }

                /// PARAMETER VALUE WIDGET ---------------------------------
                if (this->present_parameter(inout_parameter, scope)) {
                    retval = true;
                }

                ImGui::SameLine();

                /// POSTFIX ------------------------------------------------
                this->tooltip.ToolTip(this->description, ImGui::GetItemID(), 0.5f);
                this->tooltip.Marker(this->help);

                ImGui::EndGroup();
            }
        } break;
        case (ParameterPresentation::WidgetScope::GLOBAL): {

            if (this->present_parameter(inout_parameter, scope)) {
                retval = true;
            }

        } break;
        default:
            break;
        }

        ImGui::PopID();

    } catch (std::exception e) {
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


void megamol::gui::ParameterPresentation::LoadTransferFunctionTexture(
    std::vector<float>& in_texture_data, int& in_texture_width, int& in_texture_height) {

    image_widget.LoadTextureFromData(in_texture_width, in_texture_height, in_texture_data.data());
}


bool megamol::gui::ParameterPresentation::OptionButton(const std::string& id, const std::string& label, bool dirty) {

    bool retval = false;
    std::string widget_name("option_button");
    std::string widget_id = widget_name + id;
    ImGui::PushID(widget_id.c_str());

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    float knob_size = ImGui::GetFrameHeight();
    float half_knob_size = knob_size / 2.0f;
    ImVec2 widget_start_pos = ImGui::GetCursorScreenPos();

    if (!label.empty()) {
        float text_x_offset_pos = knob_size + style.ItemInnerSpacing.x;
        ImGui::SetCursorScreenPos(widget_start_pos + ImVec2(text_x_offset_pos, 0.0f));
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label.c_str());
        ImGui::SetCursorScreenPos(widget_start_pos);
    }

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_FrameBg]));
    auto child_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove;
    ImGui::BeginChild("special_button_background", ImVec2(knob_size, knob_size), false, child_flags);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    float thickness = knob_size / 5.0f;
    ImVec2 center = widget_start_pos + ImVec2(half_knob_size, half_knob_size);
    ImU32 color_front = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonActive]);
    if (dirty) {
        color_front = ImGui::ColorConvertFloat4ToU32(GUI_COLOR_BUTTON_MODIFIED);
    }
    draw_list->AddCircleFilled(center, thickness, color_front, 12);
    draw_list->AddCircle(center, 2.0f * thickness, color_front, 12, (thickness / 2.0f));

    ImVec2 rect = ImVec2(knob_size, knob_size);
    retval = ImGui::InvisibleButton("special_button", rect);

    ImGui::EndChild();
    ImGui::PopStyleColor();

    ImGui::PopID();

    return retval;
}


bool megamol::gui::ParameterPresentation::KnobButton(
    const std::string& id, float size, float& inout_value, float minval, float maxval) {

    bool retval = false;

    const float pi = 3.14159265358f;

    std::string widget_name("knob_widget_background");
    std::string widget_id = widget_name + id;
    ImGui::PushID(widget_id.c_str());

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec2 widget_start_pos = ImGui::GetCursorScreenPos();

    const float thickness = size / 15.0f;
    const float knob_radius = thickness * 2.0f;

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_FrameBg]));
    auto child_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove;
    ImGui::BeginChild("knob_widget_background", ImVec2(size, size), false, child_flags);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    // Draw Outline
    float half_knob_size = size / 2.0f;

    ImVec2 widget_center = widget_start_pos + ImVec2(half_knob_size, half_knob_size);
    float half_thickness = (thickness / 2.0f);
    ImU32 outline_color_front = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonHovered]);
    ImU32 outline_color_shadow = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_FrameBgHovered]);

    // Shadow
    draw_list->AddCircle(widget_center, half_knob_size - thickness, outline_color_shadow, 24, thickness);
    // Outline
    draw_list->AddCircle(widget_center, half_knob_size - half_thickness, outline_color_front, 24, half_thickness);

    // Draw knob
    ImU32 knob_line_color = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Button]);
    ImU32 knob_color = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonHovered]);
    ImVec2 rect = ImVec2(knob_radius * 2.0f, knob_radius * 2.0f);
    ImVec2 half_rect = ImVec2(knob_radius, knob_radius);
    float knob_center_dist = (half_knob_size - knob_radius - half_thickness);

    // Adapt scaling of one round depending on min max delta
    float scaling = 1.0f;
    if ((minval > -FLT_MAX) && (maxval < FLT_MAX) && (maxval > minval)) {
        float delta = maxval - minval;
        scaling = delta / 100.0f; // 360 degree = 1%
    }

    // Calculate knob position
    ImVec2 knob_pos = ImVec2(0.0f, -(knob_center_dist));
    float tmp_value = inout_value / scaling;
    float angle = (tmp_value - floor(tmp_value)) * pi * 2.0f;
    float cos_angle = cosf(angle);
    float sin_angle = sinf(angle);
    knob_pos =
        ImVec2((cos_angle * knob_pos.x - sin_angle * knob_pos.y), (sin_angle * knob_pos.x + cos_angle * knob_pos.y));

    ImVec2 knob_button_pos = widget_center + knob_pos - half_rect;
    ImGui::SetCursorScreenPos(knob_button_pos);
    ImGui::InvisibleButton("special_button", rect);

    if (ImGui::IsItemActive()) {
        knob_color = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonActive]);
        ImVec2 p1 = knob_pos;
        float d1 = sqrtf((p1.x * p1.x) + (p1.y * p1.y));
        p1 /= d1;
        ImVec2 p2 = ImGui::GetMousePos() - widget_center;
        float d2 = sqrtf((p2.x * p2.x) + (p2.y * p2.y));
        p2 /= d2;
        float dot = (p1.x * p2.x) + (p1.y * p2.y); // dot product
        float det = (p1.x * p2.y) - (p1.y * p2.x); // determinant
        float angle = atan2(det, dot);
        float b = angle / (2.0f * pi); // b in [0,1] for [0,360] degree
        b *= scaling;
        knob_pos = (p2 * knob_center_dist);
        inout_value = std::min(maxval, (std::max(minval, inout_value + b)));
        retval = true;
    }
    draw_list->AddLine(widget_center, widget_center + knob_pos, knob_line_color, thickness);
    draw_list->AddCircleFilled(widget_center + knob_pos, knob_radius, knob_color, 12);

    ImGui::EndChild();
    ImGui::PopStyleColor();

    ImGui::PopID();

    return retval;
}


bool megamol::gui::ParameterPresentation::ParameterExtendedModeButton(bool& inout_extended_mode) {

    bool retval = false;
    ImGui::BeginGroup();

    megamol::gui::ParameterPresentation::OptionButton("param_mode_button", "Mode");
    if (ImGui::BeginPopupContextItem("param_mode_button_context", 0)) { // 0 = left mouse button
        if (ImGui::MenuItem("Basic###param_mode_button_basic_mode", nullptr, !inout_extended_mode, true)) {
            inout_extended_mode = false;
            retval = true;
        }
        if (ImGui::MenuItem("Expert###param_mode_button_extended_mode", nullptr, inout_extended_mode, true)) {
            inout_extended_mode = true;
            retval = true;
        }
        ImGui::EndPopup();
    }
    ImGui::EndGroup();

    return retval;
}


bool megamol::gui::ParameterPresentation::present_parameter(
    megamol::gui::Parameter& inout_parameter, WidgetScope scope) {

    bool retval = false;
    bool error = true;
    std::string param_label = inout_parameter.GetName();

    // Implementation of presentation with parameter type mapping defined in
    // AbstractParamPresentation::InitPresentation().
    auto visitor = [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;

        // LOCAL -----------------------------------------------------------
        if (scope == WidgetScope::LOCAL) {
            // Set general proportional item width
            float widget_width = ImGui::GetContentRegionAvail().x * 0.65f;
            ImGui::PushItemWidth(widget_width);
            // Set read only
            if (this->IsGUIReadOnly()) {
                GUIUtils::ReadOnlyWigetStyle(true);
            }
        }

        switch (this->GetGUIPresentation()) {
            // BASIC ///////////////////////////////////////////////////
        case (Present_t::Basic): {
            // BOOL ------------------------------------------------
            if constexpr (std::is_same_v<T, bool>) {
                auto value = arg;
                if (this->widget_bool(scope, param_label, value)) {
                    inout_parameter.SetValue(value, false, true);
                    retval = true;
                }
                error = false;
            }
            // FLOAT -----------------------------------------------
            else if constexpr (std::is_same_v<T, float>) {
                auto value = arg;
                if (this->widget_float(scope, param_label, value, inout_parameter.GetMinValue<T>(),
                        inout_parameter.GetMaxValue<T>())) {
                    inout_parameter.SetValue(value, false, true);
                    retval = true;
                }
                error = false;
            } else if constexpr (std::is_same_v<T, int>) {
                switch (inout_parameter.type) {
                    // INT ---------------------------------------------
                case (Param_t::INT): {
                    auto value = arg;
                    if (this->widget_int(scope, param_label, value, inout_parameter.GetMinValue<T>(),
                            inout_parameter.GetMaxValue<T>())) {
                        inout_parameter.SetValue(value, false, true);
                        retval = true;
                    }
                    error = false;
                } break;
                    // ENUM --------------------------------------------
                case (Param_t::ENUM): {
                    auto value = arg;
                    if (this->widget_enum(scope, param_label, value, inout_parameter.GetStorage<EnumStorage_t>())) {
                        inout_parameter.SetValue(value, false, true);
                        retval = true;
                    }
                    error = false;
                } break;
                default:
                    break;
                }
            } else if constexpr (std::is_same_v<T, std::string>) {
                switch (inout_parameter.type) {
                    // STRING ------------------------------------------
                case (Param_t::STRING): {
                    auto value = arg;
                    if (this->widget_string(scope, param_label, value)) {
                        inout_parameter.SetValue(value, false, true);
                        retval = true;
                    }
                    error = false;
                } break;
                    // TRANSFER FUNCTION -------------------------------
                case (Param_t::TRANSFERFUNCTION): {
                    auto value = arg;
                    if (this->widget_string(scope, param_label, value)) {
                        inout_parameter.SetValue(value, false, true);
                        retval = true;
                    }
                    error = false;
                } break;
                    // FILE PATH ---------------------------------------
                case (Param_t::FILEPATH): {
                    auto value = arg;
                    if (this->widget_string(scope, param_label, value)) {
                        inout_parameter.SetValue(value, false, true);
                        retval = true;
                    }
                    error = false;
                } break;
                    // FLEX ENUM ---------------------------------------
                case (Param_t::FLEXENUM): {
                    auto value = arg;
                    if (this->widget_flexenum(scope, param_label, value,
                            inout_parameter.GetStorage<megamol::core::param::FlexEnumParam::Storage_t>())) {
                        inout_parameter.SetValue(value, false, true);
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
                auto value = arg;
                if (this->widget_ternary(scope, param_label, value)) {
                    inout_parameter.SetValue(value, false, true);
                    retval = true;
                }
                error = false;
            }
            // VECTOR 2 --------------------------------------------
            else if constexpr (std::is_same_v<T, glm::vec2>) {
                auto value = arg;
                if (this->widget_vector2f(scope, param_label, value, inout_parameter.GetMinValue<T>(),
                        inout_parameter.GetMaxValue<T>())) {
                    inout_parameter.SetValue(value, false, true);
                    retval = true;
                }
                error = false;
            }
            // VECTOR 3 --------------------------------------------
            else if constexpr (std::is_same_v<T, glm::vec3>) {
                auto value = arg;
                if (this->widget_vector3f(scope, param_label, value, inout_parameter.GetMinValue<T>(),
                        inout_parameter.GetMaxValue<T>())) {
                    inout_parameter.SetValue(value, false, true);
                    retval = true;
                }
                error = false;
            } else if constexpr (std::is_same_v<T, glm::vec4>) {
                switch (inout_parameter.type) {
                    // VECTOR 4 ----------------------------------------
                case (Param_t::VECTOR4F): {
                    auto value = arg;
                    if (this->widget_vector4f(scope, param_label, value, inout_parameter.GetMinValue<T>(),
                            inout_parameter.GetMaxValue<T>())) {
                        inout_parameter.SetValue(value, false, true);
                        retval = true;
                    }
                    error = false;
                } break;
                    // COLOR -------------------------------------------
                case (Param_t::COLOR): {
                    auto value = arg;
                    if (this->widget_vector4f(scope, param_label, value, glm::vec4(0.0f), glm::vec4(1.0f))) {
                        inout_parameter.SetValue(value, false, true);
                        retval = true;
                    }
                    error = false;
                } break;
                default:
                    break;
                }
            } else if constexpr (std::is_same_v<T, std::monostate>) {
                switch (inout_parameter.type) {
                    // BUTTON ------------------------------------------
                case (Param_t::BUTTON): {
                    if (this->widget_button(
                            scope, param_label, inout_parameter.GetStorage<megamol::core::view::KeyCode>())) {
                        inout_parameter.ForceSetValueDirty();
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
            auto value = inout_parameter.GetValueString();
            if (this->widget_string(scope, param_label, value)) {
                inout_parameter.SetValueString(value);
                retval = true;
            }
            error = false;
        } break;
            // COLOR ///////////////////////////////////////////////////
        case (Present_t::Color): {
            if constexpr (std::is_same_v<T, glm::vec4>) {
                switch (inout_parameter.type) {
                    // VECTOR 4 ----------------------------------------
                case (Param_t::VECTOR4F): {
                    auto value = arg;
                    if (this->widget_color(scope, param_label, value)) {
                        inout_parameter.SetValue(value, false, true);
                        retval = true;
                    }
                    error = false;
                } break;
                    // COLOR -------------------------------------------
                case (Param_t::COLOR): {
                    auto value = arg;
                    if (this->widget_color(scope, param_label, value)) {
                        inout_parameter.SetValue(value, false, true);
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
                switch (inout_parameter.type) {
                    // FILE PATH ---------------------------------------
                case (Param_t::FILEPATH): {
                    auto value = arg;
                    if (this->widget_filepath(scope, param_label, value)) {
                        inout_parameter.SetValue(value, false, true);
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
                switch (inout_parameter.type) {
                    // TRANSFER FUNCTION -------------------------------
                case (Param_t::TRANSFERFUNCTION): {
                    auto value = arg;
                    if (this->widget_transfer_function_editor(scope, inout_parameter)) {
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
        case (Present_t::PinValueToMouse): {
            bool compatible_type = false;
            // FLOAT -----------------------------------------------
            if constexpr (std::is_same_v<T, float>) {
                compatible_type = true;
            } else if constexpr (std::is_same_v<T, int>) {
                switch (inout_parameter.type) {
                    // INT ---------------------------------------------
                case (Param_t::INT): {
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
                switch (inout_parameter.type) {
                    // VECTOR 4 ----------------------------------------
                case (Param_t::VECTOR4F): {
                    compatible_type = true;
                } break;
                default:
                    break;
                }
            }
            if (compatible_type) {
                this->widget_pinvaluetomouse(scope, param_label, inout_parameter.GetValueString());
                error = false;
            }
        } break;
            // KNOB //////////////////////////////////////////////////
        case (Present_t::Knob): {
            // FLOAT -----------------------------------------------
            if constexpr (std::is_same_v<T, float>) {
                auto value = arg;
                if (this->widget_knob(scope, param_label, value, inout_parameter.GetMinValue<T>(),
                        inout_parameter.GetMaxValue<T>())) {
                    inout_parameter.SetValue(value, false, true);
                    retval = true;
                }
                error = false;
            }
        } break;
            // 3D ROTATION //////////////////////////////////////////////////
        case (Present_t::Rotation3D_Axes): {
            // FLOAT -----------------------------------------------
            if constexpr (std::is_same_v<T, glm::vec4>) {
                switch (inout_parameter.type) {
                    // VECTOR 4 ----------------------------------------
                case (Param_t::VECTOR4F): {
                    auto value = arg;
                    if (this->widget_rotation_axes(scope, param_label, value, inout_parameter.GetMinValue<T>(),
                            inout_parameter.GetMaxValue<T>())) {
                        inout_parameter.SetValue(value, false, true);
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
        case (Present_t::Rotation3D_Direction): {
            // FLOAT -----------------------------------------------
            if constexpr (std::is_same_v<T, glm::vec3>) {
                switch (inout_parameter.type) {
                    // VECTOR 3 ----------------------------------------
                case (Param_t::VECTOR3F): {
                    auto value = arg;
                    if (this->widget_rotation_direction(scope, param_label, value, inout_parameter.GetMinValue<T>(),
                            inout_parameter.GetMaxValue<T>())) {
                        inout_parameter.SetValue(value, false, true);
                        retval = true;
                    }
                    error = false;
                } break;
                default:
                    break;
                }
            }
        } break;
        default:
            break;
        }

        // LOCAL -----------------------------------------------------------
        if (scope == WidgetScope::LOCAL) {
            // Reset read only
            if (this->IsGUIReadOnly()) {
                GUIUtils::ReadOnlyWigetStyle(false);
            }
            // Reset item width
            ImGui::PopItemWidth();
        }
    };

    std::visit(visitor, inout_parameter.GetValue());

    if (error) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No widget presentation '%s' available for '%s' . [%s, %s, line %d]\n",
            this->GetPresentationName(this->GetGUIPresentation()).c_str(),
            megamol::core::param::AbstractParamPresentation::GetTypeName(inout_parameter.type).c_str(), __FILE__,
            __FUNCTION__, __LINE__);
    }

    return retval;
}


bool megamol::gui::ParameterPresentation::widget_button(megamol::gui::ParameterPresentation::WidgetScope scope,
    const std::string& label, const megamol::core::view::KeyCode& keycode) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == WidgetScope::LOCAL) {
        std::string button_hotkey = keycode.ToString();
        std::string hotkey("");
        std::string edit_label = label;

        bool hotkey_in_tooltip = false;
        bool hotkey_in_label = true;

        // Add hotkey to hover tooltip
        if (hotkey_in_tooltip) {
            if (!button_hotkey.empty()) hotkey = "\n Hotkey: " + button_hotkey;
            this->description += hotkey;
        }
        // Add hotkey to param label
        if (hotkey_in_label) {
            if (!button_hotkey.empty()) hotkey = " [" + button_hotkey + "]";
            edit_label += hotkey;
        }

        retval = ImGui::Button(edit_label.c_str());
    }
    return retval;
}


bool megamol::gui::ParameterPresentation::widget_bool(
    megamol::gui::ParameterPresentation::WidgetScope scope, const std::string& label, bool& value) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == WidgetScope::LOCAL) {
        retval = ImGui::Checkbox(label.c_str(), &value);
    }
    return retval;
}


bool megamol::gui::ParameterPresentation::widget_string(
    megamol::gui::ParameterPresentation::WidgetScope scope, const std::string& label, std::string& value) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == WidgetScope::LOCAL) {
        ImGui::BeginGroup();
        /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
        if (!std::holds_alternative<std::string>(this->widget_store)) {
            std::string utf8Str = value;
            GUIUtils::Utf8Encode(utf8Str);
            this->widget_store = utf8Str;
        }
        std::string hidden_label = "###" + label;

        // Determine multi line count of string
        int multiline_cnt = static_cast<int>(std::count(
            std::get<std::string>(this->widget_store).begin(), std::get<std::string>(this->widget_store).end(), '\n'));
        multiline_cnt = std::min(static_cast<int>(GUI_MAX_MULITLINE), multiline_cnt);
        /// if (multiline_cnt == 0) {
        ///    ImGui::InputText(hidden_label.c_str(), &std::get<std::string>(this->widget_store),
        ///    ImGuiInputTextFlags_CtrlEnterForNewLine);
        ///}
        /// else {
        ImVec2 multiline_size = ImVec2(ImGui::CalcItemWidth(),
            ImGui::GetFrameHeightWithSpacing() + (ImGui::GetFontSize() * static_cast<float>(multiline_cnt)));
        ImGui::InputTextMultiline(hidden_label.c_str(), &std::get<std::string>(this->widget_store), multiline_size,
            ImGuiInputTextFlags_CtrlEnterForNewLine);
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
    }
    return retval;
}


bool megamol::gui::ParameterPresentation::widget_color(
    megamol::gui::ParameterPresentation::WidgetScope scope, const std::string& label, glm::vec4& value) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == WidgetScope::LOCAL) {
        auto color_flags = ImGuiColorEditFlags_AlphaPreview; // | ImGuiColorEditFlags_Float;
        retval = ImGui::ColorEdit4(label.c_str(), glm::value_ptr(value), color_flags);

        this->help = "[Left Click] on the colored square to open a color picker.\n"
                     "[CTRL + Left Click] on individual component to input value.\n"
                     "[Right Click] on the individual color widget to show options.";
    }
    return retval;
}


bool megamol::gui::ParameterPresentation::widget_enum(megamol::gui::ParameterPresentation::WidgetScope scope,
    const std::string& label, int& value, EnumStorage_t storage) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == WidgetScope::LOCAL) {
        /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
        std::string utf8Str = storage[value];
        GUIUtils::Utf8Encode(utf8Str);
        auto combo_flags = ImGuiComboFlags_HeightRegular;
        if (ImGui::BeginCombo(label.c_str(), utf8Str.c_str(), combo_flags)) {
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
    }
    return retval;
}


bool megamol::gui::ParameterPresentation::widget_flexenum(megamol::gui::ParameterPresentation::WidgetScope scope,
    const std::string& label, std::string& value, megamol::core::param::FlexEnumParam::Storage_t storage) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == WidgetScope::LOCAL) {
        /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
        if (!std::holds_alternative<std::string>(this->widget_store)) {
            this->widget_store = std::string();
        }
        std::string utf8Str = value;
        GUIUtils::Utf8Encode(utf8Str);
        auto combo_flags = ImGuiComboFlags_HeightRegular;
        if (ImGui::BeginCombo(label.c_str(), utf8Str.c_str(), combo_flags)) {
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
    }
    return retval;
}


bool megamol::gui::ParameterPresentation::widget_filepath(
    megamol::gui::ParameterPresentation::WidgetScope scope, const std::string& label, std::string& value) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == WidgetScope::LOCAL) {
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
        bool button_edit = this->file_browser.Button(std::get<std::string>(this->widget_store));
        ImGui::SameLine();
        ImGui::InputText(label.c_str(), &std::get<std::string>(this->widget_store), ImGuiInputTextFlags_None);
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
    }
    return retval;
}


bool megamol::gui::ParameterPresentation::widget_ternary(
    megamol::gui::ParameterPresentation::WidgetScope scope, const std::string& label, vislib::math::Ternary& value) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == WidgetScope::LOCAL) {
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
    }
    return retval;
}


bool megamol::gui::ParameterPresentation::widget_int(megamol::gui::ParameterPresentation::WidgetScope scope,
    const std::string& label, int& value, int minval, int maxval) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == WidgetScope::LOCAL) {
        if (!std::holds_alternative<int>(this->widget_store)) {
            this->widget_store = value;
        }
        ImGui::InputInt(label.c_str(), &std::get<int>(this->widget_store), 1, 10, ImGuiInputTextFlags_None);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            this->widget_store = std::max(minval, std::min(std::get<int>(this->widget_store), maxval));
            value = std::get<int>(this->widget_store);
            retval = true;
        } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
            this->widget_store = value;
        }
    }
    return retval;
}


bool megamol::gui::ParameterPresentation::widget_float(megamol::gui::ParameterPresentation::WidgetScope scope,
    const std::string& label, float& value, float minval, float maxval) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == WidgetScope::LOCAL) {
        if (!std::holds_alternative<float>(this->widget_store)) {
            this->widget_store = value;
        }
        ImGui::InputFloat(label.c_str(), &std::get<float>(this->widget_store), 1.0f, 10.0f, this->float_format.c_str(),
            ImGuiInputTextFlags_None);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            this->widget_store = std::max(minval, std::min(std::get<float>(this->widget_store), maxval));
            value = std::get<float>(this->widget_store);
            retval = true;
        } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
            this->widget_store = value;
        }
    }
    return retval;
}


bool megamol::gui::ParameterPresentation::widget_vector2f(megamol::gui::ParameterPresentation::WidgetScope scope,
    const std::string& label, glm::vec2& value, glm::vec2 minval, glm::vec2 maxval) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == WidgetScope::LOCAL) {
        if (!std::holds_alternative<glm::vec2>(this->widget_store)) {
            this->widget_store = value;
        }
        ImGui::InputFloat2(label.c_str(), glm::value_ptr(std::get<glm::vec2>(this->widget_store)),
            this->float_format.c_str(), ImGuiInputTextFlags_None);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            auto x = std::max(minval.x, std::min(std::get<glm::vec2>(this->widget_store).x, maxval.x));
            auto y = std::max(minval.y, std::min(std::get<glm::vec2>(this->widget_store).y, maxval.y));
            this->widget_store = glm::vec2(x, y);
            value = std::get<glm::vec2>(this->widget_store);
            retval = true;
        } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
            this->widget_store = value;
        }
    }
    return retval;
}


bool megamol::gui::ParameterPresentation::widget_vector3f(megamol::gui::ParameterPresentation::WidgetScope scope,
    const std::string& label, glm::vec3& value, glm::vec3 minval, glm::vec3 maxval) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == WidgetScope::LOCAL) {
        if (!std::holds_alternative<glm::vec3>(this->widget_store)) {
            this->widget_store = value;
        }
        ImGui::InputFloat3(label.c_str(), glm::value_ptr(std::get<glm::vec3>(this->widget_store)),
            this->float_format.c_str(), ImGuiInputTextFlags_None);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            auto x = std::max(minval.x, std::min(std::get<glm::vec3>(this->widget_store).x, maxval.x));
            auto y = std::max(minval.y, std::min(std::get<glm::vec3>(this->widget_store).y, maxval.y));
            auto z = std::max(minval.z, std::min(std::get<glm::vec3>(this->widget_store).z, maxval.z));
            this->widget_store = glm::vec3(x, y, z);
            value = std::get<glm::vec3>(this->widget_store);
            retval = true;
        } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
            this->widget_store = value;
        }
    }
    return retval;
}


bool megamol::gui::ParameterPresentation::widget_vector4f(megamol::gui::ParameterPresentation::WidgetScope scope,
    const std::string& label, glm::vec4& value, glm::vec4 minval, glm::vec4 maxval) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == WidgetScope::LOCAL) {
        if (!std::holds_alternative<glm::vec4>(this->widget_store)) {
            this->widget_store = value;
        }
        ImGui::InputFloat4(label.c_str(), glm::value_ptr(std::get<glm::vec4>(this->widget_store)),
            this->float_format.c_str(), ImGuiInputTextFlags_None);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            auto x = std::max(minval.x, std::min(std::get<glm::vec4>(this->widget_store).x, maxval.x));
            auto y = std::max(minval.y, std::min(std::get<glm::vec4>(this->widget_store).y, maxval.y));
            auto z = std::max(minval.z, std::min(std::get<glm::vec4>(this->widget_store).z, maxval.z));
            auto w = std::max(minval.w, std::min(std::get<glm::vec4>(this->widget_store).w, maxval.w));
            this->widget_store = glm::vec4(x, y, z, w);
            value = std::get<glm::vec4>(this->widget_store);
            retval = true;
        } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
            this->widget_store = value;
        }
    }
    return retval;
}


bool megamol::gui::ParameterPresentation::widget_pinvaluetomouse(
    megamol::gui::ParameterPresentation::WidgetScope scope, const std::string& label, const std::string& value) {
    bool retval = false;

    // LOCAL -----------------------------------------------------------
    if (scope == ParameterPresentation::WidgetScope::LOCAL) {

        ImGui::TextDisabled(label.c_str());
    }
    // GLOBAL -----------------------------------------------------------
    else if (scope == ParameterPresentation::WidgetScope::GLOBAL) {

        auto hoverFlags = ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_AllowWhenDisabled |
                          ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem;
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


bool megamol::gui::ParameterPresentation::widget_transfer_function_editor(
    WidgetScope scope, megamol::gui::Parameter& inout_parameter) {

    bool retval = false;
    bool isActive = false;
    bool updateEditor = false;
    auto value = std::get<std::string>(inout_parameter.GetValue());
    std::string label = inout_parameter.GetName();

    ImGuiStyle& style = ImGui::GetStyle();

    if (this->use_external_tf_editor) {
        if (this->tf_editor_external_ptr == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Pointer to external transfer function editor is nullptr. [%s, %s, line %d]\n", __FILE__,
                __FUNCTION__, __LINE__);
            return false;
        }
        isActive = !(this->tf_editor_external_ptr->GetConnectedParameterName().empty());
    }

    // LOCAL -----------------------------------------------------------
    if (scope == WidgetScope::LOCAL) {
        ImGui::BeginGroup();

        if (this->use_external_tf_editor) {

            // Reduced display of value and editor state.
            if (value.empty()) {
                ImGui::TextDisabled("{    (empty)    }");
                ImGui::SameLine();
            } else {
                // Draw texture
                if (this->image_widget.IsLoaded()) {
                    this->image_widget.Widget(ImVec2(ImGui::CalcItemWidth(), ImGui::GetFrameHeight()));
                    ImGui::SameLine(ImGui::CalcItemWidth() + style.ItemInnerSpacing.x);
                } else {
                    ImGui::TextUnformatted("{ ............. }");
                    ImGui::SameLine();
                }
            }

            // Label
            ImGui::AlignTextToFramePadding();
            ImGui::TextEx(label.c_str(), ImGui::FindRenderedTextEnd(label.c_str()));
        }

        // Toggle internal and external editor, if available
        if (this->tf_editor_external_ptr != nullptr) {
            if (ImGui::RadioButton("External", this->use_external_tf_editor)) {
                this->use_external_tf_editor = true;
                /// If not this->tf_editor_external_ptr->SetConnectedParameter(&param, full_param_name); additional
                /// click on edit is required.
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Internal", !this->use_external_tf_editor)) {
                this->use_external_tf_editor = false;
                this->tf_editor_external_ptr->SetConnectedParameter(nullptr, "");
            }
            ImGui::SameLine();
        }

        if (this->use_external_tf_editor) {

            // Editor
            ImGui::PushID("Edit_");
            ImGui::PushStyleColor(ImGuiCol_Button, style.Colors[isActive ? ImGuiCol_ButtonHovered : ImGuiCol_Button]);
            ImGui::PushStyleColor(
                ImGuiCol_ButtonHovered, style.Colors[isActive ? ImGuiCol_Button : ImGuiCol_ButtonHovered]);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, style.Colors[ImGuiCol_ButtonActive]);
            if (ImGui::Button("Edit")) {
                retval = true;
            }
            ImGui::PopStyleColor(3);
            ImGui::PopID();

        } else { // Internal Editor

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
        }

        // Copy
        if (ImGui::Button("Copy")) {
#ifdef GUI_USE_GLFW
            auto glfw_win = ::glfwGetCurrentContext();
            ::glfwSetClipboardString(glfw_win, value.c_str());
#elif _WIN32
            ImGui::SetClipboardText(value.c_str());
#else // LINUX
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] No clipboard use provided. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[GUI] Transfer Function JSON String:\n%s", value.c_str());
#endif
        }
        ImGui::SameLine();

        // Paste
        if (ImGui::Button("Paste")) {
#ifdef GUI_USE_GLFW
            auto glfw_win = ::glfwGetCurrentContext();
            inout_parameter.SetValue(std::string(::glfwGetClipboardString(glfw_win)), false, true);
#elif _WIN32
            inout_parameter.SetValue(std::string(ImGui::GetClipboardText()));
#else // LINUX
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] No clipboard use provided. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
#endif
            value = std::get<std::string>(inout_parameter.GetValue());
            if (this->use_external_tf_editor) {
                if (this->tf_editor_external_ptr != nullptr) {
                    this->tf_editor_external_ptr->SetTransferFunction(value, true);
                }
            } else {
                this->tf_editor_internal.SetTransferFunction(value, false);
            }
        }

        if (!this->use_external_tf_editor) { // Internal Editor

            if (this->tf_editor_hash != inout_parameter.GetTransferFunctionHash()) {
                updateEditor = true;
            }
            // Propagate the transfer function to the editor.
            if (updateEditor) {
                this->tf_editor_internal.SetTransferFunction(value, false);
            }
            // Draw transfer function editor
            if (this->show_tf_editor) {
                if (this->tf_editor_internal.Widget(false)) {
                    std::string value;
                    if (this->tf_editor_internal.GetTransferFunction(value)) {
                        inout_parameter.SetValue(value, false, true);
                        retval = false; /// (Returning true opens external editor)
                    }
                }
            }

            this->tf_editor_hash = inout_parameter.GetTransferFunctionHash();
        }

        /// ImGui::Separator();
        ImGui::EndGroup();
    }
    // GLOBAL -----------------------------------------------------------
    else if (scope == ParameterPresentation::WidgetScope::GLOBAL) {
        if (this->use_external_tf_editor) {

            // Check for changed parameter value which should be forced to the editor once.
            if (isActive) {
                if (this->tf_editor_hash != inout_parameter.GetTransferFunctionHash()) {
                    updateEditor = true;
                }
            }
            // Propagate the transfer function to the editor.
            if (isActive && updateEditor) {
                this->tf_editor_external_ptr->SetTransferFunction(value, true);
                retval = true;
            }
            this->tf_editor_hash = inout_parameter.GetTransferFunctionHash();
        }
    }

    return retval;
}


bool megamol::gui::ParameterPresentation::widget_knob(
    WidgetScope scope, const std::string& label, float& value, float minval, float maxval) {
    bool retval = false;

    ImGuiStyle& style = ImGui::GetStyle();

    // LOCAL -----------------------------------------------------------
    if (scope == ParameterPresentation::WidgetScope::LOCAL) {

        // Draw knob
        const float knob_size =
            (1.0f * ImGui::GetTextLineHeightWithSpacing()) + (1.0f * ImGui::GetFrameHeightWithSpacing());
        if (ParameterPresentation::KnobButton("param_knob", knob_size, value, minval, maxval)) {
            retval = true;
        }

        ImGui::SameLine();

        // Draw Value
        std::string value_label;
        float left_widget_x_offset = knob_size + style.ItemInnerSpacing.x;
        ImVec2 pos = ImGui::GetCursorPos();
        ImGui::PushItemWidth(ImGui::CalcItemWidth() - left_widget_x_offset);

        if (this->widget_float(scope, label, value, minval, maxval)) {
            retval = true;
        }
        ImGui::PopItemWidth();

        // Draw min max
        ImGui::SetCursorPos(pos + ImVec2(0.0f, ImGui::GetFrameHeightWithSpacing()));
        if (minval > -FLT_MAX) {
            value_label = "Min: " + this->float_format;
            ImGui::Text(value_label.c_str(), minval);
        } else {
            ImGui::TextUnformatted("Min: -inf");
        }
        ImGui::SameLine();
        if (maxval < FLT_MAX) {
            value_label = "Max: " + this->float_format;
            ImGui::Text(value_label.c_str(), maxval);
        } else {
            ImGui::TextUnformatted("Max: inf");
        }
    }
    // GLOBAL -----------------------------------------------------------
    else if (scope == ParameterPresentation::WidgetScope::GLOBAL) {

        // no global implementation ...
    }

    return retval;
}


bool megamol::gui::ParameterPresentation::widget_rotation_axes(
    WidgetScope scope, const std::string& label, glm::vec4& value, glm::vec4 minval, glm::vec4 maxval) {

    bool retval = false;
    // LOCAL -----------------------------------------------------------
    if (scope == ParameterPresentation::WidgetScope::LOCAL) {

        auto x_cursor_pos = ImGui::GetCursorPosX();
        retval = this->widget_vector4f(scope, label, value, minval, maxval);
        ImGui::SetCursorPosX(x_cursor_pos);
        retval |= this->rotation_widget.gizmo3D_rotation_axes(value);
        // ImGui::SameLine();
        // ImGui::TextUnformatted(label.c_str());
    }
    // GLOBAL -----------------------------------------------------------
    else if (scope == ParameterPresentation::WidgetScope::GLOBAL) {

        // no global implementation ...
    }

    return retval;
}


bool megamol::gui::ParameterPresentation::widget_rotation_direction(
    WidgetScope scope, const std::string& label, glm::vec3& value, glm::vec3 minval, glm::vec3 maxval) {

    bool retval = false;
    // LOCAL -----------------------------------------------------------
    if (scope == ParameterPresentation::WidgetScope::LOCAL) {

        auto x_cursor_pos = ImGui::GetCursorPosX();
        retval = this->widget_vector3f(scope, label, value, minval, maxval);
        ImGui::SetCursorPosX(x_cursor_pos);
        retval |= this->rotation_widget.gizmo3D_rotation_direction(value);
        // ImGui::SameLine();
        // ImGui::TextUnformatted(label.c_str());

    }
    // GLOBAL -----------------------------------------------------------
    else if (scope == ParameterPresentation::WidgetScope::GLOBAL) {

        // no global implementation ...
    }

    return retval;
}
