/*
 * ParameterGroupsPresentation.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ParameterGroupsPresentation.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::gui;


megamol::gui::ParameterGroupsPresentation::ParameterGroupsPresentation(void)
    : group_widget_ids()
    , tooltip()
    , speed_knob_pos(ImVec2(0.0f, 0.0f))
    , time_knob_pos(ImVec2(0.0f, 0.0f))
    , image_buttons() {

    // Add group widget data for animation widget group
    /// View3D_2::anim
    GroupWidgetData animation;
    animation.active = false;
    animation.type.emplace(Param_t::BOOL, 1);
    animation.type.emplace(Param_t::BUTTON, 3);
    animation.type.emplace(Param_t::FLOAT, 2);
    animation.callback = [&, this](ParamPtrVector_t& params,
                             megamol::core::param::AbstractParamPresentation::Presentation presentation,
                             megamol::gui::ParameterPresentation::WidgetScope in_scope) -> bool {
        return this->group_widget_animation(params, presentation, in_scope);
    };
    group_widget_ids["anim"] = animation;
}


megamol::gui::ParameterGroupsPresentation::~ParameterGroupsPresentation(void) {}


bool megamol::gui::ParameterGroupsPresentation::PresentGUI(megamol::gui::ParamVector_t& inout_params,
    const std::string& in_module_fullname, const std::string& in_search, vislib::math::Ternary in_extended,
    bool in_indent, megamol::gui::ParameterPresentation::WidgetScope in_scope,
    const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor) {

    if (out_open_external_tf_editor != nullptr) (*out_open_external_tf_editor) = false;

    // Nothing to do if there are no parameters
    if (inout_params.empty()) return true;

    if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
        /// LOCAL

        ImGui::BeginGroup();
        if (in_indent) ImGui::Indent();
    }

    // Analyse parameter group membership and draw ungrouped parameters
    ParamGroup_t group_map;
    for (auto& param : inout_params) {
        auto param_namespace = param.GetNameSpace();
        if (!in_extended.IsUnknown()) {
            param.present.extended = in_extended.IsTrue();
        }

        if (!param_namespace.empty()) {
            // Sort parameters with namespace to group
            group_map[param_namespace].first.emplace_back(&param);
            group_map[param_namespace].second[param.type]++;
        } else {
            // Draw parameters without namespace directly at the beginning
            this->draw_parameter(
                param, in_module_fullname, in_search, in_scope, in_external_tf_editor, out_open_external_tf_editor);
        }
    }

    // Draw grouped parameters
    for (auto& group : group_map) {
        auto group_name = group.first;
        bool found_group_widget = false;

        // Draw group widget (if defined) ...
        for (auto& group_widget_id : this->group_widget_ids) {
            // Check for same group name and count of different parameter types
            /// TODO Is this check too expensive? (remove check for group_name)
            if ((group_widget_id.second.type == group.second.second) && (group_widget_id.first == group_name)) {

                found_group_widget = true;
                group_widget_id.second.active = true;

                ImGui::PushID(group_widget_id.first.c_str());

                if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {

                    if (in_extended.IsTrue()) {
                        // Visibility
                        bool visible = group_widget_id.second.IsGUIVisible();
                        if (ImGui::RadioButton("###visibile", visible)) {
                            group_widget_id.second.SetGUIVisible(!visible);
                        }
                        this->tooltip.ToolTip("Visibility", ImGui::GetItemID(), 0.5f);
                        ImGui::SameLine();

                        // Read-only option
                        bool readonly = group_widget_id.second.IsGUIReadOnly();
                        if (ImGui::Checkbox("###readonly", &readonly)) {
                            group_widget_id.second.SetGUIReadOnly(readonly);
                        }
                        this->tooltip.ToolTip("Read-Only", ImGui::GetItemID(), 0.5f);
                        ImGui::SameLine();

                        ParameterPresentation::OptionButton(
                            "param_groups", "", (group_widget_id.second.GetGUIPresentation() != Present_t::Basic));
                        if (ImGui::BeginPopupContextItem("param_present_button_context", 0)) {
                            for (auto& present_name_pair : group_widget_id.second.GetPresentationNameMap()) {
                                if (group_widget_id.second.IsPresentationCompatible(present_name_pair.first)) {
                                    if (ImGui::MenuItem(present_name_pair.second.c_str(), nullptr,
                                            (present_name_pair.first == group_widget_id.second.GetGUIPresentation()))) {
                                        group_widget_id.second.SetGUIPresentation(present_name_pair.first);
                                    }
                                }
                            }
                            ImGui::EndPopup();
                        }
                        this->tooltip.ToolTip("Presentation", ImGui::GetItemID(), 0.5f);
                        ImGui::SameLine();
                    }

                    // Call group widget draw function
                    if (group_widget_id.second.IsGUIVisible() || in_extended.IsTrue()) {

                        if (group_widget_id.second.IsGUIReadOnly()) {
                            GUIUtils::ReadOnlyWigetStyle(true);
                        }

                        if (group_widget_id.second.GetGUIPresentation() ==
                            param::AbstractParamPresentation::Presentation::Basic) {

                            this->draw_grouped_parameters(group_name, group.second.first, in_module_fullname, in_search,
                                in_scope, in_external_tf_editor, out_open_external_tf_editor);
                        } else {
                            if (!group_widget_id.second.callback(
                                    group.second.first, group_widget_id.second.GetGUIPresentation(), in_scope)) {

                                megamol::core::utility::log::Log::DefaultLog.WriteError(
                                    "[GUI] No widget presentation '%s' available for group widget '%s'. [%s, %s, line "
                                    "%d]\n",
                                    group_widget_id.second
                                        .GetPresentationName(group_widget_id.second.GetGUIPresentation())
                                        .c_str(),
                                    group_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                            }
                        }

                        if (group_widget_id.second.IsGUIReadOnly()) {
                            GUIUtils::ReadOnlyWigetStyle(false);
                        }
                    }
                } else {
                    // GLOBAL

                    group_widget_id.second.callback(
                        group.second.first, group_widget_id.second.GetGUIPresentation(), in_scope);
                }

                ImGui::PopID();
            }
        }

        // ... else draw grouped parameters with no custom group widget using namespace header.
        if (!found_group_widget) {

            if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
                /// LOCAL

                this->draw_grouped_parameters(group_name, group.second.first, in_module_fullname, in_search, in_scope,
                    in_external_tf_editor, out_open_external_tf_editor);
            } else {
                /// GLOBAL

                for (auto& param : group.second.first) {
                    this->draw_parameter((*param), in_module_fullname, in_search, in_scope, in_external_tf_editor,
                        out_open_external_tf_editor);
                }
            }
        }
    }

    if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
        /// LOCAL

        if (in_indent) ImGui::Unindent();
        ImGui::EndGroup();
    }

    return true;
}


bool megamol::gui::ParameterGroupsPresentation::ParameterGroupGUIStateToJSON(
    nlohmann::json& inout_json, const std::string& module_fullname) {

    for (auto& group_widget_id : group_widget_ids) {
        if (group_widget_id.second.active) {
            std::string param_fullname = module_fullname + "::parametergroup::" + group_widget_id.first;

            group_widget_id.second.ParameterGUIStateToJSON(inout_json, param_fullname);
        }
    }

    return false;
}


bool megamol::gui::ParameterGroupsPresentation::ParameterGroupGUIStateFromJSONString(
    const std::string& in_json_string, const std::string& module_fullname) {

    for (auto& group_widget_id : group_widget_ids) {
        std::string param_fullname = module_fullname + "::parametergroup::" + group_widget_id.first;

        if (group_widget_id.second.ParameterGUIStateFromJSONString(in_json_string, param_fullname)) {
            group_widget_id.second.active = true;
        }
    }

    return false;
}


void megamol::gui::ParameterGroupsPresentation::draw_parameter(megamol::gui::Parameter& inout_param,
    const std::string& in_module_fullname, const std::string& in_search,
    megamol::gui::ParameterPresentation::WidgetScope in_scope,
    const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor) {

    if ((inout_param.type == Param_t::TRANSFERFUNCTION) && (in_external_tf_editor != nullptr)) {
        inout_param.present.ConnectExternalTransferFunctionEditor(in_external_tf_editor);
    }

    if (in_scope == ParameterPresentation::WidgetScope::GLOBAL) {
        /// GLOBAL

        inout_param.PresentGUI(in_scope);
    } else {
        /// LOCAL

        auto param_name = inout_param.full_name;
        bool param_searched = true;
        if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
            param_searched = megamol::gui::StringSearchWidget::FindCaseInsensitiveSubstring(param_name, in_search);
        }
        bool visible = (inout_param.present.IsGUIVisible() || inout_param.present.extended) && param_searched;

        if (visible) {
            if (inout_param.PresentGUI(in_scope)) {

                // Open window calling the transfer function editor callback
                if ((inout_param.type == Param_t::TRANSFERFUNCTION) && (in_external_tf_editor != nullptr)) {
                    if (out_open_external_tf_editor != nullptr) (*out_open_external_tf_editor) = true;
                    auto param_fullname = std::string(in_module_fullname.c_str()) + "::" + inout_param.full_name;
                    in_external_tf_editor->SetConnectedParameter(&inout_param, param_fullname);
                }
            }
        }
    }
}


void megamol::gui::ParameterGroupsPresentation::draw_grouped_parameters(const std::string& in_group_name,
    ParamPtrVector_t& params, const std::string& in_module_fullname, const std::string& in_search,
    megamol::gui::ParameterPresentation::WidgetScope in_scope,
    const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor) {

    // Skip if no parameter is visible and extended mode is not set.
    bool visible = false;
    bool extended = false;
    for (auto& param : params) {
        visible = visible || param->present.IsGUIVisible();
        extended = extended || param->present.extended;
    }
    if (!visible && !extended) return;

    // Open namespace header when parameter search is active.
    if (!in_search.empty()) {
        auto headerId = ImGui::GetID(in_group_name.c_str());
        ImGui::GetStateStorage()->SetInt(headerId, 1);
    }
    if (ImGui::CollapsingHeader(in_group_name.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();
        for (auto& param : params) {
            this->draw_parameter(
                (*param), in_module_fullname, in_search, in_scope, in_external_tf_editor, out_open_external_tf_editor);
        }
        ImGui::Unindent();
    }
}


bool megamol::gui::ParameterGroupsPresentation::group_widget_animation(ParamPtrVector_t& params,
    megamol::core::param::AbstractParamPresentation::Presentation presentation,
    megamol::gui::ParameterPresentation::WidgetScope in_scope) {

    if (presentation != param::AbstractParamPresentation::Presentation::Group_Animation) return false;

    ImGuiStyle& style = ImGui::GetStyle();
    const std::string group_label("Animation");
    const ImVec2 button_size =
        ImVec2(1.5f * ImGui::GetFrameHeightWithSpacing(), 1.5f * ImGui::GetFrameHeightWithSpacing());
    const float knob_size = 2.5f * ImGui::GetFrameHeightWithSpacing();

    if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
        // LOCAL

        ImGui::TextDisabled(group_label.c_str());
        return true;
    }

    // Check required parameters
    /// Find specific parameters of group by name because of parameter type can occure multiple times.
    Parameter* param_play = nullptr;
    Parameter* param_time = nullptr;
    Parameter* param_speed = nullptr;
    for (auto& param_ptr : params) {
        if ((param_ptr->GetName() == "play") && (param_ptr->type == Param_t::BOOL)) {
            param_play = param_ptr;
        }
        if ((param_ptr->GetName() == "time") && (param_ptr->type == Param_t::FLOAT)) {
            param_time = param_ptr;
        }
        if ((param_ptr->GetName() == "speed") && (param_ptr->type == Param_t::FLOAT)) {
            param_speed = param_ptr;
        }
    }
    if ((param_play == nullptr) || (param_time == nullptr) || (param_speed == nullptr)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to find all required parameters by name for animation group widget. [%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Load button textures (once)
    if (!this->image_buttons.play.IsLoaded()) {
        this->image_buttons.play.LoadTextureFromFile("../share/resources/transport_ctrl_play.png");
    }
    if (!this->image_buttons.pause.IsLoaded()) {
        this->image_buttons.pause.LoadTextureFromFile("../share/resources/transport_ctrl_pause.png");
    }
    if (!this->image_buttons.fastforward.IsLoaded()) {
        this->image_buttons.fastforward.LoadTextureFromFile("../share/resources/transport_ctrl_fast-forward.png");
    }
    if (!this->image_buttons.fastrewind.IsLoaded()) {
        this->image_buttons.fastrewind.LoadTextureFromFile("../share/resources/transport_ctrl_fast-rewind.png");
    }
    if ((!this->image_buttons.play.IsLoaded()) || (!this->image_buttons.pause.IsLoaded()) ||
        (!this->image_buttons.fastforward.IsLoaded()) || (!this->image_buttons.fastrewind.IsLoaded())) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to load all required button textures for animation group widget. [%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // ------------------------------------------------------------------------

    if (in_scope == ParameterPresentation::WidgetScope::GLOBAL) {
        // GLOBAL

        ImGui::Begin(group_label.c_str(), nullptr,
            ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoCollapse);

    } else { // if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
        /// LOCAL
        /*
        float child_height = frame_height * 4.5f;
        auto child_flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration;
        ImGui::BeginChild("group_widget_animation", ImVec2(0.0f, child_height), true, child_flags);

        // Caption
        ImGui::TextUnformatted(group_label.c_str());
        */
    }

    // Transport Buttons ------------------------------------------------------
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, style.Colors[ImGuiCol_ButtonActive]);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, style.Colors[ImGuiCol_ButtonHovered]);

    bool play = std::get<bool>(param_play->GetValue());
    float time = std::get<float>(param_time->GetValue());
    float speed = std::get<float>(param_speed->GetValue());
    std::string button_label;

    /// PLAY - PAUSE
    if (!play) {
        if (this->image_buttons.play.Button("Play", button_size)) {
            play = !play;
        }
    } else {
        if (this->image_buttons.pause.Button("Pause", button_size)) {
            play = !play;
        }
    }
    ImGui::SameLine();

    /// SLOWER
    if (this->image_buttons.fastrewind.Button("Slower", button_size)) {
        // play = true;
        speed /= 1.5f;
    }
    this->tooltip.ToolTip(button_label, ImGui::GetItemID(), 1.0f, 5.0f);
    ImGui::SameLine();

    /// FASTER
    if (this->image_buttons.fastforward.Button("Faster", button_size)) {
        // play = true;
        speed *= 1.5f;
    }

    ImGui::PopStyleColor(2);

    // ImGui::SameLine();
    ImVec2 cursor_pos = ImGui::GetCursorPos();

    // Time -------------------------------------------------------------------
    ImGui::BeginGroup();
    std::string label("time");
    float font_size = ImGui::CalcTextSize(label.c_str()).x;
    ImGui::SetCursorPosX(cursor_pos.x + (knob_size - font_size) / 2.0f);
    ImGui::TextUnformatted(label.c_str());
    ParameterPresentation::KnobButton(
        label, knob_size, time, param_time->GetMinValue<float>(), param_time->GetMaxValue<float>());
    ImGui::Text(param_time->present.float_format.c_str(), time);
    ImGui::EndGroup();
    ImGui::SameLine();

    // Speed -------------------------------------------------------------------
    ImGui::BeginGroup();
    label = "speed";
    font_size = ImGui::CalcTextSize(label.c_str()).x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (knob_size - font_size) / 2.0f);
    ImGui::TextUnformatted(label.c_str());
    ParameterPresentation::KnobButton(
        label, knob_size, speed, param_speed->GetMinValue<float>(), param_speed->GetMaxValue<float>());
    ImGui::Text(param_speed->present.float_format.c_str(), speed);
    ImGui::EndGroup();

    // ------------------------------------------------------------------------

    param_play->SetValue(play);
    param_time->SetValue(time);
    param_speed->SetValue(speed);

    if (in_scope == ParameterPresentation::WidgetScope::GLOBAL) {
        /// GLOBAL

        ImGui::End();
    } else { // if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
        /// LOCAL
        /*
        ImGui::EndChild();
        */
    }

    return true;
}
