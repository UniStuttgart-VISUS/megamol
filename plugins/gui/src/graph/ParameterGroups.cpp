/*
 * ParameterGroups.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ParameterGroups.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::gui;


megamol::gui::ParameterGroups::ParameterGroups(void) : utils(), group_widget_ids(), button_tex_ids{0, 0, 0, 0} {

    // Add group widget data for animation
    GroupWidgetData animation;
    animation.active = false;
    animation.type.emplace(ParamType::BOOL, 1);
    animation.type.emplace(ParamType::BUTTON, 3);
    animation.type.emplace(ParamType::FLOAT, 2);
    animation.callback = [&, this](ParamPtrVectorType& params,
                             megamol::core::param::AbstractParamPresentation::Presentation presentation,
                             megamol::gui::ParameterPresentation::WidgetScope in_scope) -> bool {
        return this->group_widget_animation(params, presentation, in_scope);
    };
    group_widget_ids["anim"] = animation;
}


megamol::gui::ParameterGroups::~ParameterGroups(void) {}


bool megamol::gui::ParameterGroups::PresentGUI(megamol::gui::ParamVectorType& inout_params,
    const std::string& in_module_fullname, const std::string& in_search, bool in_extended, bool in_ignore_extended,
    megamol::gui::ParameterPresentation::WidgetScope in_scope,
    const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool& out_open_external_tf_editor) {

    out_open_external_tf_editor = false;

    // Nothing to do if there are no parameters
    if (inout_params.empty()) return true;

    if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
        /// LOCAL

        ImGui::BeginGroup();
        ImGui::Indent();
    }

    // Analyse parameter group membership and draw ungrouped parameters
    ParamGroupType group_map;
    for (auto& param : inout_params) {
        auto param_namespace = param.GetNameSpace();
        if (!in_ignore_extended) {
            param.present.extended = in_extended;
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
            /// TODO Is this check too expensive (remove check for group_name?) ...
            if ((group_widget_id.second.type == group.second.second) && (group_widget_id.first == group_name)) {

                found_group_widget = true;
                group_widget_id.second.active = true;

                ImGui::PushID(group_widget_id.first.c_str());

                if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {

                    if (in_extended) {
                        // Visibility
                        bool visible = group_widget_id.second.IsGUIVisible();
                        if (ImGui::RadioButton("###visibile", visible)) {
                            group_widget_id.second.SetGUIVisible(!visible);
                        }
                        this->utils.HoverToolTip("Visibility", ImGui::GetItemID(), 0.5f);
                        ImGui::SameLine();

                        // Read-only option
                        bool readonly = group_widget_id.second.IsGUIReadOnly();
                        if (ImGui::Checkbox("###readonly", &readonly)) {
                            group_widget_id.second.SetGUIReadOnly(readonly);
                        }
                        this->utils.HoverToolTip("Read-Only", ImGui::GetItemID(), 0.5f);
                        ImGui::SameLine();

                        ParameterPresentation::PointCircleButton(
                            "", (group_widget_id.second.GetGUIPresentation() != PresentType::Basic));
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
                        this->utils.HoverToolTip("Presentation", ImGui::GetItemID(), 0.5f);
                        ImGui::SameLine();
                    }

                    // Call group widget draw function
                    if (group_widget_id.second.IsGUIVisible() || in_extended) {

                        if (group_widget_id.second.IsGUIReadOnly()) {
                            GUIUtils::ReadOnlyWigetStyle(true);
                        }

                        if (group_widget_id.second.GetGUIPresentation() ==
                            param::AbstractParamPresentation::Presentation::Basic) {
                            found_group_widget = true;
                            this->draw_grouped_parameters(group_name, group.second.first, in_module_fullname, in_search,
                                in_scope, in_external_tf_editor, out_open_external_tf_editor);
                        } else {
                            if (!group_widget_id.second.callback(
                                    group.second.first, group_widget_id.second.GetGUIPresentation(), in_scope)) {
                                vislib::sys::Log::DefaultLog.WriteError(
                                    "No widget presentation '%s' available for group widget '%s'. [%s, %s, line %d]\n",
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

        ImGui::Unindent();
        ImGui::EndGroup();
    }

    return true;
}


bool megamol::gui::ParameterGroups::ParameterGroupGUIStateToJSON(
    nlohmann::json& inout_json, const std::string& module_fullname) {

    for (auto& group_widget_id : group_widget_ids) {
        if (group_widget_id.second.active) {
            std::string param_fullname = module_fullname + "::" + "PARAMGROUP_" + group_widget_id.first;

            group_widget_id.second.ParameterGUIStateToJSON(inout_json, param_fullname);
        }
    }

    return false;
}


bool megamol::gui::ParameterGroups::ParameterGroupGUIStateFromJSONString(
    const std::string& in_json_string, const std::string& module_fullname) {

    for (auto& group_widget_id : group_widget_ids) {
        std::string param_fullname = module_fullname + "::" + "PARAMGROUP_" + group_widget_id.first;

        if (group_widget_id.second.ParameterGUIStateFromJSONString(in_json_string, param_fullname)) {
            group_widget_id.second.active = true;
        }
    }

    return false;
}


void megamol::gui::ParameterGroups::draw_parameter(megamol::gui::Parameter& inout_param,
    const std::string& in_module_fullname, const std::string& in_search,
    megamol::gui::ParameterPresentation::WidgetScope in_scope,
    const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool& out_open_external_tf_editor) {

    if ((inout_param.type == ParamType::TRANSFERFUNCTION) && (in_external_tf_editor != nullptr)) {
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
            param_searched = megamol::gui::GUIUtils::FindCaseInsensitiveSubstring(param_name, in_search);
        }
        bool visible = (inout_param.present.IsGUIVisible() && param_searched) || inout_param.present.extended;

        if (visible) {
            if (inout_param.PresentGUI(in_scope)) {

                // Open window calling the transfer function editor callback
                if ((inout_param.type == ParamType::TRANSFERFUNCTION) && (in_external_tf_editor != nullptr)) {
                    out_open_external_tf_editor = true;
                    auto param_fullname = std::string(in_module_fullname.c_str()) + "::" + inout_param.full_name;
                    in_external_tf_editor->SetConnectedParameter(&inout_param, param_fullname);
                }
            }
        }
    }
}


void megamol::gui::ParameterGroups::draw_grouped_parameters(const std::string& in_group_name,
    ParamPtrVectorType& params, const std::string& in_module_fullname, const std::string& in_search,
    megamol::gui::ParameterPresentation::WidgetScope in_scope,
    const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool& out_open_external_tf_editor) {

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


bool megamol::gui::ParameterGroups::group_widget_animation(ParamPtrVectorType& params,
    megamol::core::param::AbstractParamPresentation::Presentation presentation,
    megamol::gui::ParameterPresentation::WidgetScope in_scope) {

    if (presentation != param::AbstractParamPresentation::Presentation::Group_Animation) return false;

    if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
        /// LOCAL

        // Check required parameters
        /// TODO Get specific parameters by name because of multiple same types
        Parameter* param_play = nullptr;
        Parameter* param_time = nullptr;
        Parameter* param_speed = nullptr;
        for (auto& param_ptr : params) {
            if ((param_ptr->GetName() == "play") && (param_ptr->type == ParamType::BOOL)) {
                param_play = param_ptr;
            }
            if ((param_ptr->GetName() == "time") && (param_ptr->type == ParamType::FLOAT)) {
                param_time = param_ptr;
            }
            if ((param_ptr->GetName() == "speed") && (param_ptr->type == ParamType::FLOAT)) {
                param_speed = param_ptr;
            }
        }
        if ((param_play == nullptr) || (param_time == nullptr) || (param_speed == nullptr)) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Unable to find all required parameters by name for animation group widget. [%s, %s, line %d]\n",
                __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        // Load button textures (once)
        if (this->button_tex_ids.play == 0) {
            megamol::gui::GUIUtils::LoadTexture(
                "../share/resources/transport_ctrl_play.png", this->button_tex_ids.play);
        }
        if (this->button_tex_ids.pause == 0) {
            megamol::gui::GUIUtils::LoadTexture(
                "../share/resources/transport_ctrl_pause.png", this->button_tex_ids.pause);
        }
        if (this->button_tex_ids.fastforward == 0) {
            megamol::gui::GUIUtils::LoadTexture(
                "../share/resources/transport_ctrl_fast-forward.png", this->button_tex_ids.fastforward);
        }
        if (this->button_tex_ids.fastrewind == 0) {
            megamol::gui::GUIUtils::LoadTexture(
                "../share/resources/transport_ctrl_fast-rewind.png", this->button_tex_ids.fastrewind);
        }
        if ((this->button_tex_ids.play == 0) || (this->button_tex_ids.pause == 0) ||
            (this->button_tex_ids.fastforward == 0) || (this->button_tex_ids.fastrewind == 0)) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Unable to load all required button textures for animation group widget. [%s, %s, line %d]\n", __FILE__,
                __FUNCTION__, __LINE__);
            return false;
        }

        // ------------------------------------------------------------------------
        ImGuiStyle& style = ImGui::GetStyle();
        float frame_height = ImGui::GetFrameHeightWithSpacing(); // ImGui::GetFrameHeight();
        float child_height = frame_height * 4.5f;
        auto child_flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration;
        ImGui::BeginChild("group_widget_animation", ImVec2(0.0f, child_height), true, child_flags);

        // Caption
        ImGui::TextUnformatted("Animation");

        // Transport Buttons ------------------------------------------------------
        ImGui::PushStyleColor(ImGuiCol_Button, style.Colors[ImGuiCol_Button]);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, style.Colors[ImGuiCol_ButtonActive]);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, style.Colors[ImGuiCol_ButtonHovered]);

        bool play = std::get<bool>(param_play->GetValue());
        float time = std::get<float>(param_time->GetValue());
        float speed = std::get<float>(param_speed->GetValue());
        std::string button_label;
        ImTextureID button_tex;

        /// PLAY - PAUSE
        button_label = "Play";
        button_tex = reinterpret_cast<ImTextureID>(this->button_tex_ids.play);
        if (play) {
            button_label = "Pause";
            button_tex = reinterpret_cast<ImTextureID>(this->button_tex_ids.pause);
        }

        if (ImGui::ImageButton(button_tex, ImVec2(frame_height, frame_height), ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f),
                1, style.Colors[ImGuiCol_Button], style.Colors[ImGuiCol_ButtonActive])) {
            play = !play;
        }
        this->utils.HoverToolTip(button_label, ImGui::GetItemID(), 1.0f, 5.0f);
        ImGui::SameLine();

        /// SLOWER
        button_label = "Slower";
        button_tex = reinterpret_cast<ImTextureID>(this->button_tex_ids.fastrewind);
        if (ImGui::ImageButton(button_tex, ImVec2(frame_height, frame_height), ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f),
                1, style.Colors[ImGuiCol_Button], style.Colors[ImGuiCol_ButtonActive])) {
            // play = true;
            speed /= 1.5f;
        }
        this->utils.HoverToolTip(button_label, ImGui::GetItemID(), 1.0f, 5.0f);
        ImGui::SameLine();

        /// FASTER
        button_label = "Faster";
        button_tex = reinterpret_cast<ImTextureID>(this->button_tex_ids.fastforward);
        if (ImGui::ImageButton(button_tex, ImVec2(frame_height, frame_height), ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f),
                1, style.Colors[ImGuiCol_Button], style.Colors[ImGuiCol_ButtonActive])) {
            // play = true;
            speed *= 1.5f;
        }
        this->utils.HoverToolTip(button_label, ImGui::GetItemID(), 1.0f, 5.0f);

        ImGui::PopStyleColor(3);

        param_play->SetValue(play);
        param_time->SetValue(time);
        param_speed->SetValue(speed);

        // Time -------------------------------------------------------------------

        param_time->PresentGUI(ParameterPresentation::WidgetScope::LOCAL);

        // Speed -------------------------------------------------------------------

        param_speed->PresentGUI(ParameterPresentation::WidgetScope::LOCAL);

        ImGui::EndChild();
    } else {
        /// GLOBAL

        // nothing to do ...
    }

    return true;
}
