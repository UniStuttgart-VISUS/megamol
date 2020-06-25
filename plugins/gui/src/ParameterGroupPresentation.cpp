/*
 * ParameterGroupPresentation.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ParameterGroupPresentation.h"


using namespace megamol;
using namespace megamol::gui;
using namespace megamol::gui::configurator;


megamol::gui::ParameterGroupPresentation::ParameterGroupPresentation(void)
    : utils(), group_widget_ids(), button_tex_ids{0, 0, 0, 0} {

    // Add group widget id for animation
    group_widget_ids["animation"].first.emplace(ParamType::BOOL, 1);
    group_widget_ids["animation"].first.emplace(ParamType::BUTTON, 3);
    group_widget_ids["animation"].first.emplace(ParamType::FLOAT, 2);
    group_widget_ids["animation"].second = [&, this](
                                               ParamPtrVectorType& params) { this->group_widget_animation(params); };
}


megamol::gui::ParameterGroupPresentation::~ParameterGroupPresentation(void) {}


bool megamol::gui::ParameterGroupPresentation::PresentGUI(megamol::gui::configurator::ParamVectorType& inout_params,
    const std::string& in_module_fullname, const std::string& in_search, bool in_extended, bool in_ignore_extended,
    megamol::gui::configurator::ParameterPresentation::WidgetScope in_scope,
    const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool& out_open_external_tf_editor) {

    out_open_external_tf_editor = false;

    // Nothing to do if there are no parameters
    if (inout_params.empty()) return true;

    if (in_scope == ParameterPresentation::WidgetScope::GLOBAL) {
        // GLOBAL

        for (auto& param : inout_params) {
            this->drawParameter(
                param, in_module_fullname, in_scope, in_external_tf_editor, out_open_external_tf_editor);
        }
    } else {
        // LOCAL

        ImGui::BeginGroup();
        ImGui::Indent();

        ParamGroupType group_map;
        for (auto& param : inout_params) {
            auto param_name = param.full_name;
            auto param_namespace = param.GetNameSpace();

            // Handle visibility of parameter as soon as possible!
            if (!in_ignore_extended) {
                param.present.extended = in_extended;
            }
            bool param_searched = true;
            if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
                param_searched = megamol::gui::GUIUtils::FindCaseInsensitiveSubstring(param_name, in_search);
            }
            bool visible = (param.present.IsGUIVisible() && param_searched) || param.present.extended;

            if (visible) {
                if (!param_namespace.empty()) {
                    // Sort parameters with namespace to group
                    group_map[param_namespace].first.emplace_back(&param);
                    group_map[param_namespace].second[param.type]++;
                } else {
                    // Draw parameters without namespace at the beginning
                    this->drawParameter(
                        param, in_module_fullname, in_scope, in_external_tf_editor, out_open_external_tf_editor);
                }
            }
        }

        // Draw grouped parameters
        for (auto& group : group_map) {
            auto group_name = group.first;
            bool found_group_widget = false;

            // Check for exisiting group widget
            for (auto& group_widget_id : this->group_widget_ids) {

                // Check for same group name and count of different parameter types
                /// XXX Is this check too expensive (also check group_name?) - Alternative?
                if (group_widget_id.second.first == group.second.second) {
                    found_group_widget = true;
                    // Call group widget draw function
                    group_widget_id.second.second(group.second.first);
                }
            }

            // Draw group parameters with no custom group widget
            if (!found_group_widget) {
                // Open namespace header when parameter search is active
                if (!in_search.empty()) {
                    auto headerId = ImGui::GetID(group_name.c_str());
                    ImGui::GetStateStorage()->SetInt(headerId, 1);
                }

                if (ImGui::CollapsingHeader(group_name.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::Indent();

                    for (auto& param : group.second.first) {
                        this->drawParameter(
                            (*param), in_module_fullname, in_scope, in_external_tf_editor, out_open_external_tf_editor);
                    }

                    // Vertical spacing
                    /// ImGui::Dummy(ImVec2(1.0f, ImGui::GetFrameHeightWithSpacing()));
                    ImGui::Unindent();
                }
            }
        }

        // Vertical spacing
        /// ImGui::Dummy(ImVec2(1.0f, ImGui::GetFrameHeightWithSpacing()));
        ImGui::Unindent();
        ImGui::EndGroup();
    }
    return true;
}


void megamol::gui::ParameterGroupPresentation::drawParameter(megamol::gui::configurator::Parameter& inout_param,
    const std::string& in_module_fullname, megamol::gui::configurator::ParameterPresentation::WidgetScope in_scope,
    const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool& out_open_external_tf_editor) {

    if ((inout_param.type == ParamType::TRANSFERFUNCTION) && (in_external_tf_editor != nullptr)) {
        inout_param.present.ConnectExternalTransferFunctionEditor(in_external_tf_editor);
    }

    if (in_scope == ParameterPresentation::WidgetScope::GLOBAL) {
        // GLOBAL
        inout_param.PresentGUI(in_scope);
    } else {
        // LOCAL
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


void megamol::gui::ParameterGroupPresentation::group_widget_animation(ParamPtrVectorType& params) {

    // Check required parameters
    Parameter* param_play = nullptr;
    Parameter* param_time = nullptr;
    Parameter* param_speed = nullptr;
    for (auto& param : params) {
        if ((param->GetName() == "play") && (param->type == ParamType::BOOL)) {
            param_play = param;
        }
        if ((param->GetName() == "time") && (param->type == ParamType::FLOAT)) {
            param_time = param;
        }
        if ((param->GetName() == "speed") && (param->type == ParamType::FLOAT)) {
            param_speed = param;
        }
    }
    if ((param_play == nullptr) || (param_time == nullptr) || (param_speed == nullptr)) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Unable to find all required parameters by name for animation group widget. [%s, %s, line %d]\n", __FILE__,
            __FUNCTION__, __LINE__);
        return;
    }

    // Load button textures
    if (this->button_tex_ids.play == 0) {
        megamol::gui::GUIUtils::LoadTexture("../share/resources/transport_ctrl_play.png", this->button_tex_ids.play);
    }
    if (this->button_tex_ids.pause == 0) {
        megamol::gui::GUIUtils::LoadTexture("../share/resources/transport_ctrl_pause.png", this->button_tex_ids.pause);
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
        return;
    }

    // ------------------------------------------------------------------------
    ImGuiStyle& style = ImGui::GetStyle();

    float frame_height = ImGui::GetFrameHeightWithSpacing(); // ImGui::GetFrameHeight();

    float child_height = frame_height * 5.0f;
    auto child_flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration;
    ImGui::BeginChild("group_widget_animation", ImVec2(0.0f, child_height), true, child_flags);

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
        std::string play_label = "Pause";
        button_tex = reinterpret_cast<ImTextureID>(this->button_tex_ids.pause);
    }

    if (ImGui::ImageButton(button_tex, ImVec2(frame_height, frame_height), ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f), 1,
            style.Colors[ImGuiCol_Button], style.Colors[ImGuiCol_ButtonActive])) {
        play = !play;
    }
    this->utils.HoverToolTip(button_label, ImGui::GetID(button_label.c_str()), 1.0f, 5.0f);
    ImGui::SameLine();

    /// SLOWER
    button_label = "Slower";
    button_tex = reinterpret_cast<ImTextureID>(this->button_tex_ids.fastrewind);
    if (ImGui::ImageButton(button_tex, ImVec2(frame_height, frame_height), ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f), 1,
            style.Colors[ImGuiCol_Button], style.Colors[ImGuiCol_ButtonActive])) {
        // play = true;
        speed /= 1.5f;
    }
    this->utils.HoverToolTip(button_label, ImGui::GetID(button_label.c_str()), 1.0f, 5.0f);
    ImGui::SameLine();

    /// FASTER
    button_label = "Faster";
    button_tex = reinterpret_cast<ImTextureID>(this->button_tex_ids.fastforward);
    if (ImGui::ImageButton(button_tex, ImVec2(frame_height, frame_height), ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f), 1,
            style.Colors[ImGuiCol_Button], style.Colors[ImGuiCol_ButtonActive])) {
        // play = true;
        speed *= 1.5f;
    }
    this->utils.HoverToolTip(button_label, ImGui::GetID(button_label.c_str()), 1.0f, 5.0f);

    ImGui::PopStyleColor(3);

    param_play->SetValue(play);
    param_time->SetValue(time);
    param_speed->SetValue(speed);

    // Time -------------------------------------------------------------------

    param_time->PresentGUI(ParameterPresentation::WidgetScope::LOCAL);
    param_speed->PresentGUI(ParameterPresentation::WidgetScope::LOCAL);

    /*
    ImGui::Value("Time", time, "%.5f");
    ImGui::SameLine();
    ImGui::Value("Speed", speed, "%.5f");
    */

    /*
    float min_time = param_time->GetMinValue<float>();
    float max_time = param_time->GetMaxValue<float>();
    if ((min_time == -FLT_MAX) || (min_time == FLT_MAX)) {
        min_time = 0.0f;
    }
    if ((max_time == -FLT_MAX) || (max_time == FLT_MAX)) {
        float frac = 10.0f;
        float tmp_time = time;
        while (tmp_time > 1.0f) {
            frac *= 10.0f;
            tmp_time /= 10.0f;
        }
        max_time = frac;
    }
    if (ImGui::SliderFloat("Time", &time, min_time, max_time, "%.3f")) {
        param_time->SetValue(time);
    }
    */

    ImGui::EndChild();
}
