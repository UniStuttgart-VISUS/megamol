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


megamol::gui::ParameterGroupPresentation::ParameterGroupPresentation(void) : group_widget_ids() {

    // Add group widget id for animation
    group_widget_ids["anim"].first.emplace(ParamType::BOOL, 1);
    group_widget_ids["anim"].first.emplace(ParamType::BUTTON, 3);
    group_widget_ids["anim"].first.emplace(ParamType::FLOAT, 2);
    group_widget_ids["anim"].second = [&, this](ParamPtrVectorType& params) { this->group_widget_animation(params); };
}


megamol::gui::ParameterGroupPresentation::~ParameterGroupPresentation(void) {}


bool megamol::gui::ParameterGroupPresentation::PresentGUI(megamol::gui::configurator::ParamVectorType& inout_params,
    const std::string& in_module_fullname, const std::string& in_search, bool in_extended, bool in_ignore_extended,
    megamol::gui::configurator::ParameterPresentation::WidgetScope in_scope,
    const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool& out_open_external_tf_editor) {

    out_open_external_tf_editor = false;

    // Nothing to do if there are no parameters
    if (inout_params.empty()) return true;

    if (in_scope == megamol::gui::configurator::ParameterPresentation::WidgetScope::GLOBAL) {
        // GLOBAL

        for (auto& param : inout_params) {
            this->drawParameter(param, in_module_fullname, in_search, in_extended, in_ignore_extended, in_scope,
                in_external_tf_editor, out_open_external_tf_editor);
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
            if (in_scope == megamol::gui::configurator::ParameterPresentation::WidgetScope::LOCAL) {
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
                    this->drawParameter(param, in_module_fullname, in_search, in_extended, in_ignore_extended, in_scope,
                        in_external_tf_editor, out_open_external_tf_editor);
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
                if ((group_widget_id.first == group_name) && (group_widget_id.second.first == group.second.second)) {
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
                        this->drawParameter((*param), in_module_fullname, in_search, in_extended, in_ignore_extended,
                            in_scope, in_external_tf_editor, out_open_external_tf_editor);
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
    const std::string& in_module_fullname, const std::string& in_search, bool in_extended, bool in_ignore_extended,
    megamol::gui::configurator::ParameterPresentation::WidgetScope in_scope,
    const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool& out_open_external_tf_editor) {

    if ((inout_param.type == ParamType::TRANSFERFUNCTION) && (in_external_tf_editor != nullptr)) {
        inout_param.present.ConnectExternalTransferFunctionEditor(in_external_tf_editor);
    }

    if (in_scope == megamol::gui::configurator::ParameterPresentation::WidgetScope::GLOBAL) {
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

    auto child_flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration;
    ImGui::BeginChild("group_widget_animation", ImVec2(0.0f, 50.0f), true, child_flags);

    ImGui::TextUnformatted("Animations...");

    ImGui::EndChild();
}
