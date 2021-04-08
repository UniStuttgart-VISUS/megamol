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


megamol::gui::ParameterGroups::ParameterGroups(void)
        : tooltip(), cube_widget_group(), animation_group(), group_widgets() {

    // Create/register available group widgets
    this->group_widgets.emplace_back(static_cast<AbstractParameterGroupWidget*>(&this->cube_widget_group));
    this->group_widgets.emplace_back(static_cast<AbstractParameterGroupWidget*>(&this->animation_group));
}


megamol::gui::ParameterGroups::~ParameterGroups(void) {}


bool megamol::gui::ParameterGroups::Draw(megamol::gui::ParamVector_t& inout_params,
    const std::string& in_module_fullname, const std::string& in_search, vislib::math::Ternary in_extended,
    bool in_indent, megamol::gui::Parameter::WidgetScope in_scope,
    const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor,
    ImGuiID in_override_header_state, PickingBuffer* inout_picking_buffer) {

    assert(ImGui::GetCurrentContext() != nullptr);

    if (out_open_external_tf_editor != nullptr)
        (*out_open_external_tf_editor) = false;

    // Nothing to do if there are no parameters
    if (inout_params.empty())
        return true;

    if (in_scope == Parameter::WidgetScope::LOCAL) {
        /// LOCAL

        ImGui::BeginGroup();
        if (in_indent)
            ImGui::Indent();
    }

    // Analyse parameter group membership and draw ungrouped parameters
    ParamGroup_t group_map;
    for (auto& param : inout_params) {
        auto param_namespace = param.GetNameSpace();
        if (!in_extended.IsUnknown()) {
            param.SetExtended(in_extended.IsTrue());
        }

        if (!param_namespace.empty()) {
            // Sort parameters with namespace to group
            group_map[param_namespace].emplace_back(&param);
        } else {
            // Draw parameters without namespace directly at the beginning
            ParameterGroups::DrawParameter(
                param, in_module_fullname, in_search, in_scope, in_external_tf_editor, out_open_external_tf_editor);
        }
    }

    // Draw grouped parameters
    for (auto& group : group_map) {
        auto group_name = group.first;
        bool found_group_widget = false;

        // Draw group widget (if defined) ...
        for (auto& group_widget_data : this->group_widgets) {

            // Check for same group name
            if ((group_widget_data->GetName() == group_name) && (group_widget_data->Check(true, group.second))) {

                found_group_widget = true;
                group_widget_data->SetActive(true);
                ImGui::PushID(group_widget_data->GetName().c_str());
                if (in_scope == Parameter::WidgetScope::LOCAL) {
                    // LOCAL

                    if (in_extended.IsTrue()) {
                        // Visibility
                        bool visible = group_widget_data->IsGUIVisible();
                        if (ImGui::RadioButton("###visibile", visible)) {
                            group_widget_data->SetGUIVisible(!visible);
                        }
                        this->tooltip.ToolTip("Visibility", ImGui::GetItemID(), 0.5f);
                        ImGui::SameLine();

                        // Read-only option
                        bool readonly = group_widget_data->IsGUIReadOnly();
                        if (ImGui::Checkbox("###readonly", &readonly)) {
                            group_widget_data->SetGUIReadOnly(readonly);
                        }
                        this->tooltip.ToolTip("Read-Only", ImGui::GetItemID(), 0.5f);
                        ImGui::SameLine();

                        // Presentation option
                        ButtonWidgets::OptionButton(
                            "param_groups", "", (group_widget_data->GetGUIPresentation() != Present_t::Basic));
                        if (ImGui::BeginPopupContextItem("param_present_button_context", 0)) {
                            for (auto& present_name_pair : group_widget_data->GetPresentationNameMap()) {
                                if (group_widget_data->IsPresentationCompatible(present_name_pair.first)) {
                                    if (ImGui::MenuItem(present_name_pair.second.c_str(), nullptr,
                                            (present_name_pair.first == group_widget_data->GetGUIPresentation()))) {
                                        group_widget_data->SetGUIPresentation(present_name_pair.first);
                                    }
                                }
                            }
                            ImGui::EndPopup();
                        }
                        this->tooltip.ToolTip("Presentation", ImGui::GetItemID(), 0.5f);
                        ImGui::SameLine();
                    }

                    // Call group widget draw function
                    if (group_widget_data->IsGUIVisible() || in_extended.IsTrue()) {
                        if (group_widget_data->IsGUIReadOnly()) {
                            GUIUtils::ReadOnlyWigetStyle(true);
                        }

                        if (!group_widget_data->Draw(
                                group.second, in_module_fullname, in_search, in_scope, inout_picking_buffer)) {

                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] No LOCAL widget presentation '%s' available for group widget '%s'. [%s, %s, "
                                "line "
                                "%d]\n",
                                group_widget_data->GetPresentationName(group_widget_data->GetGUIPresentation()).c_str(),
                                group_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                        }

                        if (group_widget_data->IsGUIReadOnly()) {
                            GUIUtils::ReadOnlyWigetStyle(false);
                        }
                    }
                } else {
                    // GLOBAL

                    group_widget_data->Draw(
                        group.second, in_module_fullname, in_search, in_scope, inout_picking_buffer);
                }

                ImGui::PopID();
            }
        }

        // ... else draw grouped parameters with no custom group widget using namespace header.
        if (!found_group_widget) {

            if (in_scope == Parameter::WidgetScope::LOCAL) {
                /// LOCAL

                ParameterGroups::DrawGroupedParameters(group_name, group.second, in_module_fullname, in_search,
                    in_scope, in_external_tf_editor, out_open_external_tf_editor, in_override_header_state);
            } else {
                /// GLOBAL

                for (auto& param : group.second) {
                    ParameterGroups::DrawParameter((*param), in_module_fullname, in_search, in_scope,
                        in_external_tf_editor, out_open_external_tf_editor);
                }
            }
        }
    }

    if (in_scope == Parameter::WidgetScope::LOCAL) {
        /// LOCAL

        if (in_indent)
            ImGui::Unindent();
        ImGui::EndGroup();
    }

    return true;
}


bool megamol::gui::ParameterGroups::StateToJSON(nlohmann::json& inout_json, const std::string& module_fullname) {

    for (auto& group_widget_data : this->group_widgets) {
        if (group_widget_data->IsActive()) {
            std::string param_fullname = module_fullname + "::ParameterGroup::" + group_widget_data->GetName();
            group_widget_data->StateToJSON(inout_json, param_fullname);
        }
    }

    return false;
}


bool megamol::gui::ParameterGroups::StateFromJSON(const nlohmann::json& in_json, const std::string& module_fullname) {

    for (auto& group_widget_data : this->group_widgets) {
        std::string param_fullname = module_fullname + "::ParameterGroup::" + group_widget_data->GetName();

        if (group_widget_data->StateFromJSON(in_json, param_fullname)) {
            group_widget_data->SetActive(true);
        }
    }

    return false;
}


void megamol::gui::ParameterGroups::DrawParameter(megamol::gui::Parameter& inout_param,
    const std::string& in_module_fullname, const std::string& in_search, megamol::gui::Parameter::WidgetScope in_scope,
    const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor) {

    if (inout_param.Type() == ParamType_t::TRANSFERFUNCTION) {
        inout_param.TransferFunctionEditor_ConnectExternal(in_external_tf_editor, false);
    }

    if (in_scope == Parameter::WidgetScope::GLOBAL) {
        /// GLOBAL

        inout_param.Draw(in_scope, in_module_fullname);
    } else {
        /// LOCAL

        auto param_name = inout_param.FullName();
        bool param_searched = true;
        bool module_searched = true;
        if (in_scope == Parameter::WidgetScope::LOCAL) {
            param_searched = megamol::gui::GUIUtils::FindCaseInsensitiveSubstring(param_name, in_search);
            module_searched = megamol::gui::GUIUtils::FindCaseInsensitiveSubstring(in_module_fullname, in_search);
        }
        bool visible = (inout_param.IsGUIVisible() || inout_param.IsExtended()) && (param_searched || module_searched);

        if (visible) {
            if (inout_param.Draw(in_scope, in_module_fullname)) {

                // Open window calling the transfer function editor callback
                if ((inout_param.Type() == ParamType_t::TRANSFERFUNCTION) && (in_external_tf_editor != nullptr)) {
                    if (out_open_external_tf_editor != nullptr) {
                        (*out_open_external_tf_editor) = true;
                    }
                    auto param_fullname = std::string(in_module_fullname.c_str()) + "::" + inout_param.FullName();
                    in_external_tf_editor->SetConnectedParameter(&inout_param, param_fullname);
                }
            }
        }
    }
}


void megamol::gui::ParameterGroups::DrawGroupedParameters(const std::string& in_group_name,
    AbstractParameterGroupWidget::ParamPtrVector_t& params, const std::string& in_module_fullname,
    const std::string& in_search, megamol::gui::Parameter::WidgetScope in_scope,
    const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor,
    ImGuiID in_override_header_state) {

    if (in_scope != Parameter::WidgetScope::LOCAL) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Parameter groups are only available in LOCAL scope. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
    }

    // Skip if no parameter is visible and extended mode is not set.
    bool visible = false;
    bool extended = false;
    for (auto& param : params) {
        visible = visible || param->IsGUIVisible();
        extended = extended || param->IsExtended();
    }
    if (!visible && !extended)
        return;

    // Open namespace header when parameter search is active.
    auto search_string = in_search;
    bool param_group_header_open = megamol::gui::GUIUtils::GroupHeader(
        megamol::gui::HeaderType::PARAMETERG_ROUP, in_group_name, search_string, in_override_header_state);

    if (param_group_header_open) {
        ImGui::Indent();
        for (auto& param : params) {
            ParameterGroups::DrawParameter((*param), in_module_fullname, search_string, in_scope, in_external_tf_editor,
                out_open_external_tf_editor);
        }
        ImGui::Unindent();
    }
}
