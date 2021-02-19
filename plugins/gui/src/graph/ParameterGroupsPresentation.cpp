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
        : group_widgets()
        , tooltip()
        , cube_widget()
        , speed_knob_pos(ImVec2(0.0f, 0.0f))
        , time_knob_pos(ImVec2(0.0f, 0.0f))
        , image_buttons() {

    // Add group widget data for animation widget group
    /// Paramter namespace: View3DGL::anim
    GroupWidgetData anim_group_widget_data(Param_t::GROUP_ANIMATION);
    anim_group_widget_data.active = false;

    anim_group_widget_data.check_callback = [&, this](bool only_check, ParamPtrVector_t& params) -> bool {
        return this->check_group_widget_animation(only_check, params);
    };

    anim_group_widget_data.draw_callback =
        [&, this](GroupWidgetData_t& group_widget_data, ParamPtrVector_t params, const std::string& in_module_fullname,
            const std::string& in_search, megamol::gui::ParameterPresentation::WidgetScope in_scope,
            const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor,
            ImGuiID in_override_header_state, PickingBuffer* inout_picking_buffer) -> bool {
        return this->draw_group_widget_animation(group_widget_data, params, in_module_fullname, in_search, in_scope,
            in_external_tf_editor, out_open_external_tf_editor, in_override_header_state, inout_picking_buffer);
    };

    /// ID string must equal parameter group name, which is used for identification
    this->group_widgets["anim"] = anim_group_widget_data;

    /// Paramter namespace: View3DGL::view
    GroupWidgetData view_group_widget_data(Param_t::GROUP_3D_CUBE);
    view_group_widget_data.active = false;

    view_group_widget_data.check_callback = [&, this](bool only_check, ParamPtrVector_t& params) -> bool {
        return this->check_group_widget_3d_cube(only_check, params);
    };

    view_group_widget_data.draw_callback =
        [&, this](GroupWidgetData_t& group_widget_data, ParamPtrVector_t params, const std::string& in_module_fullname,
            const std::string& in_search, megamol::gui::ParameterPresentation::WidgetScope in_scope,
            const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor,
            ImGuiID in_override_header_state, PickingBuffer* inout_picking_buffer) -> bool {
        return this->draw_group_widget_3d_cube(group_widget_data, params, in_module_fullname, in_search, in_scope,
            in_external_tf_editor, out_open_external_tf_editor, in_override_header_state, inout_picking_buffer);
    };

    /// ID string must equal parameter group name, which is used for identification
    this->group_widgets["view"] = view_group_widget_data;
}


megamol::gui::ParameterGroupsPresentation::~ParameterGroupsPresentation(void) {}


bool megamol::gui::ParameterGroupsPresentation::PresentGUI(megamol::gui::ParamVector_t& inout_params,
    const std::string& in_module_fullname, const std::string& in_search, vislib::math::Ternary in_extended,
    bool in_indent, megamol::gui::ParameterPresentation::WidgetScope in_scope,
    const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor,
    ImGuiID in_override_header_state, PickingBuffer* inout_picking_buffer) {

    assert(ImGui::GetCurrentContext() != nullptr);

    if (out_open_external_tf_editor != nullptr)
        (*out_open_external_tf_editor) = false;

    // Nothing to do if there are no parameters
    if (inout_params.empty())
        return true;

    if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
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
            param.present.extended = in_extended.IsTrue();
        }

        if (!param_namespace.empty()) {
            // Sort parameters with namespace to group
            group_map[param_namespace].emplace_back(&param);
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
        for (auto& group_widget_data : this->group_widgets) {

            // Check for same group name
            if ((group_widget_data.first == group_name) &&
                (group_widget_data.second.check_callback(true, group.second))) {

                found_group_widget = true;
                group_widget_data.second.active = true;
                ImGui::PushID(group_widget_data.first.c_str());
                if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
                    // LOCAL

                    if (in_extended.IsTrue()) {
                        // Visibility
                        bool visible = group_widget_data.second.IsGUIVisible();
                        if (ImGui::RadioButton("###visibile", visible)) {
                            group_widget_data.second.SetGUIVisible(!visible);
                        }
                        this->tooltip.ToolTip("Visibility", ImGui::GetItemID(), 0.5f);
                        ImGui::SameLine();

                        // Read-only option
                        bool readonly = group_widget_data.second.IsGUIReadOnly();
                        if (ImGui::Checkbox("###readonly", &readonly)) {
                            group_widget_data.second.SetGUIReadOnly(readonly);
                        }
                        this->tooltip.ToolTip("Read-Only", ImGui::GetItemID(), 0.5f);
                        ImGui::SameLine();

                        // Presentation option
                        ParameterPresentation::OptionButton(
                            "param_groups", "", (group_widget_data.second.GetGUIPresentation() != Present_t::Basic));
                        if (ImGui::BeginPopupContextItem("param_present_button_context", 0)) {
                            for (auto& present_name_pair : group_widget_data.second.GetPresentationNameMap()) {
                                if (group_widget_data.second.IsPresentationCompatible(present_name_pair.first)) {
                                    if (ImGui::MenuItem(present_name_pair.second.c_str(), nullptr,
                                            (present_name_pair.first ==
                                                group_widget_data.second.GetGUIPresentation()))) {
                                        group_widget_data.second.SetGUIPresentation(present_name_pair.first);
                                    }
                                }
                            }
                            ImGui::EndPopup();
                        }
                        this->tooltip.ToolTip("Presentation", ImGui::GetItemID(), 0.5f);
                        ImGui::SameLine();
                    }

                    // Call group widget draw function
                    if (group_widget_data.second.IsGUIVisible() || in_extended.IsTrue()) {
                        if (group_widget_data.second.IsGUIReadOnly()) {
                            GUIUtils::ReadOnlyWigetStyle(true);
                        }

                        if (!group_widget_data.second.draw_callback(group_widget_data, group.second, in_module_fullname,
                                in_search, in_scope, in_external_tf_editor, out_open_external_tf_editor,
                                in_override_header_state, inout_picking_buffer)) {

                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] No LOCAL widget presentation '%s' available for group widget '%s'. [%s, %s, "
                                "line "
                                "%d]\n",
                                group_widget_data.second
                                    .GetPresentationName(group_widget_data.second.GetGUIPresentation())
                                    .c_str(),
                                group_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                        }

                        if (group_widget_data.second.IsGUIReadOnly()) {
                            GUIUtils::ReadOnlyWigetStyle(false);
                        }
                    }
                } else {
                    // GLOBAL

                    group_widget_data.second.draw_callback(group_widget_data, group.second, in_module_fullname,
                        in_search, in_scope, in_external_tf_editor, out_open_external_tf_editor,
                        in_override_header_state, inout_picking_buffer);
                }

                ImGui::PopID();
            }
        }

        // ... else draw grouped parameters with no custom group widget using namespace header.
        if (!found_group_widget) {

            if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
                /// LOCAL

                this->draw_grouped_parameters(group_name, group.second, in_module_fullname, in_search, in_scope,
                    in_external_tf_editor, out_open_external_tf_editor, in_override_header_state);
            } else {
                /// GLOBAL

                for (auto& param : group.second) {
                    this->draw_parameter((*param), in_module_fullname, in_search, in_scope, in_external_tf_editor,
                        out_open_external_tf_editor);
                }
            }
        }
    }

    if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
        /// LOCAL

        if (in_indent)
            ImGui::Unindent();
        ImGui::EndGroup();
    }

    return true;
}


bool megamol::gui::ParameterGroupsPresentation::StateToJSON(
    nlohmann::json& inout_json, const std::string& module_fullname) {

    for (auto& group_widget_data : this->group_widgets) {
        if (group_widget_data.second.active) {
            std::string param_fullname = module_fullname + "::ParameterGroup::" + group_widget_data.first;
            group_widget_data.second.StateToJSON(inout_json, param_fullname);
        }
    }

    return false;
}


bool megamol::gui::ParameterGroupsPresentation::StateFromJSON(
    const nlohmann::json& in_json, const std::string& module_fullname) {

    for (auto& group_widget_data : this->group_widgets) {
        std::string param_fullname = module_fullname + "::ParameterGroup::" + group_widget_data.first;

        if (group_widget_data.second.StateFromJSON(in_json, param_fullname)) {
            group_widget_data.second.active = true;
        }
    }

    return false;
}


void megamol::gui::ParameterGroupsPresentation::draw_parameter(megamol::gui::Parameter& inout_param,
    const std::string& in_module_fullname, const std::string& in_search,
    megamol::gui::ParameterPresentation::WidgetScope in_scope,
    const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor) {

    if (inout_param.type == Param_t::TRANSFERFUNCTION) {
        inout_param.present.ConnectExternalTransferFunctionEditor(in_external_tf_editor);
    }

    if (in_scope == ParameterPresentation::WidgetScope::GLOBAL) {
        /// GLOBAL

        inout_param.PresentGUI(in_scope, in_module_fullname);
    } else {
        /// LOCAL

        auto param_name = inout_param.full_name;
        bool param_searched = true;
        bool module_searched = true;
        if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
            param_searched = megamol::gui::GUIUtils::FindCaseInsensitiveSubstring(param_name, in_search);
            module_searched = megamol::gui::GUIUtils::FindCaseInsensitiveSubstring(in_module_fullname, in_search);
        }
        bool visible =
            (inout_param.present.IsGUIVisible() || inout_param.present.extended) && (param_searched || module_searched);

        if (visible) {
            if (inout_param.PresentGUI(in_scope, in_module_fullname)) {

                // Open window calling the transfer function editor callback
                if ((inout_param.type == Param_t::TRANSFERFUNCTION) && (in_external_tf_editor != nullptr)) {
                    if (out_open_external_tf_editor != nullptr) {
                        (*out_open_external_tf_editor) = true;
                    }
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
    const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor,
    ImGuiID in_override_header_state) {

    if (in_scope != ParameterPresentation::WidgetScope::LOCAL) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Parameter groups are only available in LOCAL scope. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
    }

    // Skip if no parameter is visible and extended mode is not set.
    bool visible = false;
    bool extended = false;
    for (auto& param : params) {
        visible = visible || param->present.IsGUIVisible();
        extended = extended || param->present.extended;
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
            this->draw_parameter((*param), in_module_fullname, search_string, in_scope, in_external_tf_editor,
                out_open_external_tf_editor);
        }
        ImGui::Unindent();
    }
}


bool megamol::gui::ParameterGroupsPresentation::check_group_widget_animation(
    bool only_check, ParamPtrVector_t& params) {

    bool param_play = false;
    bool param_time = false;
    bool param_speed = false;
    for (auto& param_ptr : params) {
        if ((param_ptr->GetName() == "play") && (param_ptr->type == Param_t::BOOL)) {
            param_play = true;
        } else if ((param_ptr->GetName() == "time") && (param_ptr->type == Param_t::FLOAT)) {
            param_time = true;
        } else if ((param_ptr->GetName() == "speed") && (param_ptr->type == Param_t::FLOAT)) {
            param_speed = true;
        }
    }
    return (param_play && param_time && param_speed);
}

bool megamol::gui::ParameterGroupsPresentation::draw_group_widget_animation(GroupWidgetData_t& group_widget_data,
    ParamPtrVector_t params, const std::string& in_module_fullname, const std::string& in_search,
    megamol::gui::ParameterPresentation::WidgetScope in_scope,
    const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor,
    ImGuiID in_override_header_state, PickingBuffer* inout_picking_buffer) {

    // Check required parameters ----------------------------------------------
    Parameter* param_play = nullptr;
    Parameter* param_time = nullptr;
    Parameter* param_speed = nullptr;
    /// Find specific parameters of group by name because parameter type can occure multiple times.
    for (auto& param_ptr : params) {
        if ((param_ptr->GetName() == "play") && (param_ptr->type == Param_t::BOOL)) {
            param_play = param_ptr;
        } else if ((param_ptr->GetName() == "time") && (param_ptr->type == Param_t::FLOAT)) {
            param_time = param_ptr;
        } else if ((param_ptr->GetName() == "speed") && (param_ptr->type == Param_t::FLOAT)) {
            param_speed = param_ptr;
        }
    }
    if ((param_play == nullptr) || (param_time == nullptr) || (param_speed == nullptr)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to find all required parameters by name for animation group widget. [%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Parameter presentation -------------------------------------------------
    auto presentation = group_widget_data.second.GetGUIPresentation();
    if (presentation == param::AbstractParamPresentation::Presentation::Basic) {

        if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {

            this->draw_grouped_parameters(group_widget_data.first, params, in_module_fullname, in_search, in_scope,
                in_external_tf_editor, out_open_external_tf_editor, in_override_header_state);

            return true;

        } else if (in_scope == ParameterPresentation::WidgetScope::GLOBAL) {

            // no global implementation ...
            return true;
        }

    } else if (presentation == param::AbstractParamPresentation::Presentation::Group_Animation) {

        // Early exit for LOCAL widget presentation
        if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
            // LOCAL

            ImGui::TextDisabled(group_widget_data.first.c_str());
            return true;
        }
        /// else if (in_scope == ParameterPresentation::WidgetScope::GLOBAL) {

        // Load button textures (once) --------------------------------------------
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

        // DRAW -------------------------------------------------------------------
        const ImVec2 button_size =
            ImVec2(1.5f * ImGui::GetFrameHeightWithSpacing(), 1.5f * ImGui::GetFrameHeightWithSpacing());
        const float knob_size = 2.5f * ImGui::GetFrameHeightWithSpacing();

        ImGuiStyle& style = ImGui::GetStyle();
        if (in_scope == ParameterPresentation::WidgetScope::GLOBAL) {
            // GLOBAL

            ImGui::Begin(group_widget_data.first.c_str(), nullptr,
                ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar |
                    ImGuiWindowFlags_NoCollapse);
        } else { // if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
            /// LOCAL
            // Alternative LOCAL presentation

            // ImGui::BeginGroup();
            // ImGui::TextUnformatted(group_widget_data.first.c_str());
            // ImGui::Separator();
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
            // GLOBAL

            ImGui::End();
        }

        else if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
            /// LOCAL
            // Alternative LOCAL presentation

            // ImGui::EndGroup();
        }

        return true;
    }

    return false;
}


bool megamol::gui::ParameterGroupsPresentation::check_group_widget_3d_cube(bool only_check, ParamPtrVector_t& params) {

    bool param_cubeOrientation = false;
    bool param_defaultView = false;
    bool param_showCube = false;
    for (auto& param_ptr : params) {
        if ((param_ptr->GetName() == "cubeOrientation") && (param_ptr->type == Param_t::VECTOR4F)) {
            param_cubeOrientation = true;
        } else if ((param_ptr->GetName() == "defaultView") && (param_ptr->type == Param_t::ENUM)) {
            param_defaultView = true;
        } else if ((param_ptr->GetName() == "showViewCube") && (param_ptr->type == Param_t::BOOL)) {
            param_showCube = true;
        }
    }

    return (param_cubeOrientation && param_defaultView && param_showCube);
}


bool megamol::gui::ParameterGroupsPresentation::draw_group_widget_3d_cube(GroupWidgetData_t& group_widget_data,
    ParamPtrVector_t params, const std::string& in_module_fullname, const std::string& in_search,
    megamol::gui::ParameterPresentation::WidgetScope in_scope,
    const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor,
    ImGuiID in_override_header_state, PickingBuffer* inout_picking_buffer) {

    // Check required parameters ----------------------------------------------
    Parameter* param_cubeOrientation = nullptr;
    Parameter* param_defaultView = nullptr;
    Parameter* param_showCube = nullptr;
    /// Find specific parameters of group by name because parameter type can occure multiple times.
    for (auto& param_ptr : params) {
        if ((param_ptr->GetName() == "cubeOrientation") && (param_ptr->type == Param_t::VECTOR4F)) {
            param_cubeOrientation = param_ptr;
        } else if ((param_ptr->GetName() == "defaultView") && (param_ptr->type == Param_t::ENUM)) {
            param_defaultView = param_ptr;
        } else if ((param_ptr->GetName() == "showViewCube") && (param_ptr->type == Param_t::BOOL)) {
            param_showCube = param_ptr;
        }
    }
    if ((param_cubeOrientation == nullptr) || (param_defaultView == nullptr) || (param_showCube == nullptr)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[GUI] Unable to find all required parameters by name "
                                                                "for '%s' group widget. [%s, %s, line %d]\n",
            group_widget_data.first.c_str(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Parameter presentation -------------------------------------------------
    auto presentation = group_widget_data.second.GetGUIPresentation();
    // Switch presentation via parameter
    if (param_showCube->IsValueDirty()) {
        if (std::get<bool>(param_showCube->GetValue())) {
            group_widget_data.second.SetGUIPresentation(param::AbstractParamPresentation::Presentation::Group_3D_Cube);
        } else {
            group_widget_data.second.SetGUIPresentation(param::AbstractParamPresentation::Presentation::Basic);
        }
    }
    param_showCube->SetValue((presentation == param::AbstractParamPresentation::Presentation::Group_3D_Cube));

    if (presentation == param::AbstractParamPresentation::Presentation::Basic) {

        if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
            // LOCAL

            this->draw_grouped_parameters(group_widget_data.first, params, in_module_fullname, in_search, in_scope,
                in_external_tf_editor, out_open_external_tf_editor, in_override_header_state);

            return true;

        } else if (in_scope == ParameterPresentation::WidgetScope::GLOBAL) {

            // no global implementation ...
            return true;
        }

    } else if (presentation == param::AbstractParamPresentation::Presentation::Group_3D_Cube) {

        if (in_scope == ParameterPresentation::WidgetScope::LOCAL) {
            // LOCAL

            this->draw_grouped_parameters(
                group_widget_data.first, params, "", in_search, in_scope, nullptr, nullptr, GUI_INVALID_ID);

            return true;

        } else if (in_scope == ParameterPresentation::WidgetScope::GLOBAL) {
            // GLOBAL

            if (inout_picking_buffer == nullptr) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Pointer to required picking buffer is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                    __LINE__);
                return false;
            }

            auto id = param_defaultView->uid;
            inout_picking_buffer->AddInteractionObject(id, this->cube_widget.GetInteractions(id));

            ImGuiIO& io = ImGui::GetIO();
            auto default_view = std::get<int>(param_defaultView->GetValue());
            auto view_orientation = std::get<glm::vec4>(param_cubeOrientation->GetValue());
            auto viewport_dim = glm::vec2(io.DisplaySize.x, io.DisplaySize.y);
            int hovered_view = -1;
            this->cube_widget.Draw(id, default_view, hovered_view, view_orientation, viewport_dim,
                inout_picking_buffer->GetPendingManipulations());

            std::string tooltip_text;
            /// Indices must fit enum order in megamol::core::view::View3D_2::defaultview
            switch (hovered_view) {
            case (0): // DEFAULTVIEW_FRONT
                tooltip_text = "Front";
                break;
            case (1): // DEFAULTVIEW_BACK
                tooltip_text = "Back";
                break;
            case (2): // DEFAULTVIEW_RIGHT
                tooltip_text = "Right";
                break;
            case (3): // DEFAULTVIEW_LEFT
                tooltip_text = "Left";
                break;
            case (4): // DEFAULTVIEW_TOP
                tooltip_text = "Top";
                break;
            case (5): // DEFAULTVIEW_BOTTOM
                tooltip_text = "Bottom";
                break;
            default:
                break;
            }
            if (!tooltip_text.empty()) {
                ImGui::BeginTooltip();
                ImGui::TextUnformatted(tooltip_text.c_str());
                ImGui::EndTooltip();
            }

            param_defaultView->SetValue(default_view);

            return true;
        }
    }

    return false;
}
