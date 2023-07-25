/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "widgets/ParameterGroupAnimationWidget.h"
#include "graph/ParameterGroups.h"
#include "widgets/ButtonWidgets.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::utility;
using namespace megamol::gui;


megamol::gui::ParameterGroupAnimationWidget::ParameterGroupAnimationWidget()
        : AbstractParameterGroupWidget(megamol::gui::GenerateUniqueID())
        , image_buttons()
        , tooltip() {

    this->InitPresentation(ParamType_t::GROUP_ANIMATION);
    this->name = "anim";
}


bool megamol::gui::ParameterGroupAnimationWidget::Check(bool only_check, ParamPtrVector_t& params) {

    bool param_play = false;
    bool param_time = false;
    bool param_speed = false;
    for (auto& param_ptr : params) {
        if ((param_ptr->Name() == "play") && (param_ptr->Type() == ParamType_t::BOOL)) {
            param_play = true;
        } else if ((param_ptr->Name() == "time") && (param_ptr->Type() == ParamType_t::FLOAT)) {
            param_time = true;
        } else if ((param_ptr->Name() == "speed") && (param_ptr->Type() == ParamType_t::FLOAT)) {
            param_speed = true;
        }
    }
    return (param_play && param_time && param_speed);
}


bool megamol::gui::ParameterGroupAnimationWidget::Draw(ParamPtrVector_t params, const std::string& in_search,
    megamol::gui::Parameter::WidgetScope in_scope, core::utility::PickingBuffer* inout_picking_buffer,
    ImGuiID in_override_header_state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        log::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Check required parameters ----------------------------------------------
    Parameter* param_play = nullptr;
    Parameter* param_time = nullptr;
    Parameter* param_speed = nullptr;
    /// Find specific parameters of group by name because parameter type can occure multiple times.
    for (auto& param_ptr : params) {
        if ((param_ptr->Name() == "play") && (param_ptr->Type() == ParamType_t::BOOL)) {
            param_play = param_ptr;
        } else if ((param_ptr->Name() == "time") && (param_ptr->Type() == ParamType_t::FLOAT)) {
            param_time = param_ptr;
        } else if ((param_ptr->Name() == "speed") && (param_ptr->Type() == ParamType_t::FLOAT)) {
            param_speed = param_ptr;
        }
    }
    if ((param_play == nullptr) || (param_time == nullptr) || (param_speed == nullptr)) {
        utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to find all required parameters by name for animation group widget. [%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Parameter presentation -------------------------------------------------
    auto presentation = this->GetGUIPresentation();
    if (presentation == param::AbstractParamPresentation::Presentation::Basic) {

        if (in_scope == Parameter::WidgetScope::LOCAL) {

            ParameterGroups::DrawGroupedParameters(
                this->name, params, in_search, in_scope, nullptr, in_override_header_state);
            return true;

        } else if (in_scope == Parameter::WidgetScope::GLOBAL) {

            // no global implementation ...
            return true;
        }

    } else if (presentation == param::AbstractParamPresentation::Presentation::Group_Animation) {

        // Early exit for LOCAL widget presentation
        if (in_scope == Parameter::WidgetScope::LOCAL) {
            // LOCAL

            ImGui::PushID(static_cast<int>(this->uid));
            ImGui::TextDisabled(this->name.c_str());
            ImGui::PopID();

            return true;
        }
        /// else if (in_scope == Parameter::WidgetScope::GLOBAL) {

        // Load button textures (once) --------------------------------------------
        if (!this->image_buttons.play_pause.IsLoaded()) {
            this->image_buttons.play_pause.LoadTextureFromFile(
                GUI_FILENAME_TEXTURE_TRANSPORT_ICON_PLAY, GUI_FILENAME_TEXTURE_TRANSPORT_ICON_PAUSE);
        }
        if (!this->image_buttons.fastforward.IsLoaded()) {
            this->image_buttons.fastforward.LoadTextureFromFile(GUI_FILENAME_TEXTURE_TRANSPORT_ICON_FAST_FORWARD);
        }
        if (!this->image_buttons.fastrewind.IsLoaded()) {
            this->image_buttons.fastrewind.LoadTextureFromFile(GUI_FILENAME_TEXTURE_TRANSPORT_ICON_FAST_REWIND);
        }
        if ((!this->image_buttons.play_pause.IsLoaded()) || (!this->image_buttons.fastforward.IsLoaded()) ||
            (!this->image_buttons.fastrewind.IsLoaded())) {
            utility::log::Log::DefaultLog.WriteError(
                "[GUI] Unable to load all required button textures for animation group widget. [%s, %s, line %d]\n",
                __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        // DRAW -------------------------------------------------------------------
        ImGui::PushID(static_cast<int>(this->uid));

        const ImVec2 button_size =
            ImVec2(1.5f * ImGui::GetFrameHeightWithSpacing(), 1.5f * ImGui::GetFrameHeightWithSpacing());
        const float knob_size = 2.5f * ImGui::GetFrameHeightWithSpacing();

        ImGuiStyle& style = ImGui::GetStyle();
        if (in_scope == Parameter::WidgetScope::GLOBAL) {
            // GLOBAL
            std::string unique_child_name = this->name + "###" + this->name + std::to_string(this->uid);
            ImGui::Begin(unique_child_name.c_str(), nullptr,
                ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar |
                    ImGuiWindowFlags_NoCollapse);
        } else { // if (in_scope == Parameter::WidgetScope::LOCAL) {
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
        this->image_buttons.play_pause.ToggleButton(play, "Play", "Pause", button_size);
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
        ButtonWidgets::KnobButton(label, knob_size, time, param_time->GetMinValue<float>(),
            param_time->GetMaxValue<float>(), param_time->GetStepSize<float>());
        ImGui::Text(param_time->FloatFormat().c_str(), time);
        ImGui::EndGroup();
        ImGui::SameLine();

        // Speed -------------------------------------------------------------------
        ImGui::BeginGroup();
        label = "speed";
        font_size = ImGui::CalcTextSize(label.c_str()).x;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (knob_size - font_size) / 2.0f);
        ImGui::TextUnformatted(label.c_str());
        ButtonWidgets::KnobButton(label, knob_size, speed, param_speed->GetMinValue<float>(),
            param_speed->GetMaxValue<float>(), param_speed->GetStepSize<float>());
        ImGui::Text(param_speed->FloatFormat().c_str(), speed);
        ImGui::EndGroup();

        // ------------------------------------------------------------------------

        param_play->SetValue(play);
        param_time->SetValue(time);
        param_speed->SetValue(speed);

        if (in_scope == Parameter::WidgetScope::GLOBAL) {
            // GLOBAL

            ImGui::End();
        } else if (in_scope == Parameter::WidgetScope::LOCAL) {
            /// LOCAL
            // Alternative LOCAL presentation

            // ImGui::EndGroup();
        }

        ImGui::PopID();

        return true;
    }

    return false;
}
