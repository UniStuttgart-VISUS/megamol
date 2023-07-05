/*
 * CinematicUtils.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "cinematic_gl/CinematicUtils.h"

#include <imgui.h>
#include <imgui_internal.h>

using namespace megamol::cinematic_gl;


CinematicUtils::CinematicUtils()
        : core_gl::utility::RenderUtils()
        , font(megamol::core::utility::SDFFont::PRESET_ROBOTO_SANS)
        , init_once(false)
        , menu_font_size(20.0f)
        , menu_height(20.0f)
        , background_color(0.0f, 0.0f, 0.0f, 0.0f)
        , hotkey_window_setup_once(true) {}


CinematicUtils::~CinematicUtils() {}


bool CinematicUtils::Initialise(frontend_resources::RuntimeConfig const& runtimeConf) {

    if (this->init_once) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "Primitive rendering has already been initialized. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }

    // Initialise font
    if (!this->font.Initialise(runtimeConf)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Couldn't initialize the font. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->font.SetBatchDrawMode(true);

    // Initialise rendering
    if (!this->InitPrimitiveRendering(runtimeConf)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Couldn't initialize primitive rendering. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    this->init_once = true;

    return true;
}


const glm::vec4 CinematicUtils::Color(CinematicUtils::Colors c) const {

    glm::vec4 color = {0.0f, 0.0f, 0.0f, 0.0f};

    switch (c) {
    case (CinematicUtils::Colors::BACKGROUND):
        color = this->background_color;
        break;
    case (CinematicUtils::Colors::FOREGROUND): {
        glm::vec4 foreground = {1.0f, 1.0f, 1.0f, 1.0f};
        color = this->background_color;
        for (unsigned int i = 0; i < 3; i++) {
            foreground[i] -= color[i];
        }
        color = foreground;
    } break;
    case (CinematicUtils::Colors::KEYFRAME):
        color = {0.7f, 0.7f, 1.0f, 1.0f};
        break;
    case (CinematicUtils::Colors::KEYFRAME_DRAGGED):
        color = {0.5f, 0.5f, 1.0f, 1.0f};
        break;
    case (CinematicUtils::Colors::KEYFRAME_SELECTED):
        color = {0.2f, 0.2f, 1.0f, 1.0f};
        break;
    case (CinematicUtils::Colors::KEYFRAME_SPLINE):
        color = {0.4f, 0.4f, 1.0f, 1.0f};
        break;
    case (CinematicUtils::Colors::MENU):
        color = {0.0f, 0.0f, 0.5f, 1.0f};
        break;
    case (CinematicUtils::Colors::FONT):
        color = {1.0f, 1.0f, 1.0f, 1.0f};
        if (CinematicUtils::lightness(this->background_color) > 0.5f) {
            color = {0.0f, 0.0f, 0.0f, 1.0f};
        }
        break;
    case (CinematicUtils::Colors::FONT_HIGHLIGHT):
        color = {0.75f, 0.75f, 0.0f, 1.0f};
        break;
    case (CinematicUtils::Colors::LETTER_BOX):
        color = {1.0f, 1.0f, 1.0f, 1.0f};
        if (CinematicUtils::lightness(this->background_color) > 0.5f) {
            color = {0.0f, 0.0f, 0.0f, 1.0f};
        }
        break;
    case (CinematicUtils::Colors::FRAME_MARKER):
        color = {1.0f, 0.6f, 0.6f, 1.0f};
        break;
    case (CinematicUtils::Colors::MANIPULATOR_X):
        color = {1.0f, 0.0f, 0.0f, 1.0f};
        break;
    case (CinematicUtils::Colors::MANIPULATOR_Y):
        color = {0.0f, 1.0f, 0.0f, 1.0f};
        break;
    case (CinematicUtils::Colors::MANIPULATOR_Z):
        color = {0.0f, 0.0f, 1.0f, 1.0f};
        break;
    case (CinematicUtils::Colors::MANIPULATOR_VECTOR):
        color = {0.0f, 1.0f, 1.0f, 1.0f};
        break;
    case (CinematicUtils::Colors::MANIPULATOR_ROTATION):
        color = {1.0f, 1.0f, 0.0f, 1.0f};
        break;
    case (CinematicUtils::Colors::MANIPULATOR_CTRLPOINT):
        color = {1.0f, 0.0f, 1.0f, 1.0f};
        break;
    default:
        break;
    }

    return color;
}


void CinematicUtils::PushMenu(const glm::mat4& ortho, const std::string& left_label, const std::string& middle_label,
    const std::string& right_label, glm::vec2 dim_vp, float depth) {

    this->gui_update();

    // Push menu background quad
    this->PushQuadPrimitive(glm::vec3(0.0f, dim_vp.y, depth), glm::vec3(0.0f, dim_vp.y - this->menu_height, depth),
        glm::vec3(dim_vp.x, dim_vp.y - this->menu_height, depth), glm::vec3(dim_vp.x, dim_vp.y, depth),
        this->Color(CinematicUtils::Colors::MENU));

    // Push menu labels
    float vpWhalf = dim_vp.x / 2.0f;
    float new_font_size = this->menu_font_size;
    float leftLabelWidth = this->font.LineWidth(this->menu_font_size, left_label.c_str());
    float midleftLabelWidth = this->font.LineWidth(this->menu_font_size, middle_label.c_str());
    float rightLabelWidth = this->font.LineWidth(this->menu_font_size, right_label.c_str());
    while (((leftLabelWidth + midleftLabelWidth / 2.0f) > vpWhalf) ||
           ((rightLabelWidth + midleftLabelWidth / 2.0f) > vpWhalf)) {
        new_font_size -= 0.5f;
        leftLabelWidth = this->font.LineWidth(new_font_size, left_label.c_str());
        midleftLabelWidth = this->font.LineWidth(new_font_size, middle_label.c_str());
        rightLabelWidth = this->font.LineWidth(new_font_size, right_label.c_str());
    }
    float textPosY = dim_vp.y - menu_height + this->menu_font_size;
    auto current_back_color = this->Color(CinematicUtils::Colors::BACKGROUND);
    this->SetBackgroundColor(this->Color(CinematicUtils::Colors::MENU));
    auto color = this->Color(CinematicUtils::Colors::FONT);

    this->font.DrawString(ortho, glm::value_ptr(color), 0.0f, textPosY, new_font_size, false, left_label.c_str(),
        megamol::core::utility::SDFFont::ALIGN_LEFT_TOP);
    this->font.DrawString(ortho, glm::value_ptr(color), (dim_vp.x - midleftLabelWidth) / 2.0f, textPosY, new_font_size,
        false, middle_label.c_str(), megamol::core::utility::SDFFont::ALIGN_LEFT_TOP);
    this->font.DrawString(ortho, glm::value_ptr(color), (dim_vp.x - rightLabelWidth), textPosY, new_font_size, false,
        right_label.c_str(), megamol::core::utility::SDFFont::ALIGN_LEFT_TOP);
    this->SetBackgroundColor(current_back_color);
}


void CinematicUtils::HotkeyWindow(bool& inout_show, const glm::mat4& ortho, glm::vec2 dim_vp) {

    this->gui_update();

    bool valid_imgui_scope =
        ((ImGui::GetCurrentContext() != nullptr) ? (ImGui::GetCurrentContext()->WithinFrameScope) : (false));

    if (inout_show && valid_imgui_scope) {
        if (this->hotkey_window_setup_once) {
            ImGui::SetNextWindowPos(ImVec2(0.0f, this->menu_height));
            this->hotkey_window_setup_once = false;
        }
        auto window_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse;
        auto header_flags = ImGuiTreeNodeFlags_DefaultOpen;
        auto table_flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableColumnFlags_NoResize;
        auto column_flags = ImGuiTableColumnFlags_WidthStretch;

        if (ImGui::Begin("[Cinematic] HOTKEYS", &inout_show, window_flags)) {
            if (ImGui::CollapsingHeader("  GLOBAL##cinematic_header", header_flags)) {
                if (ImGui::BeginTable("cinematic_global_hotkeys", 2, table_flags)) {
                    ImGui::TableSetupColumn("", column_flags);
                    this->gui_table_row("SHIFT + A", "Apply current settings to selected/new keyframe.");
                    this->gui_table_row("SHIFT + D", "Delete selected keyframe.");
                    this->gui_table_row("SHIFT + S", "Save keyframes to file.");
                    this->gui_table_row("SHIFT + L", "Load keyframes from file.");
                    this->gui_table_row("SHIFT + Z", "Undo keyframe changes (QUERTY keyboard layout).");
                    this->gui_table_row("SHIFT + Y", "Redo keyframe changes (QUERTY keyboard layout).");
                    this->gui_table_row("LEFT Mouse Button", "Select keyframe or drag manipulator.");
                    ImGui::EndTable();
                }
            }
            if (ImGui::CollapsingHeader("  TRACKING SHOT##cinematic_header", header_flags)) {
                if (ImGui::BeginTable("cinematic_tracking_shot_hotkeys", 2, table_flags)) {
                    ImGui::TableSetupColumn("", column_flags);
                    this->gui_table_row("SHIFT + Q", "Toggle different manipulators for the selected keyframe.");
                    this->gui_table_row("SHIFT + W", "Show manipulators inside/outside of model bounding box.");
                    this->gui_table_row("SHIFT + U", "Reset look-at vector of selected keyframe.");
                    ImGui::EndTable();
                }
            }
            if (ImGui::CollapsingHeader("  CINEMATIC##cinematic_header", header_flags)) {
                if (ImGui::BeginTable("cinematic_cinematic_hotkeys", 2, table_flags)) {
                    ImGui::TableSetupColumn("", column_flags);
                    this->gui_table_row("SHIFT + R", "Start/Stop rendering complete animation.");
                    this->gui_table_row("SHIFT + SPACE", "Start/Stop animation preview.");
                    ImGui::EndTable();
                }
            }
            if (ImGui::CollapsingHeader("  TIMELINE##cinematic_header", header_flags)) {
                if (ImGui::BeginTable("cinematic_timeline_hotkeys", 2, table_flags)) {
                    ImGui::TableSetupColumn("", column_flags);
                    this->gui_table_row("SHIFT + RIGHT/LEFT Arrow", "Move selected keyframe on animation time axis.");
                    this->gui_table_row("SHIFT + F", "Snap all keyframes to animation frames.");
                    this->gui_table_row("SHIFT + G", "Snap all keyframes to simulation frames.");
                    this->gui_table_row("SHIFT + T", "Linearize simulation time between two keyframes.");
                    this->gui_table_row("SHIFT + P", "Reset shifted and scaled time axes.");
                    this->gui_table_row("LEFT Mouse Button", "Select keyframe.");
                    this->gui_table_row("MIDDLE Mouse Button", "Axes scaling in mouse direction.");
                    this->gui_table_row("RIGHT Mouse Button", "Drag & drop keyframe / pan axes.");
                    /// TODO XXX Calcualation is not correct yet ...
                    //this->gui_table_row("SHIFT + v","Set same velocity between all keyframes (Experimental).");
                    ImGui::EndTable();
                }
            }
            ImGui::End();
        }
    }
}


void CinematicUtils::Push2DText(const glm::mat4& ortho, const std::string& text, float x, float y) {

    this->gui_update();
    auto color = this->Color(CinematicUtils::Colors::FONT);
    this->font.DrawString(ortho, glm::value_ptr(color), x, y, this->menu_font_size, false, text.c_str(),
        megamol::core::utility::SDFFont::ALIGN_LEFT_TOP);
}


void CinematicUtils::DrawAll(const glm::mat4& mvp, glm::vec2 dim_vp) {

    if (!this->init_once) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Cinematic utilities must be initialized before drawing. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
        return;
    }

    this->DrawAllPrimitives(mvp, dim_vp);

    glDisable(GL_DEPTH_TEST);
    this->font.BatchDrawString(mvp);
    this->font.ClearBatchDrawCache();
    glEnable(GL_DEPTH_TEST);
}


float CinematicUtils::GetTextLineHeight() {

    this->gui_update();
    return this->font.LineHeight(this->menu_font_size);
}


float CinematicUtils::GetTextLineWidth(const std::string& text_line) {

    this->gui_update();
    return this->font.LineWidth(this->menu_font_size, text_line.c_str());
}


void CinematicUtils::SetTextRotation(float a, glm::vec3 vec) {

    this->font.SetRotation(a, vec);
}

void CinematicUtils::ResetTextRotation() {

    this->font.ResetRotation();
}


const float CinematicUtils::lightness(glm::vec4 background) const {

    return ((glm::max(background[0], glm::max(background[1], background[2])) +
                glm::min(background[0], glm::min(background[1], background[2]))) /
            2.0f);
}


void CinematicUtils::gui_update() {

    this->menu_font_size = ImGui::GetFontSize() * 1.5f;
    if (this->menu_font_size == 0.0f) {
        this->menu_font_size = 20.0f;
        //int vp[4];
        //glGetIntegerv(GL_VIEWPORT, vp);
        // font size = 2% of viewport height
        //this->menu_font_size = static_cast<float>(vp[2]) * 0.02f;
    }
    this->menu_height = this->menu_font_size; // +ImGui::GetFrameHeightWithSpacing();
}


void CinematicUtils::gui_table_row(const char* left, const char* right) {

    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(left);
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(right);
}
