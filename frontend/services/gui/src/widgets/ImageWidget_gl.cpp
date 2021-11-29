/*
 * ImageWidget_gl.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "ImageWidget_gl.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::ImageWidget::ImageWidget() : tex_ptr(nullptr), toggle_tex_ptr(nullptr), toggle_button_toggled(false), tooltip() {}


bool megamol::gui::ImageWidget::LoadTextureFromFile(const std::string& filename, const std::string& toggle_filename, GLint tex_min_filter, GLint tex_max_filter) {
    bool retval = false;
    // Primary texture
    for (auto& resource_directory : megamol::gui::gui_resource_paths) {
        std::string filename_path =
            megamol::core::utility::FileUtils::SearchFileRecursive(resource_directory, filename);
        if (!filename_path.empty()) {
            retval = megamol::core::utility::RenderUtils::LoadTextureFromFile(
                this->tex_ptr, filename_path, tex_min_filter, tex_max_filter);
            break;
        }
    }
    // Secondary toggle button texture
    if (!toggle_filename.empty()) {
        for (auto& resource_directory : megamol::gui::gui_resource_paths) {
            std::string filename_path =
                megamol::core::utility::FileUtils::SearchFileRecursive(resource_directory, toggle_filename);
            if (!filename_path.empty()) {
                retval &= megamol::core::utility::RenderUtils::LoadTextureFromFile(
                    this->toggle_tex_ptr, filename_path, tex_min_filter, tex_max_filter);
                break;
            }
        }
    }
    return retval;
}


void megamol::gui::ImageWidget::Widget(ImVec2 size, ImVec2 uv0, ImVec2 uv1) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    if (!this->IsLoaded()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No texture loaded. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    ImGui::Image(reinterpret_cast<ImTextureID>(this->tex_ptr->getName()), size, uv0, uv1,
        ImVec4(1.0f, 1.0f, 1.0f, 1.0f), style.Colors[ImGuiCol_Border]);
}


bool megamol::gui::ImageWidget::Button(const std::string& tooltip_text, ImVec2 size) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    if (!this->IsLoaded()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No texture loaded. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    auto bg = style.Colors[ImGuiCol_Button];
    auto fg = style.Colors[ImGuiCol_Text];

    bool retval = ImGui::ImageButton(reinterpret_cast<ImTextureID>(this->tex_ptr->getName()), size, ImVec2(0.0f, 0.0f),
        ImVec2(1.0f, 1.0f), 1, bg, fg);
    if (!tooltip_text.empty()) {
        this->tooltip.ToolTip(tooltip_text, ImGui::GetItemID(), 0.5f, 5.0f);
    }

    return retval;
}


bool megamol::gui::ImageWidget::ToggleButton(const std::string& tooltip_text, const std::string& toggle_tooltip_text, ImVec2 size) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    bool toggle_tex_loaded = (this->toggle_tex_ptr != nullptr) && (this->toggle_tex_ptr->getName() != 0);
    if (!this->IsLoaded() || !toggle_tex_loaded) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Not all required textures are loaded. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    auto bg = style.Colors[ImGuiCol_Button];
    auto fg = style.Colors[ImGuiCol_Text];

    bool retval = false;
    auto button_tooltip_text = tooltip_text;
    auto texture_id = this->tex_ptr->getName();
    if (this->toggle_button_toggled) {
        button_tooltip_text = toggle_tooltip_text;
        texture_id = this->toggle_tex_ptr->getName();
    }
    if (ImGui::ImageButton(reinterpret_cast<ImTextureID>(texture_id), size, ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f), 1, bg, fg)) {
        this->toggle_button_toggled = !this->toggle_button_toggled;
        retval = true;
    }
    if (!button_tooltip_text.empty()) {
        this->tooltip.ToolTip(button_tooltip_text, ImGui::GetItemID(), 0.5f, 5.0f);
    }

    return retval;
}