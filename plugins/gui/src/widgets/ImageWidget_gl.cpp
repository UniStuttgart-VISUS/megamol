/*
 * ImageWidget_gl.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ImageWidget_gl.h"


using namespace megamol;
using namespace megamol::gui;


ImageWidget::ImageWidget(void) : tex_ptr(nullptr), tooltip() {}


bool megamol::gui::ImageWidget::LoadTextureFromFile(const std::string& filename) {

    if (filename.empty()) return false;
    bool retval = false;

    static vislib::graphics::BitmapImage img;
    static sg::graphics::PngBitmapCodec pbc;
    pbc.Image() = &img;
    void* buf = nullptr;
    size_t size = 0;

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    size = megamol::gui::FileUtils::LoadRawFile(filename, &buf);
    if (size > 0) {
        if (pbc.Load(buf, size)) {
            img.Convert(vislib::graphics::BitmapImage::TemplateFloatRGBA);
            retval = megamol::gui::ImageWidget::LoadTextureFromData(img.Width(), img.Height(), img.PeekDataAs<FLOAT>());
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Unable to read texture: %s [%s, %s, line %d]\n", filename.c_str(), __FILE__, __FUNCTION__,
                __LINE__);
            retval = false;
        }
    } else {
        retval = false;
    }

    ARY_SAFE_DELETE(buf);
    return retval;
}


bool megamol::gui::ImageWidget::LoadTextureFromData(int width, int height, float* data) {

    if (data == nullptr) return false;

    glowl::TextureLayout tex_layout(GL_RGBA32F, width, height, 1, GL_RGBA, GL_FLOAT, 1);
    if (this->tex_ptr == nullptr) {
        this->tex_ptr =
            std::make_shared<glowl::Texture2D>("image_widget", tex_layout, static_cast<GLvoid*>(data), false);
    } else {
        // Reload data
        this->tex_ptr->reload(tex_layout, static_cast<GLvoid*>(data), false);
    }

    return true;
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


bool megamol::gui::ImageWidget::Button(const std::string& tooltip, ImVec2 size) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    if (!this->IsLoaded()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No texture loaded. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    bool retval = ImGui::ImageButton(reinterpret_cast<ImTextureID>(this->tex_ptr->getName()), size, ImVec2(0.0f, 0.0f),
        ImVec2(1.0f, 1.0f), 1, style.Colors[ImGuiCol_Button], style.Colors[ImGuiCol_ButtonActive]);
    this->tooltip.ToolTip(tooltip, ImGui::GetItemID(), 1.0f, 5.0f);

    return retval;
}
