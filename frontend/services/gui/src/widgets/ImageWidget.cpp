/*
 * ImageWidget_gl.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "ImageWidget.h"
#include "mmcore/utility/FileUtils.h"
#include "vislib/graphics/BitmapImage.h"
#include "vislib/graphics/PngBitmapCodec.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::ImageWidget::ImageWidget() : tooltip() {}


#ifdef WITH_GL

bool megamol::gui::ImageWidget::LoadTextureFromFile(
    const std::string& filename, const std::string& toggle_filename, GLint tex_min_filter, GLint tex_max_filter) {

    bool retval = true;

    auto load_texture_from_file = [&](const std::string& fn, std::shared_ptr<glowl::Texture2D>& tp) {
        for (auto& resource_directory : megamol::gui::gui_resource_paths) {
            std::string filename_path = megamol::core::utility::FileUtils::SearchFileRecursive(resource_directory, fn);
            if (!filename_path.empty()) {
                retval &= megamol::core_gl::utility::RenderUtils::LoadTextureFromFile(
                    tp, filename_path, tex_min_filter, tex_max_filter);
                break;
            }
        }
    };

    // Primary texture
    load_texture_from_file(filename, this->tex_ptr);

    // Secondary toggle button texture
    if (!toggle_filename.empty()) {
        load_texture_from_file(toggle_filename, this->toggle_tex_ptr);
    }

    return retval;
}

#else

bool megamol::gui::ImageWidget::LoadTextureFromFile(const std::string& filename, const std::string& toggle_filename) {

    bool retval = true;

    if (filename.empty())
        return false;

    vislib::graphics::BitmapImage img;
    sg::graphics::PngBitmapCodec pbc;
    pbc.Image() = &img;
    std::vector<char> buf;
    size_t size = 0;

    auto load_texture_from_file = [&](const std::string& fn, std::shared_ptr<CPUTexture2D<float>>& tp) {
        for (auto& resource_directory : megamol::gui::gui_resource_paths) {
            std::string filename_path = megamol::core::utility::FileUtils::SearchFileRecursive(resource_directory, fn);
            if (!filename_path.empty()) {
                if (megamol::core::utility::FileUtils::LoadRawFile(filename, buf)) {
                    if (pbc.Load(static_cast<void*>(buf.data()), buf.size())) {
                        img.Convert(vislib::graphics::BitmapImage::TemplateFloatRGBA);
                        tp = std::make_shared<CPUTexture2D<float>>();
                        tp->width = img.Width();
                        tp->height = img.Height();
                        const uint32_t size = tp->width * tp->height * 4;
                        std::vector<float> data = {img.PeekDataAs<FLOAT>(), img.PeekDataAs<FLOAT>() + size};
                        tp->data = std::move(data);
                        retval &= true;
                        break;
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "Unable to read texture: %s [%s, %s, line %d]\n", filename.c_str(), __FILE__, __FUNCTION__,
                            __LINE__);
                    }
                }
            }
        }
    };

    // Primary texture
    load_texture_from_file(filename, this->cpu_tex_ptr);

    // Secondary toggle button texture
    if (!toggle_filename.empty()) {
        load_texture_from_file(toggle_filename, this->cpu_toggle_tex_ptr);
    }

    return retval;
}

bool megamol::gui::ImageWidget::LoadTextureFromData(int width, int height, float* data) {

    cpu_tex_ptr = std::make_shared<CPUTexture2D<float>>();
    cpu_tex_ptr->width = width;
    cpu_tex_ptr->height = height;
    const uint32_t size = cpu_tex_ptr->width * cpu_tex_ptr->height * 4;
    std::vector<float> data_vec = {data, data + size};
    cpu_tex_ptr->data = std::move(data_vec);
    return true;
}

#endif


void megamol::gui::ImageWidget::Widget(ImVec2 size, ImVec2 uv0, ImVec2 uv1) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    if (!this->IsLoaded()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No texture loaded. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    ImGui::Image(this->getImTextureID(), size, uv0, uv1, ImVec4(1.0f, 1.0f, 1.0f, 1.0f), style.Colors[ImGuiCol_Border]);
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

    bool retval = ImGui::ImageButton(this->getImTextureID(), size, ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f), 1, bg, fg);
    if (!tooltip_text.empty()) {
        this->tooltip.ToolTip(tooltip_text, ImGui::GetItemID(), 0.5f, 5.0f);
    }

    return retval;
}


bool megamol::gui::ImageWidget::ToggleButton(
    bool& toggle, const std::string& tooltip_text, const std::string& toggle_tooltip_text, ImVec2 size) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    if (!this->IsLoaded() || !this->isToggleLoaded()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Not all required textures are loaded. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    auto bg = style.Colors[ImGuiCol_Button];
    auto fg = style.Colors[ImGuiCol_Text];

    bool retval = false;
    auto button_tooltip_text = tooltip_text;
    auto im_texture_id = this->getImTextureID();
    if (toggle) {
        button_tooltip_text = toggle_tooltip_text;
        im_texture_id = this->getToggleImTextureID();
    }
    if (ImGui::ImageButton(im_texture_id, size, ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f), 1, bg, fg)) {
        toggle = !toggle;
        retval = true;
    }
    if (!button_tooltip_text.empty()) {
        this->tooltip.ToolTip(button_tooltip_text, ImGui::GetItemID(), 0.5f, 5.0f);
    }

    return retval;
}
