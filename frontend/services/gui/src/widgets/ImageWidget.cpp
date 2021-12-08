/*
 * ImageWidget_gl.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "ImageWidget.h"

#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/utility/FileUtils.h"
#include "vislib/graphics/BitmapImage.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::ImageWidget::ImageWidget() : tooltip() {}

#ifdef WITH_GL
bool megamol::gui::ImageWidget::LoadTextureFromFile(
    const std::string& filename, GLint tex_min_filter, GLint tex_max_filter) {
    for (auto& resource_directory : megamol::gui::gui_resource_paths) {
        std::string filename_path =
            megamol::core::utility::FileUtils::SearchFileRecursive(resource_directory, filename);
        if (!filename_path.empty()) {
            return megamol::core_gl::utility::RenderUtils::LoadTextureFromFile(
                this->tex_ptr, filename_path, tex_min_filter, tex_max_filter);
        }
    }
    return false;
}
#else

bool megamol::gui::ImageWidget::LoadTextureFromFile(const std::string& filename) {

    if (filename.empty())
        return false;

    vislib::graphics::BitmapImage img;
    sg::graphics::PngBitmapCodec pbc;
    pbc.Image() = &img;
    std::vector<char> buf;
    size_t size = 0;
    for (auto& resource_directory : megamol::gui::gui_resource_paths) {
        std::string filename_path =
            megamol::core::utility::FileUtils::SearchFileRecursive(resource_directory, filename);
        if (!filename_path.empty()) {
            if (megamol::core::utility::FileUtils::LoadRawFile(filename, buf)) {
                if (pbc.Load(static_cast<void*>(buf.data()), buf.size())) {
                    img.Convert(vislib::graphics::BitmapImage::TemplateFloatRGBA);
                    cpu_tex_ptr = std::make_shared<CPUTexture2D<float>>();
                    cpu_tex_ptr->width = img.Width();
                    cpu_tex_ptr->height = img.Height();
                    const uint32_t size = cpu_tex_ptr->width * cpu_tex_ptr->height * 4;
                    std::vector<float> data = {img.PeekDataAs<FLOAT>(), img.PeekDataAs<FLOAT>() + size};
                    cpu_tex_ptr->data = std::move(data);
                } else {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "Unable to read texture: %s [%s, %s, line %d]\n", filename.c_str(), __FILE__, __FUNCTION__,
                        __LINE__);
                }
            }
        }
    }

    return (cpu_tex_ptr != nullptr);
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

    ImGui::Image(getImTextureID(), size, uv0, uv1, ImVec4(1.0f, 1.0f, 1.0f, 1.0f), style.Colors[ImGuiCol_Border]);
}


bool megamol::gui::ImageWidget::Button(const std::string& tooltip_text, ImVec2 size) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    if (!this->IsLoaded()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No texture loaded. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    bool retval = ImGui::ImageButton(getImTextureID(), size, ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f), 1,
        style.Colors[ImGuiCol_Button], style.Colors[ImGuiCol_ButtonActive]);
    this->tooltip.ToolTip(tooltip_text, ImGui::GetItemID(), 1.0f, 5.0f);

    return retval;
}
