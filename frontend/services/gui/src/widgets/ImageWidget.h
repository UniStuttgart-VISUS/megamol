/*
 * ImageWidget_gl.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_IMAGEWIDGET_GL_INCLUDED
#define MEGAMOL_GUI_IMAGEWIDGET_GL_INCLUDED
#pragma once

#ifdef WITH_GL
#include "mmcore_gl/utility/RenderUtils.h"
#endif // WITH_GL
#include "widgets/HoverToolTip.h"


namespace megamol {
namespace gui {

template<typename T>
struct CPUTexture2D {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<T> data;
};


/** ************************************************************************
 * OpenGL implementation of textured image widget
 */
class ImageWidget {
public:
    ImageWidget();
    ~ImageWidget() = default;

#ifdef WITH_GL
    bool IsLoaded() {
        if (this->tex_ptr == nullptr)
            return false;
        return (this->tex_ptr->getName() != 0); // OpenGL texture id
    }

    bool LoadTextureFromData(int width, int height, float* data, GLint tex_min_filter = GL_NEAREST_MIPMAP_LINEAR,
        GLint tex_max_filter = GL_LINEAR) {
        return megamol::core_gl::utility::RenderUtils::LoadTextureFromData(
            this->tex_ptr, width, height, data, tex_min_filter, tex_max_filter);
    }

    bool LoadTextureFromFile(
        const std::string& filename, GLint tex_min_filter = GL_NEAREST_MIPMAP_LINEAR, GLint tex_max_filter = GL_LINEAR);

    /**
     * Return texture id for external usage.
     */
    GLuint GetTextureID() const {
        return ((this->tex_ptr != nullptr) ? (this->tex_ptr->getName()) : (0));
    }

#else
    bool IsLoaded() const {
        if (this->cpu_tex_ptr == nullptr)
            return false;
        return true;
    }

    bool LoadTextureFromData(int width, int height, float* data);

    bool LoadTextureFromFile(const std::string& filename);

#endif

    /**
     * Draw texture as simple image.
     */
    void Widget(ImVec2 size, ImVec2 uv0 = ImVec2(0.0f, 0.0f), ImVec2 uv1 = ImVec2(1.0f, 1.0f));

    /**
     * Draw texture as button.
     */
    bool Button(const std::string& tooltip_text, ImVec2 size);

private:
    // VARIABLES --------------------------------------------------------------

    HoverToolTip tooltip;

#ifdef WITH_GL

    std::shared_ptr<glowl::Texture2D> tex_ptr = nullptr;

    ImTextureID getImTextureID() {
        return reinterpret_cast<ImTextureID>(this->tex_ptr->getName());
    }

#else

    std::shared_ptr<CPUTexture2D<float>> cpu_tex_ptr = nullptr;

    ImTextureID getImTextureID() {
        return reinterpret_cast<ImTextureID>(this->cpu_tex_ptr->data.data());
    }

#endif

};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_IMAGEWIDGET_GL_INCLUDED
