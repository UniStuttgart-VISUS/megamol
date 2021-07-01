/*
 * ImageWidget_gl.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_IMAGEWIDGET_GL_INCLUDED
#define MEGAMOL_GUI_IMAGEWIDGET_GL_INCLUDED
#pragma once


#include "mmcore/utility/RenderUtils.h"
#include "widgets/HoverToolTip.h"


namespace megamol {
namespace gui {


    /** ************************************************************************
     * OpenGL implementation of textured image widget
     */
    class ImageWidget {
    public:
        ImageWidget();
        ~ImageWidget() = default;

        bool IsLoaded() {
            if (this->tex_ptr == nullptr)
                return false;
            return (this->tex_ptr->getName() != 0); // OpenGL texture id
        }

        bool LoadTextureFromData(int width, int height, float* data, GLint tex_min_filter = GL_NEAREST_MIPMAP_LINEAR,
            GLint tex_max_filter = GL_LINEAR) {
            return megamol::core::utility::RenderUtils::LoadTextureFromData(
                this->tex_ptr, width, height, data, tex_min_filter, tex_max_filter);
        }

        bool LoadTextureFromFile(const std::string& filename, GLint tex_min_filter = GL_NEAREST_MIPMAP_LINEAR,
            GLint tex_max_filter = GL_LINEAR);

        /**
         * Draw texture as simple image.
         */
        void Widget(ImVec2 size, ImVec2 uv0 = ImVec2(0.0f, 0.0f), ImVec2 uv1 = ImVec2(1.0f, 1.0f));

        /**
         * Draw texture as button.
         */
        bool Button(const std::string& tooltip_text, ImVec2 size);

        /**
         * Return texture id for external usage.
         */
        GLuint GetTextureID() const {
            return ((this->tex_ptr != nullptr) ? (this->tex_ptr->getName()) : (0));
        }

    private:
        // VARIABLES --------------------------------------------------------------

        std::shared_ptr<glowl::Texture2D> tex_ptr;

        // Widgets
        HoverToolTip tooltip;
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_IMAGEWIDGET_GL_INCLUDED
