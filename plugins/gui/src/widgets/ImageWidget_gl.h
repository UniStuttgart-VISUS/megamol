/*
 * ImageWidget_gl.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_IMAGEWIDGET_GL_INCLUDED
#define MEGAMOL_GUI_IMAGEWIDGET_GL_INCLUDED
#pragma once


#include "mmcore/view/RenderUtils.h"
#include "widgets/HoverToolTip.h"


namespace megamol {
namespace gui {


    /**
     * OpenGL implementation of textured image widget.
     */
    class ImageWidget {
    public:
        ImageWidget(void);

        ~ImageWidget(void) = default;

        bool IsLoaded(void) {
            if (this->tex_ptr == nullptr)
                return false;
            return (this->tex_ptr->getName() != 0); // OpenGL texture id
        }

        bool LoadTextureFromData(int width, int height, float* data) {
            return megamol::core::view::RenderUtils::LoadTextureFromData(this->tex_ptr, width, height, data);
        }

        bool LoadTextureFromFile(const std::string& filename);

        /**
         * Draw texture as simple image.
         */
        void Widget(ImVec2 size, ImVec2 uv0 = ImVec2(0.0f, 0.0f), ImVec2 uv1 = ImVec2(1.0f, 1.0f));

        /**
         * Draw texture as button.
         */
        bool Button(const std::string& tooltip, ImVec2 size);

        /**
         * Return texture id for external usage.
         */
        GLuint GetTextureID(void) const {
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
