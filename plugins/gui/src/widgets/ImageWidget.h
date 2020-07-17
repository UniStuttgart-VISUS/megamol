/*
 * ImageWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_IMAGEWIDGET_INCLUDED
#define MEGAMOL_GUI_IMAGEWIDGET_INCLUDED


#include "glowl/Texture2D.hpp"

#include "FileUtils.h"
#include "GUIUtils.h"

#include "mmcore/misc/PngBitmapCodec.h"

#include "vislib/graphics/BitmapImage.h"
//#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/sys/Log.h"


namespace megamol {
namespace gui {


/**
 * String search widget.
 */
class ImageWidget {
public:
    ImageWidget(void);

    ~ImageWidget(void) = default;

    bool IsLoaded(void) {
        if (this->tex_ptr == nullptr) return false;
        return (this->tex_ptr->getName() != 0);
    }

    /**
     * Load texture from file.
     */
    static bool LoadTextureFromFile(const std::string& filename);

    /**
     * Load texture from data.
     */
    static bool LoadTextureFromData(GLsizei width, GLsizei height, const float* data);

    /**
     * ...
     */
    bool Widget(void);

    /**
     * ...
     */
    bool Button(const std::string& label);


private:
    // VARIABLES --------------------------------------------------------------

    std::shared_ptr<glowl::Texture2D> tex_ptr;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_IMAGEWIDGET_INCLUDED
