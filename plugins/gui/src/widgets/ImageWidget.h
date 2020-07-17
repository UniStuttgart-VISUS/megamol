/*
 * ImageWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_IMAGEWIDGET_INCLUDED
#define MEGAMOL_GUI_IMAGEWIDGET_INCLUDED


#include "GUIUtils.h"
#include "glowl/Texture2D.hpp"



#include "FileUtils.h"

#include "mmcore/misc/PngBitmapCodec.h"

#include "vislib/graphics/BitmapImage.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
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

    /**
     * Load texture from file.
     */
    static bool LoadTextureFromFile(const std::string& filename);

    /**
     * Load texture from data.
     */
    static bool CreateTexture(GLuint& inout_id, GLsizei width, GLsizei height, const float* data);

    /**
     * ...
     */
    bool Image(void);

    /**
     * ...
     */
    bool Button(void);


private:
    // VARIABLES --------------------------------------------------------------

    glowl::Texture2D texture;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_IMAGEWIDGET_INCLUDED
