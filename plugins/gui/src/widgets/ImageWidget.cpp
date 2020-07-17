/*
 * ImageWidget.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ImageWidget.h"


using namespace megamol;
using namespace megamol::gui;


ImageWidget::ImageWidget(void) : tex_ptr(nullptr) {}


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
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to read texture: %s [%s, %s, line %d]\n", filename.c_str(), __FILE__, __FUNCTION__, __LINE__);
            retval = false;
        }
    } else {
        retval = false;
    }

    ARY_SAFE_DELETE(buf);
    return retval;
}


bool megamol::gui::ImageWidget::LoadTextureFromData(GLsizei width, GLsizei height, const float* data) {
    if (data == nullptr) return false;

    /*
    // Delete old texture.
    if (inout_id != 0) {
        glDeleteTextures(1, &inout_id);
    }
    inout_id = 0;

    // Upload texture.
    glGenTextures(1, &inout_id);
    glBindTexture(GL_TEXTURE_2D, inout_id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, data);

    glBindTexture(GL_TEXTURE_2D, 0);
    */
    return true;
}
