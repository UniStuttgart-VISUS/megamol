/*
 * GLScreenshotSource.cpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "Screenshot_Service.hpp"

// to grab GL front buffer
#include <glad/glad.h>

void megamol::frontend_resources::GLScreenshotSource::set_read_buffer(ReadBuffer buffer) {
    m_read_buffer = buffer;
    GLenum read_buffer;

    switch (buffer) {
    default:
        [[fallthrough]];
    case ReadBuffer::FRONT:
            read_buffer = GL_FRONT;
        break;
    case ReadBuffer::BACK:
            read_buffer = GL_BACK;
        break;
    case ReadBuffer::COLOR_ATT0:
            read_buffer = GL_COLOR_ATTACHMENT0;
        break;
    case ReadBuffer::COLOR_ATT1:
            read_buffer = GL_COLOR_ATTACHMENT0+1;
        break;
    case ReadBuffer::COLOR_ATT2:
            read_buffer = GL_COLOR_ATTACHMENT0+2;
        break;
    case ReadBuffer::COLOR_ATT3:
            read_buffer = GL_COLOR_ATTACHMENT0+3;
        break;
    }
}

megamol::frontend_resources::ScreenshotImageData const& megamol::frontend_resources::GLScreenshotSource::take_screenshot() const {
    // TODO: in FBO-based rendering the FBO object carries its size and we dont need to look it up
    // simpler and more correct approach would be to observe Framebuffer_Events resource
    // but this is our naive implementation for now
    GLint viewport_dims[4] = {0};
    glGetIntegerv(GL_VIEWPORT, viewport_dims);
    GLint fbWidth = viewport_dims[2];
    GLint fbHeight = viewport_dims[3];

    static ScreenshotImageData result;
    result.resize(static_cast<size_t>(fbWidth), static_cast<size_t>(fbHeight));

    glReadBuffer(m_read_buffer);
    glReadPixels(0, 0, fbWidth, fbHeight, GL_RGBA, GL_UNSIGNED_BYTE, result.image.data());

    for (auto& pixel : result.image)
        pixel.a = megamol::frontend::Screenshot_Service::default_alpha_value;

    return result;
}

