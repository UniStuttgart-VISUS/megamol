/*
 * GLScreenshotSource.cpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "Screenshot_Service.hpp"

// to grab GL front buffer
#include <glad/gl.h>

void megamol::frontend_resources::GLScreenshotSource::set_read_buffer(ReadBuffer buffer) {
    m_read_buffer = buffer;
}

megamol::frontend_resources::ScreenshotImageData const&
megamol::frontend_resources::GLScreenshotSource::take_screenshot() const {
    // TODO: in FBO-based rendering the FBO object carries its size and we dont need to look it up
    // simpler and more correct approach would be to observe Framebuffer_Events resource
    // but this is our naive implementation for now
    GLint viewport_dims[4] = {0};
    glGetIntegerv(GL_VIEWPORT, viewport_dims);
    GLint fbWidth = viewport_dims[2];
    GLint fbHeight = viewport_dims[3];

    static ScreenshotImageData result;
    result.resize(static_cast<size_t>(fbWidth), static_cast<size_t>(fbHeight));

    GLint previous_read_fbo;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previous_read_fbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

    glReadBuffer(m_read_buffer);
    glReadPixels(0, 0, fbWidth, fbHeight, GL_RGBA, GL_UNSIGNED_BYTE, result.image.data());

    glBindFramebuffer(GL_READ_FRAMEBUFFER, previous_read_fbo);

    for (auto& pixel : result.image)
        pixel.a = megamol::frontend::Screenshot_Service::default_alpha_value;

    return result;
}
