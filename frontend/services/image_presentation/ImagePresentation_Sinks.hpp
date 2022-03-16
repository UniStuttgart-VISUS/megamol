/*
 * ImagePresentation_Sinks.hpp
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "GL_STUB.h"

#include "ImageWrapper.h"


namespace megamol::frontend {

struct glfw_window_blit {
    unsigned int gl_fbo_handle = 0;
    unsigned int fbo_width = 0, fbo_height = 0;

    glfw_window_blit() GL_STUB();
    ~glfw_window_blit() GL_STUB();

    void set_framebuffer_active() GL_STUB();
    void set_framebuffer_size(unsigned int width, unsigned int height) GL_STUB();
    void blit_texture(unsigned int gl_texture_handle, unsigned int texture_width, unsigned int texture_height)
        GL_STUB();
};

} // namespace megamol::frontend
