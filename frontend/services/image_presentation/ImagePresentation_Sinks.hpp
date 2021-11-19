/*
 * ImagePresentation_Sinks.hpp
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#pragma once
#include "ImageWrapper.h"

namespace megamol::frontend {

    struct glfw_window_blit {
    unsigned int gl_fbo_handle = 0;
    unsigned int fbo_width = 0, fbo_height = 0;

    void set_framebuffer_size(unsigned int width, unsigned int height) {
        fbo_width = width;
        fbo_height = height;
    }
#ifdef WITH_GL
    glfw_window_blit();
    ~glfw_window_blit();

    void set_framebuffer_active();
    void blit_texture(unsigned int gl_texture_handle, unsigned int texture_width, unsigned int texture_height);
#else
    glfw_window_blit() = default;
    ~glfw_window_blit() = default;

    void set_framebuffer_active() {}
    void blit_texture(unsigned int gl_texture_handle, unsigned int texture_width, unsigned int texture_height) {}
#endif
    };
}

