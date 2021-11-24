/*
 * ImagePresentation_Sinks.hpp
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#pragma once
#include "ImageWrapper.h"
#define NON_GL_DEFAULT ;
#define NON_GL_EMPTY ;
#ifndef WITH_GL
#define NON_GL_DEFAULT = default;
#define NON_GL_EMPTY {}
#endif


namespace megamol::frontend {

    struct glfw_window_blit {
    unsigned int gl_fbo_handle = 0;
    unsigned int fbo_width = 0, fbo_height = 0;

    void set_framebuffer_size(unsigned int width, unsigned int height) {
        fbo_width = width;
        fbo_height = height;
    }

    glfw_window_blit() NON_GL_DEFAULT
    ~glfw_window_blit() NON_GL_DEFAULT

    void set_framebuffer_active() NON_GL_EMPTY
    void blit_texture(unsigned int gl_texture_handle, unsigned int texture_width, unsigned int texture_height) NON_GL_EMPTY


    };
}

