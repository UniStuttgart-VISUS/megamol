/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "GL_STUB.h"
#include "ImageWrapper.h"


namespace megamol::frontend {

struct glfw_window_blit {
    unsigned int gl_fbo_handle = 0;
    unsigned int fbo_width = 0, fbo_height = 0;

    glfw_window_blit() GL_VSTUB();
    ~glfw_window_blit() GL_VSTUB();

    void set_framebuffer_active() GL_VSTUB();
    void set_framebuffer_size(unsigned int width, unsigned int height) GL_VSTUB();
    void blit_texture(unsigned int gl_texture_handle, unsigned int texture_width, unsigned int texture_height)
        GL_VSTUB();
};

} // namespace megamol::frontend
