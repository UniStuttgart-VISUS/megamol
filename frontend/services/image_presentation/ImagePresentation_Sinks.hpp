/*
 * ImagePresentation_Sinks.hpp
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "ImageWrapper.h"

namespace megamol::frontend {

struct glfw_window_blit {
    unsigned int gl_fbo_handle = 0;
    unsigned int fbo_width = 0, fbo_height = 0;

    glfw_window_blit();
    ~glfw_window_blit();

    void set_framebuffer_active();
    void set_framebuffer_size(unsigned int width, unsigned int height);
    void blit_texture(unsigned int gl_texture_handle, unsigned int texture_width, unsigned int texture_height);
};

} // namespace megamol::frontend
