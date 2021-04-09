/*
 * ImagePresentation_Sinks.cpp
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "ImagePresentation_Sinks.hpp"

#include "glad/glad.h"

using namespace megamol::frontend;

glfw_window_sink::glfw_window_sink() {
    if (gl_fbo_handle == 0)
        glGenFramebuffers(1, &gl_fbo_handle);
}

glfw_window_sink::~glfw_window_sink() {
    if (gl_fbo_handle != 0)
        glDeleteFramebuffers(1, &gl_fbo_handle);

    gl_fbo_handle = 0;
}

void glfw_window_sink::resize(unsigned int width, unsigned int height) {
    fbo_width = width;
    fbo_height = height;
}

void glfw_window_sink::blit_texture(unsigned int gl_texture_handle, unsigned int texture_width, unsigned int texture_height) {
    // credit goes to: https://stackoverflow.com/questions/31482816/opengl-is-there-an-easier-way-to-fill-window-with-a-texture-instead-using-vbo
    //glBindFramebuffer(GL_READ_FRAMEBUFFER, gl_fbo_handle);
    //glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gl_texture_handle, 0);
    //glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    //glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

    glNamedFramebufferTexture(gl_fbo_handle, GL_COLOR_ATTACHMENT0, gl_texture_handle, 0);

    glBlitNamedFramebuffer(gl_fbo_handle , 0, /*from, to*/
        0, 0, texture_width, texture_height, /* src: x0, y0, x1, y1, */
        0, 0, fbo_width, fbo_height, /* dst: x0, y0, x1, y1, */
        GL_COLOR_BUFFER_BIT, /* mask: GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_STENCIL_BUFFER_BIT */
        GL_LINEAR /* filter: GL_NEAREST, GL_LINEAR */
        );
}

