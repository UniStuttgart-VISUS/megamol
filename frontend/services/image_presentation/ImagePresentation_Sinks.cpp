/*
 * ImagePresentation_Sinks.cpp
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "ImagePresentation_Sinks.hpp"

#include "glad/glad.h"
//#include <iostream>

using namespace megamol::frontend;

glfw_window_blit::glfw_window_blit() {
    if (gl_fbo_handle == 0)
        glCreateFramebuffers(1, &gl_fbo_handle);

    GLenum draw_buffers[] = {GL_COLOR_ATTACHMENT0};
    glNamedFramebufferDrawBuffers(gl_fbo_handle, 1, draw_buffers);
}

glfw_window_blit::~glfw_window_blit() {
    if (gl_fbo_handle != 0)
        glDeleteFramebuffers(1, &gl_fbo_handle);

    gl_fbo_handle = 0;
}

void glfw_window_blit::set_framebuffer_size(unsigned int width, unsigned int height) {
    fbo_width = width;
    fbo_height = height;
}

void glfw_window_blit::blit_texture(unsigned int gl_texture_handle, unsigned int texture_width, unsigned int texture_height) {
    // credit goes to: https://stackoverflow.com/questions/31482816/opengl-is-there-an-easier-way-to-fill-window-with-a-texture-instead-using-vbo

    glNamedFramebufferTexture(gl_fbo_handle, GL_COLOR_ATTACHMENT0, gl_texture_handle, 0);

    //if (glCheckNamedFramebufferStatus(gl_fbo_handle, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    //    std::cout << "FBO ERROR: " << glCheckNamedFramebufferStatus(gl_fbo_handle, GL_FRAMEBUFFER) << std::endl;
    //    std::exit(1);
    //}

    glBlitNamedFramebuffer(gl_fbo_handle , 0, /*from, to*/
        0, 0, texture_width, texture_height, /* src: x0, y0, x1, y1, */
        0, 0, fbo_width, fbo_height, /* dst: x0, y0, x1, y1, */
        GL_COLOR_BUFFER_BIT, /* mask: GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_STENCIL_BUFFER_BIT */
        GL_LINEAR /* filter: GL_NEAREST, GL_LINEAR */
        );
}

