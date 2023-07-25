/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "ImagePresentation_Sinks.hpp"
#include <glowl/GLSLProgram.hpp>

#include "glad/gl.h"
//#include <iostream>
#include "mmcore/utility/log/Log.h"

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

void glfw_window_blit::set_framebuffer_active() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClearColor(0, 0, 0, 0);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

void glfw_window_blit::set_framebuffer_size(unsigned int width, unsigned int height) {
    fbo_width = width;
    fbo_height = height;
}

void glfw_window_blit::blit_texture(
    unsigned int gl_texture_handle, unsigned int texture_width, unsigned int texture_height) {
    // credit goes to: https://stackoverflow.com/questions/31482816/opengl-is-there-an-easier-way-to-fill-window-with-a-texture-instead-using-vbo

    // Supports no blending:
    /*
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glNamedFramebufferTexture(gl_fbo_handle, GL_COLOR_ATTACHMENT0, gl_texture_handle, 0);
    //if (glCheckNamedFramebufferStatus(gl_fbo_handle, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    //    std::cout << "FBO ERROR: " << glCheckNamedFramebufferStatus(gl_fbo_handle, GL_FRAMEBUFFER) << std::endl;
    //    std::exit(1);
    //}
    glBlitNamedFramebuffer(gl_fbo_handle , 0, // from, to
        0, 0, texture_width, texture_height, // src: x0, y0, x1, y1,
        0, 0, fbo_width, fbo_height, // dst: x0, y0, x1, y1,
        GL_COLOR_BUFFER_BIT, // mask: GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_STENCIL_BUFFER_BIT
        GL_LINEAR // filter: GL_NEAREST, GL_LINEAR
        );
    */
    /**/

    static std::unique_ptr<glowl::GLSLProgram> blit_shader;
    static GLuint vaEmpty = 0;

    if (blit_shader == nullptr) {
        std::string vertex_src = "#version 130 \n "
                                 "out vec2 uv_coord; \n "
                                 "void main() { \n "
                                 "    const vec4 vertices[6] = vec4[6](vec4(-1.0, -1.0, 0.0, 0.0), \n "
                                 "        vec4(1.0, 1.0, 1.0, 1.0), \n "
                                 "        vec4(-1.0, 1.0, 0.0, 1.0), \n "
                                 "        vec4(1.0, 1.0, 1.0, 1.0), \n "
                                 "        vec4(-1.0, -1.0, 0.0, 0.0), \n "
                                 "        vec4(1.0, -1.0, 1.0, 0.0)); \n "
                                 "    vec4 vertex = vertices[gl_VertexID]; \n "
                                 "    uv_coord = vertex.zw; \n "
                                 "    gl_Position = vec4(vertex.xy, -1.0, 1.0); \n "
                                 "} ";

        std::string fragment_src = "#version 130  \n "
                                   "#extension GL_ARB_explicit_attrib_location : require \n "
                                   "in vec2 uv_coord; \n "
                                   "uniform sampler2D col_tex; \n "
                                   "layout(location = 0) out vec4 outFragColor; \n "
                                   "void main() { \n "
                                   "    vec4 color = texture(col_tex, uv_coord).rgba; \n "
                                   "    outFragColor = color; \n "
                                   "} ";

        std::vector<std::pair<glowl::GLSLProgram::ShaderType, std::string>> shader_srcs;
        shader_srcs.push_back({glowl::GLSLProgram::ShaderType::Vertex, vertex_src});
        shader_srcs.push_back({glowl::GLSLProgram::ShaderType::Fragment, fragment_src});
        try {
            if (blit_shader != nullptr) {
                blit_shader.reset();
            }
            blit_shader = std::make_unique<glowl::GLSLProgram>(shader_srcs);
        } catch (glowl::GLSLProgramException const& exc) {
            std::string debug_label;
            if (blit_shader != nullptr) {
                debug_label = blit_shader->getDebugLabel();
            }
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Error during shader program creation of\"%s\": %s. [%s, %s, line %d]\n ", debug_label.c_str(),
                exc.what(), __FILE__, __FUNCTION__, __LINE__);
            return;
        }

        glGenVertexArrays(1, &vaEmpty);
    }

    glViewport(0, 0, fbo_width, fbo_height);
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    blit_shader->use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gl_texture_handle);
    blit_shader->setUniform("col_tex", 0);
    glBindVertexArray(vaEmpty);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    glUseProgram(0);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    /**/
}
