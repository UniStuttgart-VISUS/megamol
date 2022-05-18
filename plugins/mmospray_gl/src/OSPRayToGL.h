#pragma once

#include "mmcore_gl/view/ContextToGL.h"

namespace megamol::ospray_gl {

inline constexpr char ospraytogl_name[] = "OSPRayToGL";

inline constexpr char ospraytogl_desc[] = "Merges content to the input GL buffer";

inline constexpr auto ospray_to_gl_init_func = [](std::shared_ptr<glowl::FramebufferObject>& lhs_fbo,
                                                   std::shared_ptr<core::view::CPUFramebuffer>& fbo, int width,
                                                   int height) -> void {
    if (fbo != nullptr) {
        glDeleteTextures(1, &fbo->data.col_tex);
        glDeleteTextures(1, &fbo->data.depth_tex);
    }
    fbo = std::make_shared<core::view::CPUFramebuffer>();
    fbo->width = width;
    fbo->height = height;

    glGenTextures(1, (GLuint*)&fbo->data.col_tex);
    glBindTexture(GL_TEXTURE_2D, fbo->data.col_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenTextures(1, (GLuint*)&fbo->data.depth_tex);
    glBindTexture(GL_TEXTURE_2D, fbo->data.depth_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
};

inline constexpr auto ospray_to_gl_ren_func =
    [](std::shared_ptr<glowl::GLSLProgram>& shader, std::shared_ptr<glowl::FramebufferObject>& lhs_fbo,
        std::shared_ptr<core::view::CPUFramebuffer>& fbo, int width, int height) -> void {
    glBindTexture(GL_TEXTURE_2D, fbo->data.col_tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, fbo->colorBuffer.data());
    glBindTexture(GL_TEXTURE_2D, fbo->data.depth_tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, fbo->depthBuffer.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    core_gl::view::renderToFBO(shader, lhs_fbo, fbo->data.col_tex, fbo->data.depth_tex, width, height);
};

using OSPRayToGL = core_gl::view::ContextToGL<core::view::CallRender3D, ospray_to_gl_init_func, ospray_to_gl_ren_func,
    ospraytogl_name, ospraytogl_desc>;

} // namespace megamol::ospray_gl
