#pragma once

#include "mmcore/view/ContextToGL.h"

namespace megamol::ospray {

inline constexpr char ospraytogl_name[] = "OSPRayToGL";

inline constexpr char ospraytogl_desc[] = "Merges content to the input GL buffer";

inline constexpr auto ospray_to_gl_init_func = [](std::shared_ptr<glowl::FramebufferObject>& lhs_fbo,
                                                   std::shared_ptr<core::view::CPUFramebuffer>& fbo, int width,
                                                   int height) -> void {
    fbo = std::make_shared<core::view::CPUFramebuffer>();
};

inline constexpr auto ospray_to_gl_ren_func = [](std::shared_ptr<glowl::FramebufferObject>& lhs_fbo,
                                                  std::shared_ptr<core::view::CPUFramebuffer>& fbo,
                                                  core::view::RenderUtils& utils, int width,
                                                  int height) -> void { // module own fbo
    auto new_fbo = glowl::FramebufferObject(width, height, glowl::FramebufferObject::DEPTH32F);
    new_fbo.createColorAttachment(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);


    new_fbo.bindColorbuffer(0);
    glClear(GL_COLOR_BUFFER_BIT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, fbo->colorBuffer.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    if (fbo->depthBufferActive) {
        new_fbo.bindDepthbuffer();
        glClear(GL_DEPTH_BUFFER_BIT);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
            fbo->depthBuffer.data());
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    float right = (width + static_cast<float>(width)) / 2.0f;
    float left = (width - static_cast<float>(width)) / 2.0f;
    float bottom = (height + static_cast<float>(height)) / 2.0f;
    float up = (height - static_cast<float>(height)) / 2.0f;
    glm::vec3 pos_bottom_left = {left, bottom, 0.0f};
    glm::vec3 pos_upper_left = {left, up, 0.0f};
    glm::vec3 pos_upper_right = {right, up, 0.0f};
    glm::vec3 pos_bottom_right = {right, bottom, 0.0f};
    utils.Push2DColorTexture(
        new_fbo.getColorAttachment(0)->getName(), pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right, true);
    if (fbo->depthBufferActive) {
        /* temporary hack */
        new_fbo.bindDepthbuffer();
        GLint tx_name;
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &tx_name);
        glBindTexture(GL_TEXTURE_2D, 0);
        utils.Push2DDepthTexture(tx_name, pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right, true);
    }

    // draw into lhs fbo
    if ((lhs_fbo->getWidth() != width) || (lhs_fbo->getHeight() != height)) {
        lhs_fbo = std::make_shared<glowl::FramebufferObject>(width, height);
        lhs_fbo->createColorAttachment(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
    }

    lhs_fbo->bind();
    glm::mat4 ortho = glm::ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -1.0f, 1.0f);

    utils.DrawTextures(ortho, glm::vec2(width, height));

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

};

using OSPRayToGL = core::view::ContextToGL<core::view::CallRender3D, ospray_to_gl_init_func, ospray_to_gl_ren_func,
    ospraytogl_name, ospraytogl_desc>;

} // namespace megamol::ospray
