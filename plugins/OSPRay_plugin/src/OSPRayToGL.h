#pragma once

#include "mmcore/view/ContextToGL.h"

namespace megamol::ospray {

inline constexpr char ospraytogl_name[] = "OSPRayToGL";

inline constexpr char ospraytogl_desc[] = "Merges content to the input GL buffer";

inline constexpr auto ospray_to_gl_init_func = [](std::shared_ptr<vislib::graphics::gl::FramebufferObject>& lhs_fbo,
                                                   std::shared_ptr<core::view::CPUFramebuffer>& fbo, int width,
                                                   int height) -> void {
    fbo = std::make_shared<core::view::CPUFramebuffer>();
};

inline constexpr auto ospray_to_gl_ren_func = [](std::shared_ptr<vislib::graphics::gl::FramebufferObject>& lhs_fbo,
                                                  std::shared_ptr<core::view::CPUFramebuffer>& fbo,
                                                  core::view::RenderUtils& utils, int width,
                                                  int height) -> void { // module own fbo
    auto new_fbo = vislib::graphics::gl::FramebufferObject();
    new_fbo.Create(width, height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
        vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE, GL_DEPTH_COMPONENT);
    new_fbo.Enable();

    new_fbo.BindColourTexture();
    glClear(GL_COLOR_BUFFER_BIT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, fbo->colorBuffer.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    if (fbo->depthBufferActive) {
        new_fbo.BindDepthTexture();
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
        new_fbo.GetColourTextureID(), pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right, true);
    if (fbo->depthBufferActive) {
        utils.Push2DDepthTexture(
            new_fbo.GetDepthTextureID(), pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right, true);
    }

    new_fbo.Disable();

    // draw into lhs fbo
    if ((lhs_fbo->GetWidth() != width) || (lhs_fbo->GetHeight() != height)) {
        lhs_fbo->Release();
        lhs_fbo->Create(width, height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
            vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE, GL_DEPTH_COMPONENT);
    }
    if (lhs_fbo->IsValid() && !lhs_fbo->IsEnabled()) {
        lhs_fbo->Enable();
    }

    glm::mat4 ortho = glm::ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -1.0f, 1.0f);

    utils.DrawTextures(ortho, glm::vec2(width, height));

    if (lhs_fbo->IsValid()) {
        lhs_fbo->Disable();
    }
};

using OSPRayToGL = core::view::ContextToGL<core::view::CallRender3D, ospray_to_gl_init_func, ospray_to_gl_ren_func,
    ospraytogl_name, ospraytogl_desc>;

} // namespace megamol::ospray
