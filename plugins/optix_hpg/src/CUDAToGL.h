#pragma once

#include "mmcore/view/ContextToGL.h"

#include "CallRender3DCUDA.h"

#include "cuda.h"
#include "cudaGL.h"

#include "optix/Utils.h"

namespace megamol::optix_hpg {

inline constexpr char cudatogl_name[] = "CUDAToGL";

inline constexpr char cudatogl_desc[] = "Merges content to the input GL buffer";

inline constexpr auto cuda_to_gl_init_func = [](std::shared_ptr<vislib::graphics::gl::FramebufferObject>& lhs_fbo,
                                                 std::shared_ptr<CUDAFramebuffer>& fbo, int width, int height) -> void {
    if (fbo != nullptr) {
        CUDA_CHECK_ERROR(cuGraphicsUnmapResources(1, &fbo->data.col_tex_ref, fbo->data.exec_stream));
        CUDA_CHECK_ERROR(cuGraphicsUnmapResources(1, &fbo->data.depth_tex_ref, fbo->data.exec_stream));
        CUDA_CHECK_ERROR(cuSurfObjectDestroy(fbo->colorBuffer));
        CUDA_CHECK_ERROR(cuSurfObjectDestroy(fbo->depthBuffer));
        CUDA_CHECK_ERROR(cuGraphicsUnregisterResource(fbo->data.col_tex_ref));
        CUDA_CHECK_ERROR(cuGraphicsUnregisterResource(fbo->data.depth_tex_ref));
        glDeleteTextures(1, &fbo->data.col_tex);
        glDeleteTextures(1, &fbo->data.depth_tex);
        fbo.reset();
    }

    fbo = std::make_shared<CUDAFramebuffer>();

    glGenTextures(1, (GLuint*) &fbo->data.col_tex);
    glBindTexture(GL_TEXTURE_2D, fbo->data.col_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    CUDA_CHECK_ERROR(cuGraphicsGLRegisterImage(
        &fbo->data.col_tex_ref, fbo->data.col_tex, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_NONE));

    CUDA_CHECK_ERROR(cuGraphicsMapResources(1, &fbo->data.col_tex_ref, fbo->data.exec_stream));

    CUarray cuarr;
    CUDA_CHECK_ERROR(cuGraphicsSubResourceGetMappedArray(&cuarr, fbo->data.col_tex_ref, 0, 0));

    CUDA_RESOURCE_DESC surf_desc = {};
    surf_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    surf_desc.res.array.hArray = cuarr;
    surf_desc.flags = 0;

    CUDA_CHECK_ERROR(cuSurfObjectCreate(&fbo->colorBuffer, &surf_desc));

    glGenTextures(1, (GLuint*) &fbo->data.depth_tex);
    glBindTexture(GL_TEXTURE_2D, fbo->data.depth_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    CUDA_CHECK_ERROR(cuGraphicsGLRegisterImage(
        &fbo->data.depth_tex_ref, fbo->data.depth_tex, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_NONE));

    CUDA_CHECK_ERROR(cuGraphicsMapResources(1, &fbo->data.depth_tex_ref, fbo->data.exec_stream));

    cuarr;
    CUDA_CHECK_ERROR(cuGraphicsSubResourceGetMappedArray(&cuarr, fbo->data.depth_tex_ref, 0, 0));

    surf_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    surf_desc.res.array.hArray = cuarr;
    surf_desc.flags = 0;

    CUDA_CHECK_ERROR(cuSurfObjectCreate(&fbo->depthBuffer, &surf_desc));

    // if (lhs_fbo != nullptr) {
    //    auto color_image = lhs_fbo->GetColourTextureID();
    //    auto color_att_state = lhs_fbo->GetColorAttachmentState();
    //    GLenum color_target = GL_TEXTURE_2D;
    //    if (color_att_state.has_value()) {
    //        if (color_att_state.value() ==
    //            vislib::graphics::gl::FramebufferObject::AttachmentState::ATTACHMENT_RENDERBUFFER) {
    //            color_target = GL_RENDERBUFFER;
    //        }
    //    }
    //    CUDA_CHECK_ERROR(cuGraphicsGLRegisterImage(
    //        &fbo->data.col_att_ref, color_image, color_target, CU_GRAPHICS_REGISTER_FLAGS_NONE));

    //    /*auto depth_image = lhs_fbo->GetDepthTextureID();
    //    auto depth_att_state = lhs_fbo->GetDepthAttachmentState();
    //    GLenum depth_target = GL_TEXTURE_2D;
    //    if (depth_att_state.has_value()) {
    //        if (depth_att_state.value() ==
    //            vislib::graphics::gl::FramebufferObject::AttachmentState::ATTACHMENT_RENDERBUFFER) {
    //            depth_target = GL_RENDERBUFFER;
    //        }
    //    }
    //    cuGraphicsGLRegisterImage(
    //        &fbo->depthBuffer, depth_image, depth_target, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);*/
    //}

    fbo->width = width;
    fbo->height = height;
};

inline constexpr auto cuda_to_gl_ren_func = [](std::shared_ptr<vislib::graphics::gl::FramebufferObject>& lhs_fbo,
                                                std::shared_ptr<CUDAFramebuffer>& fbo, core::view::RenderUtils& utils,
                                                int width, int height) -> void {
    CUDA_CHECK_ERROR(cuGraphicsUnmapResources(1, &fbo->data.col_tex_ref, fbo->data.exec_stream));
    float right = (width + static_cast<float>(width)) / 2.0f;
    float left = (width - static_cast<float>(width)) / 2.0f;
    float bottom = (height + static_cast<float>(height)) / 2.0f;
    float up = (height - static_cast<float>(height)) / 2.0f;
    glm::vec3 pos_bottom_left = {left, bottom, 0.0f};
    glm::vec3 pos_upper_left = {left, up, 0.0f};
    glm::vec3 pos_upper_right = {right, up, 0.0f};
    glm::vec3 pos_bottom_right = {right, bottom, 0.0f};
    utils.Push2DColorTexture(
        fbo->data.col_tex, pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right, true);
    utils.Push2DDepthTexture(
        fbo->data.depth_tex, pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right, true);
    glm::mat4 ortho = glm::ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -1.0f, 1.0f);
    utils.DrawTextures(ortho, glm::vec2(width, height));
    CUDA_CHECK_ERROR(cuGraphicsMapResources(1, &fbo->data.col_tex_ref, fbo->data.exec_stream));
};

using CUDAToGL =
    core::view::ContextToGL<CallRender3DCUDA, cuda_to_gl_init_func, cuda_to_gl_ren_func, cudatogl_name, cudatogl_desc>;

} // namespace megamol::optix_hpg
