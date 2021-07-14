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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    CUDA_CHECK_ERROR(cuGraphicsGLRegisterImage(
        &fbo->data.depth_tex_ref, fbo->data.depth_tex, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_NONE));

    CUDA_CHECK_ERROR(cuGraphicsMapResources(1, &fbo->data.depth_tex_ref, fbo->data.exec_stream));

    CUDA_CHECK_ERROR(cuGraphicsSubResourceGetMappedArray(&cuarr, fbo->data.depth_tex_ref, 0, 0));

    surf_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    surf_desc.res.array.hArray = cuarr;
    surf_desc.flags = 0;

    CUDA_CHECK_ERROR(cuSurfObjectCreate(&fbo->depthBuffer, &surf_desc));

    fbo->width = width;
    fbo->height = height;
};

inline constexpr auto cuda_to_gl_ren_func = [](std::shared_ptr<glowl::GLSLProgram>& shader,
                                                std::shared_ptr<vislib::graphics::gl::FramebufferObject>& lhs_fbo,
                                                std::shared_ptr<CUDAFramebuffer>& fbo, int width, int height) -> void {
    CUDA_CHECK_ERROR(cuGraphicsUnmapResources(1, &fbo->data.col_tex_ref, fbo->data.exec_stream));
    CUDA_CHECK_ERROR(cuGraphicsUnmapResources(1, &fbo->data.depth_tex_ref, fbo->data.exec_stream));

    core::view::renderToFBO(shader, lhs_fbo, fbo->data.col_tex, fbo->data.depth_tex, width, height);

    CUDA_CHECK_ERROR(cuGraphicsMapResources(1, &fbo->data.col_tex_ref, fbo->data.exec_stream));
    CUDA_CHECK_ERROR(cuGraphicsMapResources(1, &fbo->data.depth_tex_ref, fbo->data.exec_stream));
};

using CUDAToGL =
    core::view::ContextToGL<CallRender3DCUDA, cuda_to_gl_init_func, cuda_to_gl_ren_func, cudatogl_name, cudatogl_desc>;

} // namespace megamol::optix_hpg
