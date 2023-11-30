#pragma once

#include <cuda.h>

#include "CallRender3DCUDA.h"
#include "mmstd/renderer/ContextToCPU.h"
#include "optix/Utils.h"

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

namespace megamol::optix_hpg {

inline constexpr char cudatocpu_name[] = "CUDAToCPU";

inline constexpr char cudatocpu_desc[] = "Merges content to the input GL buffer";

inline constexpr auto cuda_to_cpu_init_func = [](std::shared_ptr<core::view::CPUFramebuffer> const& lhs_fbo,
                                                  std::shared_ptr<CUDAFramebuffer>& fbo, int width,
                                                  int height) -> void {
    if (fbo != nullptr) {
        CUDA_CHECK_ERROR(cuSurfObjectDestroy(fbo->colorBuffer));
        CUDA_CHECK_ERROR(cuSurfObjectDestroy(fbo->depthBuffer));
        fbo.reset();
    }

    fbo = std::make_shared<CUDAFramebuffer>();

    CUDA_ARRAY_DESCRIPTOR arr_desc = {};
    arr_desc.Format = CUarray_format::CU_AD_FORMAT_FLOAT;
    arr_desc.NumChannels = 4;
    arr_desc.Width = width;
    arr_desc.Height = height;

    CUarray cuarr;
    CUDA_CHECK_ERROR(cuArrayCreate(&cuarr, &arr_desc));

    CUDA_RESOURCE_DESC surf_desc = {};
    surf_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    surf_desc.res.array.hArray = cuarr;
    surf_desc.flags = 0;

    CUDA_CHECK_ERROR(cuSurfObjectCreate(&fbo->colorBuffer, &surf_desc));

    arr_desc.Format = CUarray_format::CU_AD_FORMAT_FLOAT;
    arr_desc.NumChannels = 1;
    arr_desc.Width = width;
    arr_desc.Height = height;

    CUDA_CHECK_ERROR(cuArrayCreate(&cuarr, &arr_desc));

    surf_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    surf_desc.res.array.hArray = cuarr;
    surf_desc.flags = 0;

    CUDA_CHECK_ERROR(cuSurfObjectCreate(&fbo->depthBuffer, &surf_desc));

    fbo->width = width;
    fbo->height = height;
};

inline constexpr auto cuda_to_cpu_ren_func = [](std::shared_ptr<core::view::CPUFramebuffer>& lhs_fbo,
                                                 std::shared_ptr<CUDAFramebuffer>& fbo, int width, int height) -> void {
    //#ifdef MEGAMOL_USE_TRACY
    //    ZoneScopedN("CUDAToGL::Blit");
    //    TracyGpuZone("CUDAToGL::Blit");
    //#endif
    //    CUDA_CHECK_ERROR(cuGraphicsUnmapResources(1, &fbo->data.col_tex_ref, fbo->data.exec_stream));
    //    CUDA_CHECK_ERROR(cuGraphicsUnmapResources(1, &fbo->data.depth_tex_ref, fbo->data.exec_stream));
    //
    //    mmstd_gl::renderToFBO(shader, lhs_fbo, fbo->data.col_tex, fbo->data.depth_tex, width, height);
    //
    //    CUDA_CHECK_ERROR(cuGraphicsMapResources(1, &fbo->data.col_tex_ref, fbo->data.exec_stream));
    //    CUDA_CHECK_ERROR(cuGraphicsMapResources(1, &fbo->data.depth_tex_ref, fbo->data.exec_stream));
};

using CUDAToCPU =
    mmstd::ContextToCPU<CallRender3DCUDA, cuda_to_cpu_init_func, cuda_to_cpu_ren_func, cudatocpu_name, cudatocpu_desc>;

} // namespace megamol::optix_hpg
