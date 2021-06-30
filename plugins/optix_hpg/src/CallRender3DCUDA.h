#pragma once

#include "mmcore/view/AbstractCallRender.h"
#include "mmcore/view/Framebuffer.h"

#include "cuda.h"

namespace megamol::optix_hpg {

struct CUDAFramebufferData {
    unsigned int col_tex = 0;
    CUgraphicsResource col_tex_ref = 0;
    //CUsurfObject col_surface = 0;
    unsigned int depth_tex = 0;
    CUgraphicsResource depth_tex_ref = 0;
    //CUsurfObject depth_surface = 0;
    CUstream exec_stream = 0;
};

using CUDAFramebuffer = core::view::Framebuffer<CUsurfObject, CUsurfObject, CUDAFramebufferData>;

inline constexpr char callrender3dcuda_name[] = "CallRender3DCUDA";

inline constexpr char callrender3dcuda_desc[] = "CUDA Rendering call";

using CallRender3DCUDA = core::view::BaseCallRender<CUDAFramebuffer, callrender3dcuda_name, callrender3dcuda_desc>;

/** Description class typedef */
typedef core::factories::CallAutoDescription<CallRender3DCUDA> CallRender3DCUDADescription;

} // namespace megamol::optix_hpg
