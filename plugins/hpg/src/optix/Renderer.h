#pragma once

#include <array>
#include <limits>

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer3DModule_2.h"

#include "cuda.h"

#include "framestate.h"

#include "glowl/FramebufferObject.hpp"

#include "cudaGL.h"

#include "hpg/optix/CallContext.h"

#include "raygen.h"
#include "miss.h"

#include "SBT.h"

#include "MMOptixModule.h"

namespace megamol::hpg::optix {
class Renderer : public core::view::Renderer3DModule_2 {
public:
    static const char* ClassName(void) {
        return "OptixRenderer";
    }

    static const char* Description(void) {
        return "Renderer using OptiX framework";
    }

    static bool IsAvailable(void) {
        return true;
    }

    Renderer();

    virtual ~Renderer();

    bool Render(core::view::CallRender3D_2& call) override;

    bool GetExtents(core::view::CallRender3D_2& call) override;

protected:
    bool create() override;

    void release() override;

private:
    void setup(CallContext& ctx);

    core::CallerSlot _in_geo_slot;

    core::CallerSlot _in_ctx_slot;

    SBTRecord<device::RayGenData> _sbt_raygen_record;

    std::array<SBTRecord<device::MissData>, 2> sbt_miss_records_;

    MMOptixSBT sbt_;

    OptixPipeline _pipeline;

    MMOptixModule raygen_module_;

    MMOptixModule miss_module_;

    MMOptixModule miss_occlusion_module_;

    CUdeviceptr _frame_state_buffer;

    device::FrameState _frame_state;

    vislib::math::Rectangle<int> _current_fb_size;

    GLuint _fb_texture = 0;

    GLuint _fbo_pbo = 0;

    CUdeviceptr _old_pbo_ptr = 0;

    CUgraphicsResource _fbo_res = nullptr;

    unsigned int _frame_id = std::numeric_limits<unsigned int>::max();

    std::size_t _in_data_hash = std::numeric_limits<std::size_t>::max();

   cam_type::snapshot_type old_cam_snap;
};
} // namespace megamol::hpg::optix
