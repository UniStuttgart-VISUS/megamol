#pragma once

#include <array>
#include <limits>

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Renderer3DModuleGL.h"

#include "cuda.h"

#include "framestate.h"

#include "glowl/FramebufferObject.hpp"

#include "cudaGL.h"

#include "optix/CallContext.h"

#include "miss.h"
#include "raygen.h"
#include "picking.h"

#include "SBT.h"

#include "MMOptixModule.h"

#include "vislib/math/Rectangle.h"

#include "CUDA_Context.h"

#include "optix/Context.h"

#include "CallRender3DCUDA.h"

#include "mmcore/UniFlagCalls.h"

namespace megamol::optix_hpg {
class Renderer : public core::view::RendererModule<CallRender3DCUDA> {
public:
    std::vector<std::string> requested_lifetime_resources() override {
        auto res = core::view::RendererModule<CallRender3DCUDA>::requested_lifetime_resources();
        res.push_back(frontend_resources::CUDA_Context_Req_Name);
        return res;
    }

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

    bool Render(CallRender3DCUDA& call) override;

    bool GetExtents(CallRender3DCUDA& call) override;

protected:
    bool create() override;

    void release() override;

private:
    void setup();

    bool OnMouseButton(
        core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;

    bool OnMouseMove(double x, double y) override;

    bool is_dirty() {
        return spp_slot_.IsDirty() || max_bounces_slot_.IsDirty() || accumulate_slot_.IsDirty() ||
               intensity_slot_.IsDirty();
    }

    void reset_dirty() {
        spp_slot_.ResetDirty();
        max_bounces_slot_.ResetDirty();
        accumulate_slot_.ResetDirty();
        intensity_slot_.ResetDirty();
    }

    core::CallerSlot _in_geo_slot;

    /*core::CallerSlot flags_write_slot_;

    core::CallerSlot flags_read_slot_;*/

    core::param::ParamSlot spp_slot_;

    core::param::ParamSlot max_bounces_slot_;

    core::param::ParamSlot accumulate_slot_;

    core::param::ParamSlot intensity_slot_;

    core::param::ParamSlot enable_picking_slot_;

    float mouse_x_;

    float mouse_y_;

    SBTRecord<device::RayGenData> _sbt_raygen_record;

    SBTRecord<device::PickingData> sbt_picking_record_;

    std::array<SBTRecord<device::MissData>, 2> sbt_miss_records_;

    MMOptixSBT sbt_;

    MMOptixSBT picking_sbt_;

    OptixPipeline _pipeline;

    MMOptixModule raygen_module_;

    MMOptixModule picking_module_;

    MMOptixModule miss_module_;

    MMOptixModule miss_occlusion_module_;

    CUdeviceptr _frame_state_buffer;

    CUdeviceptr pick_state_buffer_;

    CUdeviceptr picking_pixel_buffer_ = 0;

    device::FrameState _frame_state;

    device::PickState pick_state_;

    vislib::math::Rectangle<int> _current_fb_size;

    unsigned int _frame_id = std::numeric_limits<unsigned int>::max();

    std::size_t _in_data_hash = std::numeric_limits<std::size_t>::max();

    cam_type::snapshot_type old_cam_snap;

    glm::vec4 old_bg = glm::vec4(-1);

    std::unique_ptr<Context> optix_ctx_;
};
} // namespace megamol::optix_hpg
