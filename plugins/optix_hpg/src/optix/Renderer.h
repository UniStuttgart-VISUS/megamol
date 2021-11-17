#pragma once

#include <array>
#include <limits>
#include <tuple>

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ParamSlot.h"

#include "AbstractRenderer.h"

#include "cuda.h"

#include "framestate.h"

#include "glowl/FramebufferObject.hpp"

#include "miss.h"
#include "raygen.h"

#include "SBT.h"

#include "MMOptixModule.h"

#include "vislib/math/Rectangle.h"

#include "CUDA_Context.h"

#include "optix/Context.h"

#include "CallRender3DCUDA.h"

namespace megamol::optix_hpg {
class Renderer : public AbstractRenderer {
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

protected:
private:
    void setup() override;

    bool is_dirty() override {
        return spp_slot_.IsDirty() || max_bounces_slot_.IsDirty() || accumulate_slot_.IsDirty() ||
               intensity_slot_.IsDirty();
    }

    void reset_dirty() override {
        spp_slot_.ResetDirty();
        max_bounces_slot_.ResetDirty();
        accumulate_slot_.ResetDirty();
        intensity_slot_.ResetDirty();
    }

    void on_cam_param_change(
        core::view::Camera const& cam, core::view::Camera::PerspectiveParameters const& cam_intrinsics) override;

    void on_cam_pose_change(core::view::Camera::Pose const& cam_pose) override;

    void on_change_data(OptixTraversableHandle world) override;

    void on_change_background(glm::vec4 const& bg) override;

    void on_change_programs(std::tuple<OptixProgramGroup const*, uint32_t> const& programs) override;

    void on_change_parameters() override;

    void on_change_viewport(glm::uvec2 const& viewport, std::shared_ptr<CUDAFramebuffer> fbo) override;

    void on_change_sbt(std::tuple<void const*, uint32_t, uint64_t> const& records) override;

    core::param::ParamSlot spp_slot_;

    core::param::ParamSlot max_bounces_slot_;

    core::param::ParamSlot accumulate_slot_;

    core::param::ParamSlot intensity_slot_;

    SBTRecord<device::RayGenData> sbt_raygen_record_;

    std::array<SBTRecord<device::MissData>, 2> sbt_miss_records_;

    MMOptixModule raygen_module_;

    MMOptixModule miss_module_;

    MMOptixModule miss_occlusion_module_;
};
} // namespace megamol::optix_hpg
