#pragma once

#include <vector>

#include "mmcore/param/ParamSlot.h"

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"

#include <owl/common/math/vec.h>
#include <owl/owl.h>

#include "framestate.h"
#include "particle.h"

namespace megamol::optix_owl {
class TreeletsRenderer : public mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "TreeletsRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    TreeletsRenderer();

    virtual ~TreeletsRenderer();

protected:
    bool create() override;

    void release() override;

private:
    bool Render(mmstd_gl::CallRender3DGL& call) override;

    bool GetExtents(mmstd_gl::CallRender3DGL& call) override;

    bool assertData(geocalls::MultiParticleDataCall const& call);

    void resizeFramebuffer(owl::common::vec2i const& dim);

    core::CallerSlot data_in_slot_;

    core::param::ParamSlot radius_slot_;
    core::param::ParamSlot rec_depth_slot_;
    core::param::ParamSlot spp_slot_;
    core::param::ParamSlot threshold_slot_;

    OWLContext ctx_;

    OWLModule raygen_module_;
    OWLModule pkd_module_;

    OWLRayGen raygen_;
    OWLMissProg miss_;

    OWLBuffer frameStateBuffer_;
    OWLBuffer accumBuffer_ = 0;
    OWLBuffer colorBuffer_ = 0;
    OWLBuffer particleBuffer_ = 0;
    OWLBuffer treeletBuffer_ = 0;

    OWLGroup world_;

    std::vector<device::Particle> particles_;

    owl::common::vec2i current_fb_size_;

    unsigned int frame_id_ = 0;
    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();

    device::FrameState framestate_;

    core::view::Camera::Pose old_cam_pose_;
    core::view::Camera::PerspectiveParameters old_cam_intrinsics_;
};
} // namespace megamol::optix_owl
