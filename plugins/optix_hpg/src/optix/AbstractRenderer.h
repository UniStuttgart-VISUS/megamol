#pragma once

#include "mmstd/renderer/RendererModule.h"

#include "CallRender3DCUDA.h"

#include "optix/Utils.h"

#include "framestate.h"

#include "CUDA_Context.h"
#include "SBT.h"

#include "optix/Context.h"

namespace megamol::optix_hpg {
class AbstractRenderer : public core::view::RendererModule<CallRender3DCUDA, core::Module> {
public:
    static void requested_lifetime_resources(frontend_resources::ResourceRequest& req) {
        core::view::RendererModule<CallRender3DCUDA, Module>::requested_lifetime_resources(req);
        req.require<frontend_resources::CUDA_Context>();
    }

    AbstractRenderer();

    virtual ~AbstractRenderer();

    bool Render(CallRender3DCUDA& call) override;

    bool GetExtents(CallRender3DCUDA& call) override;

protected:
    bool create() override;

    void release() override;

    device::FrameState& get_framestate() {
        return frame_state_;
    }

    std::unique_ptr<Context> const& get_context() const {
        return optix_ctx_;
    }

    MMOptixSBT& get_sbt() {
        return sbt_;
    }

    OptixPipeline& get_pipeline() {
        return pipeline_;
    }

    CUdeviceptr const& get_frame_state_buffer() const {
        return frame_state_buffer_;
    }

private:
    virtual void setup() = 0;

    virtual bool is_dirty() = 0;

    virtual void reset_dirty() = 0;

    virtual void on_cam_param_change(
        core::view::Camera const& cam, core::view::Camera::PerspectiveParameters const& cam_intrinsics) = 0;

    virtual void on_cam_pose_change(core::view::Camera::Pose const& cam_pose) = 0;

    virtual void on_change_data(OptixTraversableHandle world) = 0;

    virtual void on_change_background(glm::vec4 const& bg) = 0;

    virtual void on_change_programs(std::tuple<OptixProgramGroup const*, uint32_t> const& programs) = 0;

    virtual void on_change_parameters() = 0;

    virtual void on_change_viewport(glm::uvec2 const& viewport, std::shared_ptr<CUDAFramebuffer> fbo) = 0;

    virtual void on_change_sbt(std::tuple<void const*, uint32_t, uint64_t> const& records) = 0;

    core::CallerSlot in_geo_slot_;

    std::unique_ptr<Context> optix_ctx_;

    CUdeviceptr frame_state_buffer_;

    MMOptixSBT sbt_;

    OptixPipeline pipeline_ = 0;

    device::FrameState frame_state_;

    glm::uvec2 current_fb_size_;

    unsigned int frame_id_ = std::numeric_limits<unsigned int>::max();

    std::size_t in_data_hash_ = std::numeric_limits<std::size_t>::max();

    core::view::Camera::Pose old_cam_pose_;

    core::view::Camera::PerspectiveParameters old_cam_intrinsics_;

    glm::vec4 old_bg_ = glm::vec4(-1);
};
} // namespace megamol::optix_hpg
