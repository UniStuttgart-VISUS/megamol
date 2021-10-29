#include "AbstractRenderer.h"

#include "optix/CallGeometry.h"


megamol::optix_hpg::AbstractRenderer::AbstractRenderer() : in_geo_slot_("inGeo", "") {
    in_geo_slot_.SetCompatibleCall<CallGeometryDescription>();
    MakeSlotAvailable(&in_geo_slot_);

    // Callback should already be set by RendererModule
    this->MakeSlotAvailable(&this->chainRenderSlot);

    // Callback should already be set by RendererModule
    this->MakeSlotAvailable(&this->renderSlot);
}


megamol::optix_hpg::AbstractRenderer::~AbstractRenderer() {
    this->Release();
}


bool megamol::optix_hpg::AbstractRenderer::Render(CallRender3DCUDA& call) {
    auto const viewport = glm::uvec2(call.GetFramebuffer()->width, call.GetFramebuffer()->height);

    call.GetFramebuffer()->data.exec_stream = optix_ctx_->GetExecStream();

    auto in_geo = in_geo_slot_.CallAs<CallGeometry>();
    if (in_geo == nullptr)
        return false;

    in_geo->set_ctx(optix_ctx_.get());
    if (!(*in_geo)())
        return false;

    bool rebuild_sbt = false;

    // change viewport
    if (viewport != current_fb_size_) {
        on_change_viewport(viewport, call.GetFramebuffer());

        rebuild_sbt = true;

        current_fb_size_ = viewport;
    }

    // Camera
    core::view::Camera cam = call.GetCamera();

    auto const cam_pose = cam.get<core::view::Camera::Pose>();
    auto const cam_intrinsics = cam.get<core::view::Camera::PerspectiveParameters>();

    // change camera parameters
    if (!(cam_intrinsics == old_cam_intrinsics_)) {
        on_cam_param_change(cam, cam_intrinsics);
        old_cam_intrinsics_ = cam_intrinsics;
    }

    // change camera pose
    if (!(cam_pose == old_cam_pose_)) {
        on_cam_pose_change(cam_pose);
        old_cam_pose_ = cam_pose;
    }

    // update parameters
    if (is_dirty()) {
        on_change_parameters();
        reset_dirty();
    }

    // change background color
    if (old_bg_ != call.BackgroundColor()) {
        on_change_background(call.BackgroundColor());
        rebuild_sbt = true;
        old_bg_ = call.BackgroundColor();
    }

    // change data
    if (frame_id_ != in_geo->FrameID() || in_data_hash_ != in_geo->DataHash()) {
        on_change_data(*in_geo->get_handle());

        rebuild_sbt = true;
        frame_id_ = in_geo->FrameID();
        in_data_hash_ = in_geo->DataHash();
    }

    // change programs
    if (in_geo->has_program_update()) {
        on_change_programs(in_geo->get_program_groups());
    }

    CUDA_CHECK_ERROR(
        cuMemcpyHtoDAsync(frame_state_buffer_, &frame_state_, sizeof(frame_state_), optix_ctx_->GetExecStream()));

    if (rebuild_sbt || in_geo->has_sbt_update()) {
        on_change_sbt(in_geo->get_record());
    }

    OPTIX_CHECK_ERROR(optixLaunch(pipeline_, optix_ctx_->GetExecStream(), 0, 0, sbt_, viewport.x, viewport.y, 1));

    ++frame_state_.frameIdx;

    return true;
}


bool megamol::optix_hpg::AbstractRenderer::GetExtents(CallRender3DCUDA& call) {
    auto in_geo = in_geo_slot_.CallAs<CallGeometry>();
    if (in_geo != nullptr) {
        in_geo->SetFrameID(static_cast<unsigned int>(call.Time()));
        if (!(*in_geo)(1))
            return false;
        call.SetTimeFramesCount(in_geo->FrameCount());

        call.AccessBoundingBoxes() = in_geo->AccessBoundingBoxes();
    } else {
        call.SetTimeFramesCount(1);
        call.AccessBoundingBoxes().Clear();
    }

    return true;
}


bool megamol::optix_hpg::AbstractRenderer::create() {
    auto& cuda_res = frontend_resources.get<frontend_resources::CUDA_Context>();
    if (cuda_res.ctx_ != nullptr) {
        optix_ctx_ = std::make_unique<Context>(cuda_res);
    } else {
        return false;
    }

    on_change_parameters();

    CUDA_CHECK_ERROR(cuMemAlloc(&frame_state_buffer_, sizeof(device::FrameState)));

    setup();

    return true;
}


void megamol::optix_hpg::AbstractRenderer::release() {
    CUDA_CHECK_ERROR(cuMemFree(frame_state_buffer_));
    OPTIX_CHECK_ERROR(optixPipelineDestroy(pipeline_));
}
