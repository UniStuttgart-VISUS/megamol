#include "Renderer.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/view/Camera_2.h"

#include "raygen.h"

#include "optix/CallGeometry.h"

#include "vislib/graphics/gl/IncludeAllGL.h"

#include "optix/Utils.h"

#include "optix_stubs.h"

namespace megamol::optix_hpg {
extern "C" const char embedded_raygen_programs[];
extern "C" const char embedded_picking_programs[];
extern "C" const char embedded_miss_programs[];
} // namespace megamol::optix_hpg


megamol::optix_hpg::Renderer::Renderer()
        : _in_geo_slot("inGeo", "")
        /*, flags_write_slot_("flagsWrite", "")
        , flags_read_slot_("flagsRead", "")*/
        , spp_slot_("spp", "")
        , max_bounces_slot_("max bounces", "")
        , accumulate_slot_("accumulate", "")
        , intensity_slot_("intensity", "")
        , enable_picking_slot_("picking::enable", "") {
    _in_geo_slot.SetCompatibleCall<CallGeometryDescription>();
    MakeSlotAvailable(&_in_geo_slot);

    /*flags_write_slot_.SetCompatibleCall<core::FlagCallWrite_CPUDescription>();
    MakeSlotAvailable(&flags_write_slot_);

    flags_read_slot_.SetCompatibleCall<core::FlagCallRead_CPUDescription>();
    MakeSlotAvailable(&flags_read_slot_);*/

    spp_slot_ << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&spp_slot_);

    max_bounces_slot_ << new core::param::IntParam(0, 0);
    MakeSlotAvailable(&max_bounces_slot_);

    accumulate_slot_ << new core::param::BoolParam(true);
    MakeSlotAvailable(&accumulate_slot_);

    intensity_slot_ << new core::param::FloatParam(1.0f, std::numeric_limits<float>::min());
    MakeSlotAvailable(&intensity_slot_);

    enable_picking_slot_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&enable_picking_slot_);

    // Callback should already be set by RendererModule
    this->MakeSlotAvailable(&this->chainRenderSlot);

    // Callback should already be set by RendererModule
    this->MakeSlotAvailable(&this->renderSlot);
}


megamol::optix_hpg::Renderer::~Renderer() {
    this->Release();
}


void megamol::optix_hpg::Renderer::setup() {
    raygen_module_ = MMOptixModule(embedded_raygen_programs, optix_ctx_->GetOptiXContext(),
        &optix_ctx_->GetModuleCompileOptions(), &optix_ctx_->GetPipelineCompileOptions(),
        MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_RAYGEN,
        {{MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_GENERIC, "raygen_program"}});
    picking_module_ = MMOptixModule(embedded_picking_programs, optix_ctx_->GetOptiXContext(),
        &optix_ctx_->GetModuleCompileOptions(), &optix_ctx_->GetPipelineCompileOptions(),
        MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_RAYGEN,
        {{MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_GENERIC, "picking_program"}});
    miss_module_ = MMOptixModule(embedded_miss_programs, optix_ctx_->GetOptiXContext(),
        &optix_ctx_->GetModuleCompileOptions(), &optix_ctx_->GetPipelineCompileOptions(),
        MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_MISS,
        {{MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_GENERIC, "miss_program"}});
    miss_occlusion_module_ = MMOptixModule(embedded_miss_programs, optix_ctx_->GetOptiXContext(),
        &optix_ctx_->GetModuleCompileOptions(), &optix_ctx_->GetPipelineCompileOptions(),
        MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_MISS,
        {{MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_GENERIC, "miss_program_occlusion"}});

    OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(raygen_module_, &_sbt_raygen_record));
    OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(picking_module_, &sbt_picking_record_));
    OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(miss_module_, &sbt_miss_records_[0]));
    OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(miss_occlusion_module_, &sbt_miss_records_[1]));

    CUDA_CHECK_ERROR(cuMemAlloc(&_frame_state_buffer, sizeof(device::FrameState)));
    CUDA_CHECK_ERROR(cuMemAlloc(&pick_state_buffer_, sizeof(device::PickState)));

    _sbt_raygen_record.data.frameStateBuffer = (device::FrameState*) _frame_state_buffer;
    sbt_picking_record_.data.frameStateBuffer = (device::FrameState*) _frame_state_buffer;
    sbt_picking_record_.data.pickStateBuffer = (device::PickState*) pick_state_buffer_;

    pick_state_.primID = -1;
}


bool megamol::optix_hpg::Renderer::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {
    if (mods.test(core::view::Modifier::SHIFT) && action == core::view::MouseButtonAction::PRESS &&
        enable_picking_slot_.Param<core::param::BoolParam>()->Value()) {
        /*auto const screenX = mouse_x_ / _current_fb_size.Width();
        auto const screenY = 1.f - (mouse_y_ / _current_fb_size.Height());*/
        auto const screenX = mouse_x_;
        auto const screenY = _current_fb_size.Height() - mouse_y_;
        //auto const screenY = mouse_y_;

        std::vector<int> tmp_pick_buf(_current_fb_size.Area());
        CUDA_CHECK_ERROR(cuMemcpyDtoHAsync(tmp_pick_buf.data(), picking_pixel_buffer_,
            sizeof(int32_t) * _current_fb_size.Area(), optix_ctx_->GetExecStream()));

        pick_state_.primID = tmp_pick_buf[screenX + screenY * _current_fb_size.Width()];


        pick_state_.mouseCoord = glm::uvec2(screenX, screenY);
        pick_state_.primID = -1;

        CUDA_CHECK_ERROR(
            cuMemcpyHtoDAsync(pick_state_buffer_, &pick_state_, sizeof(pick_state_), optix_ctx_->GetExecStream()));

        OPTIX_CHECK_ERROR(optixLaunch(_pipeline, optix_ctx_->GetExecStream(), 0, 0, picking_sbt_, 1, 1, 1));

        CUDA_CHECK_ERROR(
            cuMemcpyDtoHAsync(&pick_state_, pick_state_buffer_, sizeof(pick_state_), optix_ctx_->GetExecStream()));

        core::utility::log::Log::DefaultLog.WriteInfo("[OptiXRenderer]: Picking result -> %d", pick_state_.primID);

        return true;
    }

    return false;
}


bool megamol::optix_hpg::Renderer::OnMouseMove(double x, double y) {
    this->mouse_x_ = static_cast<float>(x);
    this->mouse_y_ = static_cast<float>(y);
    return false;
}


bool megamol::optix_hpg::Renderer::Render(CallRender3DCUDA& call) {
    auto const width = call.GetFramebuffer()->width;
    auto const height = call.GetFramebuffer()->height;
    auto viewport = vislib::math::Rectangle<int>(0, 0, width, height);

    static bool not_init = true;

    if (not_init) {
        setup();

        _sbt_raygen_record.data.fbSize = glm::uvec2(viewport.Width(), viewport.Height());
        _sbt_raygen_record.data.col_surf = call.GetFramebuffer()->colorBuffer;
        _sbt_raygen_record.data.depth_surf = call.GetFramebuffer()->depthBuffer;

        sbt_picking_record_.data.fbSize = glm::uvec2(viewport.Width(), viewport.Height());

        call.GetFramebuffer()->data.exec_stream = optix_ctx_->GetExecStream();

        _current_fb_size = viewport;
        if (picking_pixel_buffer_ != 0) {
            CUDA_CHECK_ERROR(cuMemFree(picking_pixel_buffer_));
        }
        CUDA_CHECK_ERROR(cuMemAlloc(&picking_pixel_buffer_, sizeof(int32_t) * viewport.Width() * viewport.Height()));
        CUDA_CHECK_ERROR(cuMemsetD32Async(picking_pixel_buffer_, -1, viewport.Area(), optix_ctx_->GetExecStream()));
        _sbt_raygen_record.data.picking_buffer = (int*) picking_pixel_buffer_;

        not_init = false;
    }

    auto in_geo = _in_geo_slot.CallAs<CallGeometry>();
    if (in_geo == nullptr)
        return false;

    in_geo->set_ctx(optix_ctx_.get());
    if (!(*in_geo)())
        return false;

    bool rebuild_sbt = false;

    if (viewport != _current_fb_size) {
        _sbt_raygen_record.data.fbSize = glm::uvec2(viewport.Width(), viewport.Height());
        _sbt_raygen_record.data.col_surf = call.GetFramebuffer()->colorBuffer;
        _sbt_raygen_record.data.depth_surf = call.GetFramebuffer()->depthBuffer;
        _frame_state.frameIdx = 0;

        sbt_picking_record_.data.fbSize = glm::uvec2(viewport.Width(), viewport.Height());

        rebuild_sbt = true;

        _current_fb_size = viewport;
        if (picking_pixel_buffer_ != 0) {
            CUDA_CHECK_ERROR(cuMemFree(picking_pixel_buffer_));
        }
        CUDA_CHECK_ERROR(cuMemAlloc(&picking_pixel_buffer_, sizeof(int32_t) * viewport.Width() * viewport.Height()));
        CUDA_CHECK_ERROR(cuMemsetD32Async(picking_pixel_buffer_, -1, viewport.Area(), optix_ctx_->GetExecStream()));
        _sbt_raygen_record.data.picking_buffer = (int*)picking_pixel_buffer_;
    }

    // Camera
    core::view::Camera_2 cam;
    call.GetCamera(cam);
    cam_type::snapshot_type snapshot;
    cam_type::matrix_type viewTemp, projTemp;
    // Generate complete snapshot and calculate matrices
    cam.calc_matrices(snapshot, viewTemp, projTemp, core::thecam::snapshot_content::all);
    auto const depth_A = projTemp(2, 2);
    auto const depth_B = projTemp(2, 3);
    auto const depth_C = projTemp(3, 2);
    auto const depth_params = glm::vec3(depth_A, depth_B, depth_C);
    auto curCamPos = snapshot.position;
    auto curCamView = snapshot.view_vector;
    auto curCamRight = snapshot.right_vector;
    auto curCamUp = snapshot.up_vector;
    // auto curCamNearClip = snapshot.frustum_near;
    auto curCamNearClip = 100;
    auto curCamAspect = snapshot.resolution_aspect;
    auto hfov = cam.half_aperture_angle_radians();

    auto th = std::tan(hfov) * curCamNearClip;
    auto rw = th * curCamAspect;

    _frame_state.camera_center = glm::vec3(curCamPos.x(), curCamPos.y(), curCamPos.z());
    _frame_state.camera_front = glm::vec3(curCamView.x(), curCamView.y(), curCamView.z());
    _frame_state.camera_right = glm::vec3(curCamRight.x(), curCamRight.y(), curCamRight.z());
    _frame_state.camera_up = glm::vec3(curCamUp.x(), curCamUp.y(), curCamUp.z());

    _frame_state.rw = rw;
    _frame_state.th = th;
    _frame_state.near = curCamNearClip;

    _frame_state.samplesPerPixel = spp_slot_.Param<core::param::IntParam>()->Value();
    _frame_state.maxBounces = max_bounces_slot_.Param<core::param::IntParam>()->Value();
    _frame_state.accumulate = accumulate_slot_.Param<core::param::BoolParam>()->Value();

    _frame_state.depth_params = depth_params;
    _frame_state.intensity = intensity_slot_.Param<core::param::FloatParam>()->Value();

    if (old_cam_snap.position != snapshot.position || old_cam_snap.view_vector != snapshot.view_vector ||
        old_cam_snap.right_vector != snapshot.right_vector || old_cam_snap.up_vector != snapshot.up_vector ||
        is_dirty()) {
        _frame_state.frameIdx = 0;
        old_cam_snap = snapshot;
        reset_dirty();
    } else {
        ++_frame_state.frameIdx;
    }

    if (old_bg != call.BackgroundColor()) {
        _frame_state.background = call.BackgroundColor();
        sbt_miss_records_[0].data.bg = _frame_state.background;
        old_bg = call.BackgroundColor();
        _frame_state.frameIdx = 0;
        rebuild_sbt = true;
    }

    CUDA_CHECK_ERROR(
        cuMemcpyHtoDAsync(_frame_state_buffer, &_frame_state, sizeof(_frame_state), optix_ctx_->GetExecStream()));

    if (in_geo->FrameID() != _frame_id || in_geo->DataHash() != _in_data_hash) {
        _sbt_raygen_record.data.world = *in_geo->get_handle();
        sbt_picking_record_.data.world = *in_geo->get_handle();
        _frame_state.frameIdx = 0;

        rebuild_sbt = true;

        auto num_groups = 2 + in_geo->get_num_programs();
        std::vector<OptixProgramGroup> groups;
        groups.reserve(num_groups);
        groups.push_back(raygen_module_);
        groups.push_back(picking_module_);
        groups.push_back(miss_module_);
        std::for_each(in_geo->get_program_groups(), in_geo->get_program_groups() + in_geo->get_num_programs(),
            [&groups](OptixProgramGroup const el) { groups.push_back(el); });

        std::size_t log_size = 2048;
        std::string log;
        log.resize(log_size);

        OPTIX_CHECK_ERROR(optixPipelineCreate(optix_ctx_->GetOptiXContext(), &optix_ctx_->GetPipelineCompileOptions(),
            &optix_ctx_->GetPipelineLinkOptions(), groups.data(), groups.size(), log.data(), &log_size, &_pipeline));


        _frame_id = in_geo->FrameID();
        _in_data_hash = in_geo->DataHash();
    }

    if (rebuild_sbt) {
        sbt_.SetSBT(&_sbt_raygen_record, sizeof(_sbt_raygen_record), nullptr, 0, sbt_miss_records_.data(),
            sizeof(SBTRecord<device::MissData>), sbt_miss_records_.size(), in_geo->get_record(),
            in_geo->get_record_stride(), in_geo->get_num_records(), nullptr, 0, 0, optix_ctx_->GetExecStream());
        picking_sbt_.SetSBT(&sbt_picking_record_, sizeof(sbt_picking_record_), nullptr, 0, sbt_miss_records_.data(),
            sizeof(SBTRecord<device::MissData>), sbt_miss_records_.size(), in_geo->get_record(),
            in_geo->get_record_stride(), in_geo->get_num_records(), nullptr, 0, 0, optix_ctx_->GetExecStream());
    }

    OPTIX_CHECK_ERROR(
        optixLaunch(_pipeline, optix_ctx_->GetExecStream(), 0, 0, sbt_, viewport.Width(), viewport.Height(), 1));

    if (enable_picking_slot_.Param<core::param::BoolParam>()->Value()) {
        if (pick_state_.primID != -1) {
            in_geo->set_pick_idx(pick_state_.primID);
            pick_state_.primID = -1;
        }
        //auto fcr = flags_read_slot_.CallAs<core::FlagCallRead_CPU>();
        //auto fcw = flags_write_slot_.CallAs<core::FlagCallWrite_CPU>();
        //if (fcr != nullptr && fcw != nullptr && pick_state_.primID != -1) {
        //    if ((*fcr)(0)) {
        //        auto flags = fcr->getData();
        //        auto version = fcr->version();
        //        try {
        //            auto& flag = flags->flags->at(pick_state_.primID);
        //            if (flag == core::FlagStorage::SELECTED) {
        //                flag = core::FlagStorage::ENABLED;
        //            } else {
        //                flag = core::FlagStorage::SELECTED;
        //            }
        //            fcw->setData(flags, version + 1);
        //            //fcr->setData(flags, version + 1);
        //            (*fcw)(0);
        //            pick_state_.primID = -1;
        //        } catch (...) { return true; }
        //    }
        //}
    }

    return true;
}


bool megamol::optix_hpg::Renderer::GetExtents(CallRender3DCUDA& call) {
    auto in_geo = _in_geo_slot.CallAs<CallGeometry>();
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


bool megamol::optix_hpg::Renderer::create() {
    auto& cuda_res = frontend_resources.get<frontend_resources::CUDA_Context>();
    if (cuda_res.ctx_ != nullptr) {
        optix_ctx_ = std::make_unique<Context>(cuda_res);
    } else {
        return false;
    }
    return true;
}


void megamol::optix_hpg::Renderer::release() {
    CUDA_CHECK_ERROR(cuMemFree(_frame_state_buffer));
    OPTIX_CHECK_ERROR(optixPipelineDestroy(_pipeline));
}
