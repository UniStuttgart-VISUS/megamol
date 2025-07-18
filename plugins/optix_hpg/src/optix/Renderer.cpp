#include "Renderer.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include "raygen.h"

#include "optix/CallGeometry.h"

#include "optix/Utils.h"

#include "optix_stubs.h"

namespace megamol::optix_hpg {
extern "C" const char embedded_raygen_programs[];
extern "C" const char embedded_miss_programs[];
extern "C" const char embedded_miss_occlusion_programs[];
} // namespace megamol::optix_hpg


megamol::optix_hpg::Renderer::Renderer()
        : AbstractRenderer()
        , spp_slot_("spp", "")
        , max_bounces_slot_("max bounces", "")
        , accumulate_slot_("accumulate", "")
        , intensity_slot_("intensity", "") {
    spp_slot_ << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&spp_slot_);

    max_bounces_slot_ << new core::param::IntParam(0, 0);
    MakeSlotAvailable(&max_bounces_slot_);

    accumulate_slot_ << new core::param::BoolParam(true);
    MakeSlotAvailable(&accumulate_slot_);

    intensity_slot_ << new core::param::FloatParam(1.0f, std::numeric_limits<float>::min());
    MakeSlotAvailable(&intensity_slot_);
}


megamol::optix_hpg::Renderer::~Renderer() {
    this->Release();
}


void megamol::optix_hpg::Renderer::setup() {
    auto const& optix_ctx = get_context();
    raygen_module_ = MMOptixModule(embedded_raygen_programs, optix_ctx->GetOptiXContext(),
        &optix_ctx->GetModuleCompileOptions(), &optix_ctx->GetPipelineCompileOptions(),
        MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_RAYGEN,
        {{MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_GENERIC, "raygen_program"}});
    miss_module_ = MMOptixModule(embedded_miss_programs, optix_ctx->GetOptiXContext(),
        &optix_ctx->GetModuleCompileOptions(), &optix_ctx->GetPipelineCompileOptions(),
        MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_MISS,
        {{MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_GENERIC, "miss_program"}});
    miss_occlusion_module_ = MMOptixModule(embedded_miss_occlusion_programs, optix_ctx->GetOptiXContext(),
        &optix_ctx->GetModuleCompileOptions(), &optix_ctx->GetPipelineCompileOptions(),
        MMOptixModule::MMOptixProgramGroupKind::MMOPTIX_PROGRAM_GROUP_KIND_MISS,
        {{MMOptixModule::MMOptixNameKind::MMOPTIX_NAME_GENERIC, "miss_program_occlusion"}});

    OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(raygen_module_, &sbt_raygen_record_));
    OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(miss_module_, &sbt_miss_records_[0]));
    OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(miss_occlusion_module_, &sbt_miss_records_[1]));

    sbt_raygen_record_.data.frameStateBuffer = (device::FrameState*) get_frame_state_buffer();
}


void megamol::optix_hpg::Renderer::on_cam_param_change(
    core::view::Camera const& cam, core::view::Camera::PerspectiveParameters const& cam_intrinsics) {
    auto proj = cam.getProjectionMatrix();
    // Generate complete snapshot and calculate matrices
    // is this a) correct and b) actually needed for the new cam?
    auto const depth_A = proj[2][2];
    auto const depth_B = proj[2][3];
    auto const depth_C = proj[3][2];
    auto const depth_params = glm::vec3(depth_A, depth_B, depth_C);
    auto& frame_state = get_framestate();
    frame_state.depth_params = depth_params;

    auto const curCamNearClip = 100;
    auto const curCamAspect = cam_intrinsics.aspect;
    auto const hfov = 0.5f * cam_intrinsics.fovy;

    auto const th = std::tan(hfov) * curCamNearClip;
    auto const rw = th * curCamAspect;

    frame_state.rw = rw;
    frame_state.th = th;
    frame_state.near = curCamNearClip;

    frame_state.frameIdx = 0;
}


void megamol::optix_hpg::Renderer::on_cam_pose_change(core::view::Camera::Pose const& cam_pose) {
    auto& frame_state = get_framestate();
    frame_state.camera_center = glm::vec3(cam_pose.position.x, cam_pose.position.y, cam_pose.position.z);
    frame_state.camera_front = glm::vec3(cam_pose.direction.x, cam_pose.direction.y, cam_pose.direction.z);
    frame_state.camera_up = glm::vec3(cam_pose.up.x, cam_pose.up.y, cam_pose.up.z);
    auto const curCamRight = glm::cross(cam_pose.direction, cam_pose.up);
    frame_state.camera_right = glm::vec3(curCamRight.x, curCamRight.y, curCamRight.z);

    frame_state.frameIdx = 0;
}


void megamol::optix_hpg::Renderer::on_change_data(OptixTraversableHandle world) {
    auto& frame_state = get_framestate();
    sbt_raygen_record_.data.world = world;
    frame_state.frameIdx = 0;
}


void megamol::optix_hpg::Renderer::on_change_background(glm::vec4 const& bg) {
    auto& frame_state = get_framestate();
    frame_state.background = bg;
    sbt_miss_records_[0].data.bg = bg;

    frame_state.frameIdx = 0;
}


void megamol::optix_hpg::Renderer::on_change_programs(std::tuple<OptixProgramGroup const*, uint32_t> const& programs) {
    auto& frame_state = get_framestate();
    frame_state.frameIdx = 0;

    auto const& [geo_progs, num_geo_progs] = programs;

    auto num_groups = 3 + num_geo_progs;
    std::vector<OptixProgramGroup> groups;
    groups.reserve(num_groups);
    groups.push_back(raygen_module_);
    groups.push_back(miss_module_);
    groups.push_back(miss_occlusion_module_);
    std::for_each(
        geo_progs, geo_progs + num_geo_progs, [&groups](OptixProgramGroup const el) { groups.push_back(el); });

    std::size_t log_size = 2048;
    std::string log;
    log.resize(log_size);

    auto const& optix_ctx = get_context();
    auto& pipeline = get_pipeline();
    cuStreamSynchronize(optix_ctx->GetExecStream());
    if (pipeline) {
        OPTIX_CHECK_ERROR(optixPipelineDestroy(pipeline));
    }

    OPTIX_CHECK_ERROR(optixPipelineCreate(optix_ctx->GetOptiXContext(), &optix_ctx->GetPipelineCompileOptions(),
        &optix_ctx->GetPipelineLinkOptions(), groups.data(), groups.size(), log.data(), &log_size, &pipeline));

    core::utility::log::Log::DefaultLog.WriteInfo("[OptiX]: {}", log);
}


void megamol::optix_hpg::Renderer::on_change_parameters() {
    auto& frame_state = get_framestate();
    frame_state.samplesPerPixel = spp_slot_.Param<core::param::IntParam>()->Value();
    frame_state.maxBounces = max_bounces_slot_.Param<core::param::IntParam>()->Value();
    frame_state.accumulate = accumulate_slot_.Param<core::param::BoolParam>()->Value();

    frame_state.intensity = intensity_slot_.Param<core::param::FloatParam>()->Value();

    frame_state.frameIdx = 0;
}


void megamol::optix_hpg::Renderer::on_change_viewport(
    glm::uvec2 const& viewport, std::shared_ptr<CUDAFramebuffer> fbo) {
    auto& frame_state = get_framestate();
    sbt_raygen_record_.data.fbSize = viewport;
    sbt_raygen_record_.data.col_surf = fbo->colorBuffer;
    sbt_raygen_record_.data.depth_surf = fbo->depthBuffer;
    frame_state.frameIdx = 0;
}


void megamol::optix_hpg::Renderer::on_change_sbt(std::tuple<void const*, uint32_t, uint64_t> const& records) {
    auto const& [geo_records, num_geo_records, geo_records_stride] = records;
    auto const& optix_ctx = get_context();
    auto& sbt = get_sbt();
    sbt.SetSBT(&sbt_raygen_record_, sizeof(sbt_raygen_record_), nullptr, 0, sbt_miss_records_.data(),
        sizeof(SBTRecord<device::MissData>), sbt_miss_records_.size(), geo_records, geo_records_stride, num_geo_records,
        nullptr, 0, 0, optix_ctx->GetExecStream());
}
