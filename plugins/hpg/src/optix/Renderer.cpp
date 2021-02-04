#include "stdafx.h"
#include "Renderer.h"

#include "mmcore/param/IntParam.h"
#include "mmcore/view/Camera_2.h"
#include "mmcore/param/BoolParam.h"

#include "raygen.h"

#include "hpg/optix/CallGeometry.h"

#include "vislib/graphics/gl/IncludeAllGL.h"

#include "hpg/optix/Utils.h"

#include "optix_stubs.h"

namespace megamol::hpg::optix {
extern "C" const char embedded_raygen_programs[];
extern "C" const char embedded_miss_programs[];
} // namespace megamol::hpg::optix


megamol::hpg::optix::Renderer::Renderer()
        : _in_geo_slot("inGeo", "")
        , _in_ctx_slot("inCtx", "")
        , spp_slot_("spp", "")
        , max_bounces_slot_("max bounces", "")
        , accumulate_slot_("accumulate", "") {
    _in_geo_slot.SetCompatibleCall<CallGeometryDescription>();
    MakeSlotAvailable(&_in_geo_slot);

    _in_ctx_slot.SetCompatibleCall<CallContextDescription>();
    MakeSlotAvailable(&_in_ctx_slot);

    spp_slot_ << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&spp_slot_);

    max_bounces_slot_ << new core::param::IntParam(0, 0);
    MakeSlotAvailable(&max_bounces_slot_);

    accumulate_slot_ << new core::param::BoolParam(true);
    MakeSlotAvailable(&accumulate_slot_);
}


megamol::hpg::optix::Renderer::~Renderer() {
    this->Release();
}


void megamol::hpg::optix::Renderer::setup(CallContext& ctx) {
    raygen_module_ = MMOptixModule(embedded_raygen_programs, ctx.get_ctx(), ctx.get_module_options(),
        ctx.get_pipeline_options(), OPTIX_PROGRAM_GROUP_KIND_RAYGEN, {"raygen_program"});
    miss_module_ = MMOptixModule(embedded_miss_programs, ctx.get_ctx(), ctx.get_module_options(),
        ctx.get_pipeline_options(), OPTIX_PROGRAM_GROUP_KIND_MISS, {"miss_program"});
    miss_occlusion_module_ = MMOptixModule(embedded_miss_programs, ctx.get_ctx(), ctx.get_module_options(),
        ctx.get_pipeline_options(), OPTIX_PROGRAM_GROUP_KIND_MISS, {"miss_program_occlusion"});

    OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(raygen_module_, &_sbt_raygen_record));
    OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(miss_module_, &sbt_miss_records_[0]));
    OPTIX_CHECK_ERROR(optixSbtRecordPackHeader(miss_occlusion_module_, &sbt_miss_records_[1]));

    CUDA_CHECK_ERROR(cuMemAlloc(&_frame_state_buffer, sizeof(device::FrameState)));

    _sbt_raygen_record.data.frameStateBuffer = (device::FrameState*) _frame_state_buffer;
}


bool megamol::hpg::optix::Renderer::Render(core::view::CallRender3D_2& call) {
    auto viewport = call.GetViewport();

    static bool not_init = true;

    auto in_ctx = _in_ctx_slot.CallAs<CallContext>();
    if (in_ctx == nullptr)
        return false;
    if (!(*in_ctx)(0))
        return false;

    if (not_init) {
        setup(*in_ctx);

        _sbt_raygen_record.data.fbSize = glm::uvec2(viewport.Width(), viewport.Height());

        _current_fb_size = viewport;

        not_init = false;
    }

    auto in_geo = _in_geo_slot.CallAs<CallGeometry>();
    if (in_geo == nullptr)
        return false;

    if (!(*in_geo)())
        return false;

    bool rebuild_sbt = false;

    if (_fb_texture == 0) {
        glGenTextures(1, &_fb_texture);
        glBindTexture(GL_TEXTURE_2D, _fb_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA32F, viewport.Width(), viewport.Height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    }

    if (_fbo_pbo == 0) {
        glGenBuffers(1, &_fbo_pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _fbo_pbo);
        glBufferData(
            GL_PIXEL_UNPACK_BUFFER, sizeof(float) * 4 * viewport.Width() * viewport.Height(), nullptr, GL_STREAM_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        if (_fbo_res != nullptr) {
            cuGraphicsUnregisterResource(_fbo_res);
        }
        cuGraphicsGLRegisterBuffer(&_fbo_res, _fbo_pbo, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
    }

    if (viewport != _current_fb_size) {
        _sbt_raygen_record.data.fbSize = glm::uvec2(viewport.Width(), viewport.Height());
        rebuild_sbt = true;

        glDeleteTextures(1, &_fb_texture);
        glGenTextures(1, &_fb_texture);
        glBindTexture(GL_TEXTURE_2D, _fb_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA32F, viewport.Width(), viewport.Height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        glDeleteBuffers(1, &_fbo_pbo);
        glGenBuffers(1, &_fbo_pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _fbo_pbo);
        glBufferData(
            GL_PIXEL_UNPACK_BUFFER, sizeof(float) * 4 * viewport.Width() * viewport.Height(), nullptr, GL_STREAM_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        if (_fbo_res != nullptr) {
            cuGraphicsUnregisterResource(_fbo_res);
        }
        cuGraphicsGLRegisterBuffer(&_fbo_res, _fbo_pbo, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

        _current_fb_size = viewport;
    }

    // Camera
    core::view::Camera_2 cam;
    call.GetCamera(cam);
    cam_type::snapshot_type snapshot;
    cam_type::matrix_type viewTemp, projTemp;
    // Generate complete snapshot and calculate matrices
    cam.calc_matrices(snapshot, viewTemp, projTemp, core::thecam::snapshot_content::all);
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

    _frame_state.camera_lens_center = glm::vec3(curCamPos.x(), curCamPos.y(), curCamPos.z());
    _frame_state.camera_screen_00 = glm::vec3(curCamView.x(), curCamView.y(), curCamView.z());
    _frame_state.camera_screen_du = glm::vec3(curCamRight.x(), curCamRight.y(), curCamRight.z());
    _frame_state.camera_screen_dv = glm::vec3(curCamUp.x(), curCamUp.y(), curCamUp.z());

    _frame_state.rw = rw;
    _frame_state.th = th;
    _frame_state.near = curCamNearClip;

    _frame_state.samplesPerPixel = spp_slot_.Param<core::param::IntParam>()->Value();
    _frame_state.maxBounces = max_bounces_slot_.Param<core::param::IntParam>()->Value();
    _frame_state.accumulate = accumulate_slot_.Param<core::param::BoolParam>()->Value();

    if (old_cam_snap.position != snapshot.position || old_cam_snap.view_vector != snapshot.view_vector ||
        old_cam_snap.right_vector != snapshot.right_vector || old_cam_snap.up_vector != snapshot.up_vector || is_dirty()) {
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
        cuMemcpyHtoDAsync(_frame_state_buffer, &_frame_state, sizeof(_frame_state), in_ctx->get_exec_stream()));
    // owlBufferUpload(_frame_state_buffer, &_frame_state);

    if (in_geo->FrameID() != _frame_id || in_geo->DataHash() != _in_data_hash) {
        _sbt_raygen_record.data.world = *in_geo->get_handle();

        rebuild_sbt = true;

        auto num_groups = 2 + in_geo->get_num_programs();
        std::vector<OptixProgramGroup> groups;
        groups.reserve(num_groups);
        groups.push_back(raygen_module_);
        groups.push_back(miss_module_);
        std::for_each(in_geo->get_program_groups(), in_geo->get_program_groups() + in_geo->get_num_programs(),
            [&groups](OptixProgramGroup const el) { groups.push_back(el); });

        std::size_t log_size = 2048;
        std::string log;
        log.resize(log_size);

        OPTIX_CHECK_ERROR(optixPipelineCreate(in_ctx->get_ctx(), in_ctx->get_pipeline_options(),
            in_ctx->get_pipeline_link_options(), groups.data(), groups.size(), log.data(), &log_size, &_pipeline));


        _frame_id = in_geo->FrameID();
        _in_data_hash = in_geo->DataHash();
    }

    cuGraphicsMapResources(1, &_fbo_res, in_ctx->get_exec_stream());
    CUdeviceptr pbo_ptr = 0;
    std::size_t pbo_size = 0;
    cuGraphicsResourceGetMappedPointer(&pbo_ptr, &pbo_size, _fbo_res);
    if (_old_pbo_ptr != pbo_ptr) {
        _sbt_raygen_record.data.colorBufferPtr = (glm::vec4*) pbo_ptr;

        _old_pbo_ptr = pbo_ptr;
        rebuild_sbt = true;
    }

    if (rebuild_sbt) {
        sbt_.SetSBT(&_sbt_raygen_record, sizeof(_sbt_raygen_record), nullptr, 0, sbt_miss_records_.data(),
            sizeof(SBTRecord<device::MissData>), sbt_miss_records_.size(), in_geo->get_record(),
            in_geo->get_record_stride(), in_geo->get_num_records(), nullptr, 0, 0, in_ctx->get_exec_stream());
    }

    OPTIX_CHECK_ERROR(
        optixLaunch(_pipeline, in_ctx->get_exec_stream(), 0, 0, sbt_, viewport.Width(), viewport.Height(), 1));

    cuGraphicsUnmapResources(1, &_fbo_res, in_ctx->get_exec_stream());

    glDisable(GL_LIGHTING);
    glColor3f(1, 1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, _fb_texture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _fbo_pbo);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, viewport.Width(), viewport.Height(), GL_RGBA, GL_FLOAT, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    // glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, viewport.Width(), viewport.Height(), GL_RGBA,
    //     GL_FLOAT, colorPtr);

    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, viewport.Width(), viewport.Height());

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, (float) viewport.Width(), 0.f, (float) viewport.Height(), -1.f, 1.f);

    glBegin(GL_QUADS);
    {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);

        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float) viewport.Height(), 0.f);

        glTexCoord2f(1.f, 1.f);
        glVertex3f((float) viewport.Width(), (float) viewport.Height(), 0.f);

        glTexCoord2f(1.f, 0.f);
        glVertex3f((float) viewport.Width(), 0.f, 0.f);
    }
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_TEXTURE_2D);


    return true;
}


bool megamol::hpg::optix::Renderer::GetExtents(core::view::CallRender3D_2& call) {
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


bool megamol::hpg::optix::Renderer::create() {
    return true;
}


void megamol::hpg::optix::Renderer::release() {
    CUDA_CHECK_ERROR(cuMemFree(_frame_state_buffer));
    OPTIX_CHECK_ERROR(optixPipelineDestroy(_pipeline));
}
