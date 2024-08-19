#include "BaseRenderer.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FilePathParam.h"

#include <owl/common/math/box.h>
#include <owl/common/math/vec.h>

#include "framestate.h"
#include "raygen.h"

#include <glad/gl.h>

#include <tbb/parallel_for.h>

#include <cuda_runtime.h>

namespace megamol::optix_owl {
extern "C" const unsigned char raygenPrograms_ptx[];

BaseRenderer::BaseRenderer()
        : data_in_slot_("dataIn", "")
        , radius_slot_("radius", "")
        , rec_depth_slot_("rec_depth", "")
        , spp_slot_("spp", "")
        , accumulate_slot_("accumulate", "")
        , dump_debug_info_slot_("debug::dump", "")
        , debug_rdf_slot_("debug::rdf", "")
        , debug_output_path_slot_("debug::outpath", "") {
    data_in_slot_.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&data_in_slot_);

    radius_slot_ << new core::param::FloatParam(0.5f, std::numeric_limits<float>::min());
    MakeSlotAvailable(&radius_slot_);

    rec_depth_slot_ << new core::param::IntParam(0, 0, 20);
    MakeSlotAvailable(&rec_depth_slot_);

    spp_slot_ << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&spp_slot_);

    accumulate_slot_ << new core::param::BoolParam(true);
    MakeSlotAvailable(&accumulate_slot_);

    dump_debug_info_slot_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&dump_debug_info_slot_);

    debug_rdf_slot_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&debug_rdf_slot_);

    debug_output_path_slot_ << new core::param::FilePathParam(
        "./debug/", core::param::FilePathParam::FilePathFlags_::Flag_Directory);
    MakeSlotAvailable(&debug_output_path_slot_);
}

BaseRenderer::~BaseRenderer() {
    this->Release();
}

bool BaseRenderer::create() {
    ctx_ = owlContextCreate(nullptr, 1);
    owlContextSetRayTypeCount(ctx_, 1);

    raygen_module_ = owlModuleCreate(ctx_, reinterpret_cast<const char*>(raygenPrograms_ptx));

    frameStateBuffer_ = owlDeviceBufferCreate(ctx_, OWL_USER_TYPE(device::FrameState), 1, nullptr);

    OWLVarDecl rayGenVars[] = {{"colorBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::RayGenData, colorBufferPtr)},
        {"accumBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::RayGenData, accumBufferPtr)},
        {"frameStateBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::RayGenData, frameStateBuffer)},
        {"fbSize", OWL_INT2, OWL_OFFSETOF(device::RayGenData, fbSize)},
        {"world", OWL_GROUP, OWL_OFFSETOF(device::RayGenData, world)},
        {"rec_depth", OWL_INT, OWL_OFFSETOF(device::RayGenData, rec_depth)}, {/* sentinel to mark end of list */}};

    raygen_ = owlRayGenCreate(ctx_, raygen_module_, "raygen", sizeof(device::RayGenData), rayGenVars, -1);

    OWLVarDecl missProgVars[] = {{/* sentinel to mark end of list */}};

    miss_ = owlMissProgCreate(ctx_, raygen_module_, "miss", 0, missProgVars, -1);

    owlRayGenSet1i(raygen_, "rec_depth", rec_depth_slot_.Param<core::param::IntParam>()->Value());
    owlRayGenSetBuffer(raygen_, "frameStateBuffer", frameStateBuffer_);

    resizeFramebuffer(owl::common::vec2i(1920, 1080));

    framestate_.accumID = 0;
    framestate_.samplesPerPixel = spp_slot_.Param<core::param::IntParam>()->Value();

    return true;
}

void BaseRenderer::release() {
    owlBufferDestroy(particleBuffer_);
    owlBufferDestroy(accumBuffer_);
    owlBufferDestroy(colorBuffer_);
    owlBufferDestroy(frameStateBuffer_);
    owlModuleRelease(raygen_module_);
    owlContextDestroy(ctx_);
}

bool BaseRenderer::Render(mmstd_gl::CallRender3DGL& call) {
    auto in_data = data_in_slot_.CallAs<geocalls::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    in_data->SetFrameID(call.Time());
    if (!(*in_data)(1))
        return false;
    if (!(*in_data)(0))
        return false;

    if (current_fb_size_ != owl::common::vec2i(call.GetViewResolution().x, call.GetViewResolution().y)) {
        resizeFramebuffer(owl::common::vec2i(call.GetViewResolution().x, call.GetViewResolution().y));
        owlBuildPrograms(ctx_);
        owlBuildPipeline(ctx_);
        owlBuildSBT(ctx_);
    }

    core::view::Camera cam = call.GetCamera();

    auto const cam_pose = cam.get<core::view::Camera::Pose>();
    auto const cam_intrinsics = cam.get<core::view::Camera::PerspectiveParameters>();
    if (!(cam_pose == old_cam_pose_) || !(cam_intrinsics == old_cam_intrinsics_)) {
        framestate_.camera_lens_center =
            owl::common::vec3f(cam_pose.position.x, cam_pose.position.y, cam_pose.position.z);
        framestate_.camera_screen_du = owl::common::vec3f(cam_pose.right.x, cam_pose.right.y, cam_pose.right.z);
        framestate_.camera_screen_dv = owl::common::vec3f(cam_pose.up.x, cam_pose.up.y, cam_pose.up.z);
        auto vz = owl::common::vec3f(cam_pose.direction.x, cam_pose.direction.y, cam_pose.direction.z);

        /*const float aspect = static_cast<float>(current_fb_size_.x) / static_cast<float>(current_fb_size_.y);
        const float minFocalDistance = 1e6f * max(computeStableEpsilon(framestate_.camera_lens_center),
                                                  computeStableEpsilon(framestate_.camera_screen_du));
        float screen_height =
            2.f * tanf(70.f / 2 * (float) M_PI / 180.f) * max(minFocalDistance, 1.f);
        auto const vertical = screen_height * framestate_.camera_screen_dv;
        auto const horizontal = screen_height * aspect * framestate_.camera_screen_du;

        framestate_.camera_screen_00 = max(minFocalDistance, 1.f) * vz - 0.5f * vertical -
                                       0.5f * horizontal;*/


        /*auto const d = static_cast<float>(current_fb_size_.x) / tanf(2.f*cam_intrinsics.fovy);
        vz *= d;*/

        auto const curCamRight = glm::cross(cam_pose.direction, cam_pose.up);
        framestate_.camera_screen_du = owl::common::vec3f(curCamRight.x, curCamRight.y, curCamRight.z);

        framestate_.camera_screen_dz = vz;

        auto const curCamNearClip = 100;
        auto const curCamAspect = cam_intrinsics.aspect;
        auto const hfov = 0.5f * cam_intrinsics.fovy;

        auto const th = std::tan(hfov) * curCamNearClip;
        auto const rw = th * curCamAspect;

        framestate_.rw = rw;
        framestate_.th = th;
        framestate_.near_plane = curCamNearClip;

        framestate_.accumID = 0;

        old_cam_pose_ = cam_pose;
        old_cam_intrinsics_ = cam_intrinsics;
    }

    if (in_data->FrameID() != frame_id_ || in_data->DataHash() != in_data_hash_ || data_param_is_dirty()) {
        if (!assertData(*in_data))
            return false;
        frame_id_ = in_data->FrameID();
        in_data_hash_ = in_data->DataHash();
        data_param_reset_dirty();

        framestate_.accumID = 0;

        owlRayGenSetGroup(raygen_, "world", world_);

        owlBuildPrograms(ctx_);
        owlBuildPipeline(ctx_);
        owlBuildSBT(ctx_);
    }

    if (rec_depth_slot_.IsDirty()) {
        owlRayGenSet1i(raygen_, "rec_depth", rec_depth_slot_.Param<core::param::IntParam>()->Value());
        framestate_.accumID = 0;

        owlBuildPrograms(ctx_);
        owlBuildPipeline(ctx_);
        owlBuildSBT(ctx_);

        rec_depth_slot_.ResetDirty();
    }

    framestate_.samplesPerPixel = spp_slot_.Param<core::param::IntParam>()->Value();

    owlBufferUpload(frameStateBuffer_, &framestate_);
    owlRayGenLaunch2D(raygen_, current_fb_size_.x, current_fb_size_.y);
    cudaStreamSynchronize(owlContextGetStream(ctx_, 0));

    if (colorBuffer_) {
        auto color_ptr = reinterpret_cast<uint8_t const*>(owlBufferGetPointer(colorBuffer_, 0));
        glViewport(0, 0, current_fb_size_.x, current_fb_size_.y);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

        static unsigned int gl_tex_id = 0;
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        glDisable(GL_DEPTH_TEST);
        glEnable(GL_TEXTURE_2D);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, current_fb_size_.x, current_fb_size_.y, 0, GL_RGBA, GL_UNSIGNED_BYTE,
            color_ptr);

        glEnable(GL_TEXTURE_2D);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f);
        glVertex2f(0.0f, 0.0f);

        glTexCoord2f(1.0f, 0.0f);
        glVertex2f(1.0f, 0.0f);

        glTexCoord2f(1.0f, 1.0f);
        glVertex2f(1.0f, 1.0f);

        glTexCoord2f(0.0f, 1.0f);
        glVertex2f(0.0f, 1.0f);
        glEnd();

        glDisable(GL_TEXTURE_2D);
        glEnable(GL_DEPTH_TEST);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    if (accumulate_slot_.Param<core::param::BoolParam>()->Value()) {
        ++framestate_.accumID;
    } else {
        framestate_.accumID = 0;
    }

    return true;
}
   

bool BaseRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {
    auto in_data = data_in_slot_.CallAs<geocalls::MultiParticleDataCall>();
    if (in_data == nullptr)
        return false;

    in_data->SetFrameID(call.Time());
    if (!(*in_data)(1))
        return false;

    call.AccessBoundingBoxes() = in_data->GetBoundingBoxes();
    call.SetTimeFramesCount(in_data->FrameCount());

    return true;
}

void BaseRenderer::resizeFramebuffer(owl::common::vec2i const& dim) {
    if (!accumBuffer_)
        accumBuffer_ = owlDeviceBufferCreate(ctx_, OWL_FLOAT4, dim.x * dim.y, nullptr);
    owlBufferResize(accumBuffer_, dim.x * dim.y);

    owlRayGenSetBuffer(raygen_, "accumBuffer", accumBuffer_);

    if (!colorBuffer_)
        colorBuffer_ = owlHostPinnedBufferCreate(ctx_, OWL_INT, dim.x * dim.y);
    owlBufferResize(colorBuffer_, dim.x * dim.y);

    owlRayGenSetBuffer(raygen_, "colorBuffer", colorBuffer_);
    owlRayGenSet2i(raygen_, "fbSize", dim.x, dim.y);

    current_fb_size_ = dim;
    framestate_.accumID = 0;
}
} // namespace megamol::optix_owl
