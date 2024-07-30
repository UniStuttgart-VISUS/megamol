#include "PKDRenderer.h"

#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include <owl/common/math/vec.h>
#include <owl/common/math/box.h>

#include "PKDCreate.h"

#include "pkd.h"
#include "framestate.h"
#include "raygen.h"

#include <glad/gl.h>

namespace megamol::optix_owl {
extern "C" const unsigned char raygenPrograms_ptx[];
extern "C" const unsigned char pkdPrograms_ptx[];

PKDRenderer::PKDRenderer() : data_in_slot_("dataIn", ""), radius_slot_("radius", ""), rec_depth_slot_("rec_depth", ""), spp_slot_("spp", "") {
    data_in_slot_.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&data_in_slot_);

    radius_slot_ << new core::param::FloatParam(0.5f, std::numeric_limits<float>::min());
    MakeSlotAvailable(&radius_slot_);

    rec_depth_slot_ << new core::param::IntParam(0, 0, 20);
    MakeSlotAvailable(&rec_depth_slot_);

    spp_slot_ << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&spp_slot_);
}

PKDRenderer::~PKDRenderer() {
    this->Release();
}

bool PKDRenderer::create() {
    ctx_ = owlContextCreate(nullptr, 1);
    owlContextSetRayTypeCount(ctx_, 1);

    raygen_module_ = owlModuleCreate(ctx_, reinterpret_cast<const char*>(raygenPrograms_ptx));
    pkd_module_ = owlModuleCreate(ctx_, reinterpret_cast<const char*>(pkdPrograms_ptx));

    frameStateBuffer_ = owlDeviceBufferCreate(ctx_, OWL_USER_TYPE(device::FrameState), 1, nullptr);

    OWLVarDecl rayGenVars[] = {
        {"colorBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::RayGenData, colorBufferPtr)},
        {"accumBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::RayGenData, accumBufferPtr)},
        {"particleBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::RayGenData, particleBuffer)},
        {"frameStateBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::RayGenData, frameStateBuffer)},
        {"fbSize", OWL_INT2, OWL_OFFSETOF(device::RayGenData, fbSize)},
        {"world", OWL_GROUP, OWL_OFFSETOF(device::RayGenData, world)},
        {"rec_depth", OWL_INT, OWL_OFFSETOF(device::RayGenData, rec_depth)}, {/* sentinel to mark end of list */}};

    raygen_ = owlRayGenCreate(ctx_, raygen_module_, "raygen", sizeof(device::RayGenData), rayGenVars, -1);

    OWLVarDecl missProgVars[] = {{/* sentinel to mark end of list */}};

    miss_ = owlMissProgCreate(ctx_, raygen_module_, "miss", 0, missProgVars, -1);

    owlRayGenSet1i(raygen_, "rec_depth", rec_depth_slot_.Param<core::param::IntParam>()->Value());
    owlRayGenSetBuffer(raygen_, "frameStateBuffer", frameStateBuffer_);

    resizeFramebuffer(vec2i(1920, 1080));

    framestate_.accumID = 0;
    framestate_.samplesPerPixel = spp_slot_.Param<core::param::IntParam>()->Value();

    return true;
}

void PKDRenderer::release() {
    owlBufferDestroy(particleBuffer_);
    owlBufferDestroy(accumBuffer_);
    owlBufferDestroy(colorBuffer_);
    owlBufferDestroy(frameStateBuffer_);
    owlModuleRelease(raygen_module_);
    owlModuleRelease(pkd_module_);
    owlContextDestroy(ctx_);
}

float computeStableEpsilon(float f) {
    return abs(f) * float(1. / (1 << 21));
}

float computeStableEpsilon(const owl::common::vec3f v) {
    return max(max(computeStableEpsilon(v.x), computeStableEpsilon(v.y)), computeStableEpsilon(v.z));
}

bool PKDRenderer::Render(mmstd_gl::CallRender3DGL& call) {
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

    if (in_data->FrameID() != frame_id_ || in_data->DataHash() != in_data_hash_) {
        if (!assertData(*in_data))
            return false;
        frame_id_ = in_data->FrameID();
        in_data_hash_ = in_data->DataHash();

        owlRayGenSetGroup(raygen_, "world", world_);
        owlRayGenSetBuffer(raygen_, "particleBuffer", particleBuffer_);

        owlBuildPrograms(ctx_);
        owlBuildPipeline(ctx_);
        owlBuildSBT(ctx_);
    }

    owlBufferUpload(frameStateBuffer_, &framestate_);
    owlRayGenLaunch2D(raygen_, current_fb_size_.x, current_fb_size_.y);

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

    return true;
}

bool PKDRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {
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

bool PKDRenderer::assertData(geocalls::MultiParticleDataCall const& call) {
    auto const pl_count = call.GetParticleListCount();

    particles_.clear();
    owl::common::box3f total_bounds;
    auto const global_radius = radius_slot_.Param<core::param::FloatParam>()->Value();

    for (unsigned int pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto const& particles = call.AccessParticles(pl_idx);

        auto const p_count = particles.GetCount();
        if (p_count == 0)
            continue;
        /*if (particles.GetVertexDataType() == geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR)
            continue;*/

        std::vector<device::Particle> data(p_count);

        auto x_acc = particles.GetParticleStore().GetXAcc();
        auto y_acc = particles.GetParticleStore().GetYAcc();
        auto z_acc = particles.GetParticleStore().GetZAcc();

        owl::common::box3f bounds;

        for (std::size_t i = 0; i < p_count; ++i) {
            data[i].pos = owl::common::vec3f(x_acc->Get_f(i), y_acc->Get_f(i), z_acc->Get_f(i));
            bounds.extend(
                owl::common::box3f().including(data[i].pos - global_radius).including(data[i].pos + global_radius));
        }

        particles_.insert(particles_.end(), data.begin(), data.end());
        total_bounds.extend(bounds);
    }

    makePKD(particles_, total_bounds);
    if (particleBuffer_)
        owlBufferDestroy(particleBuffer_);
    particleBuffer_ = owlDeviceBufferCreate(ctx_, OWL_USER_TYPE(particles_[0]), particles_.size(), particles_.data());

    core::utility::log::Log::DefaultLog.WriteInfo(
        "[PKDRenderer] Rendering %d particles", particles_.size());

    OWLVarDecl allPKDVars[] = {{"particleBuffer", OWL_BUFPTR, OWL_OFFSETOF(device::PKDGeomData, particleBuffer)},
        {"particleRadius", OWL_FLOAT, OWL_OFFSETOF(device::PKDGeomData, particleRadius)},
        {"particleCount", OWL_INT, OWL_OFFSETOF(device::PKDGeomData, particleCount)},
        {"bounds.lower", OWL_FLOAT3, OWL_OFFSETOF(device::PKDGeomData, worldBounds.lower)},
        {"bounds.upper", OWL_FLOAT3, OWL_OFFSETOF(device::PKDGeomData, worldBounds.upper)},
        {/* sentinel to mark end of list */}};

    OWLGeomType allPKDType = owlGeomTypeCreate(ctx_, OWL_GEOMETRY_USER, sizeof(device::PKDGeomData), allPKDVars, -1);
    owlGeomTypeSetBoundsProg(allPKDType, pkd_module_, "pkd_bounds");
    owlGeomTypeSetIntersectProg(allPKDType, 0, pkd_module_, "pkd_intersect");
    owlGeomTypeSetClosestHit(allPKDType, 0, pkd_module_, "pkd_ch");

    OWLGeom geom = owlGeomCreate(ctx_, allPKDType);

    owlGeomSetPrimCount(geom, 1);

    owlGeomSetBuffer(geom, "particleBuffer", particleBuffer_);
    owlGeomSet1f(geom, "particleRadius", global_radius);
    owlGeomSet1i(geom, "particleCount", (int) particles_.size());
    owlGeomSet3f(geom, "bounds.lower", (const owl3f&) total_bounds.lower);
    owlGeomSet3f(geom, "bounds.upper", (const owl3f&) total_bounds.upper);

    owlBuildPrograms(ctx_);

    OWLGroup ug = owlUserGeomGroupCreate(ctx_, 1, &geom);
    owlGroupBuildAccel(ug);

    world_ = owlInstanceGroupCreate(ctx_, 1, &ug);

    owlGroupBuildAccel(world_);

    return true;
}

void PKDRenderer::resizeFramebuffer(owl::common::vec2i const& dim) {
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
}
} // namespace megamol::optix_owl
