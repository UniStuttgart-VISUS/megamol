/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "SurfaceLICRenderer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <random>

#include "geometry_calls/VolumetricDataCall.h"
#include "mmcore/Call.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmstd/renderer/AbstractCallRender.h"
#include "mmstd_gl/renderer/CallGetTransferFunctionGL.h"
#include "mmstd_gl/renderer/TransferFunctionGL.h"

using megamol::core::utility::log::Log;

namespace megamol::volume_gl {

SurfaceLICRenderer::SurfaceLICRenderer()
        : input_renderer("input_renderer", "Renderer producing the surface and depth used for drawing the LIC upon")
        , input_velocities("input_velocities", "Grid with velocities")
        , input_transfer_function(
              "m_input_transfer_function", "Transfer function to color the LIC according to the velocity magnitude")
        , arc_length("arc_length", "Length of the streamlines relative to the domain size")
        , num_advections("num_advections", "Number of advections for reaching the desired arc length")
        , epsilon("epsilon", "Threshold for detecting coherent structures")
        , noise_bands("noise_bands", "Number of noise bands for LOD noise")
        , noise_scale("noise_scale", "Noise scalar for fine-tuning LOD noise")
        , coloring("coloring", "Different options on velocity coloring")
        , ka("lighting::ka", "Ambient part for Phong lighting")
        , kd("lighting::kd", "Diffuse part for Phong lighting")
        , ks("lighting::ks", "Specular part for Phong lighting")
        , shininess("lighting::shininess", "Shininess for Phong lighting")
        , ambient_color("lighting::ambient color", "Ambient color")
        , specular_color("lighting::specular color", "Specular color")
        , light_color("lighting::light color", "Light color")
        , fbo(nullptr)
        , hash(-1) {

    this->input_renderer.SetCompatibleCall<mmstd_gl::CallRender3DGLDescription>();
    this->MakeSlotAvailable(&this->input_renderer);

    this->input_velocities.SetCompatibleCall<geocalls::VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->input_velocities);

    this->input_transfer_function.SetCompatibleCall<mmstd_gl::CallGetTransferFunctionGLDescription>();
    this->MakeSlotAvailable(&this->input_transfer_function);

    this->arc_length << new core::param::FloatParam(0.03f);
    this->MakeSlotAvailable(&this->arc_length);

    this->num_advections << new core::param::IntParam(100, 1);
    this->MakeSlotAvailable(&this->num_advections);

    this->noise_bands << new core::param::IntParam(2, 1);
    this->MakeSlotAvailable(&this->noise_bands);

    this->noise_scale << new core::param::FloatParam(5.0f);
    this->MakeSlotAvailable(&this->noise_scale);

    this->epsilon << new core::param::FloatParam(0.03f);
    this->MakeSlotAvailable(&this->epsilon);

    this->coloring << new core::param::EnumParam(0);
    this->coloring.Param<core::param::EnumParam>()->SetTypePair(0, "Original");
    this->coloring.Param<core::param::EnumParam>()->SetTypePair(1, "Projected");
    this->coloring.Param<core::param::EnumParam>()->SetTypePair(2, "Difference");
    this->MakeSlotAvailable(&this->coloring);

    this->ka << new core::param::FloatParam(0.2f, 0.0f);
    this->MakeSlotAvailable(&this->ka);

    this->kd << new core::param::FloatParam(0.6f, 0.0f);
    this->MakeSlotAvailable(&this->kd);

    this->ks << new core::param::FloatParam(0.2f, 0.0f);
    this->MakeSlotAvailable(&this->ks);

    this->shininess << new core::param::FloatParam(10.0f, 0.0f);
    this->MakeSlotAvailable(&this->shininess);

    this->ambient_color << new core::param::ColorParam(1.0f, 1.0f, 1.0f, 1.0f);
    this->MakeSlotAvailable(&this->ambient_color);

    this->specular_color << new core::param::ColorParam(1.0f, 1.0f, 1.0f, 1.0f);
    this->MakeSlotAvailable(&this->specular_color);

    this->light_color << new core::param::ColorParam(1.0f, 1.0f, 1.0f, 1.0f);
    this->MakeSlotAvailable(&this->light_color);
}

SurfaceLICRenderer::~SurfaceLICRenderer() {
    this->Release();
}

bool SurfaceLICRenderer::create() {
    auto const shaderOptions = msf::ShaderFactoryOptionsOpenGL(GetCoreInstance()->GetShaderPaths());

    try {
        this->pre_compute_shdr =
            core::utility::make_glowl_shader("pre_compute_shdr", shaderOptions, "volume_gl/SurfaceLIC-Pre.comp.glsl");
        this->lic_compute_shdr =
            core::utility::make_glowl_shader("lic_compute_shdr", shaderOptions, "volume_gl/SurfaceLIC-Lic.comp.glsl");
        this->render_to_framebuffer_shdr = core::utility::make_glowl_shader("render_to_framebuffer_shdr", shaderOptions,
            "volume_gl/RaycastVolumeRenderer.vert.glsl", "volume_gl/RaycastVolumeRenderer.frag.glsl");
    } catch (std::exception& e) {
        Log::DefaultLog.WriteError(("SurfaceLICRenderer: " + std::string(e.what())).c_str());
        return false;
    }

    return true;
}

void SurfaceLICRenderer::release() {}

bool SurfaceLICRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {
    auto ci = this->input_renderer.CallAs<mmstd_gl::CallRender3DGL>();
    auto cd = this->input_velocities.CallAs<geocalls::VolumetricDataCall>();

    if (ci == nullptr)
        return false;
    if (cd == nullptr)
        return false;

    int const req_frame = static_cast<int>(call.Time());

    ci->SetTime(req_frame);
    cd->SetFrameID(req_frame, true);

    *ci = call;

    if (!(*ci)(mmstd_gl::CallRender3DGL::FnGetExtents))
        return false;
    if (!(*cd)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS))
        return false;

    call.SetTimeFramesCount(cd->FrameCount());
    call.AccessBoundingBoxes().SetBoundingBox(
        this->combineBoundingBoxes({cd->GetBoundingBoxes().ObjectSpaceBBox(), ci->GetBoundingBoxes().BoundingBox()}));
    call.AccessBoundingBoxes().SetClipBox(
        this->combineBoundingBoxes({cd->GetBoundingBoxes().ObjectSpaceClipBox(), ci->GetBoundingBoxes().ClipBox()}));

    // TODO is this order correct?
    call = *ci;

    return true;
}

bool SurfaceLICRenderer::Render(mmstd_gl::CallRender3DGL& call) {
    const auto req_frame = call.Time();

    // Get input rendering
    auto ci = this->input_renderer.CallAs<mmstd_gl::CallRender3DGL>();
    if (ci == nullptr)
        return false;

    ci->SetTime(req_frame);
    core::view::Camera cam = call.GetCamera();
    ci->SetCamera(cam);

    auto viewport = call.GetViewResolution();
    if (this->fbo == nullptr || this->fbo->getWidth() != viewport.x || this->fbo->getHeight() != viewport.y) {
        this->fbo =
            std::make_shared<glowl::FramebufferObject>(viewport.x, viewport.y, glowl::FramebufferObject::DEPTH24);

        this->fbo->createColorAttachment(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
        this->fbo->createColorAttachment(GL_RGBA32F, GL_RGBA, GL_FLOAT);
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    ci->SetFramebuffer(this->fbo);

    if (!(*ci)(mmstd_gl::CallRender3DGL::FnRender))
        return false;
    call.SetTimeFramesCount(ci->TimeFramesCount());

    // Get input velocities
    auto cd = this->input_velocities.CallAs<geocalls::VolumetricDataCall>();
    if (cd == nullptr)
        return false;

    if (!(*cd)(geocalls::VolumetricDataCall::IDX_GET_DATA))
        return false;

    cd->SetFrameID(req_frame, true);

    if (cd->GetComponents() != 3) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Input velocities must be vectors with 3 components");
        return false;
    }

    if (this->velocity_texture == nullptr || this->velocity_texture->getWidth() != cd->GetResolution(0) ||
        this->velocity_texture->getHeight() != cd->GetResolution(1) ||
        this->velocity_texture->getDepth() != cd->GetResolution(2) || this->hash != cd->DataHash()) {

        this->hash = cd->DataHash();

        glowl::TextureLayout velocity_layout(GL_RGB32F, cd->GetResolution(0), cd->GetResolution(1),
            cd->GetResolution(2), GL_RGB, GL_FLOAT, 1,
            {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
                {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
                {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
            {});

        this->velocity_texture = std::make_unique<glowl::Texture3D>("velocity_texture", velocity_layout, cd->GetData());
    }

    // Get input transfer function
    auto ct = this->input_transfer_function.CallAs<mmstd_gl::CallGetTransferFunctionGL>();

    GLuint tf_texture = 0;

    if (ct != nullptr && (*ct)()) {
        tf_texture = ct->OpenGLTexture();
    }

    // Create velocity target texture
    if (this->velocity_target == nullptr || this->velocity_target->getWidth() != call.GetViewResolution().x ||
        this->velocity_target->getHeight() != call.GetViewResolution().y) {

        glowl::TextureLayout velocity_tgt_layout(GL_RGBA32F, call.GetViewResolution().x, call.GetViewResolution().y, 1,
            GL_RGBA, GL_FLOAT, 1,
            {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
                {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
                {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
            {});

        this->velocity_target = std::make_unique<glowl::Texture2D>("velocity_target", velocity_tgt_layout, nullptr);
    }

    // Create render target texture
    if (this->render_target == nullptr || this->render_target->getWidth() != call.GetViewResolution().x ||
        this->render_target->getHeight() != call.GetViewResolution().y) {

        glowl::TextureLayout render_tgt_layout(GL_RGBA8, call.GetViewResolution().x, call.GetViewResolution().y, 1,
            GL_RGBA, GL_UNSIGNED_BYTE, 1,
            {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
                {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
                {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
            {});

        this->render_target = std::make_unique<glowl::Texture2D>("render_target", render_tgt_layout, nullptr);
    }

    // Create noise texture
    if (this->noise_texture == nullptr) {
        glowl::TextureLayout noise_layout(GL_R32F, 64, 64, 64, GL_RED, GL_FLOAT, 1,
            {{GL_TEXTURE_WRAP_S, GL_REPEAT}, {GL_TEXTURE_WRAP_T, GL_REPEAT}, {GL_TEXTURE_WRAP_R, GL_REPEAT},
                {GL_TEXTURE_MIN_FILTER, GL_LINEAR}, {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
            {});

        this->noise.resize(64 * 64 * 64);

        std::random_device r;
        std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
        std::mt19937 engine(seed);
        // std::normal_distribution<float> normal_dist(0.0f, 0.5f);
        std::uniform_real_distribution<float> normal_dist(0.0f, 1.0f);

        auto random = [&engine, &normal_dist](
                          float& value) { value = std::min(std::max(normal_dist(engine), 0.0f), 1.0f); };

        std::for_each(this->noise.begin(), this->noise.end(), random);

        this->noise_texture = std::make_unique<glowl::Texture3D>("noise_texture", noise_layout, this->noise.data());
    }

    // Get camera
    glm::mat4 view = cam.getViewMatrix();
    glm::mat4 proj = cam.getProjectionMatrix();

    const auto intrinsics = cam.get<core::view::Camera::PerspectiveParameters>();

    const auto cam_near = intrinsics.near_plane.value();
    const auto cam_far = intrinsics.far_plane.value();

    const std::array<float, 2> rt_resolution{
        static_cast<float>(this->render_target->getWidth()), static_cast<float>(this->render_target->getHeight())};

    const std::array<float, 3> origin{call.GetBoundingBoxes().BoundingBox().Left(),
        call.GetBoundingBoxes().BoundingBox().Bottom(), call.GetBoundingBoxes().BoundingBox().Back()};
    const std::array<float, 3> resolution{call.GetBoundingBoxes().BoundingBox().Width(),
        call.GetBoundingBoxes().BoundingBox().Height(), call.GetBoundingBoxes().BoundingBox().Depth()};

    std::array<float, 4> light = {0.0f, 0.0f, 1.0f, 1.0f};
    glGetLightfv(GL_LIGHT0, GL_POSITION, light.data());

    // Transform velocities to 2D in a pre-computation step
    this->pre_compute_shdr->use();

    glUniformMatrix4fv(this->pre_compute_shdr->getUniformLocation("view_mx"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(this->pre_compute_shdr->getUniformLocation("proj_mx"), 1, GL_FALSE, glm::value_ptr(proj));

    glUniform2fv(this->pre_compute_shdr->getUniformLocation("rt_resolution"), 1, rt_resolution.data());

    glUniform3fv(this->pre_compute_shdr->getUniformLocation("origin"), 1, origin.data());
    glUniform3fv(this->pre_compute_shdr->getUniformLocation("resolution"), 1, resolution.data());

    glActiveTexture(GL_TEXTURE0);
    this->fbo->bindDepthbuffer();
    glUniform1i(this->pre_compute_shdr->getUniformLocation("depth_tx2D"), 0);

    glActiveTexture(GL_TEXTURE1);
    this->fbo->bindColorbuffer(1);
    glUniform1i(this->pre_compute_shdr->getUniformLocation("normal_tx2D"), 1);

    glActiveTexture(GL_TEXTURE2);
    this->velocity_texture->bindTexture();
    glUniform1i(this->pre_compute_shdr->getUniformLocation("velocity_tx3D"), 2);

    this->velocity_target->bindImage(0, GL_WRITE_ONLY);

    glDispatchCompute(
        static_cast<int>(std::ceil(rt_resolution[0] / 8.0f)), static_cast<int>(std::ceil(rt_resolution[1] / 8.0f)), 1);

    glBindImageTexture(0, 0, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R8);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_3D, 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glUseProgram(0);

    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    // Compute surface LIC
    this->lic_compute_shdr->use();

    glUniformMatrix4fv(this->lic_compute_shdr->getUniformLocation("view_mx"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(this->lic_compute_shdr->getUniformLocation("proj_mx"), 1, GL_FALSE, glm::value_ptr(proj));

    glUniform1f(this->lic_compute_shdr->getUniformLocation("cam_near"), cam_near);
    glUniform1f(this->lic_compute_shdr->getUniformLocation("cam_far"), cam_far);

    glUniform2fv(this->lic_compute_shdr->getUniformLocation("rt_resolution"), 1, rt_resolution.data());

    glUniform3fv(this->lic_compute_shdr->getUniformLocation("origin"), 1, origin.data());
    glUniform3fv(this->lic_compute_shdr->getUniformLocation("resolution"), 1, resolution.data());

    glUniform1i(this->lic_compute_shdr->getUniformLocation("noise_bands"),
        this->noise_bands.Param<core::param::IntParam>()->Value());
    glUniform1f(this->lic_compute_shdr->getUniformLocation("noise_scale"),
        this->noise_scale.Param<core::param::FloatParam>()->Value());
    glUniform1f(this->lic_compute_shdr->getUniformLocation("arc_length"),
        this->arc_length.Param<core::param::FloatParam>()->Value());
    glUniform1i(this->lic_compute_shdr->getUniformLocation("num_advections"),
        this->num_advections.Param<core::param::IntParam>()->Value());
    glUniform1f(
        this->lic_compute_shdr->getUniformLocation("epsilon"), this->epsilon.Param<core::param::FloatParam>()->Value());

    glUniform1i(this->lic_compute_shdr->getUniformLocation("coloring"),
        this->coloring.Param<core::param::EnumParam>()->Value());

    glUniform1f(this->lic_compute_shdr->getUniformLocation("max_magnitude"),
        static_cast<float>(cd->GetMetadata()->MaxValues[0]));

    glUniform1f(this->lic_compute_shdr->getUniformLocation("ka"), this->ka.Param<core::param::FloatParam>()->Value());
    glUniform1f(this->lic_compute_shdr->getUniformLocation("kd"), this->kd.Param<core::param::FloatParam>()->Value());
    glUniform1f(this->lic_compute_shdr->getUniformLocation("ks"), this->ks.Param<core::param::FloatParam>()->Value());
    glUniform1f(this->lic_compute_shdr->getUniformLocation("shininess"),
        this->shininess.Param<core::param::FloatParam>()->Value());
    glUniform3fv(this->lic_compute_shdr->getUniformLocation("light"), 1, light.data());
    glUniform3fv(this->lic_compute_shdr->getUniformLocation("ambient_col"), 1,
        this->ambient_color.Param<core::param::ColorParam>()->Value().data());
    glUniform3fv(this->lic_compute_shdr->getUniformLocation("specular_col"), 1,
        this->specular_color.Param<core::param::ColorParam>()->Value().data());
    glUniform3fv(this->lic_compute_shdr->getUniformLocation("light_col"), 1,
        this->light_color.Param<core::param::ColorParam>()->Value().data());

    glActiveTexture(GL_TEXTURE0);
    this->fbo->bindDepthbuffer();
    glUniform1i(this->lic_compute_shdr->getUniformLocation("depth_tx2D"), 0);

    glActiveTexture(GL_TEXTURE1);
    this->velocity_target->bindTexture();
    glUniform1i(this->lic_compute_shdr->getUniformLocation("velocity_tx2D"), 1);

    glActiveTexture(GL_TEXTURE2);
    this->fbo->bindColorbuffer(1);
    glUniform1i(this->lic_compute_shdr->getUniformLocation("normal_tx2D"), 2);

    glActiveTexture(GL_TEXTURE3);
    this->noise_texture->bindTexture();
    glUniform1i(this->lic_compute_shdr->getUniformLocation("noise_tx3D"), 3);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_1D, tf_texture);
    glUniform1i(this->lic_compute_shdr->getUniformLocation("tf_tx1D"), 4);

    this->render_target->bindImage(0, GL_WRITE_ONLY);

    glDispatchCompute(
        static_cast<int>(std::ceil(rt_resolution[0] / 8.0f)), static_cast<int>(std::ceil(rt_resolution[1] / 8.0f)), 1);

    glBindImageTexture(0, 0, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R8);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_1D, 0);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_3D, 0);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glUseProgram(0);

    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    // Render to framebuffer
    call.GetFramebuffer()->bind();

    bool state_depth_test = glIsEnabled(GL_DEPTH_TEST);
    bool state_blend = glIsEnabled(GL_BLEND);

    if (!state_depth_test)
        glEnable(GL_DEPTH_TEST);
    if (state_blend)
        glDisable(GL_BLEND);

    this->render_to_framebuffer_shdr->use();

    glActiveTexture(GL_TEXTURE0);
    this->render_target->bindTexture();
    glUniform1i(this->render_to_framebuffer_shdr->getUniformLocation("src_tx2D"), 0);

    glActiveTexture(GL_TEXTURE1);
    this->fbo->bindDepthbuffer();
    glUniform1i(this->render_to_framebuffer_shdr->getUniformLocation("depth_tx2D"), 1);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glUseProgram(0);

    if (state_blend)
        glEnable(GL_BLEND);
    if (!state_depth_test)
        glDisable(GL_DEPTH_TEST);
}

vislib::math::Cuboid<float> SurfaceLICRenderer::combineBoundingBoxes(std::vector<vislib::math::Cuboid<float>> boxVec) {
    auto out = boxVec.front();
    for (auto& bb : boxVec) {
        out.Union(bb);
    }
    return out;
}

} // namespace megamol::volume_gl
