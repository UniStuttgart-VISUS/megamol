/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "ResolutionScalingRenderer2D.h"

#include <vector>

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"

using namespace megamol::infovis_gl;
using megamol::core::utility::log::Log;

ResolutionScalingRenderer2D::ResolutionScalingRenderer2D()
        : BaseAmortizedRenderer2D()
        , amortLevelParam("AmortLevel", "Level of Amortization")
        , skipInterpolationParam("SkipInterpolation", "Do not interpolate missing pixels.") {

    amortLevelParam << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&amortLevelParam);

    skipInterpolationParam << new core::param::BoolParam(false);
    MakeSlotAvailable(&skipInterpolationParam);
}

ResolutionScalingRenderer2D::~ResolutionScalingRenderer2D() {
    Release();
}

bool ResolutionScalingRenderer2D::createImpl(const msf::ShaderFactoryOptionsOpenGL& shaderOptions) {
    try {
        shader_ = core::utility::make_glowl_shader("amort_resolutionscaling", shaderOptions,
            "infovis_gl/amort/amort_quad.vert.glsl", "infovis_gl/amort/amort_resolutionscaling.frag.glsl");
    } catch (std::exception& e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, ("ResolutionScalingRenderer2D: " + std::string(e.what())).c_str());
        return false;
    }

    lowResFBO_ = std::make_shared<glowl::FramebufferObject>(1, 1);
    lowResFBO_->createColorAttachment(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);

    // Store texture layout for later resize
    texLayout_ = glowl::TextureLayout(GL_RGBA8, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, 1,
        {
            {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_MIN_FILTER, GL_NEAREST},
            {GL_TEXTURE_MAG_FILTER, GL_NEAREST},
        },
        {});
    distTexLayout_ = glowl::TextureLayout(GL_R32F, 1, 1, 1, GL_RED, GL_FLOAT, 1,
        {
            {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_MIN_FILTER, GL_NEAREST},
            {GL_TEXTURE_MAG_FILTER, GL_NEAREST},
        },
        {});

    texRead_ = std::make_unique<glowl::Texture2D>("texStoreA", texLayout_, nullptr);
    texWrite_ = std::make_unique<glowl::Texture2D>("texStoreB", texLayout_, nullptr);
    distTexRead_ = std::make_unique<glowl::Texture2D>("distTexR", distTexLayout_, nullptr);
    distTexWrite_ = std::make_unique<glowl::Texture2D>("distTexW", distTexLayout_, nullptr);

    auto err = glGetError();
    if (err != GL_NO_ERROR) {
        Log::DefaultLog.WriteError("GL_ERROR in BaseAmortizedRenderer2D: %i", err);
    }

    return true;
}

void ResolutionScalingRenderer2D::releaseImpl() {
    // nothing to do
}

bool ResolutionScalingRenderer2D::renderImpl(core_gl::view::CallRender2DGL& nextRendererCall,
    std::shared_ptr<core_gl::view::CallRender2DGL::FBO_TYPE> fbo, core::view::Camera cam) {

    const int a = amortLevelParam.Param<core::param::IntParam>()->Value();
    const int w = fbo->getWidth();
    const int h = fbo->getHeight();

    if (a != oldAmortLevel_ || w != oldWidth_ || h != oldHeight_) {
        updateSize(a, w, h);
        oldAmortLevel_ = a;
        oldWidth_ = w;
        oldHeight_ = h;
    }

    lowResFBO_->bind();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    setupCamera(cam, w, h, a);

    nextRendererCall.SetFramebuffer(lowResFBO_);
    nextRendererCall.SetCamera(cam);
    (nextRendererCall)(core::view::AbstractCallRender::FnRender);

    reconstruct(fbo, a);

    return true;
}

void ResolutionScalingRenderer2D::updateSize(int a, int w, int h) {
    frameIdx_ = 0;
    shiftMx_ = glm::mat4(1.0f);
    lastProjViewMx_ = glm::mat4(1.0f);
    camOffsets_.resize(a * a);
    for (int j = 0; j < a; j++) {
        for (int i = 0; i < a; i++) {
            const float x = (static_cast<float>(a) - 1.0f - 2.0f * static_cast<float>(i)) / static_cast<float>(w);
            const float y = (static_cast<float>(a) - 1.0f - 2.0f * static_cast<float>(j)) / static_cast<float>(h);
            camOffsets_[j * a + i] = glm::vec3(x, y, 0.0f);
        }
    }

    lowResFBO_->resize(static_cast<int>(std::ceil(static_cast<float>(w) / static_cast<float>(a))),
        static_cast<int>(std::ceil(static_cast<float>(h) / static_cast<float>(a))));

    const std::vector<uint32_t> nullData(w * h, 0); // Fits for RGBA8 and R32F

    texLayout_.width = w;
    texLayout_.height = h;
    texRead_ = std::make_unique<glowl::Texture2D>("texRead", texLayout_, nullData.data());
    texWrite_ = std::make_unique<glowl::Texture2D>("texWrite", texLayout_, nullData.data());

    distTexLayout_.width = w;
    distTexLayout_.height = h;
    distTexRead_ = std::make_unique<glowl::Texture2D>("distTexRead", distTexLayout_, nullData.data());
    distTexWrite_ = std::make_unique<glowl::Texture2D>("distTexWrite", distTexLayout_, nullData.data());

    samplingSequence_ = std::vector<int>(a * a);
    samplingSequence_[0] = 0;
    for (int i = 1; i < a * a; i++) {
        samplingSequence_[i] = (samplingSequence_[i - 1] + (a + 1) * (a + 1)) % (a * a);
    }
    samplingSequencePosition_ = 0;
}

void ResolutionScalingRenderer2D::setupCamera(core::view::Camera& cam, int width, int height, int a) {
    auto const projViewMx = cam.getProjectionMatrix() * cam.getViewMatrix();

    auto intrinsics = cam.get<core::view::Camera::OrthographicParameters>();
    glm::vec3 adj_offset = glm::vec3(-intrinsics.aspect * intrinsics.frustrum_height * camOffsets_[frameIdx_].x,
        -intrinsics.frustrum_height * camOffsets_[frameIdx_].y, 0.0f);

    shiftMx_ = lastProjViewMx_ * inverse(projViewMx);
    lastProjViewMx_ = projViewMx;

    auto p = cam.get<core::view::Camera::Pose>();
    p.position = p.position + 0.5f * adj_offset;

    const float ha = static_cast<float>(height) / static_cast<float>(a);
    const float wa = static_cast<float>(width) / static_cast<float>(a);
    const float hAdj = std::ceil(ha) / ha;
    const float wAdj = std::ceil(wa) / wa;

    const float hOffs = hAdj * intrinsics.frustrum_height - intrinsics.frustrum_height;
    const float wOffs =
        wAdj * intrinsics.aspect * intrinsics.frustrum_height - intrinsics.aspect * intrinsics.frustrum_height;
    p.position = p.position + glm::vec3(0.5f * wOffs, 0.5f * hOffs, 0.0f);
    intrinsics.frustrum_height = hAdj * intrinsics.frustrum_height.value();
    intrinsics.aspect = wAdj / hAdj * intrinsics.aspect;

    cam.setOrthographicProjection(intrinsics);
    cam.setPose(p);
}

void ResolutionScalingRenderer2D::reconstruct(std::shared_ptr<glowl::FramebufferObject> const& fbo, int a) {
    int w = fbo->getWidth();
    int h = fbo->getHeight();

    glViewport(0, 0, w, h);
    fbo->bind();

    shader_->use();
    shader_->setUniform("amortLevel", a);
    shader_->setUniform("w", w);
    shader_->setUniform("h", h);
    shader_->setUniform("frameIdx", frameIdx_);
    shader_->setUniform("shiftMx", shiftMx_);
    shader_->setUniform(
        "skipInterpolation", static_cast<int>(skipInterpolationParam.Param<core::param::BoolParam>()->Value()));

    glActiveTexture(GL_TEXTURE0);
    lowResFBO_->bindColorbuffer(0);
    shader_->setUniform("texLowResFBO", 0);

    texRead_->bindImage(0, GL_READ_ONLY);
    texWrite_->bindImage(1, GL_WRITE_ONLY);
    distTexRead_->bindImage(2, GL_READ_ONLY);
    distTexWrite_->bindImage(3, GL_WRITE_ONLY);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glUseProgram(0);

    samplingSequencePosition_ = (samplingSequencePosition_ + 1) % (a * a);
    frameIdx_ = samplingSequence_[samplingSequencePosition_];
    texRead_.swap(texWrite_);
    distTexRead_.swap(distTexWrite_);
}
