/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "ImageSpaceAmortization2D.h"

#include <numeric>
#include <vector>

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"

using namespace megamol::mmstd_gl;
using megamol::core::utility::log::Log;

ImageSpaceAmortization2D::ImageSpaceAmortization2D()
        : BaseAmortization2D()
        , amortModeParam("AmortMode", "Amortization Mode")
        , amortLevelParam("AmortLevel", "Level of Amortization")
        , autoLevelParam("AutoLevel", "Automatic level selection.")
        , targetFpsParam("TargetFPS", "Target FPS for automatic level selection.")
        , skipInterpolationParam("debug::SkipInterpolation", "Do not interpolate missing pixels.")
        , showQuadMarkerParam("debug::ShowQuadMarker", "Mark bottom left pixel of amortization quad.")
        , resetParam("debug::Reset", "Reset all textures and buffers.")
        , viewProjMx_(glm::mat4())
        , lastViewProjMx_(glm::mat4()) {

    amortModeParam << new core::param::EnumParam(MODE_2D);
    amortModeParam.Param<core::param::EnumParam>()->SetTypePair(MODE_2D, "2D");
    amortModeParam.Param<core::param::EnumParam>()->SetTypePair(MODE_HORIZONTAL, "Horizontal");
    amortModeParam.Param<core::param::EnumParam>()->SetTypePair(MODE_VERTICAL, "Vertical");
    MakeSlotAvailable(&amortModeParam);

    amortLevelParam << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&amortLevelParam);

    autoLevelParam << new core::param::BoolParam(false);
    MakeSlotAvailable(&autoLevelParam);

    targetFpsParam << new core::param::IntParam(15, 1);
    MakeSlotAvailable(&targetFpsParam);

    skipInterpolationParam << new core::param::BoolParam(false);
    MakeSlotAvailable(&skipInterpolationParam);

    showQuadMarkerParam << new core::param::BoolParam(false);
    MakeSlotAvailable(&showQuadMarkerParam);

    resetParam << new core::param::ButtonParam();
    resetParam.SetUpdateCallback(&ImageSpaceAmortization2D::resetCallback);
    MakeSlotAvailable(&resetParam);
}

ImageSpaceAmortization2D::~ImageSpaceAmortization2D() {
    Release();
}

bool ImageSpaceAmortization2D::createImpl(const msf::ShaderFactoryOptionsOpenGL& shaderOptions) {
    try {
        shader_ = core::utility::make_glowl_shader("amort_resolutionscaling", shaderOptions,
            "mmstd_gl/upscaling/image_space_amortization.vert.glsl",
            "mmstd_gl/upscaling/image_space_amortization.frag.glsl");
    } catch (std::exception& e) {
        Log::DefaultLog.WriteError(("ImageSpaceAmortization2D: " + std::string(e.what())).c_str());
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
    distTexLayout_ = glowl::TextureLayout(GL_RG32F, 1, 1, 1, GL_RG, GL_FLOAT, 1,
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
        Log::DefaultLog.WriteError("GL_ERROR in ImageSpaceAmortization2D: %i", err);
    }

    return true;
}

void ImageSpaceAmortization2D::releaseImpl() {
    // nothing to do
}

bool ImageSpaceAmortization2D::renderImpl(CallRender2DGL& call, CallRender2DGL& nextRendererCall) {

    auto const& fbo = call.GetFramebuffer();
    auto const& cam = call.GetCamera();
    auto const& bg = call.BackgroundColor();

    const auto m = static_cast<AmortMode>(amortModeParam.Param<core::param::EnumParam>()->Value());
    int aParam = amortLevelParam.Param<core::param::IntParam>()->Value();

    // auto-scaling algorithm params
    constexpr int minFramesAfterLevelChange = 10;
    constexpr int frameTimeBufferSize = 100;
    constexpr float targetBiasPercentage = 0.05f;

    const bool autoAmort = autoLevelParam.Param<core::param::BoolParam>()->Value();
    if (autoAmort) {
        amortLevelParam.Param<core::param::IntParam>()->SetGUIReadOnly(true);
        const int targetFps = targetFpsParam.Param<core::param::IntParam>()->Value();
        const float targetTimeMs = 1000.0f / static_cast<float>(targetFps);

        auto now = std::chrono::steady_clock::now();
        if (lastTime_.has_value()) {
            const float frame_time =
                std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(now - lastTime_.value()).count();
            lastFrameTimes_.push_back(frame_time);

            // TODO this is a bad operation for std::vector
            while (lastFrameTimes_.size() > frameTimeBufferSize) {
                lastFrameTimes_.erase(lastFrameTimes_.begin());
            }

            if (lastFrameTimes_.size() >= minFramesAfterLevelChange) {
                auto avg_frame_time = std::reduce(lastFrameTimes_.begin(), lastFrameTimes_.end()) /
                                      static_cast<float>(lastFrameTimes_.size());

                if (lastFrameTimeAvg_.has_value()) {
                    const auto a_was_inc = std::get<0>(lastFrameTimeAvg_.value());
                    const auto lastT = std::get<1>(lastFrameTimeAvg_.value());
                    if (a_was_inc) {
                        frameTimePrediction_ = avg_frame_time / lastT;
                    } else {
                        frameTimePrediction_ = lastT / avg_frame_time;
                    }
                    lastFrameTimeAvg_ = std::nullopt;
                    std::cout << frameTimePrediction_ << std::endl;
                }
                frameTimePrediction_ = std::clamp(frameTimePrediction_, 0.25f, 0.90f);

                int a = aParam;
                if (avg_frame_time > targetTimeMs) {
                    float expected_frame_time = avg_frame_time * frameTimePrediction_;
                    float biasedTargetTime = targetTimeMs;
                    if (expected_frame_time < targetTimeMs) {
                        biasedTargetTime = targetTimeMs + targetBiasPercentage * (avg_frame_time - expected_frame_time);
                    }
                    float dist_exp = std::abs(expected_frame_time - biasedTargetTime);
                    float dist_cur = std::abs(avg_frame_time - biasedTargetTime);
                    if (dist_exp < dist_cur) {
                        a++;
                    }
                } else {
                    auto expected_frame_time = avg_frame_time / frameTimePrediction_;
                    float biasedTargetTime = targetTimeMs;
                    if (expected_frame_time > targetTimeMs) {
                        biasedTargetTime = targetTimeMs - targetBiasPercentage * (expected_frame_time - avg_frame_time);
                    }
                    auto dist_exp = std::abs(expected_frame_time - biasedTargetTime);
                    auto dist_cur = std::abs(avg_frame_time - biasedTargetTime);
                    if (dist_exp < dist_cur) {
                        a--;
                    }
                }

                // TODO do we want to add maximum convergence time constraint as user param here?

                a = std::clamp(a, 1, 16);
                if (a != aParam) {
                    lastFrameTimeAvg_ = std::make_tuple(a > aParam, avg_frame_time);
                    aParam = a;
                    amortLevelParam.Param<core::param::IntParam>()->SetValue(aParam);
                    lastFrameTimes_.clear();
                }
            }
        }
        lastTime_ = now;
    } else {
        amortLevelParam.Param<core::param::AbstractParam>()->SetGUIReadOnly(false);
        lastTime_ = std::nullopt;
        lastFrameTimes_.clear();
    }

    const glm::ivec2 a((m == MODE_VERTICAL) ? 1 : aParam, (m == MODE_HORIZONTAL) ? 1 : aParam);
    const int w = fbo->getWidth();
    const int h = fbo->getHeight();

    if (a != oldAmortLevel_ || w != oldWidth_ || h != oldHeight_) {
        if (w != oldWidth_ || h != oldHeight_) {
            updateTextureSize(w, h);
        }
        updateAmortSize(a, w, h);
        oldAmortLevel_ = a;
        oldWidth_ = w;
        oldHeight_ = h;
    }

    lowResFBO_->bind();
    glClearColor(bg.r * bg.a, bg.g * bg.a, bg.b * bg.a, bg.a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    lastViewProjMx_ = viewProjMx_;
    viewProjMx_ = cam.getProjectionMatrix() * cam.getViewMatrix();

    auto lowResCam = cam;
    setupCamera(lowResCam, w, h, a);

    nextRendererCall.SetFramebuffer(lowResFBO_);
    nextRendererCall.SetCamera(lowResCam);
    (nextRendererCall)(core::view::AbstractCallRender::FnRender);

    reconstruct(fbo, cam, a);

    return true;
}

bool ImageSpaceAmortization2D::resetCallback(core::param::ParamSlot& slot) {
    oldAmortLevel_ = glm::ivec2(-1);
    oldWidth_ = -1;
    oldHeight_ = -1;
    return true;
}

void ImageSpaceAmortization2D::updateTextureSize(int w, int h) {
    viewProjMx_ = glm::mat4(1.0f);
    lastViewProjMx_ = glm::mat4(1.0f);

    texLayout_.width = w;
    texLayout_.height = h;
    const std::vector<uint32_t> zeroData(w * h, 0); // uin32_t <=> RGBA8.
    texRead_ = std::make_unique<glowl::Texture2D>("texRead", texLayout_, zeroData.data());
    texWrite_ = std::make_unique<glowl::Texture2D>("texWrite", texLayout_, zeroData.data());

    distTexLayout_.width = w;
    distTexLayout_.height = h;
    const std::vector<float> posInit(2 * w * h, std::numeric_limits<float>::lowest()); // RG32F
    distTexRead_ = std::make_unique<glowl::Texture2D>("distTexRead", distTexLayout_, posInit.data());
    distTexWrite_ = std::make_unique<glowl::Texture2D>("distTexWrite", distTexLayout_, posInit.data());
}

void ImageSpaceAmortization2D::updateAmortSize(glm::ivec2 a, int w, int h) {
    camOffsets_.resize(a.x * a.y);
    for (int j = 0; j < a.y; j++) {
        for (int i = 0; i < a.x; i++) {
            const float x = static_cast<float>(2 * i - a.x + 1) / static_cast<float>(w);
            const float y = static_cast<float>(2 * j - a.y + 1) / static_cast<float>(h);
            camOffsets_[j * a.x + i] = glm::vec3(x, y, 0.0f);
        }
    }

    lowResFBO_->resize((w + a.x - 1) / a.x, (h + a.y - 1) / a.y); // Integer division with round up.

    samplingSequence_.clear();

    const int nextPowerOfTwoExp = static_cast<int>(std::ceil(std::log2(std::max(a.x, a.y))));
    const int nextPowerOfTwoVal = static_cast<int>(std::pow(2, nextPowerOfTwoExp));

    std::array<std::array<int, 2>, 4> offsetPattern{{{0, 0}, {1, 1}, {0, 1}, {1, 0}}};
    std::vector<int> offsetLength(nextPowerOfTwoExp, 0);
    for (int i = 0; i < nextPowerOfTwoExp; i++) {
        offsetLength[i] = static_cast<int>(std::pow(2, nextPowerOfTwoExp - i - 1));
    }

    for (int i = 0; i < nextPowerOfTwoVal * nextPowerOfTwoVal; i++) {
        int x = 0;
        int y = 0;
        for (int j = 0; j < nextPowerOfTwoExp; j++) {
            const int levelIndex = (i / static_cast<int>(std::pow(4, j))) % 4;
            x += offsetPattern[levelIndex][0] * offsetLength[j];
            y += offsetPattern[levelIndex][1] * offsetLength[j];
        }
        if (x < a.x && y < a.y) {
            samplingSequence_.push_back(x + y * a.x);
        }
    }

    samplingSequencePosition_ = 0;
    frameIdx_ = samplingSequence_[samplingSequencePosition_];
}

void ImageSpaceAmortization2D::setupCamera(core::view::Camera& cam, int width, int height, glm::ivec2 a) {
    auto intrinsics = cam.get<core::view::Camera::OrthographicParameters>();
    auto pose = cam.get<core::view::Camera::Pose>();

    const float aspect = intrinsics.aspect.value();
    const float frustumHeight = intrinsics.frustrum_height.value();
    const float frustumWidth = aspect * frustumHeight;

    float wOffs = 0.5f * frustumWidth * camOffsets_[frameIdx_].x;
    float hOffs = 0.5f * frustumHeight * camOffsets_[frameIdx_].y;

    const int lowResWidth = (width + a.x - 1) / a.x;
    const int lowResHeight = (height + a.y - 1) / a.y;

    float wAdj = static_cast<float>(lowResWidth * a.x) / static_cast<float>(width);
    float hAdj = static_cast<float>(lowResHeight * a.y) / static_cast<float>(height);

    wOffs += 0.5f * (wAdj - 1.0f) * frustumWidth;
    hOffs += 0.5f * (hAdj - 1.0f) * frustumHeight;

    pose.position += glm::vec3(wOffs, hOffs, 0.0f);
    intrinsics.frustrum_height = hAdj * frustumHeight;
    intrinsics.aspect = wAdj / hAdj * aspect;

    cam.setOrthographicProjection(intrinsics);
    cam.setPose(pose);
}

void ImageSpaceAmortization2D::reconstruct(
    std::shared_ptr<glowl::FramebufferObject> const& fbo, core::view::Camera const& cam, glm::ivec2 a) {
    int w = fbo->getWidth();
    int h = fbo->getHeight();

    glViewport(0, 0, w, h);
    fbo->bind();

    // Calculate inverse and shift matrix in double precision
    const glm::dmat4 shiftMx = glm::dmat4(lastViewProjMx_) * glm::inverse(glm::dmat4(viewProjMx_));

    const auto intrinsics = cam.get<core::view::Camera::OrthographicParameters>();

    shader_->use();
    shader_->setUniform("amortLevel", a);
    shader_->setUniform("resolution", w, h);
    shader_->setUniform("lowResResolution", lowResFBO_->getWidth(), lowResFBO_->getHeight());
    shader_->setUniform("frameIdx", frameIdx_);
    shader_->setUniform("shiftMx", glm::mat4(shiftMx));
    shader_->setUniform("camCenter", cam.getPose().position);
    shader_->setUniform("camAspect", intrinsics.aspect.value());
    shader_->setUniform("frustumHeight", intrinsics.frustrum_height.value());
    shader_->setUniform(
        "skipInterpolation", static_cast<int>(skipInterpolationParam.Param<core::param::BoolParam>()->Value()));
    shader_->setUniform(
        "showQuadMarker", static_cast<int>(showQuadMarkerParam.Param<core::param::BoolParam>()->Value()));

    glActiveTexture(GL_TEXTURE0);
    lowResFBO_->bindColorbuffer(0);
    shader_->setUniform("texLowResFBO", 0);

    texRead_->bindImage(0, GL_READ_ONLY);
    texWrite_->bindImage(1, GL_WRITE_ONLY);
    distTexRead_->bindImage(2, GL_READ_ONLY);
    distTexWrite_->bindImage(3, GL_WRITE_ONLY);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glUseProgram(0);

    samplingSequencePosition_ = (samplingSequencePosition_ + 1) % samplingSequence_.size();
    frameIdx_ = samplingSequence_[samplingSequencePosition_];
    texRead_.swap(texWrite_);
    distTexRead_.swap(distTexWrite_);
}
