/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "ImageSpaceAmortization2D.h"

#include <vector>

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"

using namespace megamol::mmstd_gl;
using megamol::core::utility::log::Log;

ImageSpaceAmortization2D::ImageSpaceAmortization2D()
        : BaseAmortization2D()
        , amortModeParam("AmortMode", "Amortization Mode")
        , amortLevelParam("AmortLevel", "Level of Amortization")
        , skipInterpolationParam("SkipInterpolation", "Do not interpolate missing pixels.")
        , viewProjMx_(glm::mat4())
        , lastViewProjMx_(glm::mat4()) {

    amortModeParam << new core::param::EnumParam(MODE_2D);
    amortModeParam.Param<core::param::EnumParam>()->SetTypePair(MODE_2D, "2D");
    amortModeParam.Param<core::param::EnumParam>()->SetTypePair(MODE_HORIZONTAL, "Horizontal");
    amortModeParam.Param<core::param::EnumParam>()->SetTypePair(MODE_VERTICAL, "Vertical");
    MakeSlotAvailable(&amortModeParam);

    amortLevelParam << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&amortLevelParam);

    skipInterpolationParam << new core::param::BoolParam(false);
    MakeSlotAvailable(&skipInterpolationParam);
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
    const int a = amortLevelParam.Param<core::param::IntParam>()->Value();
    const int w = fbo->getWidth();
    const int h = fbo->getHeight();

    if (m != oldAmortMode_ || a != oldAmortLevel_ || w != oldWidth_ || h != oldHeight_) {
        updateSize(m, a, w, h);
        oldAmortMode_ = m;
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
    setupCamera(lowResCam, w, h, m, a);

    nextRendererCall.SetFramebuffer(lowResFBO_);
    nextRendererCall.SetCamera(lowResCam);
    (nextRendererCall)(core::view::AbstractCallRender::FnRender);

    reconstruct(fbo, cam, m, a);

    return true;
}

void ImageSpaceAmortization2D::updateSize(AmortMode m, int a, int w, int h) {
    viewProjMx_ = glm::mat4(1.0f);
    lastViewProjMx_ = glm::mat4(1.0f);
    const int a_x = (m == MODE_VERTICAL) ? 1 : a;
    const int a_y = (m == MODE_HORIZONTAL) ? 1 : a;

    camOffsets_.resize(a_x * a_y);
    for (int j = 0; j < a_y; j++) {
        for (int i = 0; i < a_x; i++) {
            const float x = static_cast<float>(2 * i - a_x + 1) / static_cast<float>(w);
            const float y = static_cast<float>(2 * j - a_y + 1) / static_cast<float>(h);
            camOffsets_[j * a_x + i] = glm::vec3(x, y, 0.0f);
        }
    }

    lowResFBO_->resize((w + a_x - 1) / a_x, (h + a_y - 1) / a_y); // Integer division with round up.

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

    samplingSequence_.clear();

    if (m == MODE_2D) {
        const int nextPowerOfTwoExp = static_cast<int>(std::ceil(std::log2(a)));
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
            if (x < a && y < a) {
                samplingSequence_.push_back(x + y * a);
            }
        }
    } else {
        // TODO linear sampling pattern is not nice
        for (int i = 0; i < a; i++) {
            samplingSequence_.push_back(i);
        }
    }

    samplingSequencePosition_ = 0;
    frameIdx_ = samplingSequence_[samplingSequencePosition_];
}

void ImageSpaceAmortization2D::setupCamera(core::view::Camera& cam, int width, int height, AmortMode m, int a) {
    auto intrinsics = cam.get<core::view::Camera::OrthographicParameters>();
    auto pose = cam.get<core::view::Camera::Pose>();

    const float aspect = intrinsics.aspect.value();
    const float frustumHeight = intrinsics.frustrum_height.value();
    const float frustumWidth = aspect * frustumHeight;

    float wOffs = 0.5f * frustumWidth * camOffsets_[frameIdx_].x;
    float hOffs = 0.5f * frustumHeight * camOffsets_[frameIdx_].y;

    const int a_x = (m == MODE_VERTICAL) ? 1 : a;
    const int a_y = (m == MODE_HORIZONTAL) ? 1 : a;

    const int lowResWidth = (width + a_x - 1) / a_x;
    const int lowResHeight = (height + a_y - 1) / a_y;

    float wAdj = static_cast<float>(lowResWidth * a_x) / static_cast<float>(width);
    float hAdj = static_cast<float>(lowResHeight * a_y) / static_cast<float>(height);

    wOffs += 0.5f * (wAdj - 1.0f) * frustumWidth;
    hOffs += 0.5f * (hAdj - 1.0f) * frustumHeight;

    pose.position += glm::vec3(wOffs, hOffs, 0.0f);
    intrinsics.frustrum_height = hAdj * frustumHeight;
    intrinsics.aspect = wAdj / hAdj * aspect;

    cam.setOrthographicProjection(intrinsics);
    cam.setPose(pose);
}

void ImageSpaceAmortization2D::reconstruct(
    std::shared_ptr<glowl::FramebufferObject> const& fbo, core::view::Camera const& cam, AmortMode m, int a) {
    int w = fbo->getWidth();
    int h = fbo->getHeight();

    glViewport(0, 0, w, h);
    fbo->bind();

    // Calculate inverse and shift matrix in double precision
    const glm::dmat4 shiftMx = glm::dmat4(lastViewProjMx_) * glm::inverse(glm::dmat4(viewProjMx_));

    const auto intrinsics = cam.get<core::view::Camera::OrthographicParameters>();

    shader_->use();
    if (m == MODE_HORIZONTAL) {
        shader_->setUniform("amortLevel", a, 1);
    } else if (m == MODE_VERTICAL) {
        shader_->setUniform("amortLevel", 1, a);
    } else {
        shader_->setUniform("amortLevel", a, a);
    }
    shader_->setUniform("resolution", w, h);
    shader_->setUniform("lowResResolution", lowResFBO_->getWidth(), lowResFBO_->getHeight());
    shader_->setUniform("frameIdx", frameIdx_);
    shader_->setUniform("shiftMx", glm::mat4(shiftMx));
    shader_->setUniform("camCenter", cam.getPose().position);
    shader_->setUniform("camAspect", intrinsics.aspect.value());
    shader_->setUniform("frustumHeight", intrinsics.frustrum_height.value());
    shader_->setUniform(
        "skipInterpolation", static_cast<int>(skipInterpolationParam.Param<core::param::BoolParam>()->Value()));

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
