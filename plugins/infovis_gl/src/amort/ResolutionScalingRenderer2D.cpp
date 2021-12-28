#include "ResolutionScalingRenderer2D.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"

using namespace megamol::infovis_gl;
using megamol::core::utility::log::Log;

ResolutionScalingRenderer2D::ResolutionScalingRenderer2D()
        : BaseAmortizedRenderer2D()
        , amortLevelParam("AmortLevel", "Level of Amortization")
        , debugParam("Debug", "some")
        , debugFloatParam("debugInt", "some") {

    this->amortLevelParam << new core::param::IntParam(1, 1);
    this->MakeSlotAvailable(&amortLevelParam);
    this->debugParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&debugParam);
    this->debugFloatParam << new core::param::FloatParam(0.0, 0.0);
    this->MakeSlotAvailable(&debugFloatParam);
}

ResolutionScalingRenderer2D::~ResolutionScalingRenderer2D() {
    this->Release();
}

bool ResolutionScalingRenderer2D::createImpl(const msf::ShaderFactoryOptionsOpenGL& shaderOptions) {
    try {
        shader_ = core::utility::make_glowl_shader("amort_resolutionscaling", shaderOptions,
            "infovis_gl/amort/amort_quad.vert.glsl", "infovis_gl/amort/amort_resolutionscaling.frag.glsl");
        linshader_ = core::utility::make_glowl_shader("amort_linearscaling", shaderOptions,
            "infovis_gl/amort/amort_quad.vert.glsl", "infovis_gl/amort/amort_linearscaling.frag.glsl");

    } catch (std::exception& e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, ("BaseAmortizedRenderer2D: " + std::string(e.what())).c_str());
        return false;
    }

    lowResFBO_ = std::make_shared<glowl::FramebufferObject>(1, 1);
    lowResFBO_->createColorAttachment(GL_RGBA8, GL_RGBA, GL_FLOAT);

    // Store texture layout for later resize
    texLayout_ = glowl::TextureLayout(GL_RGBA8, 1, 1, 1, GL_RGBA, GL_FLOAT, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
            {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});
    texA_ = std::make_unique<glowl::Texture2D>("texStoreA", texLayout_, nullptr);
    texB_ = std::make_unique<glowl::Texture2D>("texStoreB", texLayout_, nullptr);

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

    int a = amortLevelParam.Param<core::param::IntParam>()->Value();
    int w = fbo->getWidth();
    int h = fbo->getHeight();

    if (a != oldLevel_ || w != oldWidth_ || h != oldHeight_) {
        updateSize(a, w, h);
        oldLevel_ = a;
        oldWidth_ = w;
        oldHeight_ = h;
    }

    lowResFBO_->bind();
    glClear(GL_COLOR_BUFFER_BIT);

    setupCamera(cam, w, h, a);

    nextRendererCall.SetFramebuffer(lowResFBO_);
    nextRendererCall.SetCamera(cam);
    (nextRendererCall)(core::view::AbstractCallRender::FnRender);

    reconstruct(fbo, a);

    return true;
}

void ResolutionScalingRenderer2D::updateSize(int a, int w, int h) {
    frameIdx_ = 0;
    movePush_ = glm::mat4(1.0);
    lastProjViewMx_ = glm::mat4(1.0);
    camOffsets_.resize(a * a);
    for (int j = 0; j < a; j++) {
        for (int i = 0; i < a; i++) {
            camOffsets_[j * a + i] = glm::fvec3(((float)a - 1.0 - 2.0 * i) / w, ((float)a - 1.0 - 2.0 * j) / h, 0.0);
        }
    }

    lowResFBO_->resize(ceil(float(w) / float(a)), ceil(float(h) / float(a)));

    texLayout_.width = w;
    texLayout_.height = h;
    texA_ = std::make_unique<glowl::Texture2D>("texStoreA", texLayout_, nullptr);
    texA_->bindTexture();
    GLfloat temp[4]{0.0};
    //glTextureParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, temp);
    //glBindTexture(GL_TEXTURE_2D, 0);
    texB_ = std::make_unique<glowl::Texture2D>("texStoreB", texLayout_, nullptr);
    texB_->bindTexture();
    //glTextureParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, temp);
    //glBindTexture(GL_TEXTURE_2D, 0);
}

void ResolutionScalingRenderer2D::setupCamera(core::view::Camera& cam, int width, int height, int a) {
    auto const projViewMx = cam.getProjectionMatrix() * cam.getViewMatrix();

    auto intrinsics = cam.get<core::view::Camera::OrthographicParameters>();
    glm::vec3 adj_offset = glm::vec3(-intrinsics.aspect * intrinsics.frustrum_height * camOffsets_[frameIdx_].x,
        -intrinsics.frustrum_height * camOffsets_[frameIdx_].y, 0.0);

    movePush_ = lastProjViewMx_ * inverse(projViewMx);
    lastProjViewMx_ = projViewMx;

    auto p = cam.get<core::view::Camera::Pose>();
    p.position = p.position + 0.5f * adj_offset;


    float hAdj = ceil(float(height) / float(a)) / (float(height) / float(a));
    float wAdj = ceil(float(width) / float(a)) / (float(width) / float(a));

    float hOffs = hAdj * intrinsics.frustrum_height - intrinsics.frustrum_height;
    float wOffs =
        wAdj * intrinsics.aspect * intrinsics.frustrum_height - intrinsics.aspect * intrinsics.frustrum_height;
    p.position = p.position + glm::vec3(0.5 * wOffs, 0.5 * hOffs, 0);
    intrinsics.frustrum_height = hAdj * intrinsics.frustrum_height.value();
    intrinsics.aspect = wAdj / hAdj * intrinsics.aspect;


    cam.setOrthographicProjection(intrinsics);

    cam.setPose(p);
}

void ResolutionScalingRenderer2D::reconstruct(std::shared_ptr<glowl::FramebufferObject>& fbo, int a) {
    int w = fbo->getWidth();
    int h = fbo->getHeight();

    glViewport(0, 0, w, h);
    fbo->bind();
    if (!debugParam.Param<core::param::BoolParam>()->Value()) {
        shader_->use();
        shader_->setUniform("amortLevel", a);
        shader_->setUniform("w", w);
        shader_->setUniform("h", h);
        shader_->setUniform("frametype", frameIdx_);
        shader_->setUniform("moveM", movePush_);
        glActiveTexture(GL_TEXTURE4);
        lowResFBO_->bindColorbuffer(0);
        shader_->setUniform("src_tex2D", 4);

    } else {
        linshader_->use();
        linshader_->setUniform("amortLevel", a);
        linshader_->setUniform("w", w);
        linshader_->setUniform("h", h);
        linshader_->setUniform("frametype", frameIdx_);
        linshader_->setUniform("moveM", movePush_);
        glActiveTexture(GL_TEXTURE4);
        lowResFBO_->bindColorbuffer(0);
        linshader_->setUniform("src_tex2D", 4);
    }

    if (!debugParam.Param<core::param::BoolParam>()->Value()) {
        texA_->bindImage(6, GL_READ_ONLY);
        texB_->bindImage(7, GL_WRITE_ONLY);
    } else {
        glActiveTexture(GL_TEXTURE6);
        texA_->bindTexture();
        linshader_->setUniform("Store", 6);
        texB_->bindImage(7, GL_WRITE_ONLY);
    }
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glUseProgram(0);

    frameIdx_ = (frameIdx_ + (a - 1) * (a - 1)) % (a * a);
    texA_.swap(texB_);
}
