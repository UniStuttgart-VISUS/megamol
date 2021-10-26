#include "ResolutionScalingRenderer2D.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"

using namespace megamol::infovis;
using megamol::core::utility::log::Log;

ResolutionScalingRenderer2D::ResolutionScalingRenderer2D()
        : BaseAmortizedRenderer2D(), amortLevelParam("AmortLevel", "Level of Amortization") {

    this->amortLevelParam << new core::param::IntParam(1, 1);
    this->MakeSlotAvailable(&amortLevelParam);
}

ResolutionScalingRenderer2D::~ResolutionScalingRenderer2D() {
    this->Release();
}

bool ResolutionScalingRenderer2D::createImpl(const msf::ShaderFactoryOptionsOpenGL& shaderOptions) {
    try {
        amort_reconstruction_shdr_array[6] = core::utility::make_glowl_shader("amort_reconstruction6", shaderOptions,
            "infovis/amort/amort_reconstruction.vert.glsl", "infovis/amort/amort_reconstruction6.frag.glsl");
    } catch (std::exception& e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, ("BaseAmortizedRenderer2D: " + std::string(e.what())).c_str());
        return false;
    }

    if (glowlFBO == nullptr) {
        glowlFBO = std::make_shared<glowl::FramebufferObject>(1, 1);
        glowlFBO->createColorAttachment(GL_RGBA32F, GL_RGBA, GL_FLOAT);
    }

    if (texA == nullptr || texB == nullptr) {
        texstore_layout = glowl::TextureLayout(GL_RGBA32F, 1, 1, 1, GL_RGBA, GL_FLOAT, 1,
            {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
                {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_NEAREST},
                {GL_TEXTURE_MAG_FILTER, GL_NEAREST}},
            {});
        texA = std::make_unique<glowl::Texture2D>("texStoreA", texstore_layout, nullptr);
        texB = std::make_unique<glowl::Texture2D>("texStoreB", texstore_layout, nullptr);
    }

    auto err = glGetError();
    if (err != GL_NO_ERROR) {
        Log::DefaultLog.WriteError("GL_ERROR in BaseAmortizedRenderer2D: %i", err);
    }

    return true;
}

void ResolutionScalingRenderer2D::releaseImpl() {}

bool ResolutionScalingRenderer2D::renderImpl(core::view::CallRender2DGL& targetRendererCall,
    std::shared_ptr<core::view::CallRender2DGL::FBO_TYPE> fbo, core::view::Camera cam) {
    mvMatrix = cam.getViewMatrix();
    projMatrix = cam.getProjectionMatrix();
    this->fbo = fbo;

    int a = amortLevelParam.Param<core::param::IntParam>()->Value();
    int w = fbo->getWidth();
    int h = fbo->getHeight();

    // check if amortization mode changed
    if (w != oldW || h != oldH || a != oldaLevel) {
        resizeArrays(w, h);
    }
    setupAccel(w, h, &cam);
    targetRendererCall.SetFramebuffer(glowlFBO);
    targetRendererCall.SetCamera(cam);

    // send call to next renderer in line
    (targetRendererCall)(core::view::AbstractCallRender::FnRender);

    doReconstruction(w, h);

    // to avoid excessive resizing, retain last render variables and check if changed
    oldH = h;
    oldW = w;
    oldaLevel = a;
}

void ResolutionScalingRenderer2D::resizeArrays(int w, int h) {

    int a = this->amortLevelParam.Param<core::param::IntParam>()->Value();
    framesNeeded = a * a;
    frametype = 0;
    movePush = glm::mat4(1.0);
    lastPmvm = glm::mat4(1.0);
    invMatrices.resize(framesNeeded);
    moveMatrices.resize(framesNeeded);
    camOffsets.resize(framesNeeded);
    for (int j = 0; j < a; j++) {
        for (int i = 0; i < a; i++) {
            camOffsets[j * a + i] = glm::fvec3(((float) a - 1.0 - 2.0 * i) / w, ((float) a - 1.0 - 2.0 * j) / h, 0.0);
        }
    }
    // glActiveTexture(GL_TEXTURE4);
    glowlFBO->resize(w / a, h / a);

    texstore_layout.width = w;
    texstore_layout.height = h;

    texA = std::make_unique<glowl::Texture2D>("texStoreA", texstore_layout, nullptr);
    texB = std::make_unique<glowl::Texture2D>("texStoreB", texstore_layout, nullptr);
}

void ResolutionScalingRenderer2D::setupAccel(int ow, int oh, core::view::Camera* cam) {
    int a = this->amortLevelParam.Param<core::param::IntParam>()->Value();
    glm::mat4 pm;
    glm::mat4 mvm;
    auto pmvm = projMatrix * mvMatrix;

    auto intrinsics = cam->get<core::view::Camera::OrthographicParameters>();
    glm::vec3 adj_offset = glm::vec3(-intrinsics.aspect * intrinsics.frustrum_height * camOffsets[frametype].x,
        -intrinsics.frustrum_height * camOffsets[frametype].y, 0.0);

    // glm::mat4 jit = glm::translate(glm::mat4(1.0f), adj_offset);
    movePush = lastPmvm * inverse(pmvm);
    // movePush = glm::mat4(1.0);
    lastPmvm = pmvm;

    auto p = cam->get<core::view::Camera::Pose>();
    p.position = p.position + 0.5f * adj_offset;
    cam->setPose(p);
    glowlFBO->bind();

    glClear(GL_COLOR_BUFFER_BIT);
}

void ResolutionScalingRenderer2D::doReconstruction(int w, int h) {
    glViewport(0, 0, w, h);

    amort_reconstruction_shdr_array[6]->use();

    fbo->bind();
    int a = this->amortLevelParam.Param<core::param::IntParam>()->Value();

    amort_reconstruction_shdr_array[6]->setUniform("h", h);
    amort_reconstruction_shdr_array[6]->setUniform("w", w);
    amort_reconstruction_shdr_array[6]->setUniform("amortLevel", a);

    glUniformMatrix4fv(
        amort_reconstruction_shdr_array[6]->getUniformLocation("moveMatrices"), 4, GL_FALSE, &moveMatrices[0][0][0]);

    if (parity == 0) {
        texA->bindImage(6, GL_READ_ONLY);
        amort_reconstruction_shdr_array[6]->setUniform("StoreA", 6);
        texB->bindImage(7, GL_WRITE_ONLY);
        amort_reconstruction_shdr_array[6]->setUniform("StoreB", 7);
    } else {
        texA->bindImage(7, GL_WRITE_ONLY);
        amort_reconstruction_shdr_array[6]->setUniform("StoreA", 7);
        texB->bindImage(6, GL_READ_ONLY);
        amort_reconstruction_shdr_array[6]->setUniform("StoreB", 6);
    }
    glActiveTexture(GL_TEXTURE4);
    // glBindTexture(GL_TEXTURE_2D, glowlFBO->getColorAttachment(0)->getName());
    glowlFBO->bindColorbuffer(0);
    amort_reconstruction_shdr_array[6]->setUniform("src_tex2D", 4);
    amort_reconstruction_shdr_array[6]->setUniform("moveM", movePush);
    amort_reconstruction_shdr_array[6]->setUniform("frametype", frametype);

    glDrawArrays(GL_TRIANGLES, 0, 6);
    glUseProgram(0);

    frametype = (frametype + (this->amortLevelParam.Param<core::param::IntParam>()->Value() - 1) *
                                 (this->amortLevelParam.Param<core::param::IntParam>()->Value() - 1)) %
                framesNeeded;
    parity = (parity + 1) % 2;
}
