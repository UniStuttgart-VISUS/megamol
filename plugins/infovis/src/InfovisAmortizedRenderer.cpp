#include "InfovisAmortizedRenderer.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/ShaderFactory.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/view/CallRender2DGL.h"
#include "mmcore/view/MouseFlags.h"

using namespace megamol;
using namespace megamol::infovis;

InfovisAmortizedRenderer::InfovisAmortizedRenderer()
        : Renderer2D()
        , nextRendererSlot("nextRenderer", "connects to following Renderers, that will render in reduced resolution.")
        , halveRes("Halvres", "Turn on switch")
        , approachEnumSlot("AmortizationMode", "Which amortization approach to use.")
        , superSamplingLevelSlot("SSLevel", "Level of Supersampling")
        , amortLevel("AmortLevel", "Level of Amortization") {
    this->nextRendererSlot.SetCompatibleCall<megamol::core::view::CallRender2DGLDescription>();
    this->MakeSlotAvailable(&this->nextRendererSlot);

    this->halveRes << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&halveRes);

    auto* approachMappings = new core::param::EnumParam(6);
    approachMappings->SetTypePair(MS_AR, "MS-AR");
    approachMappings->SetTypePair(QUAD_AR, "4xAR");
    approachMappings->SetTypePair(QUAD_AR_C, "4xAR-C");
    approachMappings->SetTypePair(SS_AR, "SS-AR");
    approachMappings->SetTypePair(PARAMETER_AR, "Parameterized-AR");
    approachMappings->SetTypePair(DEBUG_PLACEHOLDER, "debug");
    approachMappings->SetTypePair(PUSH_AR, "Push-AR");
    this->approachEnumSlot << approachMappings;
    this->MakeSlotAvailable(&approachEnumSlot);

    this->superSamplingLevelSlot << new core::param::IntParam(1, 1);
    this->MakeSlotAvailable(&superSamplingLevelSlot);

    this->amortLevel << new core::param::IntParam(1, 1);
    this->MakeSlotAvailable(&amortLevel);
}

InfovisAmortizedRenderer::~InfovisAmortizedRenderer() {
    this->Release();
}

bool megamol::infovis::InfovisAmortizedRenderer::create() {
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("ignore: %i", glGetError());

    if (!makeShaders()) {
        return false;
    }
    setupBuffers();
    return true;
}

// TODO
void InfovisAmortizedRenderer::release() {}

std::vector<glm::fvec3> InfovisAmortizedRenderer::calculateHammersley(int until, int ow, int oh) {
    // calculation of Positions according to hammersley sequence
    // https://www.researchgate.net/publication/244441430_Sampling_with_Hammersley_and_Halton_Points
    std::vector<glm::fvec3> outputArray(until);
    float p = 0;
    float u;
    float v = 0;
    for (int k = 0; k < until; k++) {
        u = 0;
        p = (float) 0.5;
        for (int kk = k; kk > 0; p *= 0.5, kk >>= 1) {
            if (kk % 2 == 1) {
                u += p;
            }
            v = (float) ((k + 0.5) / until);
        }
        outputArray[k] = glm::vec3(u - floor(2 * u), v - floor(2 * v), 0);
    }
    std::vector<glm::fvec3> camOffsets(4 * until);
    for (int i = 0; i < until; i++) {
        glm::vec3 temp = (outputArray[i] + glm::vec3(-1.0, +1.0, 0.0));
        camOffsets[4 * i] = glm::vec3(temp.x / ow, temp.y / oh, 0);
        temp = (outputArray[i] + glm::vec3(+1.0, +1.0, 0.0));
        camOffsets[4 * i + 1] = glm::vec3(temp.x / ow, temp.y / oh, 0);
        temp = (outputArray[i] + glm::vec3(-1.0, -1.0, 0.0));
        camOffsets[4 * i + 2] = glm::vec3(temp.x / ow, temp.y / oh, 0);
        temp = (outputArray[i] + glm::vec3(+1.0, -1.0, 0.0));
        camOffsets[4 * i + 3] = glm::vec3(temp.x / ow, temp.y / oh, 0);
    }
    // jit = glm::translate(glm::mat4(1.0f), glm::vec3((camOffsets[f].x - 1 + 2 * (frametype % 2)) / ow,
    // (camOffsets[f].y + 1 - 2 * floor((frametype % 4) / 2)) / oh, 0));

    return camOffsets;
}

bool InfovisAmortizedRenderer::makeShaders() {
    auto const shader_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());

    try {
        amort_reconstruction_shdr_array[0] = core::utility::make_glowl_shader("amort_reconstruction0", shader_options,
            "infovis/amort_reconstruction.vert.glsl", "infovis/amort_reconstruction0.frag.glsl");
        amort_reconstruction_shdr_array[1] = core::utility::make_glowl_shader("amort_reconstruction1", shader_options,
            "infovis/amort_reconstruction.vert.glsl", "infovis/amort_reconstruction1.frag.glsl");
        amort_reconstruction_shdr_array[2] = core::utility::make_glowl_shader("amort_reconstruction2", shader_options,
            "infovis/amort_reconstruction.vert.glsl", "infovis/amort_reconstruction2.frag.glsl");
        amort_reconstruction_shdr_array[3] = core::utility::make_glowl_shader("amort_reconstruction3", shader_options,
            "infovis/amort_reconstruction.vert.glsl", "infovis/amort_reconstruction3.frag.glsl");
        amort_reconstruction_shdr_array[4] = core::utility::make_glowl_shader("amort_reconstruction4", shader_options,
            "infovis/amort_reconstruction.vert.glsl", "infovis/amort_reconstruction4.frag.glsl");
        amort_reconstruction_shdr_array[5] = core::utility::make_glowl_shader("amort_reconstruction5", shader_options,
            "infovis/amort_reconstruction.vert.glsl", "infovis/amort_reconstruction5.frag.glsl");
        amort_reconstruction_shdr_array[6] = core::utility::make_glowl_shader("amort_reconstruction6", shader_options,
            "infovis/amort_reconstruction.vert.glsl", "infovis/amort_reconstruction6.frag.glsl");
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            ("InfovisAmortizedRenderer: " + std::string(e.what())).c_str());
        return false;
    }
    return true;
}

void InfovisAmortizedRenderer::setupBuffers() {
    glGenFramebuffers(1, &amortizedFboA);
    glGenFramebuffers(1, &amortizedMsaaFboA);
    // glGenFramebuffers(1, &amortizedPushFBO);
    if (glowlFBO == nullptr) {
        glowlFBO = std::make_shared<glowl::FramebufferObject>(1, 1);
        glowlFBO->createColorAttachment(GL_RGBA32F, GL_RGBA, GL_FLOAT);
        glDisable(GL_MULTISAMPLE);
        glowlFBOms = std::make_shared<glowl::FramebufferObject>(1, 1);
        glowlFBOms->createColorAttachment(GL_RGBA32F, GL_RGBA, GL_FLOAT);
        glowlFBOms->bind();
        glEnable(GL_MULTISAMPLE);
    }

    glGenTextures(1, &imageArrayA);
    glGenTextures(1, &pushImage);
    glGenTextures(1, &imStoreArray);

    // glGenTextures(1, &imStoreA);
    // glGenTextures(1, &imStoreB);


    if (texA == nullptr || texB == nullptr) {
        texstore_layout = glowl::TextureLayout(GL_RGBA32F, 1, 1, 1, GL_RGBA, GL_FLOAT, 1,
            {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
                {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_NEAREST},
                {GL_TEXTURE_MAG_FILTER, GL_NEAREST}},
            {});
        texA = std::make_unique<glowl::Texture2D>("texStoreA", texstore_layout, nullptr);
        texB = std::make_unique<glowl::Texture2D>("texStoreB", texstore_layout, nullptr);
    }
    glGenBuffers(1, &ssboMatrices);

    glBindFramebuffer(GL_FRAMEBUFFER, amortizedMsaaFboA);
    glEnable(GL_MULTISAMPLE);

    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, glowlFBOms->getColorAttachment(0)->getName());
    glTexImage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 2, GL_RGB, 1, 1, 2, GL_TRUE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

    glBindTexture(GL_TEXTURE_2D_ARRAY, imageArrayA);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB, 1, 1, 1, 0, GL_RGB, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

    glBindTexture(GL_TEXTURE_2D, pushImage);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

    glBindTexture(GL_TEXTURE_2D_ARRAY, imStoreArray);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB, 1, 1, 1, 0, GL_RGB, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

    glBindTexture(GL_TEXTURE_2D, imStoreA);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

    glBindTexture(GL_TEXTURE_2D, imStoreB);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

    auto err = glGetError();
    if (err != GL_NO_ERROR) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("GL_ERROR in InfovisAmortizedRenderer: " + err);
    }
}

void InfovisAmortizedRenderer::setupAccel(int approach, int ow, int oh, int ssLevel, core::view::Camera* cam) {
    int a = this->amortLevel.Param<core::param::IntParam>()->Value();
    int w = ceil(float(ow) / a);
    int h = ceil(float(oh) / a);
    glm::mat4 pm;
    glm::mat4 mvm;
    auto pmvm = projMatrix * mvMatrix;

    if (approach == 0) {
        invMatrices[frametype] = pmvm;
        for (int i = 0; i < framesNeeded; i++)
            moveMatrices[i] = invMatrices[i] * glm::inverse(pmvm);

        glowlFBOms->bind();
        glActiveTexture(GL_TEXTURE10);
        glEnable(GL_MULTISAMPLE);

        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, glowlFBOms->getColorAttachment(0)->getName());
        glTexImage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 2, GL_RGBA8, w, h, 2, GL_TRUE);
        glFramebufferTextureLayer(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, glowlFBOms->getColorAttachment(0)->getName(), 0, frametype);

        glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_PROGRAMMABLE_SAMPLE_LOCATIONS_ARB, 1);
        const float tbl[4] = {0.25 + (frametype * 0.5), 0.25, 0.75 - (frametype * 0.5), 0.75};
        glFramebufferSampleLocationsfvARB(GL_FRAMEBUFFER, 0u, 2, tbl);
        // glowlFBO->bind();

    } else {
        glDisable(GL_MULTISAMPLE);
    }
    if (approach == 1 || approach == 2 || approach == 3) {
        glm::mat4 jit;
        glm::mat4 pmvm = pm * mvm;

        jit = glm::translate(glm::mat4(1.0f), camOffsets[frametype]);
        invMatrices[frametype] = jit * pmvm;
        for (int i = 0; i < framesNeeded; i++)
            moveMatrices[i] = invMatrices[i] * glm::inverse(pmvm);

        pm = jit * pm;
        for (int i = 0; i < 16; i++)
            projMatrix_column[i] = glm::value_ptr(pm)[i];

        glBindFramebuffer(GL_FRAMEBUFFER, amortizedFboA);
        glActiveTexture(GL_TEXTURE10);
        glBindTexture(GL_TEXTURE_2D_ARRAY, imageArrayA);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB, w, h, 4 * ssLevel, 0, GL_RGB, GL_FLOAT, 0);
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, imageArrayA, 0, frametype);
    }
    if (approach == 4) {
        glm::mat4 jit;
        glm::mat4 pmvm = pm * mvm;
        int a = this->amortLevel.Param<core::param::IntParam>()->Value();

        jit = glm::translate(glm::mat4(1.0f), camOffsets[frametype]);
        // invMatrices[frametype] = jit * pmvm;
        invMatrices[frametype] = pmvm;
        for (int i = 0; i < framesNeeded; i++)
            moveMatrices[i] = invMatrices[i] * glm::inverse(pmvm);

        pm = jit * pm;
        for (int i = 0; i < 16; i++)
            projMatrix_column[i] = glm::value_ptr(pm)[i];

        glBindFramebuffer(GL_FRAMEBUFFER, amortizedFboA);
        glActiveTexture(GL_TEXTURE10);
        glBindTexture(GL_TEXTURE_2D_ARRAY, imageArrayA);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB, w, h, a * a, 0, GL_RGB, GL_FLOAT, 0);

        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, imageArrayA, 0, frametype);
    }
    if (approach == 5) {
        glBindFramebuffer(GL_FRAMEBUFFER, amortizedMsaaFboA);
        glActiveTexture(GL_TEXTURE10);
        glEnable(GL_MULTISAMPLE);

        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, msImageArray);
        glTexImage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 2, GL_RGBA8, w, h, 1, GL_TRUE);
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, msImageArray, 0, 0);

        glm::mat4 pmvm = pm * mvm;
        int a = this->amortLevel.Param<core::param::IntParam>()->Value();

        movePush = lastPmvm * inverse(pmvm);
        lastPmvm = pmvm;

        glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_PROGRAMMABLE_SAMPLE_LOCATIONS_ARB, 1);
        const float tbl[2] = {(1.0 + (frametype % a) * 2.0) / (2.0 * a), (1.0 + int(frametype / a) * 2.0) / (2.0 * a)};
        glFramebufferSampleLocationsfvARB(GL_FRAMEBUFFER, 0u, 1, tbl);
    }
    if (approach == 6) {
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
    }
    glClear(GL_COLOR_BUFFER_BIT);
}

void InfovisAmortizedRenderer::resizeArrays(int approach, int w, int h, int ssLevel) {
    // glDeleteTextures(1, &imStoreA);
    // glDeleteTextures(1, &imStoreB);

    if (approach == 0) {
        framesNeeded = 2;
        if (invMatrices.size() != framesNeeded) {
            invMatrices.resize(framesNeeded);
            moveMatrices.resize(framesNeeded);
            frametype = 0;
        }
    }
    if (approach == 1) {
        framesNeeded = 4;
        if (invMatrices.size() != framesNeeded) {
            invMatrices.resize(framesNeeded);
            moveMatrices.resize(framesNeeded);
            frametype = 0;
            camOffsets.resize(4);
            camOffsets = {glm::vec3(-1.0 / w, 1.0 / h, 0), glm::vec3(1.0 / w, 1.0 / h, 0),
                glm::vec3(-1.0 / w, -1.0 / h, 0), glm::vec3(1.0 / w, -1.0 / h, 0)};
        }
        glActiveTexture(GL_TEXTURE10);
        glBindTexture(GL_TEXTURE_2D_ARRAY, imageArrayA);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB, w, h, 4 * ssLevel, 0, GL_RGB, GL_FLOAT, 0);
    }
    if (approach == 2) {

        framesNeeded = 4;
        if (invMatrices.size() != framesNeeded) {
            invMatrices.resize(framesNeeded);
            moveMatrices.resize(framesNeeded);
            frametype = 0;
            camOffsets.resize(framesNeeded);
            camOffsets = {glm::vec3(-2.0 / w, 2.0 / h, 0), glm::vec3(0.0 / w, 2.0 / h, 0),
                glm::vec3(-2.0 / w, 0 / h, 0), glm::vec3(0.0 / w, 0.0 / h, 0)};
        }
        glActiveTexture(GL_TEXTURE10);
        glBindTexture(GL_TEXTURE_2D_ARRAY, imageArrayA);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB, w, h, 4 * ssLevel, 0, GL_RGB, GL_FLOAT, 0);
    }
    if (approach == 3) {
        framesNeeded = 4 * ssLevel;
        if (invMatrices.size() != framesNeeded || camOffsets.size() != ssLevel) {
            invMatrices.resize(framesNeeded);
            moveMatrices.resize(framesNeeded);
            camOffsets.resize(4 * ssLevel);
            camOffsets = calculateHammersley(ssLevel, w, h);
            frametype = 0;
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMatrices);
        glBufferData(
            GL_SHADER_STORAGE_BUFFER, framesNeeded * sizeof(moveMatrices[0]), &moveMatrices[0][0][0], GL_STATIC_READ);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, ssboMatrices);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        glActiveTexture(GL_TEXTURE10);
        glBindTexture(GL_TEXTURE_2D_ARRAY, imageArrayA);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB, w, h, 4 * ssLevel, 0, GL_RGB, GL_FLOAT, 0);
    }
    if (approach == 4) {
        int a = this->amortLevel.Param<core::param::IntParam>()->Value();
        framesNeeded = a * a;
        if (invMatrices.size() != framesNeeded) {
            invMatrices.resize(framesNeeded);
            moveMatrices.resize(framesNeeded);
            frametype = 0;
            camOffsets.resize(framesNeeded);
            for (int j = 0; j < a; j++) {
                for (int i = 0; i < a; i++) {
                    camOffsets[j * a + i] =
                        glm::fvec3(((float) a - 1.0 - 2.0 * i) / w, ((float) a - 1.0 - 2.0 * j) / h, 0.0);
                }
            }
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMatrices);
        glBufferData(
            GL_SHADER_STORAGE_BUFFER, framesNeeded * sizeof(moveMatrices[0]), &moveMatrices[0][0][0], GL_STATIC_READ);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, ssboMatrices);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
    if (approach == 5) {
        int a = this->amortLevel.Param<core::param::IntParam>()->Value();
        framesNeeded = a * a;
        frametype = 0;
        movePush = glm::mat4(1.0);
        lastPmvm = glm::mat4(1.0);
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D, imStoreA);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, w, h);
        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_2D, imStoreB);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, w, h);
    }
    if (approach == 6) {
        int a = this->amortLevel.Param<core::param::IntParam>()->Value();
        framesNeeded = a * a;
        frametype = 0;
        movePush = glm::mat4(1.0);
        lastPmvm = glm::mat4(1.0);
        invMatrices.resize(framesNeeded);
        moveMatrices.resize(framesNeeded);
        camOffsets.resize(framesNeeded);
        for (int j = 0; j < a; j++) {
            for (int i = 0; i < a; i++) {
                camOffsets[j * a + i] =
                    glm::fvec3(((float) a - 1.0 - 2.0 * i) / w, ((float) a - 1.0 - 2.0 * j) / h, 0.0);
            }
        }
        // glActiveTexture(GL_TEXTURE4);
        glowlFBO->resize(w / a, h / a);

        texstore_layout.width = w;
        texstore_layout.height = h;

        texA = std::make_unique<glowl::Texture2D>("texStoreA", texstore_layout, nullptr);
        texB = std::make_unique<glowl::Texture2D>("texStoreB", texstore_layout, nullptr);
    }
}

bool InfovisAmortizedRenderer::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {
    auto* cr = this->nextRendererSlot.CallAs<megamol::core::view::CallRender2DGL>();
    if (cr) {
        megamol::core::view::InputEvent evt;
        evt.tag = megamol::core::view::InputEvent::Tag::MouseButton;
        evt.mouseButtonData.button = button;
        evt.mouseButtonData.action = action;
        evt.mouseButtonData.mods = mods;
        cr->SetInputEvent(evt);
        return (*cr)(megamol::core::view::CallRender2DGL::FnOnMouseButton);
    }
    return false;
}

bool InfovisAmortizedRenderer::OnMouseMove(double x, double y) {
    auto* cr = this->nextRendererSlot.CallAs<megamol::core::view::CallRender2DGL>();
    if (cr) {
        megamol::core::view::InputEvent evt;
        evt.tag = megamol::core::view::InputEvent::Tag::MouseMove;
        evt.mouseMoveData.x = x;
        evt.mouseMoveData.y = y;
        cr->SetInputEvent(evt);
        return (*cr)(megamol::core::view::CallRender2DGL::FnOnMouseMove);
    }
    return false;
}

bool InfovisAmortizedRenderer::OnMouseScroll(double dx, double dy) {
    auto* cr = this->nextRendererSlot.CallAs<megamol::core::view::CallRender2DGL>();
    if (cr) {
        megamol::core::view::InputEvent evt;
        evt.tag = megamol::core::view::InputEvent::Tag::MouseScroll;
        evt.mouseScrollData.dx = dx;
        evt.mouseScrollData.dy = dy;
        cr->SetInputEvent(evt);
        return (*cr)(megamol::core::view::CallRender2DGL::FnOnMouseScroll);
    }
    return false;
}

bool InfovisAmortizedRenderer::OnChar(unsigned int codePoint) {
    auto* cr = this->nextRendererSlot.CallAs<megamol::core::view::CallRender2DGL>();
    if (cr) {
        megamol::core::view::InputEvent evt;
        evt.tag = megamol::core::view::InputEvent::Tag::Char;
        evt.charData.codePoint = codePoint;
        cr->SetInputEvent(evt);
        return (*cr)(megamol::core::view::CallRender2DGL::FnOnChar);
    }
    return false;
}

bool InfovisAmortizedRenderer::OnKey(
    megamol::core::view::Key key, megamol::core::view::KeyAction action, megamol::core::view::Modifiers mods) {
    auto* cr = this->nextRendererSlot.CallAs<megamol::core::view::CallRender2DGL>();
    if (cr) {
        megamol::core::view::InputEvent evt;
        evt.tag = megamol::core::view::InputEvent::Tag::Key;
        evt.keyData.key = key;
        evt.keyData.action = action;
        evt.keyData.mods = mods;
        cr->SetInputEvent(evt);
        return (*cr)(megamol::core::view::CallRender2DGL::FnOnKey);
    }
    return false;
}

void InfovisAmortizedRenderer::doReconstruction(int approach, int w, int h, int ssLevel) {
    glViewport(0, 0, w, h);

    amort_reconstruction_shdr_array[approach]->use();

    fbo->bind();
    int a = this->amortLevel.Param<core::param::IntParam>()->Value();

    amort_reconstruction_shdr_array[approach]->setUniform("h", h);
    amort_reconstruction_shdr_array[approach]->setUniform("w", w);
    amort_reconstruction_shdr_array[approach]->setUniform(
        "amortLevel", this->amortLevel.Param<core::param::IntParam>()->Value());

    if (approach == 0 || approach == 1 || approach == 2 || approach == 3) {
        amort_reconstruction_shdr_array[approach]->setUniform("src_tex2D", 10);
        amort_reconstruction_shdr_array[approach]->setUniform("ssLevel", ssLevel);
        glUniformMatrix4fv(amort_reconstruction_shdr_array[approach]->getUniformLocation("moveMatrices"), a * a,
            GL_FALSE, &moveMatrices[0][0][0]);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMatrices);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, ssboMatrices);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
    if (approach == 4) {
        glUniformMatrix4fv(amort_reconstruction_shdr_array[approach]->getUniformLocation("moveMatrices"), a * a,
            GL_FALSE, &moveMatrices[0][0][0]);
    } else {
        glUniformMatrix4fv(amort_reconstruction_shdr_array[approach]->getUniformLocation("moveMatrices"), 4, GL_FALSE,
            &moveMatrices[0][0][0]);
    }
    if (approach == 5) {
        amort_reconstruction_shdr_array[approach]->setUniform("frametype", frametype);
        amort_reconstruction_shdr_array[approach]->setUniform("src_tex2D", 10);
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D, imStoreA);
        glBindImageTexture(5, imStoreA, 0, GL_TRUE, 0, GL_READ_WRITE, GL_RGBA32F);
        amort_reconstruction_shdr_array[approach]->setUniform("StoreA", 5);

        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_2D, imStoreB);
        glBindImageTexture(6, imStoreB, 0, GL_TRUE, 0, GL_READ_WRITE, GL_RGBA32F);
        amort_reconstruction_shdr_array[approach]->setUniform("StoreB", 6);

        amort_reconstruction_shdr_array[approach]->setUniform("moveM", movePush);
    }
    if (approach == 6) {
        if (parity == 0) {
            texA->bindImage(6, GL_READ_ONLY);
            amort_reconstruction_shdr_array[approach]->setUniform("StoreA", 6);
            texB->bindImage(7, GL_WRITE_ONLY);
            amort_reconstruction_shdr_array[approach]->setUniform("StoreB", 7);
        } else {
            texA->bindImage(7, GL_WRITE_ONLY);
            amort_reconstruction_shdr_array[approach]->setUniform("StoreA", 7);
            texB->bindImage(6, GL_READ_ONLY);
            amort_reconstruction_shdr_array[approach]->setUniform("StoreB", 6);
        }
        glActiveTexture(GL_TEXTURE4);
        // glBindTexture(GL_TEXTURE_2D, glowlFBO->getColorAttachment(0)->getName());
        glowlFBO->bindColorbuffer(0);
        amort_reconstruction_shdr_array[approach]->setUniform("src_tex2D", 4);
        amort_reconstruction_shdr_array[approach]->setUniform("moveM", movePush);
        amort_reconstruction_shdr_array[approach]->setUniform("frametype", frametype);
    }

    glDrawArrays(GL_TRIANGLES, 0, 6);
    glUseProgram(0);

    if (approach == 6) {
        frametype = (frametype + (this->amortLevel.Param<core::param::IntParam>()->Value() - 1) *
                                     (this->amortLevel.Param<core::param::IntParam>()->Value() - 1)) %
                    framesNeeded;
        parity = (parity + 1) % 2;
    } else {
        frametype = (frametype + 1) % framesNeeded;
        parity = (parity + 1) % 2;
    }
}

bool InfovisAmortizedRenderer::Render(core::view::CallRender2DGL& call) {
    core::view::CallRender2DGL* cr2d = this->nextRendererSlot.CallAs<core::view::CallRender2DGL>();

    if (cr2d == nullptr) {
        // Nothing to do really
        return true;
    }

    // get camera
    core::view::Camera cam = call.GetCamera();
    mvMatrix = cam.getViewMatrix();
    projMatrix = cam.getProjectionMatrix();

    cr2d->SetTime(call.Time());
    cr2d->SetInstanceTime(call.InstanceTime());
    cr2d->SetLastFrameTime(call.LastFrameTime());

    auto bg = call.BackgroundColor();

    backgroundColor[0] = bg[0];
    backgroundColor[1] = bg[1];
    backgroundColor[2] = bg[2];
    backgroundColor[3] = 1.0;
    fbo = call.GetFramebuffer();
    cr2d->SetBackgroundColor(call.BackgroundColor());
    cr2d->AccessBoundingBoxes() = call.GetBoundingBoxes();
    cr2d->SetViewResolution(call.GetViewResolution());

    if (this->halveRes.Param<core::param::BoolParam>()->Value()) {
        int a = amortLevel.Param<core::param::IntParam>()->Value();
        int w = fbo->getWidth();
        int h = fbo->getHeight();
        int ssLevel = this->superSamplingLevelSlot.Param<core::param::IntParam>()->Value();
        int approach = this->approachEnumSlot.Param<core::param::EnumParam>()->Value();

        // check if amortization mode changed
        if (approach != oldApp || w != oldW || h != oldH || ssLevel != oldssLevel || a != oldaLevel) {
            resizeArrays(approach, w, h, ssLevel);
        }
        setupAccel(approach, w, h, ssLevel, &cam);
        if (approach == 0) {
            cr2d->SetFramebuffer(glowlFBOms);
        } else {
            cr2d->SetFramebuffer(glowlFBO);
        }
        cr2d->SetCamera(cam);
        cr2d->SetBackgroundColor(call.BackgroundColor());

        // send call to next renderer in line
        (*cr2d)(core::view::AbstractCallRender::FnRender);

        doReconstruction(approach, w, h, ssLevel);

        // to avoid excessive resizing, retain last render variables and check if changed
        oldApp = approach;
        oldssLevel = ssLevel;
        oldH = h;
        oldW = w;
        oldaLevel = a;
    } else {
        cr2d->SetFramebuffer(fbo);
        cr2d->SetCamera(cam);

        // send call to next renderer in line
        (*cr2d)(core::view::AbstractCallRender::FnRender);
    }
    return true;
}

bool megamol::infovis::InfovisAmortizedRenderer::GetExtents(core::view::CallRender2DGL& call) {
    core::view::CallRender2DGL* cr2d = this->nextRendererSlot.CallAs<core::view::CallRender2DGL>();
    if (cr2d == nullptr) {
        return false;
    }

    if (!(*cr2d)(core::view::CallRender2DGL::FnGetExtents)) {
        return false;
    }

    cr2d->SetTimeFramesCount(call.TimeFramesCount());
    cr2d->SetIsInSituTime(call.IsInSituTime());

    call.AccessBoundingBoxes() = cr2d->GetBoundingBoxes();

    return true;
}
