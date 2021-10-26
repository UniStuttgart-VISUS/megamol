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
using megamol::core::utility::log::Log;

InfovisAmortizedRenderer::InfovisAmortizedRenderer()
        : Renderer2D()
        , nextRendererSlot("nextRenderer", "connects to following Renderers, that will render in reduced resolution.")
        , enabledParam("Enabled", "Turn on switch")
        , approachParam("AmortizationMode", "Which amortization approach to use.")
        , amortLevelParam("AmortLevel", "Level of Amortization") {
    this->nextRendererSlot.SetCompatibleCall<megamol::core::view::CallRender2DGLDescription>();
    this->MakeSlotAvailable(&this->nextRendererSlot);

    this->enabledParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&enabledParam);

    auto* approachMappings = new core::param::EnumParam(6);
    approachMappings->SetTypePair(MS_AR, "MS-AR");
    approachMappings->SetTypePair(QUAD_AR, "4xAR");
    approachMappings->SetTypePair(QUAD_AR_C, "4xAR-C");
    approachMappings->SetTypePair(SS_AR, "SS-AR");
    approachMappings->SetTypePair(PARAMETER_AR, "Parameterized-AR");
    approachMappings->SetTypePair(DEBUG_PLACEHOLDER, "debug");
    approachMappings->SetTypePair(PUSH_AR, "Push-AR");
    this->approachParam << approachMappings;
    this->MakeSlotAvailable(&approachParam);

    this->amortLevelParam << new core::param::IntParam(1, 1);
    this->MakeSlotAvailable(&amortLevelParam);
}

InfovisAmortizedRenderer::~InfovisAmortizedRenderer() {
    this->Release();
}

bool megamol::infovis::InfovisAmortizedRenderer::create() {
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        Log::DefaultLog.WriteWarn("Ignore glError() from previous modules: %i", error);
    }

    if (!createShaders()) {
        return false;
    }
    if (!createBuffers()) {
        return false;
    }
    return true;
}

// TODO
void InfovisAmortizedRenderer::release() {}

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

    fbo = call.GetFramebuffer();
    cr2d->SetBackgroundColor(call.BackgroundColor());
    cr2d->AccessBoundingBoxes() = call.GetBoundingBoxes();
    cr2d->SetViewResolution(call.GetViewResolution());

    if (this->enabledParam.Param<core::param::BoolParam>()->Value()) {
        int a = amortLevelParam.Param<core::param::IntParam>()->Value();
        int w = fbo->getWidth();
        int h = fbo->getHeight();
        int approach = this->approachParam.Param<core::param::EnumParam>()->Value();

        // check if amortization mode changed
        if (approach != oldApp || w != oldW || h != oldH || a != oldaLevel) {
            resizeArrays(approach, w, h);
        }
        setupAccel(approach, w, h, &cam);
        cr2d->SetFramebuffer(glowlFBO);
        cr2d->SetCamera(cam);
        cr2d->SetBackgroundColor(call.BackgroundColor());

        // send call to next renderer in line
        (*cr2d)(core::view::AbstractCallRender::FnRender);

        doReconstruction(approach, w, h);

        // to avoid excessive resizing, retain last render variables and check if changed
        oldApp = approach;
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

bool InfovisAmortizedRenderer::createShaders() {
    auto const shader_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
    try {
        amort_reconstruction_shdr_array[6] = core::utility::make_glowl_shader("amort_reconstruction6", shader_options,
            "infovis/amort_reconstruction.vert.glsl", "infovis/amort_reconstruction6.frag.glsl");
    } catch (std::exception& e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, ("InfovisAmortizedRenderer: " + std::string(e.what())).c_str());
        return false;
    }
    return true;
}

bool InfovisAmortizedRenderer::createBuffers() {

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
        Log::DefaultLog.WriteError("GL_ERROR in InfovisAmortizedRenderer: %i", err);
    }

    return true;
}

void InfovisAmortizedRenderer::resizeArrays(int approach, int w, int h) {
    if (approach == 6) {
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

void InfovisAmortizedRenderer::setupAccel(int approach, int ow, int oh, core::view::Camera* cam) {
    int a = this->amortLevelParam.Param<core::param::IntParam>()->Value();
    int w = ceil(float(ow) / a);
    int h = ceil(float(oh) / a);
    glm::mat4 pm;
    glm::mat4 mvm;
    auto pmvm = projMatrix * mvMatrix;

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

void InfovisAmortizedRenderer::doReconstruction(int approach, int w, int h) {
    glViewport(0, 0, w, h);

    amort_reconstruction_shdr_array[approach]->use();

    fbo->bind();
    int a = this->amortLevelParam.Param<core::param::IntParam>()->Value();

    amort_reconstruction_shdr_array[approach]->setUniform("h", h);
    amort_reconstruction_shdr_array[approach]->setUniform("w", w);
    amort_reconstruction_shdr_array[approach]->setUniform("amortLevel", a);

    glUniformMatrix4fv(amort_reconstruction_shdr_array[approach]->getUniformLocation("moveMatrices"), 4, GL_FALSE,
        &moveMatrices[0][0][0]);

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
        frametype = (frametype + (this->amortLevelParam.Param<core::param::IntParam>()->Value() - 1) *
                                     (this->amortLevelParam.Param<core::param::IntParam>()->Value() - 1)) %
                    framesNeeded;
    } else {
        frametype = (frametype + 1) % framesNeeded;
    }
    parity = (parity + 1) % 2;
}
