#include "stdafx.h"
#include "mmcore/view/AmortizedRenderer.h"

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/ShaderSource.h"

#include "mmcore/param/BoolParam.h"

#include <glm/gtc/type_ptr.hpp>
#include "glm/gtc/functions.hpp"
#include "glm/gtc/matrix_transform.hpp"

using namespace megamol::core::view;

AmortizedRenderer::AmortizedRenderer(void)
        : view::AbstractRenderingView()
        , nextRendererSlot("nextRenderer", "connects to following Renderers, that will render in reduced resolution.")
        , inputCall("previousViewCall", "call from previous view")
        , halveRes("halve Resolution", "halve Resolution for FPS test")
        , approachSlot("Approach", "Numerical Value assigned to each temporal reconstruction approach")
        , superSamplingLevelSlot("SupersamplingLevel", "Level of Supersampling Quality") {
    this->nextRendererSlot.SetCompatibleCall<CallRender2DDescription>();
    this->MakeSlotAvailable(&this->nextRendererSlot);

   
    this->MakeSlotAvailable(&this->inputCall);
}

AmortizedRenderer::~AmortizedRenderer(void) {
    this->release();
}

// TODO
void AmortizedRenderer::release(void) {}

bool AmortizedRenderer::create(void) {
    return true;
}

// TODO
bool AmortizedRenderer::GetExtents(core::view::CallRender2D& call) {
    return true;
}

unsigned int megamol::core::view::AmortizedRenderer::GetCameraSyncNumber(void) const {
    return 0;
}

void AmortizedRenderer::setupBuffers() {
    glGenFramebuffers(1, &amortizedFboA);
    glGenFramebuffers(1, &amortizedFboB);
    glGenFramebuffers(1, &amortizedMsaaFboA);
    glGenFramebuffers(1, &amortizedMsaaFboB);
    glGenTextures(1, &msImageStorageA);
    glGenTextures(1, &imageArrayA);
    glGenTextures(1, &imageArrayB);
    glGenTextures(1, &msImageArray);
    glGenBuffers(1, &ssboMatrices);

    glBindFramebuffer(GL_FRAMEBUFFER, amortizedMsaaFboA);
    // glBindTexture(GL_TEXTURE_2D, imStoreI);
    glEnable(GL_MULTISAMPLE);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, msImageStorageA);
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_FLOAT, 0);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 1, GL_RGB, 1, 1, GL_TRUE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    // glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, imStoreI, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, msImageStorageA, 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);

    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, msImageArray);
    glTexImage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 2, GL_RGB, 1, 1, 2, GL_TRUE);
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

    glBindTexture(GL_TEXTURE_2D_ARRAY, imageArrayB);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB, 1, 1, 1, 0, GL_RGB, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
}

std::vector<glm::fvec2> AmortizedRenderer::calculateHammersley(int until) {
    // calculation of Positions according to hammersley sequence
    // https://www.researchgate.net/publication/244441430_Sampling_with_Hammersley_and_Halton_Points
    std::vector<glm::fvec2> outputArray(until);
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
        outputArray[k] = glm::vec2(u - floor(2 * u), v - floor(2 * v));
    }
    return outputArray;
}

void AmortizedRenderer::setupAccel(int approach, int ow, int oh, int ssLevel) {
    int w = ow / 2;
    int h = oh / 2;

    glm::mat4 pm;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            pm[j][i] = projMatrix_column[i + j * 4];
        }
    }
    glm::mat4 mvm;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            mvm[j][i] = modelViewMatrix_column[i + j * 4];
        }
    }
    auto pmvm = pm * mvm;
    int framesNeeded;
    if (approach == 0 && this->halveRes.Param<core::param::BoolParam>()->Value()) {
        framesNeeded = 2;
        if (invMatrices.size() != framesNeeded) {
            invMatrices.resize(framesNeeded);
            moveMatrices.resize(framesNeeded);
        }

        glClearColor(0, 0, 0, 0);
        // glClearColor(backgroundColor[0], backgroundColor[1], backgroundColor[2], backgroundColor[3]);
        glm::mat4 jit;

        invMatrices[frametype] = pmvm;

        glm::mat4 inversePMVM = glm::inverse(pmvm);
        for (int i = 0; i < framesNeeded; i++)
            moveMatrices[i] = invMatrices[i] * inversePMVM;

        glBindFramebuffer(GL_FRAMEBUFFER, amortizedMsaaFboA);
        glActiveTexture(GL_TEXTURE11);
        glEnable(GL_MULTISAMPLE);

        if (frametype == 0) {
            glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, msImageArray);
            glTexImage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 2, GL_RGBA8, w, h, 2, GL_TRUE);
            glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, msImageArray, 0, 0);

            glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_PROGRAMMABLE_SAMPLE_LOCATIONS_ARB, 1);
            const float tbl[4] = {0.25, 0.25, 0.75, 0.75};
            glFramebufferSampleLocationsfvARB(GL_FRAMEBUFFER, 0u, 2, tbl);

            // glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, imStoreI, 0);
            glClearColor(0, 0, 0, 0);
            // glClearColor(backgroundColor[0], backgroundColor[1], backgroundColor[2], backgroundColor[3]);
        }
        if (frametype == 1) {
            glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, msImageArray);
            glTexImage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 2, GL_RGBA8, w, h, 2, GL_TRUE);
            glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, msImageArray, 0, 1);

            glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_PROGRAMMABLE_SAMPLE_LOCATIONS_ARB, 1);
            const float tblb[4] = {0.75, 0.25, 0.25, 0.75};
            glFramebufferSampleLocationsfvARB(GL_FRAMEBUFFER, 0u, 2, tblb);

            glClearColor(0, 0, 0, 0);
            // glClearColor(backgroundColor[0], backgroundColor[1], backgroundColor[2], backgroundColor[3]);
        }
        if (glGetError() == GL_INVALID_ENUM)
            megamol::core::utility::log::Log::DefaultLog.WriteError("%i", glGetError());

        glClear(GL_COLOR_BUFFER_BIT);

        glViewport(0, 0, w, h);
    }

    // non cbr quarter res 4 frame restoration
    if (approach == 1 && this->halveRes.Param<core::param::BoolParam>()->Value()) {
        framesNeeded = 4;
        if (invMatrices.size() != framesNeeded) {
            invMatrices.resize(framesNeeded);
            moveMatrices.resize(framesNeeded);
        }

        // glClearColor(backgroundColor[0], backgroundColor[1], backgroundColor[2], backgroundColor[3]);
        glClearColor(0, 0, 0, 0);

        glm::mat4 jit;
        glm::mat4 pmvm = pm * mvm;
        if (frametype == 0) {
            jit = glm::translate(glm::mat4(1.0f), glm::vec3(-1.0 / ow, 1.0 / oh, 0));
            invMatrices[frametype] = jit * pmvm;
        }
        if (frametype == 1) {
            jit = glm::translate(glm::mat4(1.0f), glm::vec3(1.0 / ow, 1.0 / oh, 0));
            invMatrices[frametype] = jit * pmvm;
        }
        if (frametype == 2) {
            jit = glm::translate(glm::mat4(1.0f), glm::vec3(-1.0 / ow, -1.0 / oh, 0));
            invMatrices[frametype] = jit * pmvm;
        }
        if (frametype == 3) {
            jit = glm::translate(glm::mat4(1.0f), glm::vec3(1.0 / ow, -1.0 / oh, 0));
            invMatrices[frametype] = jit * pmvm;
        }

        for (int i = 0; i < framesNeeded; i++)
            moveMatrices[i] = invMatrices[i] * glm::inverse(pmvm);
        // moveMatrices[i] = glm::mat4(1.0);


        pm = jit * pm;
        for (int i = 0; i < 16; i++)
            projMatrix_column[i] = glm::value_ptr(pm)[i];

        glBindFramebuffer(GL_FRAMEBUFFER, amortizedFboA);
        glActiveTexture(GL_TEXTURE10);
        glBindTexture(GL_TEXTURE_2D_ARRAY, imageArrayA);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB, w, h, 4, 0, GL_RGB, GL_FLOAT, 0);

        if (frametype == 0) {
            glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, imageArrayA, 0, 0);
        }
        if (frametype == 1) {
            glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, imageArrayA, 0, 1);
        }
        if (frametype == 2) {
            glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, imageArrayA, 0, 2);
        }
        if (frametype == 3) {
            glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, imageArrayA, 0, 3);
        }
        glClearColor(0, 0, 0, 0);
        // glClearColor(backgroundColor[0], backgroundColor[1], backgroundColor[2], backgroundColor[3]);


        glClear(GL_COLOR_BUFFER_BIT);

        glViewport(0, 0, w, h);
    }

    if (approach == 2 && this->halveRes.Param<core::param::BoolParam>()->Value()) {
        framesNeeded = 4;
        if (invMatrices.size() != framesNeeded) {
            invMatrices.resize(framesNeeded);
            moveMatrices.resize(framesNeeded);
        }
        glClearColor(0, 0, 0, 0);
        // glClearColor(backgroundColor[0], backgroundColor[1], backgroundColor[2], backgroundColor[3]);

        glm::mat4 jit;
        glm::mat4 pmvm = pm * mvm;
        if (frametype == 0) {
            jit = glm::translate(glm::mat4(1.0f), glm::vec3(-2.0 / ow, 2.0 / oh, 0));
        }
        if (frametype == 1) {
            jit = glm::translate(glm::mat4(1.0f), glm::vec3(0 / ow, 2.0 / oh, 0));
        }
        if (frametype == 2) {
            jit = glm::translate(glm::mat4(1.0f), glm::vec3(-2.0 / ow, 0 / oh, 0));
        }
        if (frametype == 3) {
            jit = glm::translate(glm::mat4(1.0f), glm::vec3(0.0 / ow, 0 / oh, 0));
        }
        invMatrices[frametype] = jit * pmvm;
        for (int i = 0; i < framesNeeded; i++)
            moveMatrices[i] = invMatrices[i] * glm::inverse(pmvm);
        // moveMatrices[i] = glm::mat4(1.0);


        pm = jit * pm;
        for (int i = 0; i < 16; i++)
            projMatrix_column[i] = glm::value_ptr(pm)[i];

        glBindFramebuffer(GL_FRAMEBUFFER, amortizedFboB);
        glActiveTexture(GL_TEXTURE11);
        glBindTexture(GL_TEXTURE_2D_ARRAY, imageArrayB);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB, w, h, 4, 0, GL_RGB, GL_FLOAT, 0);

        if (frametype == 0) {
            glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, imageArrayB, 0, 0);
        }
        if (frametype == 1) {
            glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, imageArrayB, 0, 1);
        }
        if (frametype == 2) {
            glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, imageArrayB, 0, 2);
        }
        if (frametype == 3) {
            glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, imageArrayB, 0, 3);
        }
        glClearColor(0, 0, 0, 0);
        // glClearColor(backgroundColor[0], backgroundColor[1], backgroundColor[2], backgroundColor[3]);
        glClear(GL_COLOR_BUFFER_BIT);

        glViewport(0, 0, w, h);
    }

    if (approach == 3 && this->halveRes.Param<core::param::BoolParam>()->Value()) {
        framesNeeded = 4 * ssLevel;
        if (invMatrices.size() != framesNeeded || hammerPositions.size() != ssLevel) {
            invMatrices.resize(framesNeeded);
            moveMatrices.resize(framesNeeded);
            hammerPositions.resize(ssLevel);
            hammerPositions = calculateHammersley(ssLevel);
        }
        glClearColor(0, 0, 0, 0);
        // glClearColor(backgroundColor[0], backgroundColor[1], backgroundColor[2], backgroundColor[3]);

        glBindFramebuffer(GL_FRAMEBUFFER, amortizedFboA);
        glActiveTexture(GL_TEXTURE10);
        glBindTexture(GL_TEXTURE_2D_ARRAY, imageArrayA);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB, w, h, 4 * ssLevel, 0, GL_RGB, GL_FLOAT, 0);

        glm::mat4 jit;
        glm::mat4 pmvm = pm * mvm;
        int f = floor(frametype / 4);
        if (frametype % 4 == 0) {
            jit = glm::translate(
                glm::mat4(1.0f), glm::vec3((hammerPositions[f].x - 1) / ow, (hammerPositions[f].y + 1) / oh, 0));
            invMatrices[frametype] = glm::translate(glm::mat4(1.0f), glm::vec3(-1.0 / ow, 1.0 / oh, 0)) * pmvm;
        }
        if (frametype % 4 == 1) {
            jit = glm::translate(
                glm::mat4(1.0f), glm::vec3((hammerPositions[f].x + 1) / ow, (hammerPositions[f].y + 1) / oh, 0));
            invMatrices[frametype] = glm::translate(glm::mat4(1.0f), glm::vec3(1.0 / ow, 1.0 / oh, 0)) * pmvm;
        }
        if (frametype % 4 == 2) {
            jit = glm::translate(
                glm::mat4(1.0f), glm::vec3((hammerPositions[f].x - 1) / ow, (hammerPositions[f].y - 1) / oh, 0));
            invMatrices[frametype] = glm::translate(glm::mat4(1.0f), glm::vec3(-1.0 / ow, -1.0 / oh, 0)) * pmvm;
        }
        if (frametype % 4 == 3) {
            jit = glm::translate(
                glm::mat4(1.0f), glm::vec3((hammerPositions[f].x + 1) / ow, (hammerPositions[f].y - 1) / oh, 0));
            invMatrices[frametype] = glm::translate(glm::mat4(1.0f), glm::vec3(1.0 / ow, -1.0 / oh, 0)) * pmvm;
        }
        glm::mat4 invM = glm::inverse(pmvm);
        for (int i = 0; i < framesNeeded; i++)
            moveMatrices[i] = invMatrices[i] * invM;
        // moveMatrices[i] = glm::mat4(1.0);

        pm = jit * pm;
        for (int i = 0; i < 16; i++)
            projMatrix_column[i] = glm::value_ptr(pm)[i];

        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, imageArrayA, 0, frametype);
        glClear(GL_COLOR_BUFFER_BIT);

        glViewport(0, 0, w, h);
    }
}

void AmortizedRenderer::doReconstruction(int approach, int w, int h, int ssLevel) {

    if (approach == 0 && this->halveRes.Param<core::param::BoolParam>()->Value()) {
        glViewport(0, 0, w, h);

        pc_reconstruction0_shdr->Enable();

        glUniform1i(pc_reconstruction0_shdr->ParameterLocation("src_tex2D"), 11);

        glBindFramebuffer(GL_FRAMEBUFFER, origFBO);
        glUniform1i(pc_reconstruction0_shdr->ParameterLocation("h"), h);
        glUniform1i(pc_reconstruction0_shdr->ParameterLocation("w"), w);
        glUniform1i(pc_reconstruction0_shdr->ParameterLocation("approach"), approach);
        glUniform1i(pc_reconstruction0_shdr->ParameterLocation("frametype"), frametype);

        glUniformMatrix4fv(
            pc_reconstruction0_shdr->ParameterLocation("moveMatrices"), 2, GL_FALSE, &moveMatrices[0][0][0]);

        glDrawArrays(GL_TRIANGLES, 0, 6);
        pc_reconstruction0_shdr->Disable();
        frametype = (frametype + 1) % framesNeeded;
    }

    if (approach == 1 && this->halveRes.Param<core::param::BoolParam>()->Value()) {
        glViewport(0, 0, w, h);

        pc_reconstruction1_shdr->Enable();

        glActiveTexture(GL_TEXTURE10);
        glBindTexture(GL_TEXTURE_2D_ARRAY, imageArrayA);
        glUniform1i(pc_reconstruction1_shdr->ParameterLocation("tx2D_array"), 10);

        glBindFramebuffer(GL_FRAMEBUFFER, origFBO);
        glUniform1i(pc_reconstruction1_shdr->ParameterLocation("h"), h);
        glUniform1i(pc_reconstruction1_shdr->ParameterLocation("w"), w);
        glUniform1i(pc_reconstruction1_shdr->ParameterLocation("approach"), approach);
        glUniform1i(pc_reconstruction1_shdr->ParameterLocation("frametype"), frametype);

        glUniformMatrix4fv(
            pc_reconstruction1_shdr->ParameterLocation("mMatrices"), 4, GL_FALSE, &moveMatrices[0][0][0]);

        glDrawArrays(GL_TRIANGLES, 0, 6);
        pc_reconstruction1_shdr->Disable();
        frametype = (frametype + 1) % framesNeeded;
    }

    if (approach == 2 && this->halveRes.Param<core::param::BoolParam>()->Value()) {
        glViewport(0, 0, w, h);

        pc_reconstruction2_shdr->Enable();

        glActiveTexture(GL_TEXTURE11);
        glBindTexture(GL_TEXTURE_2D_ARRAY, imageArrayB);
        glUniform1i(pc_reconstruction2_shdr->ParameterLocation("src_tx2Da"), 11);

        glBindFramebuffer(GL_FRAMEBUFFER, origFBO);
        glUniform1i(pc_reconstruction2_shdr->ParameterLocation("h"), h);
        glUniform1i(pc_reconstruction2_shdr->ParameterLocation("w"), w);
        glUniform1i(pc_reconstruction2_shdr->ParameterLocation("approach"), approach);
        glUniform1i(pc_reconstruction2_shdr->ParameterLocation("frametype"), frametype);

        glUniformMatrix4fv(
            pc_reconstruction2_shdr->ParameterLocation("mMatrices"), framesNeeded, GL_FALSE, &moveMatrices[0][0][0]);

        glDrawArrays(GL_TRIANGLES, 0, 6);
        pc_reconstruction2_shdr->Disable();
        frametype = (frametype + 1) % framesNeeded;
    }

    if (approach == 3 && this->halveRes.Param<core::param::BoolParam>()->Value()) {
        glViewport(0, 0, w, h);

        pc_reconstruction3_shdr->Enable();

        glActiveTexture(GL_TEXTURE10);
        glBindTexture(GL_TEXTURE_2D_ARRAY, imageArrayA);
        glUniform1i(pc_reconstruction3_shdr->ParameterLocation("tx2D_array"), 10);

        glBindFramebuffer(GL_FRAMEBUFFER, origFBO);

        glUniform1i(pc_reconstruction3_shdr->ParameterLocation("h"), h);
        glUniform1i(pc_reconstruction3_shdr->ParameterLocation("w"), w);
        glUniform1i(pc_reconstruction3_shdr->ParameterLocation("approach"), approach);
        glUniform1i(pc_reconstruction3_shdr->ParameterLocation("frametype"), frametype);
        glUniform1i(pc_reconstruction3_shdr->ParameterLocation("ssLevel"), ssLevel);

        // glUniformMatrix4fv(
        //    pc_reconstruction3_shdr->ParameterLocation("mMatrices"), framesNeeded, GL_FALSE, &moveMatrices[0][0][0]);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMatrices);
        glBufferData(
            GL_SHADER_STORAGE_BUFFER, framesNeeded * sizeof(moveMatrices[0]), &moveMatrices[0][0][0], GL_STATIC_READ);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, ssboMatrices);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        glDrawArrays(GL_TRIANGLES, 0, 6);
        pc_reconstruction3_shdr->Disable();

        frametype = (frametype + 1) % framesNeeded;
    }
}

void AmortizedRenderer::Render(const mmcRenderViewContext& context) {}

bool AmortizedRenderer::Render(core::view::CallRender2D& call) {
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &origFBO);
    // this is the apex of suck and must die
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    // end suck

    setupAccel(0, call.GetViewport().Width(), call.GetViewport().Height(), 1);

    // drawstuff

    doReconstruction(0, call.GetViewport().Width(), call.GetViewport().Height(), 1);

    return true;
}

//desperation ahead

void megamol::core::view::AmortizedRenderer::SerialiseCamera(vislib::Serialiser& serialiser) const {}

void megamol::core::view::AmortizedRenderer::DeserialiseCamera(vislib::Serialiser& serialiser) {}

void megamol::core::view::AmortizedRenderer::ResetView(void) {}

void megamol::core::view::AmortizedRenderer::Resize(unsigned int width, unsigned int height) {}

bool megamol::core::view::AmortizedRenderer::OnRenderView(Call& call) {
    return false;
}

void megamol::core::view::AmortizedRenderer::UpdateFreeze(bool freeze) {}

bool megamol::core::view::AmortizedRenderer::OnKey(Key key, KeyAction action, Modifiers mods) {
    return false;
}

bool megamol::core::view::AmortizedRenderer::OnChar(unsigned int codePoint) {
    return false;
}

bool megamol::core::view::AmortizedRenderer::OnMouseButton(
    MouseButton button, MouseButtonAction action, Modifiers mods) {
    return false;
}

bool megamol::core::view::AmortizedRenderer::OnMouseMove(double x, double y) {
    return false;
}

bool megamol::core::view::AmortizedRenderer::OnMouseScroll(double dx, double dy) {
    return false;
}
