/*
 * ReplacementRenderer.cpp
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "ReplacementRenderer.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::cinematic_gl;

using namespace vislib;


ReplacementRenderer::ReplacementRenderer(void)
        : megamol::core::view::RendererModule<mmstd_gl::CallRender3DGL, mmstd_gl::ModuleGL>()
        , alphaParam("alpha", "The alpha value of the replacement rendering.")
        , replacementRenderingParam("replacement", "Show/hide replacement rendering for chained renderer.")
        , toggleReplacementParam("toggleReplacement", "Toggle replacement rendering.")
        , draw_replacement(false)
        , utils()
        , bbox() {

    // Make render slots available
    this->MakeSlotAvailable(&this->chainRenderSlot);
    this->MakeSlotAvailable(&this->renderSlot);

    // Init parameters
    alphaParam.SetParameter(new param::FloatParam(0.75f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&alphaParam);

    this->toggleReplacementParam.SetParameter(new param::ButtonParam(
        frontend_resources::KeyCode{frontend_resources::Key::KEY_1, frontend_resources::Modifier::ALT}));
    this->MakeSlotAvailable(&this->toggleReplacementParam);

    this->replacementRenderingParam.SetParameter(new param::BoolParam(this->draw_replacement));
    this->MakeSlotAvailable(&this->replacementRenderingParam);
}


ReplacementRenderer::~ReplacementRenderer(void) {

    this->Release();
}


void ReplacementRenderer::release(void) {}


bool ReplacementRenderer::create(void) {

    // Initialise render utils
    if (!this->utils.Initialise(this->GetCoreInstance())) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[REPLACEMENT RENDERER] [create] Couldn't initialize the font. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
        return false;
    }

    return true;
}


bool ReplacementRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {

    auto cr3d_out = this->chainRenderSlot.CallAs<mmstd_gl::CallRender3DGL>();
    if (cr3d_out != nullptr) {
        *cr3d_out = call;
        if ((*cr3d_out)(view::AbstractCallRender::FnGetExtents)) {
            call = *cr3d_out;
            this->bbox = call.AccessBoundingBoxes().BoundingBox();
            return true;
        }
    } else {
        call.SetTimeFramesCount(1);
        call.SetTime(0.0f);
        this->bbox = vislib::math::Cuboid<float>(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        call.AccessBoundingBoxes().SetBoundingBox(this->bbox);
        return true;
    }
    return false;
}


bool ReplacementRenderer::Render(mmstd_gl::CallRender3DGL& call) {

    if (this->replacementRenderingParam.IsDirty()) {
        this->replacementRenderingParam.ResetDirty();
        this->draw_replacement = this->replacementRenderingParam.Param<param::BoolParam>()->Value();
    }

    if (this->toggleReplacementParam.IsDirty()) {
        this->toggleReplacementParam.ResetDirty();
        this->draw_replacement = !this->draw_replacement;
        this->replacementRenderingParam.Param<param::BoolParam>()->SetValue(this->draw_replacement, false);
    }

    if (this->draw_replacement) {
        // Render bounding box as replacement

        auto const lhsFBO = call.GetFramebuffer();
        lhsFBO->bind();

        glViewport(0, 0, lhsFBO->getWidth(), lhsFBO->getHeight());
        glm::vec2 vp_dim = {lhsFBO->getWidth(), lhsFBO->getHeight()};

        // Camera
        core::view::Camera camera = call.GetCamera();
        glm::mat4 proj = camera.getProjectionMatrix();
        glm::mat4 view = camera.getViewMatrix();
        glm::mat4 mvp = proj * view;

        float alpha = alphaParam.Param<param::FloatParam>()->Value();

        glm::vec4 front = {0.0f, 0.0f, 1.0f, alpha};
        glm::vec4 back = {0.0f, 1.0f, 1.0f, alpha};
        glm::vec4 right = {1.0f, 0.0f, 0.0f, alpha};
        glm::vec4 left = {1.0f, 0.0f, 1.0f, alpha};
        glm::vec4 top = {0.0f, 1.0f, 0.0f, alpha};
        glm::vec4 bottom = {1.0f, 1.0f, 0.0f, alpha};

        glm::vec3 left_top_back = {this->bbox.Left(), this->bbox.Top(), this->bbox.Back()};
        glm::vec3 left_bottom_back = {this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back()};
        glm::vec3 right_top_back = {this->bbox.Right(), this->bbox.Top(), this->bbox.Back()};
        glm::vec3 right_bottom_back = {this->bbox.Right(), this->bbox.Bottom(), this->bbox.Back()};
        glm::vec3 left_top_front = {this->bbox.Left(), this->bbox.Top(), this->bbox.Front()};
        glm::vec3 left_bottom_front = {this->bbox.Left(), this->bbox.Bottom(), this->bbox.Front()};
        glm::vec3 right_top_front = {this->bbox.Right(), this->bbox.Top(), this->bbox.Front()};
        glm::vec3 right_bottom_front = {this->bbox.Right(), this->bbox.Bottom(), this->bbox.Front()};

        this->utils.PushQuadPrimitive(
            left_bottom_front, right_bottom_front, right_top_front, left_top_front, front); // Front
        this->utils.PushQuadPrimitive(left_bottom_back, left_top_back, right_top_back, right_bottom_back, back); // Back
        this->utils.PushQuadPrimitive(left_top_back, left_top_front, right_top_front, right_top_back, top);      // Top
        this->utils.PushQuadPrimitive(
            left_bottom_back, right_bottom_back, right_bottom_front, left_bottom_front, bottom); // Bottom
        this->utils.PushQuadPrimitive(left_bottom_back, left_bottom_front, left_top_front, left_top_back, left); // Left
        this->utils.PushQuadPrimitive(
            right_bottom_back, right_top_back, right_top_front, right_bottom_front, right); // Right

        this->utils.DrawQuadPrimitives(mvp, vp_dim);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    } else {

        auto cr3d_out = this->chainRenderSlot.CallAs<mmstd_gl::CallRender3DGL>();
        if (cr3d_out != nullptr) {
            *cr3d_out = call;
            return (*cr3d_out)(core::view::AbstractCallRender::FnRender);
        }
    }

    return true;
}
