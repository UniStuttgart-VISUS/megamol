/**
 * MegaMol
 * Copyright (c) 2018, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd_gl/renderer/Renderer3DModuleGL.h"

#include "mmstd/view/AbstractView.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"

using namespace megamol::mmstd_gl;

/*
 * Renderer3DModuleGL::Renderer3DModuleGL
 */
Renderer3DModuleGL::Renderer3DModuleGL(void) : RendererModule<CallRender3DGL, ModuleGL>() {
    // Callback should already be set by RendererModule
    this->MakeSlotAvailable(&this->chainRenderSlot);

    // Callback should already be set by RendererModule
    this->MakeSlotAvailable(&this->renderSlot);
}

/*
 * Renderer3DModuleGL::~Renderer3DModuleGL
 */
Renderer3DModuleGL::~Renderer3DModuleGL(void) {
    // intentionally empty
}

/*
 * Renderer3DModuleGL::GetExtentsChain
 */
bool Renderer3DModuleGL::GetExtentsChain(CallRender3DGL& call) {
    CallRender3DGL* chainedCall = this->chainRenderSlot.CallAs<CallRender3DGL>();
    if (chainedCall != nullptr) {
        // copy the incoming call to the output
        *chainedCall = call;

        // set bounding boxes to invalid before calling to the right, in case the chained renderer
        // does not provide them at all. this would result in the possibly outdated bounding box
        // from the left hand side being used
        chainedCall->AccessBoundingBoxes().Clear();

        // chain through the get extents call
        (*chainedCall)(core::view::AbstractCallRender::FnGetExtents);
    }

    // TODO extents magic


    // get our own extents
    // set bounding boxes to invalid before getting own extent, in case the function
    // does not provide them at all. this would result in the possibly outdated bounding box
    // from the left hand side being used
    call.AccessBoundingBoxes().Clear();
    this->GetExtents(call);

    if (chainedCall != nullptr) {
        const auto& mybb = call.AccessBoundingBoxes().BoundingBox();
        const auto& otherbb = chainedCall->AccessBoundingBoxes().BoundingBox();
        const auto& mycb = call.AccessBoundingBoxes().ClipBox();
        const auto& othercb = chainedCall->AccessBoundingBoxes().ClipBox();

        auto newbb = mybb;
        auto newcb = mycb;

        if (call.GetBoundingBoxes().IsBoundingBoxValid() && chainedCall->GetBoundingBoxes().IsBoundingBoxValid()) {
            newbb.Union(otherbb);
        } else if (chainedCall->GetBoundingBoxes().IsBoundingBoxValid()) {
            newbb = otherbb; // just override for the call
        }                    // we ignore the other two cases as they both lead to usage of the already set mybb

        if (call.GetBoundingBoxes().IsClipBoxValid() && chainedCall->GetBoundingBoxes().IsClipBoxValid()) {
            newcb.Union(othercb);
        } else if (chainedCall->GetBoundingBoxes().IsClipBoxValid()) {
            newcb = othercb; // just override for the call
        }                    // we ignore the other two cases as they both lead to usage of the already set mycb

        call.AccessBoundingBoxes().SetBoundingBox(newbb);
        call.AccessBoundingBoxes().SetClipBox(newcb);

        // TODO machs richtig
        call.SetTimeFramesCount(chainedCall->TimeFramesCount());
    }

    return true;
}

/*
 * Renderer3DModuleGL::RenderChain
 */
bool Renderer3DModuleGL::RenderChain(CallRender3DGL& call) {

    this->PreRender(call);

    CallRender3DGL* chainedCall = this->chainRenderSlot.CallAs<CallRender3DGL>();

    if (chainedCall != nullptr) {
        // copy the incoming call to the output
        *chainedCall = call;

        // chain through the render call
        (*chainedCall)(core::view::AbstractCallRender::FnRender);


        call = *chainedCall;
    }

    // bind fbo and set viewport before rendering your own stuff
    auto fbo = call.GetFramebuffer();
    fbo->bind();
    glViewport(0, 0, fbo->getWidth(), fbo->getHeight());

    // render our own stuff
    this->Render(call);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return true;
}

/*
 * Renderer3DModuleGL::PreRender
 */
void Renderer3DModuleGL::PreRender(CallRender3DGL& call) {
    // intentionally empty
}
