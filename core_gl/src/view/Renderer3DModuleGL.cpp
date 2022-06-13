/*
 * Renderer3DModuleGL.cpp
 *
 * Copyright (C) 2018, 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore_gl/view/Renderer3DModuleGL.h"
#include "mmcore/view/AbstractView.h"
#include "mmcore_gl/view/CallRender3DGL.h"

using namespace megamol::core_gl::view;

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

        // chain through the get extents call
        (*chainedCall)(core::view::AbstractCallRender::FnGetExtents);
    }

    // TODO extents magic


    // get our own extents
    this->GetExtents(call);

    if (chainedCall != nullptr) {
        auto mybb = call.AccessBoundingBoxes().BoundingBox();
        auto otherbb = chainedCall->AccessBoundingBoxes().BoundingBox();
        auto mycb = call.AccessBoundingBoxes().ClipBox();
        auto othercb = chainedCall->AccessBoundingBoxes().ClipBox();

        if (call.AccessBoundingBoxes().IsBoundingBoxValid() &&
            chainedCall->AccessBoundingBoxes().IsBoundingBoxValid()) {
            mybb.Union(otherbb);
        } else if (chainedCall->AccessBoundingBoxes().IsBoundingBoxValid()) {
            mybb = otherbb; // just override for the call
        }                   // we ignore the other two cases as they both lead to usage of the already set mybb

        if (call.AccessBoundingBoxes().IsClipBoxValid() && chainedCall->AccessBoundingBoxes().IsClipBoxValid()) {
            mycb.Union(othercb);
        } else if (chainedCall->AccessBoundingBoxes().IsClipBoxValid()) {
            mycb = othercb; // just override for the call
        }                   // we ignore the other two cases as they both lead to usage of the already set mycb

        call.AccessBoundingBoxes().SetBoundingBox(mybb);
        call.AccessBoundingBoxes().SetClipBox(mycb);

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
