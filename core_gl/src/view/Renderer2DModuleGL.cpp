/*
 * Renderer2DModuleGL.cpp
 *
 * Copyright (C) 2018, 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore_gl/view/Renderer2DModuleGL.h"
#include "mmcore/view/AbstractView.h"
#include "mmcore_gl/view/CallRender2DGL.h"
#include "stdafx.h"

using namespace megamol::core_gl::view;

/*
 * Renderer2DModuleGL::Renderer2DModuleGL
 */
Renderer2DModuleGL::Renderer2DModuleGL(void) : RendererModule<CallRender2DGL, ModuleGL>() {
    // Callback should already be set by RendererModule
    this->MakeSlotAvailable(&this->chainRenderSlot);

    // Callback should already be set by RendererModule
    this->MakeSlotAvailable(&this->renderSlot);
}

/*
 * Renderer2DModuleGL::~Renderer2DModuleGL
 */
Renderer2DModuleGL::~Renderer2DModuleGL() {
    // intentionally empty
}

/*
 * Renderer2DModuleGL::GetExtentsChain
 */
bool Renderer2DModuleGL::GetExtentsChain(CallRender2DGL& call) {
    CallRender2DGL* chainedCall = this->chainRenderSlot.CallAs<CallRender2DGL>();
    if (chainedCall != nullptr) {
        // copy the incoming call to the output
        *chainedCall = call;

        // chain through the get extents call
        (*chainedCall)(core::view::AbstractCallRender::FnGetExtents);
    }

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

        call.SetTimeFramesCount(std::max(call.TimeFramesCount(), chainedCall->TimeFramesCount()));
    }

    return true;
}

/*
 * Renderer2DModuleGL::RenderChain
 */
bool Renderer2DModuleGL::RenderChain(CallRender2DGL& call) {

    this->PreRender(call);

    CallRender2DGL* chainedCall = this->chainRenderSlot.CallAs<CallRender2DGL>();

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
 * Renderer2DModuleGL::PreRender
 */
void Renderer2DModuleGL::PreRender(CallRender2DGL& call) {
    // intentionally empty
}
