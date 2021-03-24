/*
 * Renderer3DModuleGL.cpp
 *
 * Copyright (C) 2018, 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/Renderer3DModuleGL.h"
#include "mmcore/view/AbstractView.h"
#include "mmcore/view/CallRender3DGL.h"

using namespace megamol::core::view;

/*
 * Renderer3DModuleGL::Renderer3DModuleGL
 */
Renderer3DModuleGL::Renderer3DModuleGL(void)
    : RendererModule<CallRender3DGL>() {
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
        (*chainedCall)(view::AbstractCallRender::FnGetExtents);
    }

    // TODO extents magic


    // get our own extents
    this->GetExtents(call);

    if (chainedCall != nullptr) {
        auto mybb = call.AccessBoundingBoxes().BoundingBox();
        mybb.Union(chainedCall->AccessBoundingBoxes().BoundingBox());
        auto mycb = call.AccessBoundingBoxes().ClipBox();
        mycb.Union(chainedCall->AccessBoundingBoxes().ClipBox());
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

    // Camera
    view::Camera_2 cam;
    call.GetCamera(cam);
    cam_type::snapshot_type snapshot;
    cam_type::matrix_type viewTemp, projTemp;
    cam.calc_matrices(snapshot, viewTemp, projTemp, thecam::snapshot_content::all);

    this->PreRender(call);

    CallRender3DGL* chainedCall = this->chainRenderSlot.CallAs<CallRender3DGL>();

    if (chainedCall != nullptr) {
        // copy the incoming call to the output
        *chainedCall = call;

        // chain through the render call
        (*chainedCall)(view::AbstractCallRender::FnRender);

        call = *chainedCall;
    }

    // render our own stuff
    this->Render(call);

    return true;
}

/*
 * Renderer3DModuleGL::PreRender
 */
void Renderer3DModuleGL::PreRender(CallRender3DGL& call) {
    // intentionally empty
}
