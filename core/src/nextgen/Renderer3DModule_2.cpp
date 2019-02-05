/*
 * Renderer3DModule_2.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/nextgen/Renderer3DModule_2.h"
#include "mmcore/nextgen/CallRender3D_2.h"
#include "mmcore/view/InputCall.h"

using namespace megamol::core;
using namespace megamol::core::nextgen;
using namespace megamol::core::view;

/*
 * Renderer3DModule_2::Renderer3DModule_2
 */
Renderer3DModule_2::Renderer3DModule_2(void)
    : RendererModule<CallRender3D_2>()
    , chainRenderSlot("chainRendering", "Connects the renderer to and additional renderer") {

    // TODO key callbacks, etc
    // this->renderSlot.SetCallback(CallRender3D_2::ClassName(),
    //    AbstractCallRender::FunctionName(AbstractCallRender::FnRender), &Renderer3DModule_2::RenderChain);
    // this->renderSlot.SetCallback(CallRender3D_2::ClassName(),
    //    AbstractCallRender::FunctionName(AbstractCallRender::FnGetExtents), &Renderer3DModule_2::GetExtentsChain);
    // this->renderSlot.

    /*this->renderSlot.SetCallback(CallRender3D_2::ClassName(),
        AbstractCallRender::FunctionName(AbstractCallRender::FnRender),
        &RendererModule<CallRender3D_2>::RenderCallback);*/

    this->chainRenderSlot.SetCompatibleCall<CallRender3D_2Description>();
    this->MakeSlotAvailable(&this->chainRenderSlot);

    this->MakeSlotAvailable(&this->renderSlot);
}

/*
 * Renderer3DModule_2::~Renderer3DModule_2
 */
Renderer3DModule_2::~Renderer3DModule_2(void) {
    // intentionally empty
}

/*
 * Renderer3DModule_2::GetExtentsChain
 */
bool Renderer3DModule_2::GetExtentsChain(Call& call) {
    CallRender3D_2* cr3d = dynamic_cast<CallRender3D_2*>(&call);
    if (cr3d == nullptr) return false;

    CallRender3D_2* chainedCall = this->chainRenderSlot.CallAs<CallRender3D_2>();
    if (chainedCall != nullptr) {
        // copy the incoming call to the output
        *chainedCall = *cr3d;

        // chain through the the get extents call
        (*chainedCall)(1);
    }

    // TODO extents magic

    // get our own extents
    this->GetExtents(*cr3d);

    return true;
}

/*
 * Renderer3DModule_2::RenderChain
 */
bool Renderer3DModule_2::RenderChain(Call& call) {
    CallRender3D_2* cr3d = dynamic_cast<CallRender3D_2*>(&call);
    if (cr3d == nullptr) return false;

    CallRender3D_2* chainedCall = this->chainRenderSlot.CallAs<CallRender3D_2>();
    if (chainedCall != nullptr) {
        // copy the incoming call to the output
        *chainedCall = *cr3d;

        // chain through the the render call
        (*chainedCall)(0);
    } else {
        // TODO move this behind the fbo magic?

        auto vp = cr3d->GetViewport();
        glViewport(vp.Left(), vp.Bottom(), vp.Width(), vp.Height());
        auto backCol = cr3d->BackgroundColor();
        glClearColor(backCol.x, backCol.y, backCol.z, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    // TODO FBO magic

    // render our own stuff
    this->Render(*cr3d);

    return true;
}

/*
 * Renderer3DModule_2::MouseEventChain
 */
bool Renderer3DModule_2::MouseEventChain(float x, float y, view::MouseFlags flags) { return false; }
