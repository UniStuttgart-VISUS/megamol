/*
 * Renderer3DModule_2.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/Renderer3DModule_2.h"
#include "mmcore/view/CallRender3D_2.h"

using namespace megamol::core;

/*
 * view::Renderer3DModule_2::Renderer3DModule_2
 */
view::Renderer3DModule_2::Renderer3DModule_2(void)
    : Module()
    , renderSlot("rendering", "Connects the renderer to a view")
    , chainRenderSlot("chainRendering", "Connects the renderer to and additional renderer") {

    this->chainRenderSlot.SetCompatibleCall<CallRender3D_2Description>();
    this->MakeSlotAvailable(&this->chainRenderSlot);

    this->renderSlot.SetCallback(
        CallRender3D_2::ClassName(), CallRender3D_2::FunctionName(0), &Renderer3DModule_2::RenderCallback);
    this->renderSlot.SetCallback(
        CallRender3D_2::ClassName(), CallRender3D_2::FunctionName(1), &Renderer3DModule_2::GetExtentsCallback);
    this->renderSlot.SetCallback(
        CallRender3D_2::ClassName(), CallRender3D_2::FunctionName(2), &Renderer3DModule_2::OnMouseEventCallback);
    this->MakeSlotAvailable(&this->renderSlot);
}

/*
 * view::Renderer3DModule_2::~Renderer3DModule_2
 */
view::Renderer3DModule_2::~Renderer3DModule_2(void) {
    // intentionally empty
}

/*
 * view::Renderer3DModule_2::MouseEvent
 */
bool view::Renderer3DModule_2::MouseEvent(float x, float y, MouseFlags flags) { return false; }

/*
 * view::Renderer3DModule_2::OnMouseEventCallback
 */
bool view::Renderer3DModule_2::OnMouseEventCallback(Call& call) {
    try {
        view::CallRender3D_2& cr3d = dynamic_cast<view::CallRender3D_2&>(call);
        return this->MouseEvent(cr3d.GetMouseX(), cr3d.GetMouseY(), cr3d.GetMouseFlags());
    } catch (...) {
        ASSERT("OnMouseEventCallback call cast failed\n");
    }
    return false;
}

/*
 * view::Renderer3DModule_2::GetExtentsChain
 */
bool view::Renderer3DModule_2::GetExtentsChain(Call& call) {
    view::CallRender3D_2* cr3d = dynamic_cast<view::CallRender3D_2*>(&call);
    if (cr3d == nullptr) return false;

    view::CallRender3D_2* chainedCall = this->chainRenderSlot.CallAs<view::CallRender3D_2>();
    if (chainedCall != nullptr) {
        // copy the incoming call to the output
        *chainedCall = *cr3d;

        // chain through the the get extents call
        (*chainedCall)(1);
    }

    // TODO extents magic

    // get our own extents
    this->GetExtents(call);

    return true;
}

/*
 * view::Renderer3DModule_2::RenderChain
 */
bool view::Renderer3DModule_2::RenderChain(Call& call) { 
    view::CallRender3D_2* cr3d = dynamic_cast<view::CallRender3D_2*>(&call);
    if (cr3d == nullptr) return false;

    view::CallRender3D_2* chainedCall = this->chainRenderSlot.CallAs<view::CallRender3D_2>();
    if (chainedCall != nullptr) {
        // copy the incoming call to the output
        *chainedCall = *cr3d;

        // chain through the the render call
        (*chainedCall)(0);
    }

    // TODO FBO magic

    // render our own stuff
    this->Render(call);

    return true;
}

/*
 * view::Renderer3DModule_2::MouseEventChain
 */
bool view::Renderer3DModule_2::MouseEventChain(float x, float y, view::MouseFlags flags) { return false; }
