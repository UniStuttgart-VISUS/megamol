/*
 * Renderer3DModule2000GT.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/Renderer3DModule2000GT.h"
#include "mmcore/view/CallRender3D2000GT.h"

using namespace megamol::core;

/* 
 * view::Renderer3DModule2000GT::Renderer3DModule2000GT
 */
view::Renderer3DModule2000GT::Renderer3DModule2000GT(void)
    : Module(), renderSlot("rendering", "Connects the renderer to a view") {

    this->renderSlot.SetCallback(
        CallRender3D2000GT::ClassName(), CallRender3D2000GT::FunctionName(0), &Renderer3DModule2000GT::RenderCallback);
    this->renderSlot.SetCallback(CallRender3D2000GT::ClassName(), CallRender3D2000GT::FunctionName(1),
        &Renderer3DModule2000GT::GetExtentsCallback);
    this->renderSlot.SetCallback(CallRender3D2000GT::ClassName(), CallRender3D2000GT::FunctionName(2),
        &Renderer3DModule2000GT::GetCapabilitiesCallback);
    this->renderSlot.SetCallback(CallRender3D2000GT::ClassName(), CallRender3D2000GT::FunctionName(3),
        &Renderer3DModule2000GT::OnMouseEventCallback);
    this->MakeSlotAvailable(&this->renderSlot);
}

/* 
 * view::Renderer3DModule2000GT::~Renderer3DModule2000GT
 */
view::Renderer3DModule2000GT::~Renderer3DModule2000GT(void) {
    // intentionally empty
}

/* 
 * view::Renderer3DModule2000GT::MouseEvent
 */
bool view::Renderer3DModule2000GT::MouseEvent(float x, float y, MouseFlags flags) { return false; }

/* 
 * view::Renderer3DModule2000GT::OnMouseEventCallback
 */
bool view::Renderer3DModule2000GT::OnMouseEventCallback(Call& call) {
    try {
        view::CallRender3D2000GT& cr3d = dynamic_cast<view::CallRender3D2000GT&>(call);
        return this->MouseEvent(cr3d.GetMouseX(), cr3d.GetMouseY(), cr3d.GetMouseFlags());
    } catch (...) {
        ASSERT("OnMouseEventCallback call cast failed\n");
    }
}
