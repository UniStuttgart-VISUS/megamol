/*
 * Renderer3DModule.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/CallRender3D.h"


using namespace megamol::core;


/*
 * view::Renderer3DModule::Renderer3DModule
 */
view::Renderer3DModule::Renderer3DModule(void) : Module(),
        renderSlot("rendering", "Connects the Renderer to a view") {

    this->renderSlot.SetCallback(CallRender3D::ClassName(),
        CallRender3D::FunctionName(0), &Renderer3DModule::RenderCallback);
    this->renderSlot.SetCallback(CallRender3D::ClassName(),
        CallRender3D::FunctionName(1), &Renderer3DModule::GetExtentsCallback);
    this->renderSlot.SetCallback(CallRender3D::ClassName(),
        CallRender3D::FunctionName(2),
        &Renderer3DModule::GetCapabilitiesCallback);
    this->renderSlot.SetCallback(CallRender3D::ClassName(),
        CallRender3D::FunctionName(3),
        &Renderer3DModule::OnMouseEventCallback);
    this->MakeSlotAvailable(&this->renderSlot);
}


/*
 * view::Renderer3DModule::~Renderer3DModule
 */
view::Renderer3DModule::~Renderer3DModule(void) {
    // intentionally empty
}


/*
 * view::Renderer3DModule::MouseEvent
 */
bool view::Renderer3DModule::MouseEvent(float x, float y, MouseFlags flags) {
    return false;
}


/*
 * view::Renderer3DModule::OnMouseEventCallback
 */
bool view::Renderer3DModule::OnMouseEventCallback(Call& call) {
    try {
        view::CallRender3D &cr3d = dynamic_cast<view::CallRender3D&>(call);
        return this->MouseEvent(cr3d.GetMouseX(), cr3d.GetMouseY(), cr3d.GetMouseFlags());
    } catch (...) {
        ASSERT("OnMouseEventCallback call cast failed\n");
    }
    return false;
}
