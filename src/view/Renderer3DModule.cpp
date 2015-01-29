/*
 * Renderer3DModule.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "view/Renderer3DModule.h"
#include "view/CallRender3D.h"


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
    this->MakeSlotAvailable(&this->renderSlot);
}


/*
 * view::Renderer3DModule::~Renderer3DModule
 */
view::Renderer3DModule::~Renderer3DModule(void) {
    // intentionally empty
}
