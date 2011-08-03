/*
 * Renderer3DModuleDS.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Renderer3DModuleDS.h"
#include "CallRender3D.h"
#include "CallRenderDeferred3D.h"


using namespace megamol::core;


/*
 * view::Renderer3DModuleDS::Renderer3DModuleDS
 */
view::Renderer3DModuleDS::Renderer3DModuleDS(void) : Module(),
        renderSlot("rendering", "Connects the renderer to a view"),
        renderSlotDS("renderingDS", "Connects the renderer to a calling renderer"){

    this->renderSlot.SetCallback(CallRender3D::ClassName(),
        CallRender3D::FunctionName(0), &Renderer3DModuleDS::RenderCallback);
    this->renderSlot.SetCallback(CallRender3D::ClassName(),
        CallRender3D::FunctionName(1), &Renderer3DModuleDS::GetExtentsCallback);
    this->renderSlot.SetCallback(CallRender3D::ClassName(),
        CallRender3D::FunctionName(2),
        &Renderer3DModuleDS::GetCapabilitiesCallback);
    this->MakeSlotAvailable(&this->renderSlot);


    this->renderSlotDS.SetCallback(CallRenderDeferred3D::ClassName(),
        CallRenderDeferred3D::FunctionName(0), &Renderer3DModuleDS::RenderCallback);
    this->renderSlotDS.SetCallback(CallRenderDeferred3D::ClassName(),
        CallRenderDeferred3D::FunctionName(1), &Renderer3DModuleDS::GetExtentsCallback);
    this->renderSlotDS.SetCallback(CallRenderDeferred3D::ClassName(),
        CallRenderDeferred3D::FunctionName(2),
        &Renderer3DModuleDS::GetCapabilitiesCallback);
    this->MakeSlotAvailable(&this->renderSlotDS);
}


/*
 * view::Renderer3DModuleDS::~Renderer3DModuleDS
 */
view::Renderer3DModuleDS::~Renderer3DModuleDS(void) {
    // intentionally empty
}
