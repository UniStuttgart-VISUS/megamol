/*
 * Renderer3DModuleDS.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/Renderer3DModuleDS.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/view/CallRenderDeferred3D.h"


using namespace megamol::core;


/*
 * view::Renderer3DModuleDS::Renderer3DModuleDS
 */
view::Renderer3DModuleDS::Renderer3DModuleDS(void) : Renderer3DModule(),
        renderSlotDS("renderingDS", "Connects the renderer to a calling renderer") {
    this->renderSlotDS.SetCallback(CallRenderDeferred3D::ClassName(),
        CallRenderDeferred3D::FunctionName(0), &Renderer3DModuleDS::RenderChainCallback);
    this->renderSlotDS.SetCallback(CallRenderDeferred3D::ClassName(),
        CallRenderDeferred3D::FunctionName(1), &Renderer3DModuleDS::GetExtentsChainCallback);
    this->MakeSlotAvailable(&this->renderSlotDS);
}
