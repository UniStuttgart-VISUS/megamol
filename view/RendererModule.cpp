/*
 * RendererModule.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "RendererModule.h"
#include "CallRender.h"


using namespace megamol::core;


/*
 * view::RendererModule::RendererModule
 */
view::RendererModule::RendererModule(void) : Module(),
        renderSlot("rendering", "Connects the Renderer to a view") {

    CallRenderDescription crd;
    this->renderSlot.SetCallback(crd.ClassName(), "Render",
        &RendererModule::RenderCallback);
    this->MakeSlotAvailable(&this->renderSlot);
}


/*
 * view::RendererModule::~RendererModule
 */
view::RendererModule::~RendererModule(void) {
    // intentionally empty
}
