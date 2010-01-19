/*
 * Renderer2DModule.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Renderer2DModule.h"
#include "vislib/assert.h"


using namespace megamol::core;


/*
 * view::Renderer2DModule::Renderer2DModule
 */
view::Renderer2DModule::Renderer2DModule(void) : Module(),
        renderSlot("rendering", "Connects the Renderer to a view") {

    this->renderSlot.SetCallback(CallRender2D::ClassName(),
        CallRender2D::FunctionName(0),
        &Renderer2DModule::onRenderCallback);
    this->renderSlot.SetCallback(CallRender2D::ClassName(),
        CallRender2D::FunctionName(1),
        &Renderer2DModule::onGetExtentsCallback);
    this->MakeSlotAvailable(&this->renderSlot);
}


/*
 * view::Renderer2DModule::~Renderer2DModule
 */
view::Renderer2DModule::~Renderer2DModule(void) {
    // intentionally empty
}


/*
 * view::Renderer2DModule::onGetExtentsCallback
 */
bool view::Renderer2DModule::onGetExtentsCallback(Call& call) {
    try {
        view::CallRender2D &cr2d = dynamic_cast<view::CallRender2D&>(call);
        return this->GetExtents(cr2d);
    } catch(...) {
        ASSERT("onGetExtentsCallback call cast failed\n");
    }
    return false;
}
        

/*
 * view::Renderer2DModule::onRenderCallback
 */
bool view::Renderer2DModule::onRenderCallback(Call& call) {
    try {
        view::CallRender2D &cr2d = dynamic_cast<view::CallRender2D&>(call);
        return this->Render(cr2d);
    } catch(...) {
        ASSERT("onRenderCallback call cast failed\n");
    }
    return false;
}
