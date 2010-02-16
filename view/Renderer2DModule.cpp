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

    this->renderSlot.SetCallback("CallRender2D", "Render",
        &Renderer2DModule::onRenderCallback);
    this->renderSlot.SetCallback("CallRender2D", "GetExtents",
        &Renderer2DModule::onGetExtentsCallback);
    this->renderSlot.SetCallback("CallRender2D", "MouseEvent",
        &Renderer2DModule::onMouseEventCallback);
    this->MakeSlotAvailable(&this->renderSlot);
}


/*
 * view::Renderer2DModule::~Renderer2DModule
 */
view::Renderer2DModule::~Renderer2DModule(void) {
    // intentionally empty
}


/*
 * view::Renderer2DModule::MouseEvent
 */
bool view::Renderer2DModule::MouseEvent(float x, float y, view::MouseFlags flags) {
    return false;
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


/*
 * view::Renderer2DModule::onMouseEventCallback
 */
bool view::Renderer2DModule::onMouseEventCallback(Call& call) {
    try {
        view::CallRender2D &cr2d = dynamic_cast<view::CallRender2D&>(call);
        return this->MouseEvent(cr2d.GetMouseX(), cr2d.GetMouseY(), cr2d.GetMouseFlags());
    } catch(...) {
        ASSERT("onRenderCallback call cast failed\n");
    }
    return false;
}
