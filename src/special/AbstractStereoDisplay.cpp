/*
 * AbstractStereoDisplay.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "AbstractStereoDisplay.h"
#include "mmcore/view/CallCursorInput.h"
#include "vislib/math/mathfunctions.h"

using namespace megamol::core;


/*
 * special::AbstractStereoDisplay::~AbstractStereoDisplay
 */
special::AbstractStereoDisplay::~AbstractStereoDisplay(void) {
    this->Release();
}


/*
 * special::AbstractStereoDisplay::ResetView
 */
void special::AbstractStereoDisplay::ResetView(void) {
    view::CallCursorInput *cci
        = this->cursorInputSlot.CallAs<view::CallCursorInput>();
    if (cci != NULL) (*cci)(3);
}


/*
 * special::AbstractStereoDisplay::Resize
 */
void special::AbstractStereoDisplay::Resize(unsigned int width,
        unsigned int height) {
    this->viewportWidth = width;
    if (this->viewportWidth < 1) this->viewportWidth = 1;
    this->viewportHeight = height;
    if (this->viewportHeight < 1) this->viewportHeight = 1;
}


/*
 * special::AbstractStereoDisplay::SetCursor2DButtonState
 */
void special::AbstractStereoDisplay::SetCursor2DButtonState(unsigned int btn,
        bool down) {
    view::CallCursorInput *cci
        = this->cursorInputSlot.CallAs<view::CallCursorInput>();
    if (cci != NULL) {
        cci->Btn() = btn;
        cci->Down() = down;
        (*cci)(0);
    }
}


/*
 * special::AbstractStereoDisplay::SetCursor2DPosition
 */
void special::AbstractStereoDisplay::SetCursor2DPosition(float x, float y) {
    view::CallCursorInput *cci
        = this->cursorInputSlot.CallAs<view::CallCursorInput>();
    if (cci != NULL) {
        float s = vislib::math::Min(
            static_cast<float>(this->viewportWidth),
            static_cast<float>(this->viewportHeight));
        cci->X() = (x - 0.5f * this->viewportWidth) / s;
        cci->Y() = (y - 0.5f * this->viewportHeight) / s;
        (*cci)(1);
    }
}


/*
 * special::AbstractStereoDisplay::SetInputModifier
 */
void special::AbstractStereoDisplay::SetInputModifier(mmcInputModifier mod,
        bool down) {
    view::CallCursorInput *cci
        = this->cursorInputSlot.CallAs<view::CallCursorInput>();
    if (cci != NULL) {
        cci->Mod() = mod;
        cci->Down() = down;
        (*cci)(2);
    }
}


/*
 * special::AbstractStereoDisplay::UpdateFreeze
 */
void special::AbstractStereoDisplay::UpdateFreeze(bool freeze) {
    view::CallRenderView *crv = this->renderViewSlot.CallAs<view::CallRenderView>();
    if (crv == NULL) return;
    (*crv)(freeze ? 1 : 2);
}


/*
 * special::AbstractStereoDisplay::AbstractStereoDisplay
 */
special::AbstractStereoDisplay::AbstractStereoDisplay(void) : view::AbstractView(),
        Module(), viewportWidth(1), viewportHeight(1),
        renderViewSlot("doRender", "Slot connecting to the view to be rendered"),
        cursorInputSlot("cursorInput", "Slot for sending the cursor input") {
    this->MakeSlotAvailable(view::AbstractView::getRenderViewSlot());

    this->renderViewSlot.SetCompatibleCall<view::CallRenderViewDescription>();
    this->MakeSlotAvailable(&this->renderViewSlot);

    this->cursorInputSlot.SetCompatibleCall<view::CallCursorInputDescription>();
    this->MakeSlotAvailable(&this->cursorInputSlot);

}


/*
 * special::AbstractStereoDisplay::create
 */
bool special::AbstractStereoDisplay::create(void) {
    // intentionally empty
    return true;
}


/*
 * special::AbstractStereoDisplay::release
 */
void special::AbstractStereoDisplay::release(void) {
    // intentionally empty
}


/*
 * special::AbstractStereoDisplay::onRenderView
 */
bool special::AbstractStereoDisplay::onRenderView(Call& call) {
    view::CallRenderView *crv = dynamic_cast<view::CallRenderView *>(&call);
    if (crv == NULL) return false;
    view::CallRenderView *crv2 = this->renderViewSlot.CallAs<view::CallRenderView>();
    if (crv2 == NULL) return false;
    (*crv2)=(*crv);
    return (*crv2)();
}
