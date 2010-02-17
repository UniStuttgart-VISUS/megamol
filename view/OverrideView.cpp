/*
 * OverrideView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "OverrideView.h"
#include "param/EnumParam.h"
#include "param/Vector2fParam.h"
#include "param/Vector4fParam.h"

using namespace megamol::core;
using vislib::graphics::CameraParameters;


/*
 * view::OverrideView::OverrideView
 */
view::OverrideView::OverrideView(void) : AbstractOverrideView(),
        eye(CameraParameters::LEFT_EYE),
        eyeSlot("eye", "The stereo projection eye"),
        projType(CameraParameters::MONO_PERSPECTIVE),
        projTypeSlot("projType", "The stereo projection type"),
        tileH(100.0f), tileSlot("tile", "The rendering tile"),
        tileW(100.0f), tileX(0.0f), tileY(0.0f), virtHeight(100.0f),
        virtSizeSlot("virtSize", "The virtual viewport size"),
        virtWidth(0.0f), viewportWidth(1), viewportHeight(1) {

    param::EnumParam *eyeParam = new param::EnumParam(
        static_cast<int>(CameraParameters::LEFT_EYE));
    eyeParam->SetTypePair(static_cast<int>(CameraParameters::LEFT_EYE), "Left Eye");
    eyeParam->SetTypePair(static_cast<int>(CameraParameters::RIGHT_EYE), "Right Eye");
    this->eyeSlot << eyeParam;
    this->MakeSlotAvailable(&this->eyeSlot);

    param::EnumParam *projParam = new param::EnumParam(
        static_cast<int>(CameraParameters::MONO_PERSPECTIVE));
    projParam->SetTypePair(static_cast<int>(CameraParameters::MONO_PERSPECTIVE), "Mono");
    projParam->SetTypePair(static_cast<int>(CameraParameters::STEREO_OFF_AXIS), "Stereo OffAxis");
    projParam->SetTypePair(static_cast<int>(CameraParameters::STEREO_PARALLEL), "Stereo Parallel");
    projParam->SetTypePair(static_cast<int>(CameraParameters::STEREO_TOE_IN), "Stereo ToeIn");
    this->projTypeSlot << projParam;
    this->MakeSlotAvailable(&this->projTypeSlot);

    this->tileSlot << new param::Vector4fParam(vislib::math::Vector<float, 4>());
    this->MakeSlotAvailable(&this->tileSlot);

    this->virtSizeSlot << new param::Vector2fParam(vislib::math::Vector<float, 2>());
    this->MakeSlotAvailable(&this->virtSizeSlot);

}


/*
 * view::OverrideView::~OverrideView
 */
view::OverrideView::~OverrideView(void) {
    this->Release();
}


/*
 * view::OverrideView::Render
 */
void view::OverrideView::Render(void) {
    view::CallRenderView *crv = this->getCallRenderView();
    if (crv == NULL) return; // false ?

    if (this->eyeSlot.IsDirty()) {
        this->eyeSlot.ResetDirty();
        this->eye = static_cast<CameraParameters::StereoEye>(
            this->eyeSlot.Param<param::EnumParam>()->Value());
    }
    if (this->projTypeSlot.IsDirty()) {
        this->projTypeSlot.ResetDirty();
        this->projType = static_cast<CameraParameters::ProjectionType>(
            this->projTypeSlot.Param<param::EnumParam>()->Value());
    }
    if (this->tileSlot.IsDirty()) {
        this->tileSlot.ResetDirty();
        const vislib::math::Vector<float, 4> &val
            = this->tileSlot.Param<param::Vector4fParam>()->Value();
        this->tileX = val[0];
        this->tileY = val[1];
        this->tileW = val[2];
        this->tileH = val[3];
    }
    if (this->virtSizeSlot.IsDirty()) {
        this->virtSizeSlot.ResetDirty();
        const vislib::math::Vector<float, 2> &val
            = this->virtSizeSlot.Param<param::Vector2fParam>()->Value();
        this->virtWidth = val[0];
        this->virtHeight = val[1];
    }

    crv->ResetAll();
    crv->SetProjection(this->projType, this->eye);
    if ((this->virtWidth != 0) && (this->virtHeight != 0) && (this->tileW != 0) && (this->tileH != 0)) {
        crv->SetTile(this->virtWidth, this->virtHeight, this->tileX, this->tileY, this->tileW, this->tileH);
    }
    crv->SetViewportSize(this->viewportWidth, this->viewportHeight);
    (*crv)(view::CallRenderView::CALL_RENDER);
}


/*
 * view::OverrideView::ResetView
 */
void view::OverrideView::ResetView(void) {
    // resets camera, not override values
    view::CallRenderView *crv = this->getCallRenderView();
    if (crv != NULL) {
        (*crv)(view::CallRenderView::CALL_RESETVIEW);
    }
}


/*
 * view::OverrideView::Resize
 */
void view::OverrideView::Resize(unsigned int width, unsigned int height) {
    this->viewportWidth = width;
    if (this->viewportWidth < 1) this->viewportWidth = 1;
    this->viewportHeight = height;
    if (this->viewportHeight < 1) this->viewportHeight = 1;
}


/*
 * view::OverrideView::SetCursor2DButtonState
 */
void view::OverrideView::SetCursor2DButtonState(unsigned int btn, bool down) {
    view::CallRenderView *crv = this->getCallRenderView();
    if (crv != NULL) {
        crv->SetMouseButton(btn, down);
        (*crv)(view::CallRenderView::CALL_SETCURSOR2DBUTTONSTATE);
    }
}


/*
 * view::OverrideView::SetCursor2DPosition
 */
void view::OverrideView::SetCursor2DPosition(float x, float y) {
    view::CallRenderView *crv = this->getCallRenderView();
    if (crv != NULL) {
        this->packMouseCoordinates(x, y);
        crv->SetMousePosition(x, y);
        (*crv)(view::CallRenderView::CALL_SETCURSOR2DPOSITION);
    }
}


/*
 * view::OverrideView::SetInputModifier
 */
void view::OverrideView::SetInputModifier(mmcInputModifier mod, bool down) {
    view::CallRenderView *crv = this->getCallRenderView();
    if (crv != NULL) {
        crv->SetInputModifier(mod, down);
        (*crv)(view::CallRenderView::CALL_SETINPUTMODIFIER);
    }
}


/*
 * view::OverrideView::UpdateFreeze
 */
void view::OverrideView::UpdateFreeze(bool freeze) {
    view::CallRenderView *crv = this->getCallRenderView();
    if (crv != NULL) {
        (*crv)(freeze
            ? view::CallRenderView::CALL_FREEZE
            : view::CallRenderView::CALL_UNFREEZE);
    }
}


/*
 * view::OverrideView::create
 */
bool view::OverrideView::create(void) {
    // intentionally empty
    return true;
}


/*
 * view::OverrideView::release
 */
void view::OverrideView::release(void) {
    // intentionally empty
}


/*
 * view::OverrideView::packMouseCoordinates
 */
void view::OverrideView::packMouseCoordinates(float &x, float &y) {
    x /= this->viewportWidth;
    y /= this->viewportHeight;
    //y = 1.0f - y;
    x = this->tileX + this->tileW * x;
    y = this->tileY + this->tileH * y;
}
