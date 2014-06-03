//
// LinkedView3D.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: May 31, 2013
//     Author: scharnkn
//

#include <stdafx.h>
#include "LinkedView3D.h"
#include "vislib/mathfunctions.h"
#include "CallCamParams.h"
#include "vislib/CameraParamsStore.h"

using namespace megamol;
using namespace megamol::protein;


/*
 * LinkedView3D::LinkedView3D
 */
LinkedView3D::LinkedView3D(void) : core::view::View3D(),
        sharedCamParamsSlot("shareCamParams", "Obtain read and write access to shared camera parameters"),
        drag(false), camChanged(false),  oldPosX(-1.0), oldPosY(-1.0), cam() {

    // Data caller slota for the potential maps
    this->sharedCamParamsSlot.SetCompatibleCall<CallCamParamsDescription>();
    this->MakeSlotAvailable(&this->sharedCamParamsSlot);

}


/*
 * LinkedView3D::~LinkedView3D
 */
LinkedView3D::~LinkedView3D(void) {
    this->Release();
}


/*
 * LinkedView3D::Render
 */
void LinkedView3D::Render(const mmcRenderViewContext& context) {

    // Get camera parameters call
    CallCamParams *cp = this->sharedCamParamsSlot.CallAs<CallCamParams>();
    if (cp == NULL) {
        return;
    }

    cp->SetCameraParameters(this->camParams); // Set pointer

    // Update shared camera parameters if necessary
    if (this->camChanged) {
        // Obtain current shared camera parameters from cam params call
        if (!(*cp)(CallCamParams::CallForSetCamParams)) {
            return;
        }
        this->camChanged = false;
    }

    // Obtain current shared camera parameters from cam params call
    if (!(*cp)(CallCamParams::CallForGetCamParams)) {
        return;
    }

    core::view::View3D::Render(context); // Call parent
}


/*
 * LinkedView3D::SetCursor2DButtonState
 */
void LinkedView3D::SetCursor2DButtonState(unsigned int btn, bool down) {

    this->drag = down;

    core::view::View3D::SetCursor2DButtonState(btn, down); // Call parent
}


/*
 * LinkedView3D::SetCursor2DPosition
 */
void LinkedView3D::SetCursor2DPosition(float x, float y) {

    using namespace vislib;

    if ((this->drag) && ((!math::IsEqual(x, this->oldPosX)) || (!math::IsEqual(y, this->oldPosY)))){
        this->camChanged = true;
    }

    this->oldPosX = x;
    this->oldPosY = y;

    core::view::View3D::SetCursor2DPosition(x, y); // Call parent
}



