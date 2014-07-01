//
// LinkedView3D.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//

#include <stdafx.h>
#include "LinkedView3D.h"
#include "vislib/mathfunctions.h"
#include "CallCamParams.h"
#include "vislib/CameraParamsStore.h"

using namespace megamol;
using namespace megamol::core;


/*
 * view::LinkedView3D::LinkedView3D
 */
view::LinkedView3D::LinkedView3D(void) : core::view::View3D(),
        sharedCamParamsSlot("shareCamParams", "Obtain read and write access to shared camera parameters") {

    using namespace vislib::graphics;

    // Data caller slot to share camera parameters
    this->sharedCamParamsSlot.SetCompatibleCall<CallCamParamsDescription>();
    this->MakeSlotAvailable(&this->sharedCamParamsSlot);

    this->observableCamParams = new ObservableCameraParams();
    this->observableCamParams->CopyFrom(this->camParams);
    this->observableCamParams->AddCameraParameterObserver(&this->observer);
}


/*
 * view::LinkedView3D::~LinkedView3D
 */
view::LinkedView3D::~LinkedView3D(void) {
    this->Release();
}


/*
 * view::LinkedView3D::Render
 */
void view::LinkedView3D::Render(const mmcRenderViewContext& context) {

    this->observableCamParams->CopyChangedParamsFrom(this->camParams);

    // Get camera parameters call
    CallCamParams *cp = this->sharedCamParamsSlot.CallAs<CallCamParams>();
    if (cp == NULL) {
        return;
    }

    cp->SetCameraParameters(this->camParams); // Set pointer

    // Update shared camera parameters if necessary
    if (this->observer.HasCamChanged()) {
        // Obtain current shared camera parameters from cam params call
        if (!(*cp)(CallCamParams::CallForSetCamParams)) {
            return;
        }
    }

    // Obtain current shared camera parameters from cam params call
    if (!(*cp)(CallCamParams::CallForGetCamParams)) {
        return;
    }

    // Update observable parameters and reset flag
    this->observableCamParams->CopyFrom(this->camParams);
    this->observer.ResetCamChanged();

    core::view::View3D::Render(context); // Call parent
}



