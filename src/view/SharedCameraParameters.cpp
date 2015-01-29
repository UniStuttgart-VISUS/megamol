//
// SharedCameraParameters.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//

#include <stdafx.h>
#include "view/SharedCameraParameters.h"
#include "view/CallCamParams.h"

using namespace megamol;
using namespace megamol::core;

/*
 * view::SharedCameraParameters::SharedCameraParameters
 */
view::SharedCameraParameters::SharedCameraParameters(void) : core::Module(),
        camParamsSlot("camParams", "Provides read and write access to shared camera parameters"),
        valid(false) {

    this->camParamsSlot.SetCallback(CallCamParams::ClassName(),
            CallCamParams::FunctionName(CallCamParams::CallForGetCamParams),
            &SharedCameraParameters::getCamParams);
    this->camParamsSlot.SetCallback(CallCamParams::ClassName(),
            CallCamParams::FunctionName(CallCamParams::CallForSetCamParams),
            &SharedCameraParameters::setCamParams);
    this->MakeSlotAvailable(&this->camParamsSlot);
}


/*
 * view::SharedCameraParameters::~SharedCameraParameters
 */
view::SharedCameraParameters::~SharedCameraParameters(void) {
    this->Release();
}


/*
 * view::SharedCameraParameters::create
 */
bool view::SharedCameraParameters::create(void) {
    return true;
}


/*
 * view::SharedCameraParameters::release
 */
void view::SharedCameraParameters::release(void) {
}


/*
 * view::SharedCameraParameters::getCamParams
 */
bool view::SharedCameraParameters::getCamParams(megamol::core::Call& call) {

    // Get cam params call
    CallCamParams *cp = dynamic_cast<CallCamParams*>(&call);

    if (cp == NULL) {
        return false;
    }

    // Set camera parameters for the caller if current params are valid
    if (this->valid) {
        cp->CopyFrom(this);
    }

    return true;
}


/*
 * view::SharedCameraParameters::setCamParams
 */
bool view::SharedCameraParameters::setCamParams(megamol::core::Call& call) {

    // Get cam params call
    CallCamParams *cp = dynamic_cast<CallCamParams*>(&call);

    if (cp == NULL) {
        return false;
    }

    // Get camera parameters from the caller
    this->CopyFrom(cp->GetCamParams());

    this->valid = true;

    return true;
}



