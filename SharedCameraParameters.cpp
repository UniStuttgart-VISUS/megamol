//
// SharedCameraParameters.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: May 31, 2013
//     Author: scharnkn
//

#include <stdafx.h>
#include "SharedCameraParameters.h"
#include "CallCamParams.h"

using namespace megamol;
using namespace megamol::protein;


/*
 * SharedCameraParameters::SharedCameraParameters
 */
SharedCameraParameters::SharedCameraParameters(void) : core::Module(),
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
 * SharedCameraParameters::~SharedCameraParameters
 */
SharedCameraParameters::~SharedCameraParameters(void) {
    this->Release();
}


/*
 * SharedCameraParameters::create
 */
bool SharedCameraParameters::create(void) {
    return true;
}


/*
 * SharedCameraParameters::release
 */
void SharedCameraParameters::release(void) {
}


/*
 * SharedCameraParameters::getCamParams
 */
bool SharedCameraParameters::getCamParams(megamol::core::Call& call) {

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
 * SharedCameraParameters::setCamParams
 */
bool SharedCameraParameters::setCamParams(megamol::core::Call& call) {

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



