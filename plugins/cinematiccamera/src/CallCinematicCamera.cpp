/*
* CallCinematicCamera.cpp
*
*/
#include "stdafx.h"

#include "CallCinematicCamera.h"

using namespace megamol;
using namespace megamol::cinematiccamera;


/*
* CallCinematicCamera::CallCinematicCamera
*/
CallCinematicCamera::CallCinematicCamera(void) : core::AbstractGetDataCall(), 
    keyframes(NULL), boundingbox(NULL), interpolCamPos(NULL),
    selectedKeyframe(), cameraParam()
    {

    this->selectedTime     = 0.0f;
    this->totalTime        = 1.0f;
    this->interpolSteps    = 10;
}


/*
* CallCinematicCamera::~CallCinematicCamera
*/
CallCinematicCamera::~CallCinematicCamera(void) {
	this->keyframes      = NULL;
    this->interpolCamPos = NULL;
    this->boundingbox    = NULL;
}
