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
    keyframes(NULL), boundingbox(NULL), selectedKeyframe(NULL), interpolatedKeyframe(NULL), cameraParam()
    {

    this->selectedTime     = 0.0f;
    this->interpolatedTime = 0.0f;
    this->totalTime        = 1.0f;
    this->keyframesChanged = false;
    this->selTimeChanged   = false;
    this->intTimeChanged   = false;
    this->totTimeChanged   = false;
    this->camParamChanged  = false;

}


/*
* CallCinematicCamera::~CallCinematicCamera
*/
CallCinematicCamera::~CallCinematicCamera(void) {
	this->keyframes            = NULL;
    this->boundingbox          = NULL;
    this->selectedKeyframe     = NULL;
    this->interpolatedKeyframe = NULL;
}
