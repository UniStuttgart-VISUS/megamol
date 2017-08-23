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
    keyframes(NULL), boundingbox(NULL), selectedKeyframe(), interpolatedKeyframe(), cameraParam()
    {

    this->selectedTime     = 0.0f;
    this->interpolatedTime = 0.0f;
    this->totalTime        = 1.0f;

}


/*
* CallCinematicCamera::~CallCinematicCamera
*/
CallCinematicCamera::~CallCinematicCamera(void) {
	this->keyframes            = NULL;
    this->boundingbox          = NULL;
}
