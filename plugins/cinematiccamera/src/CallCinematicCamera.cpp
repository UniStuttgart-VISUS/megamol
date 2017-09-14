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

    this->totalTime        = 1.0f;
    this->interpolSteps    = 10;
    this->maxAnimTime      = 1.0f;
    this->dropTime         = 0.0f;
    this->bboxCenter       = vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f);
}


/*
* CallCinematicCamera::~CallCinematicCamera
*/
CallCinematicCamera::~CallCinematicCamera(void) {
	this->keyframes      = NULL;
    this->interpolCamPos = NULL;
    this->boundingbox    = NULL;
}
