/*
* CallCinematicCamera.cpp
*
* Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"

#include "CallCinematicCamera.h"

#include "vislib/graphics/gl/IncludeAllGL.h"


using namespace megamol;
using namespace megamol::cinematiccamera;


/*
* CallCinematicCamera::CallCinematicCamera
*/
CallCinematicCamera::CallCinematicCamera(void) : core::AbstractGetDataCall()  {

    this->keyframes         = nullptr;
    this->boundingbox       = nullptr;
    this->interpolCamPos    = nullptr;
    this->cameraParam       = nullptr;
    this->selectedKeyframe  = Keyframe();
    this->totalAnimTime     = 1.0f;
    this->totalSimTime      = 1.0f;
    this->dropAnimTime      = 0.0f;
    this->dropSimTime       = 0.0f;
    this->interpolSteps     = 10;
    this->fps               = 24;
    this->bboxCenter        = vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f);
    this->firstCtrllPos     = vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f);
    this->lastCtrllPos      = vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f);
}


/*
* CallCinematicCamera::~CallCinematicCamera
*/
CallCinematicCamera::~CallCinematicCamera(void) {
	this->keyframes      = nullptr;
    this->interpolCamPos = nullptr;
    this->boundingbox    = nullptr;
}
