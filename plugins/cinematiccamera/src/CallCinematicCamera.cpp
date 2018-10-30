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
CallCinematicCamera::CallCinematicCamera(void) : core::AbstractGetDataCall(),
    cameraParam(nullptr),
    interpolCamPos(nullptr),
    keyframes(nullptr),
    boundingbox(nullptr),
    interpolSteps(10),
    selectedKeyframe(),
    dropAnimTime(0.0f),
    dropSimTime(0.0f),
    totalAnimTime(1.0f),
    totalSimTime(1.0f),
    bboxCenter(0.0f, 0.0f, 0.0f),
    fps(24),
    startCtrllPos(0.0f, 0.0f, 0.0f),
    endCtrllPos(0.0f, 0.0f, 0.0f)
{


}


/*
* CallCinematicCamera::~CallCinematicCamera
*/
CallCinematicCamera::~CallCinematicCamera(void) {
	this->keyframes      = nullptr;
    this->interpolCamPos = nullptr;
    this->boundingbox    = nullptr;
}
