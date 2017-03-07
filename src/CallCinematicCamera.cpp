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
CallCinematicCamera::CallCinematicCamera(void) : core::AbstractGetDataCall(), keyframes(NULL) {

}


/*
* CallCinematicCamera::~CallCinematicCamera
*/
CallCinematicCamera::~CallCinematicCamera(void) {
	this->keyframes = NULL;
}


/** function name for getting all Keyframes */
const unsigned int CallForGetKeyframes = 0;
/** function name for getting selected Keyframes */
const unsigned int CallForGetSelectedKeyframe = 1;
/** function name for setting the selected Keyframe */
const unsigned int CallForSelectKeyframe = 2;
/**function name for getting interpolated Keyframe */
const unsigned int CallForInterpolatedKeyframe = 3;
/**function name for getting total time */
const unsigned int CallForGetTotalTime = 4;
/**function name for getting a keyframe at a certain time*/
const unsigned int CallForGetKeyframeAtTime = 5;
/**function name for adding a keyframe at a certain position*/
const unsigned int CallForNewKeyframeAtPosition = 6;