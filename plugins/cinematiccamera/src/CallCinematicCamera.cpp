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
