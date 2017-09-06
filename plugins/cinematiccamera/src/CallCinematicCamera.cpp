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
    this->bboxCenter       = vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f);

    // initialise color table
    this->colorTable.Clear();
    this->colorTable.AssertCapacity(100);
    this->colorTable.Add(vislib::math::Vector<float, 3>(0.4f, 0.4f, 1.0f)); // COL_SPLINE          = 0,
    this->colorTable.Add(vislib::math::Vector<float, 3>(0.7f, 0.7f, 1.0f)); // COL_KEYFRAME        = 1,
    this->colorTable.Add(vislib::math::Vector<float, 3>(0.1f, 0.1f, 1.0f)); // COL_KEYFRAME_SELECT = 2,
    this->colorTable.Add(vislib::math::Vector<float, 3>(0.1f, 0.1f, 1.0f)); // COL_KEYFRAME_DRAG   = 3,
    this->colorTable.Add(vislib::math::Vector<float, 3>(0.3f, 0.8f, 0.8f)); // COL_MANIP_LOOKAT    = 4,
    this->colorTable.Add(vislib::math::Vector<float, 3>(0.8f, 0.0f, 0.8f)); // COL_MANIP_UP        = 5,
    this->colorTable.Add(vislib::math::Vector<float, 3>(0.8f, 0.1f, 0.0f)); // COL_MANIP_X_AXIS    = 6,
    this->colorTable.Add(vislib::math::Vector<float, 3>(0.8f, 0.8f, 0.0f)); // COL_MANIP_Y_AXIS    = 7,
    this->colorTable.Add(vislib::math::Vector<float, 3>(0.1f, 0.8f, 0.0f)); // COL_MANIP_Z_AXIS    = 8
    this->colorTable.Add(vislib::math::Vector<float, 3>(0.8f, 0.0f, 0.0f)); // COL_ANIM_REPEAT     = 9

}


/*
* CallCinematicCamera::~CallCinematicCamera
*/
CallCinematicCamera::~CallCinematicCamera(void) {
	this->keyframes      = NULL;
    this->interpolCamPos = NULL;
    this->boundingbox    = NULL;
}


/*
* CallCinematicCamera::getColor
*/
vislib::math::Vector<float, 3> CallCinematicCamera::getColor(CallCinematicCamera::colType  c) {
    return this->colorTable[(int)c];
}