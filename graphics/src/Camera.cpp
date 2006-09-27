/*
 * Camera.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/Camera.h"

#include "vislib/IllegalParamException.h"
#include "vislib/mathfunctions.h"


#define CAMERA_SCREENSPACE_DELTA 0.1f


/*
 * vislib::graphics::Camera::Camera
 */
vislib::graphics::Camera::Camera(void) : holder(NULL), updateCounter(0) {
    this->SetDefaultValues();
}


/**
 * vislib::graphics::Camera::~Camera
 */
vislib::graphics::Camera::~Camera(void) {
    delete this->holder;
}


/*
 * vislib::graphics::Camera::SetDefaultValues
 */
void vislib::graphics::Camera::SetDefaultValues(void) {
    this->apertureAngle = 30.0f;
    this->farClip = 10.0f;
    this->focalDistance = 1.0f;
    this->nearClip = 0.1f;
    this->stereoDisparity = 0.1f;
    this->stereoProjectionType = vislib::graphics::Camera::OFF_AXIS_PROJECTION;
    this->virtualWidth = 800;
    this->virtualHeight = 600;
}


/*
 * vislib::graphics::Camera::SetApertureAngle
 */
void vislib::graphics::Camera::SetApertureAngle(math::AngleDeg apertureAngle) {
    if ((apertureAngle <= math::AngleDeg(0)) || (apertureAngle >= math::AngleDeg(180))) {
        throw IllegalParamException("apertureAngle", __FILE__, __LINE__);
    }
    this->apertureAngle = apertureAngle;
    this->updateCounter++;
}


/*
 * vislib::graphics::Camera::SetFarClipDistance
 */
void vislib::graphics::Camera::SetFarClipDistance(SceneSpaceValue farClip) {
    if (farClip <= this->nearClip) {
        farClip = this->nearClip + CAMERA_SCREENSPACE_DELTA;
    }
    this->farClip = farClip;
    this->updateCounter++;
}


/*
 * vislib::graphics::Camera::SetFocalDistance
 */
void vislib::graphics::Camera::SetFocalDistance(SceneSpaceValue focalDistance) {
    if (focalDistance <= SceneSpaceValue(0)) {
        focalDistance = CAMERA_SCREENSPACE_DELTA;
    }
    this->focalDistance = focalDistance;
    this->updateCounter++;
}


/*
 * vislib::graphics::Camera::SetNearClipDistance
 */
void vislib::graphics::Camera::SetNearClipDistance(SceneSpaceValue nearClip) {
    if (nearClip <= SceneSpaceValue(0)) {
        nearClip = CAMERA_SCREENSPACE_DELTA;
    }
    this->nearClip = nearClip;
    if (this->farClip <= nearClip) {
        this->farClip = nearClip + CAMERA_SCREENSPACE_DELTA;
    }
    this->updateCounter++;
}


/*
 * vislib::graphics::Camera::SetStereoDisparity
 */
void vislib::graphics::Camera::SetStereoDisparity(SceneSpaceValue stereoDisparity) {
    this->stereoDisparity = stereoDisparity;
    if (this->stereoDisparity < SceneSpaceValue(0)) {
        this->stereoDisparity = -this->stereoDisparity;
    }
    this->updateCounter++;
}


/*
 * vislib::graphics::Camera::SetStereoProjectionType
 */
void vislib::graphics::Camera::SetStereoProjectionType(vislib::graphics::Camera::StereoProjectionType stereoProjectionType) {
    this->stereoProjectionType = stereoProjectionType;
    this->updateCounter++;
}


/*
 * vislib::graphics::Camera::SetVirtualWidth
 */
void vislib::graphics::Camera::SetVirtualWidth(ImageSpaceValue virtualWidth) {
    this->virtualWidth = math::Max(virtualWidth, ImageSpaceValue(1));
    this->updateCounter++;
}


/*
 * vislib::graphics::Camera::SetVirtualHeight
 */
void vislib::graphics::Camera::SetVirtualHeight(ImageSpaceValue virtualHeight) {
    this->virtualHeight = math::Max(virtualHeight, ImageSpaceValue(1));
    this->updateCounter++;
}
