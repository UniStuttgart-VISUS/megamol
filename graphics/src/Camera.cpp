/*
 * Camera.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/Camera.h"

#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/mathfunctions.h"
#include "vislib/assert.h"


#define CAMERA_SCENESPACE_DELTA 0.01f


/*
 * vislib::graphics::Camera::Camera
 */
vislib::graphics::Camera::Camera(void) : holder(NULL), updateCounter(0) {
    this->SetDefaultValues();
}


/**
 * copy ctor
 */
vislib::graphics::Camera::Camera(const vislib::graphics::Camera &rhs) 
        : holder(NULL), updateCounter(0) {
    *this = rhs;
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
    this->halfApertureAngle = math::AngleDeg2Rad(15.0f);
    this->farClip = 10.0f;
    this->focalDistance = 1.0f;
    this->nearClip = 0.1f;
    this->halfStereoDisparity = 0.01f;
    this->projectionType = vislib::graphics::Camera::MONO_PERSPECTIVE;
    this->virtualHalfWidth = static_cast<ImageSpaceValue>(400);
    this->virtualHalfHeight = static_cast<ImageSpaceValue>(300);
    this->ResetTileRectangle();
    this->updateCounter = 0;
    this->membersChanged = true;
}


/*
 * vislib::graphics::Camera::SetApertureAngle
 */
void vislib::graphics::Camera::SetApertureAngle(math::AngleDeg apertureAngle) {
    if ((apertureAngle <= math::AngleDeg(0)) || (apertureAngle >= math::AngleDeg(180))) {
        throw IllegalParamException("apertureAngle", __FILE__, __LINE__);
    }
    this->halfApertureAngle = math::AngleDeg2Rad(apertureAngle * 0.5f);
    this->membersChanged = true;
}


/*
 * vislib::graphics::Camera::SetFarClipDistance
 */
void vislib::graphics::Camera::SetFarClipDistance(SceneSpaceValue farClip) {
    if (farClip <= this->nearClip) {
        farClip = this->nearClip + CAMERA_SCENESPACE_DELTA;
    }
    this->farClip = farClip;
    this->membersChanged = true;
}


/*
 * vislib::graphics::Camera::SetFocalDistance
 */
void vislib::graphics::Camera::SetFocalDistance(SceneSpaceValue focalDistance) {
    if (focalDistance <= SceneSpaceValue(0)) {
        focalDistance = CAMERA_SCENESPACE_DELTA;
    }
    this->focalDistance = focalDistance;
    this->membersChanged = true;
}


/*
 * vislib::graphics::Camera::SetNearClipDistance
 */
void vislib::graphics::Camera::SetNearClipDistance(SceneSpaceValue nearClip) {
    if (nearClip <= SceneSpaceValue(0)) {
        nearClip = CAMERA_SCENESPACE_DELTA;
    }
    this->nearClip = nearClip;
    if (this->farClip <= nearClip) {
        this->farClip = nearClip + CAMERA_SCENESPACE_DELTA;
    }
    this->membersChanged = true;
}


/*
 * vislib::graphics::Camera::SetStereoDisparity
 */
void vislib::graphics::Camera::SetStereoDisparity(SceneSpaceValue stereoDisparity) {
    this->halfStereoDisparity = stereoDisparity * 0.5f;
    if (this->halfStereoDisparity < SceneSpaceValue(0)) {
        this->halfStereoDisparity = -this->halfStereoDisparity;
    }
    this->membersChanged = true;
}


/*
 * vislib::graphics::Camera::SetProjectionType
 */
void vislib::graphics::Camera::SetProjectionType(vislib::graphics::Camera::ProjectionType projectionType) {
    this->projectionType = projectionType;
    this->membersChanged = true;
}


/*
 * vislib::graphics::Camera::SetVirtualWidth
 */
void vislib::graphics::Camera::SetVirtualWidth(ImageSpaceValue virtualWidth) {
    if ((math::IsEqual(this->tileRect.Left(), 0.0f)) && (math::IsEqual(this->tileRect.Right(), this->virtualHalfWidth * 2.0f))) {
        this->tileRect.SetRight(math::Max(virtualWidth, ImageSpaceValue(CAMERA_SCENESPACE_DELTA)));
    }
    this->virtualHalfWidth = math::Max(virtualWidth * 0.5f, ImageSpaceValue(CAMERA_SCENESPACE_DELTA));
    this->membersChanged = true;
}


/*
 * vislib::graphics::Camera::SetVirtualHeight
 */
void vislib::graphics::Camera::SetVirtualHeight(ImageSpaceValue virtualHeight) {
    if ((math::IsEqual(this->tileRect.Bottom(), 0.0f)) && (math::IsEqual(this->tileRect.Top(), this->virtualHalfHeight * 2.0f))) {
        this->tileRect.SetTop(math::Max(virtualHeight, ImageSpaceValue(CAMERA_SCENESPACE_DELTA)));
    }
    this->virtualHalfHeight = math::Max<ImageSpaceValue>(virtualHeight * 0.5f, ImageSpaceValue(CAMERA_SCENESPACE_DELTA));
    this->membersChanged = true;
}


/*
 * vislib::graphics::Camera::operator=
 */
vislib::graphics::Camera & vislib::graphics::Camera::operator=(
        const vislib::graphics::Camera &rhs) {
    this->holder = rhs.holder->Clone();

    this->halfApertureAngle = rhs.halfApertureAngle;
    this->farClip = rhs.farClip;
    this->focalDistance = rhs.focalDistance;
    this->nearClip = rhs.nearClip;
    this->halfStereoDisparity = rhs.halfStereoDisparity;
    this->projectionType = rhs.projectionType;
    this->virtualHalfWidth = rhs.virtualHalfWidth;
    this->virtualHalfHeight = rhs.virtualHalfHeight;
    this->tileRect = rhs.tileRect;
    this->updateCounter = 0;
    this->membersChanged = true;

    return *this;
}


/*
 * vislib::graphics::Camera::CalcFrustrumParameters
 */
void vislib::graphics::Camera::CalcFrustrumParameters(SceneSpaceValue &outLeft,
        SceneSpaceValue &outRight, SceneSpaceValue &outBottom,
        SceneSpaceValue &outTop, SceneSpaceValue &outNearClip,
        SceneSpaceValue &outFarClip) {
    float w, h;

    if (!this->holder) {
        throw IllegalStateException("Camera is not associated with a beholer", __FILE__, __LINE__);
    }

    outNearClip = this->nearClip;
    outFarClip = this->farClip;

    switch(this->projectionType) {
        case MONO_PERSPECTIVE: // no break
        case STEREO_PARALLEL_LEFT: // no break
        case STEREO_PARALLEL_RIGHT: // no break
        case STEREO_TOE_IN_LEFT: // no break
        case STEREO_TOE_IN_RIGHT: {
            h = tan(this->halfApertureAngle) * this->nearClip;
            w = h * this->virtualHalfWidth / this->virtualHalfHeight;

            outLeft = this->tileRect.GetLeft() * w / this->virtualHalfWidth;
            outRight = this->tileRect.GetRight() * w/ this->virtualHalfWidth;
            outBottom = this->tileRect.GetBottom() * h / this->virtualHalfHeight;
            outTop = this->tileRect.GetTop() * h / this->virtualHalfHeight;

            outLeft -= w;
            outRight -= w;
            outBottom -= h;
            outTop -= h;
        } break;
        case MONO_ORTHOGRAPHIC:
            // return shifted tile
            outLeft = this->tileRect.GetLeft() - this->virtualHalfWidth;
            outRight = this->tileRect.GetRight() - this->virtualHalfWidth;
            outBottom = this->tileRect.GetBottom() - this->virtualHalfHeight;
            outTop = this->tileRect.GetTop() - this->virtualHalfHeight;
            // return;
            break;
        default:
            // projection parameter calculation still not implemeneted
            ASSERT(false);
    }
}


/*
 * vislib::graphics::Camera::CalcViewParameters
 */
void vislib::graphics::Camera::CalcViewParameters(
        math::Point3D<SceneSpaceValue> &outPosition,
        math::Vector3D<SceneSpaceValue> &outFront,
        math::Vector3D<SceneSpaceValue> &outUp) {
    if (!this->holder) {
        throw IllegalStateException("Camera is not associated with a beholer", __FILE__, __LINE__);
    }

    this->holder->ReturnUpVector(outUp);
    this->holder->ReturnPosition(outPosition);
    this->holder->ReturnFrontVector(outFront);

    if ((this->projectionType == STEREO_PARALLEL_LEFT) 
            || (this->projectionType == STEREO_OFF_AXIS_LEFT) 
            || (this->projectionType == STEREO_TOE_IN_LEFT)) {
        
        math::Vector3D<SceneSpaceValue> right;

        this->holder->ReturnRightVector(right);
        right *= this->halfStereoDisparity;
        outPosition -= right;
    }

    if ((this->projectionType == STEREO_PARALLEL_RIGHT) 
            || (this->projectionType == STEREO_OFF_AXIS_RIGHT) 
            || (this->projectionType == STEREO_TOE_IN_RIGHT)) {

        math::Vector3D<SceneSpaceValue> right;

        this->holder->ReturnRightVector(right);
        right *= this->halfStereoDisparity;
        outPosition += right;
    }

    if ((this->projectionType == STEREO_TOE_IN_LEFT) 
            || (this->projectionType == STEREO_TOE_IN_RIGHT)) {

        math::Point3D<SceneSpaceValue> lookAt;

        this->holder->ReturnPosition(lookAt);
        lookAt += outFront * this->focalDistance;
        outFront = lookAt - outPosition;
        outFront.Normalise(); // unsure if this is necessary
    }
}
