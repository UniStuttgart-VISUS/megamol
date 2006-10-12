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


#define CAMERA_SCENESPACE_DELTA static_cast<SceneSpaceType>(0.01)


/*
 * vislib::graphics::Camera::Camera
 */
vislib::graphics::Camera::Camera(void) : beholder(NULL), updateCounter(0) {
    this->SetDefaultValues();
}


/**
 * copy ctor
 */
vislib::graphics::Camera::Camera(const vislib::graphics::Camera &rhs) 
        : beholder(NULL), updateCounter(0) {
    *this = rhs;
}


/*
 * vislib::graphics::Camera::Camera
 */
vislib::graphics::Camera::Camera(Beholder *beholder) : updateCounter(0) {
    this->beholder = beholder;
    this->SetDefaultValues();
 }


/**
 * vislib::graphics::Camera::~Camera
 */
vislib::graphics::Camera::~Camera(void) {
    this->beholder = NULL; // Do not delete!
}


/*
 * vislib::graphics::Camera::SetDefaultValues
 */
void vislib::graphics::Camera::SetDefaultValues(void) {
    this->halfApertureAngle = math::AngleDeg2Rad(15.0);
    this->farClip = static_cast<SceneSpaceType>(10.0); // some arbitrary values
    this->focalDistance = static_cast<SceneSpaceType>(1.0); // some arbitrary values
    this->nearClip = static_cast<SceneSpaceType>(0.1); // some arbitrary values
    this->halfStereoDisparity = static_cast<SceneSpaceType>(0.01);
    this->projectionType = vislib::graphics::Camera::MONO_PERSPECTIVE;
    this->eye = vislib::graphics::Camera::LEFT_EYE;
    this->virtualHalfWidth = static_cast<ImageSpaceType>(400); // some arbitrary values
    this->virtualHalfHeight = static_cast<ImageSpaceType>(300); // some arbitrary values
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
    this->halfApertureAngle = math::AngleDeg2Rad(apertureAngle * static_cast<math::AngleDeg>(0.5));
    this->membersChanged = true;
}


/*
 * vislib::graphics::Camera::SetFarClipDistance
 */
void vislib::graphics::Camera::SetFarClipDistance(SceneSpaceType farClip) {
    if (farClip <= this->nearClip) {
        farClip = this->nearClip + CAMERA_SCENESPACE_DELTA;
    }
    this->farClip = farClip;
    this->membersChanged = true;
}


/*
 * vislib::graphics::Camera::SetFocalDistance
 */
void vislib::graphics::Camera::SetFocalDistance(SceneSpaceType focalDistance) {
    if (focalDistance <= static_cast<SceneSpaceType>(0)) {
        focalDistance = CAMERA_SCENESPACE_DELTA;
    }
    this->focalDistance = focalDistance;
    this->membersChanged = true;
}


/*
 * vislib::graphics::Camera::SetNearClipDistance
 */
void vislib::graphics::Camera::SetNearClipDistance(SceneSpaceType nearClip) {
    if (nearClip <= static_cast<SceneSpaceType>(0)) {
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
void vislib::graphics::Camera::SetStereoDisparity(SceneSpaceType stereoDisparity) {
    this->halfStereoDisparity = stereoDisparity * static_cast<SceneSpaceType>(0.5);
    if (this->halfStereoDisparity < static_cast<SceneSpaceType>(0)) {
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
 * vislib::graphics::Camera::SetStereoEye
 */
void vislib::graphics::Camera::SetStereoEye(StereoEye eye) {
    this->eye = eye;
    this->membersChanged = true;
}


/*
 * vislib::graphics::Camera::SetVirtualWidth
 */
void vislib::graphics::Camera::SetVirtualWidth(ImageSpaceType virtualWidth) {
    if ((math::IsEqual(this->tileRect.Left(), static_cast<SceneSpaceType>(0.0))) 
            && (math::IsEqual(this->tileRect.Right(), this->virtualHalfWidth * static_cast<SceneSpaceType>(2.0)))) {
        this->tileRect.SetRight(math::Max(virtualWidth, static_cast<ImageSpaceType>(CAMERA_SCENESPACE_DELTA)));
    }
    this->virtualHalfWidth = math::Max(virtualWidth * static_cast<ImageSpaceType>(0.5), 
        static_cast<ImageSpaceType>(CAMERA_SCENESPACE_DELTA));
    this->membersChanged = true;
}


/*
 * vislib::graphics::Camera::SetVirtualHeight
 */
void vislib::graphics::Camera::SetVirtualHeight(ImageSpaceType virtualHeight) {
    if ((math::IsEqual(this->tileRect.Bottom(), static_cast<SceneSpaceType>(0.0))) 
            && (math::IsEqual(this->tileRect.Top(), this->virtualHalfHeight * static_cast<SceneSpaceType>(2.0)))) {
        this->tileRect.SetTop(math::Max(virtualHeight, static_cast<ImageSpaceType>(CAMERA_SCENESPACE_DELTA)));
    }
    this->virtualHalfHeight = math::Max(virtualHeight * static_cast<ImageSpaceType>(0.5), 
        static_cast<ImageSpaceType>(CAMERA_SCENESPACE_DELTA));
    this->membersChanged = true;
}


/*
 * vislib::graphics::Camera::operator=
 */
vislib::graphics::Camera & vislib::graphics::Camera::operator=(
        const vislib::graphics::Camera &rhs) {

    this->beholder = rhs.beholder;
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
 * vislib::graphics::Camera::SetBeholder
 */
void vislib::graphics::Camera::SetBeholder(Beholder *beholder) {
    this->beholder = beholder;
    this->updateCounter = 0;
}


/*
 * vislib::graphics::Camera::CalcFrustumParameters
 */
void vislib::graphics::Camera::CalcFrustumParameters(SceneSpaceType &outLeft,
        SceneSpaceType &outRight, SceneSpaceType &outBottom,
        SceneSpaceType &outTop, SceneSpaceType &outNearClip,
        SceneSpaceType &outFarClip) {
    SceneSpaceType w, h;

    if (!this->beholder) {
        throw IllegalStateException("Camera is not associated with a beholer", __FILE__, __LINE__);
    }

    // clipping distances
    outNearClip = this->nearClip;
    outFarClip = this->farClip;

    switch(this->projectionType) {
        case MONO_PERSPECTIVE: // no break
        case STEREO_PARALLEL: // no break
        case STEREO_TOE_IN: {
            // symmetric main frustum
            h = tan(this->halfApertureAngle) * this->nearClip;
            w = h * this->virtualHalfWidth / this->virtualHalfHeight;

            // recalc tile rect on near clipping plane
            outLeft = this->tileRect.GetLeft() * w / this->virtualHalfWidth;
            outRight = this->tileRect.GetRight() * w/ this->virtualHalfWidth;
            outBottom = this->tileRect.GetBottom() * h / this->virtualHalfHeight;
            outTop = this->tileRect.GetTop() * h / this->virtualHalfHeight;
          
            // cut out local frustum for tile rect
            outLeft -= w;
            outRight -= w;
            outBottom -= h;
            outTop -= h;
        } break;
        case STEREO_OFF_AXIS: {
            // symmetric main frustum
            h = tan(this->halfApertureAngle) * this->nearClip;
            w = h * this->virtualHalfWidth / this->virtualHalfHeight;

            // recalc tile rect on near clipping plane
            outLeft = this->tileRect.GetLeft() * w / this->virtualHalfWidth;
            outRight = this->tileRect.GetRight() * w/ this->virtualHalfWidth;
            outBottom = this->tileRect.GetBottom() * h / this->virtualHalfHeight;
            outTop = this->tileRect.GetTop() * h / this->virtualHalfHeight;

            // shear frustum
            w += static_cast<SceneSpaceType>(((this->eye == LEFT_EYE) ? -1.0 : 1.0) 
                * (this->nearClip * this->halfStereoDisparity) / this->focalDistance);

            // cut out local frustum for tile rect
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
        math::Point<SceneSpaceType, 3> &outPosition,
        math::Vector<SceneSpaceType, 3> &outFront,
        math::Vector<SceneSpaceType, 3> &outUp) {
    if (!this->beholder) {
        throw IllegalStateException("Camera is not associated with a beholer", __FILE__, __LINE__);
    }

    // mono projection position information
    outUp = this->beholder->GetUpVector();
    outPosition = this->beholder->GetPosition();
    outFront = this->beholder->GetFrontVector();

    if ((this->projectionType == STEREO_PARALLEL) 
            || (this->projectionType == STEREO_OFF_AXIS) 
            || (this->projectionType == STEREO_TOE_IN)) {
        // shift eye for stereo
        math::Vector<SceneSpaceType, 3> right = this->beholder->GetRightVector() * this->halfStereoDisparity;

        if (this->eye == LEFT_EYE) {
            // left eye
            outPosition -= right;
        } else {
            // right eye
            outPosition += right;
        }

        if (this->projectionType == STEREO_TOE_IN) {
            // rotate camera parameters for toe in
            outFront = (this->beholder->GetPosition() + (outFront * this->focalDistance)) - outPosition;
            // outFront.Normalise(); // unsure if this is necessary
        }
    }
}
