/*
 * ZoomToRectangle.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */


#include "vislib/ZoomToRectangle.h"
#include "vislib/Camera.h"
#include "vislib/IllegalStateException.h"
#include "vislib/mathfunctions.h"
#include "vislib/Vector.h"


/*
 * vislib::graphics::ZoomToRectangle::ZoomToRectangle
 */
vislib::graphics::ZoomToRectangle::ZoomToRectangle(
        const vislib::SmartPtr<vislib::graphics::CameraParameters>& 
        cameraParams) : AbstractCameraController(), fixFocus(true), 
        mode(ZOOM_PAN_DOLLY), resizeOrthoCams(false), targetRect() {
}


/*
 * vislib::graphics::ZoomToRectangle::ZoomToRectangle
 */
vislib::graphics::ZoomToRectangle::ZoomToRectangle(
        const vislib::graphics::ZoomToRectangle& rhs) 
        : AbstractCameraController(rhs.CameraParams()), fixFocus(true), 
        mode(ZOOM_PAN_DOLLY), resizeOrthoCams(false), targetRect() {
    *this = rhs;
}


/*
 * vislib::graphics::ZoomToRectangle::~ZoomToRectangle
 */
vislib::graphics::ZoomToRectangle::~ZoomToRectangle(void) {
    // intentionally empty
}


/*
 * vislib::graphics::ZoomToRectangle::Zoom
 */
void vislib::graphics::ZoomToRectangle::Zoom(void) {
    // check preconditions
    if (!this->IsCameraParamsValid()) {
        throw IllegalStateException("Camera not set", __FILE__, __LINE__);
    }
    if (this->targetRect.IsEmpty()) {
        throw IllegalStateException("Zoom target rectangle is empty", 
            __FILE__, __LINE__);
    }

    if (this->CameraParams()->Projection() != CameraParameters::MONO_ORTHOGRAPHIC) {
        this->zoomProjectiveCamera();
    } else {
        this->zoomOrthographicCamera();
    }
}


/*
 * vislib::graphics::ZoomToRectangle::operator=
 */
vislib::graphics::ZoomToRectangle& 
        vislib::graphics::ZoomToRectangle::operator=(
        const ZoomToRectangle& rhs) {

    this->SetCameraParams(rhs.CameraParams());
    this->fixFocus = rhs.fixFocus;
    this->mode = rhs.mode;
    this->resizeOrthoCams = rhs.resizeOrthoCams;
    this->targetRect = rhs.targetRect;

    return *this;
}


/*
 * vislib::graphics::ZoomToRectangle::zoomOrthographicCamera
 */
void vislib::graphics::ZoomToRectangle::zoomOrthographicCamera(void) {
    ASSERT(this->IsCameraParamsValid());
    ASSERT(this->CameraParams()->Projection() 
        == CameraParameters::MONO_ORTHOGRAPHIC);
    BECAUSE_I_KNOW(this->CameraParams()->VirtualViewSize().Width() > 0.0f);
    BECAUSE_I_KNOW(this->CameraParams()->VirtualViewSize().Height() > 0.0f);

    // move the camera to center the target area
    math::Point<ImageSpaceType, 2> center = this->targetRect.CalcCenter();
    center.Set(
        center.X() - this->CameraParams()->VirtualViewSize().Width() * 0.5f,
        center.Y() - this->CameraParams()->VirtualViewSize().Height() * 0.5f);
    this->CameraParams()->SetView(
        this->CameraParams()->Position() 
        + this->CameraParams()->Right() * center.X()
        - this->CameraParams()->Up() * center.Y(), 
        this->CameraParams()->LookAt() 
        + this->CameraParams()->Right() * center.X()
        - this->CameraParams()->Up() * center.Y(), 
        this->CameraParams()->Up());

    if ((this->resizeOrthoCams) 
            && (this->CameraParams()->VirtualViewSize().Height() > 0.0f)) {
        // resize virtual camera image and tile
        float scale = 1.0f;

        if (this->targetRect.Width() <= 0.0f) {
            scale = this->targetRect.Height() 
                / float(this->CameraParams()->VirtualViewSize().Height());

        } else if (this->targetRect.Height() <= 0.0f) {
            scale = this->targetRect.Width() 
                / float(this->CameraParams()->VirtualViewSize().Width());

        } else {
            float aCam = float(this->CameraParams()->VirtualViewSize().Width())
                / float(this->CameraParams()->VirtualViewSize().Height());

            if (aCam > float(this->targetRect.AspectRatio())) {
                scale = this->targetRect.Height() 
                    / float(this->CameraParams()->VirtualViewSize().Height());

            } else {
                scale = this->targetRect.Width() 
                    / float(this->CameraParams()->VirtualViewSize().Width());

            }
        }

        math::Rectangle<ImageSpaceType> r = this->CameraParams()->TileRect();
        r.Set(r.Left() * scale, r.Bottom() * scale, 
            r.Right() * scale, r.Top() * scale);
        this->CameraParams()->SetTileRect(r);
        this->CameraParams()->SetVirtualViewSize(
            this->CameraParams()->VirtualViewSize().Width() * scale,
            this->CameraParams()->VirtualViewSize().Height() * scale);
    }
}


/*
 * vislib::graphics::ZoomToRectangle::zoomProjectiveCamera
 */
void vislib::graphics::ZoomToRectangle::zoomProjectiveCamera(void) {
    ASSERT(this->IsCameraParamsValid());
    ASSERT(this->CameraParams()->Projection() 
        != CameraParameters::MONO_ORTHOGRAPHIC);
    BECAUSE_I_KNOW(this->CameraParams()->VirtualViewSize().Width() > 0.0f);
    BECAUSE_I_KNOW(this->CameraParams()->VirtualViewSize().Height() > 0.0f);

    // calculate distance of the image plane
    SceneSpaceType imgDist = (this->CameraParams()->VirtualViewSize().Height()
        * 0.5f) / tanf(this->CameraParams()->HalfApertureAngle());

    if ((this->mode == ZOOM_TRACK_DOLLY) || (this->mode == ZOOM_TRACK_ZOOM)) {
        // Track:

        // center of the target rectangle
        math::Point<SceneSpaceType, 3> center = this->targetRect.CalcCenter();
        // make it relative to the point on the view line

        center.Set(center.X() 
            - this->CameraParams()->VirtualViewSize().Width() * 0.5f, 
            center.Y() - this->CameraParams()->VirtualViewSize().Height()
            * 0.5f, 0.0f);

        // bring relative center point on the focus plane
        center.Set(static_cast<ImageSpaceType>(center.X() 
            * this->CameraParams()->FocalDistance() / imgDist), 
            static_cast<ImageSpaceType>(center.Y() 
            * this->CameraParams()->FocalDistance() / imgDist), 0.0f);

        // calculate the real target height on the image plane
        ImageSpaceType targetHeightHalf;

        // camera aspect ratio
        double aCam = double(this->CameraParams()->VirtualViewSize().Width()) 
            / double(this->CameraParams()->VirtualViewSize().Height());

        // ... compared to the target aspect ration
        if (aCam > this->targetRect.AspectRatio()) {
            targetHeightHalf = 0.5f * this->targetRect.Height();
        } else {
            targetHeightHalf = 0.5f * (this->targetRect.Width() / float(aCam));
        }

        if (this->mode == ZOOM_TRACK_ZOOM) {
            // Zoom:

            // adopt aperture angle
            this->CameraParams()->SetApertureAngle(math::AngleRad2Deg(
                static_cast<math::AngleRad>(2.0 * atan(targetHeightHalf 
                / imgDist))));

        } else { // ZOOM_TRACK_DOLLY
            // Dolly:

            // calculate needed distance to image plane
            SceneSpaceType dist = targetHeightHalf / tanf(
                this->CameraParams()->HalfApertureAngle());

            // transfer relative zu focus plane
            dist = (dist * this->CameraParams()->FocalDistance()) / imgDist;

            // setup dolly
            center.SetZ(this->CameraParams()->FocalDistance() - dist);

        }

        // move the camera
        this->CameraParams()->SetView(
            this->CameraParams()->Position() 
            + this->CameraParams()->Right() * center.X()
            - this->CameraParams()->Up() * center.Y()
            + this->CameraParams()->Front() * center.Z(), 
            this->CameraParams()->LookAt() 
            + this->CameraParams()->Right() * center.X()
            - this->CameraParams()->Up() * center.Y()
            + this->CameraParams()->Front() * center.Z(), 
            this->CameraParams()->Up());

        // correct focus distance in case of dolly movement
        if (this->fixFocus) {
            this->CameraParams()->SetFocalDistance(
                this->CameraParams()->FocalDistance() - center.Z());
        }

    } else { // ZOOM_PAN_*
        // Pan:

        // move the image space origin to the middle of the virtual camera image
        this->targetRect.Move(
            -this->CameraParams()->VirtualViewSize().Width() * 0.5f, 
            -this->CameraParams()->VirtualViewSize().Height() * 0.5f);

        // target corner points in world space.
        math::Point<SceneSpaceType, 3> tp1 = this->CameraParams()->Position()
            + this->CameraParams()->Front() 
            + this->CameraParams()->Right() * this->targetRect.Left() / imgDist
            - this->CameraParams()->Up() * this->targetRect.Top() / imgDist;
        math::Point<SceneSpaceType, 3> tp2 = this->CameraParams()->Position()
            + this->CameraParams()->Front() + this->CameraParams()->Right() 
            * this->targetRect.Right() / imgDist
            - this->CameraParams()->Up() * this->targetRect.Top() / imgDist;
        math::Point<SceneSpaceType, 3> tp3 = this->CameraParams()->Position()
            + this->CameraParams()->Front() 
            + this->CameraParams()->Right() * this->targetRect.Left() / imgDist
            - this->CameraParams()->Up() * this->targetRect.Bottom() / imgDist;
        math::Point<SceneSpaceType, 3> tp4 = this->CameraParams()->Position()
            + this->CameraParams()->Front() + this->CameraParams()->Right() 
            * this->targetRect.Right() / imgDist
            - this->CameraParams()->Up() * this->targetRect.Bottom() / imgDist;
        
        // horizontal pan
        math::AngleRad angle;
        angle = static_cast<math::AngleRad>((atan2(static_cast<SceneSpaceType>(
            this->targetRect.Left()), imgDist) + atan2(
            static_cast<SceneSpaceType>(this->targetRect.Right()), imgDist)) 
            * 0.5f);
        SceneSpaceType lookAtDist = (this->CameraParams()->LookAt() 
            - this->CameraParams()->Position()).Length();
        math::Vector<SceneSpaceType, 3> lookAtDir 
            = this->CameraParams()->Front() + this->CameraParams()->Right() 
            * tan(angle);
        lookAtDir.Normalise();
        this->CameraParams()->SetLookAt(this->CameraParams()->Position() 
            + (lookAtDir * lookAtDist)); 

        // target corner vectors in camera space front-up-plane
        math::Vector<SceneSpaceType, 3> ctp1 = tp1 
            - this->CameraParams()->Position();
        ctp1 = ctp1 - this->CameraParams()->Right() 
            * ctp1.Dot(this->CameraParams()->Right());
        math::Vector<SceneSpaceType, 3> ctp2 = tp2 
            - this->CameraParams()->Position();
        ctp2 = ctp2 - this->CameraParams()->Right() 
            * ctp2.Dot(this->CameraParams()->Right());
        math::Vector<SceneSpaceType, 3> ctp3 = tp3 
            - this->CameraParams()->Position();
        ctp3 = ctp3 - this->CameraParams()->Right() 
            * ctp3.Dot(this->CameraParams()->Right());
        math::Vector<SceneSpaceType, 3> ctp4 = tp4 
            - this->CameraParams()->Position();
        ctp4 = ctp4 - this->CameraParams()->Right() 
            * ctp4.Dot(this->CameraParams()->Right());

        // angles between the corner vectors and the view vector
        math::AngleRad a1 = ctp1.Angle(this->CameraParams()->Front());
        math::AngleRad a2 = ctp2.Angle(this->CameraParams()->Front());
        math::AngleRad a3 = ctp3.Angle(this->CameraParams()->Front());
        math::AngleRad a4 = ctp4.Angle(this->CameraParams()->Front());

        // correct signs of the angles
        if (ctp1.Dot(this->CameraParams()->Up()) < 0.0f) a1 = -a1;
        if (ctp2.Dot(this->CameraParams()->Up()) < 0.0f) a2 = -a2;
        if (ctp3.Dot(this->CameraParams()->Up()) < 0.0f) a3 = -a3;
        if (ctp4.Dot(this->CameraParams()->Up()) < 0.0f) a4 = -a4;

        // find the centered angle
        angle = (math::Max(math::Max(a1, a2), math::Max(a3, a4)) 
            + math::Min(math::Min(a1, a2), math::Min(a3, a4))) * 0.5f;

        // vertical tilt
        lookAtDir = this->CameraParams()->Front() + this->CameraParams()->Up()
            * tan(angle);
        lookAtDir.Normalise();
        this->CameraParams()->SetLookAt(this->CameraParams()->Position() 
            + (lookAtDir * lookAtDist)); 

        // target corner vectors in camera space (right, up, front)
        ctp1 = tp1 - this->CameraParams()->Position();
        ctp1.Set(this->CameraParams()->Right().Dot(ctp1), 
            this->CameraParams()->Up().Dot(ctp1), 
            this->CameraParams()->Front().Dot(ctp1));
        ctp2 = tp2 - this->CameraParams()->Position();
        ctp2.Set(this->CameraParams()->Right().Dot(ctp2), 
            this->CameraParams()->Up().Dot(ctp2), 
            this->CameraParams()->Front().Dot(ctp2));
        ctp3 = tp3 - this->CameraParams()->Position();
        ctp3.Set(this->CameraParams()->Right().Dot(ctp3), 
            this->CameraParams()->Up().Dot(ctp3), 
            this->CameraParams()->Front().Dot(ctp3));
        ctp4 = tp4 - this->CameraParams()->Position();
        ctp4.Set(this->CameraParams()->Right().Dot(ctp4), 
            this->CameraParams()->Up().Dot(ctp4), 
            this->CameraParams()->Front().Dot(ctp4));

        // scale target corner vectors to (front = 1.0)
        ctp1 /= ctp1.GetZ();
        ctp2 /= ctp2.GetZ();
        ctp3 /= ctp3.GetZ();
        ctp4 /= ctp4.GetZ();

        // aperture angles based on the y extends
        a1 = static_cast<math::AngleRad>(math::Abs(atan(ctp1.Y())));
        a2 = static_cast<math::AngleRad>(math::Abs(atan(ctp2.Y())));
        a3 = static_cast<math::AngleRad>(math::Abs(atan(ctp3.Y())));
        a4 = static_cast<math::AngleRad>(math::Abs(atan(ctp4.Y())));
        angle = math::Max(math::Max(a1, a2), math::Max(a3, a4));

        // aperture angles based on the x extends
        a1 = static_cast<math::AngleRad>(math::Abs(atan(ctp1.X()
            * this->CameraParams()->VirtualViewSize().Height()
            / this->CameraParams()->VirtualViewSize().Width())));
        a2 = static_cast<math::AngleRad>(math::Abs(atan(ctp2.X()
            * this->CameraParams()->VirtualViewSize().Height()
            / this->CameraParams()->VirtualViewSize().Width())));
        a3 = static_cast<math::AngleRad>(math::Abs(atan(ctp3.X()
            * this->CameraParams()->VirtualViewSize().Height()
            / this->CameraParams()->VirtualViewSize().Width())));
        a4 = static_cast<math::AngleRad>(math::Abs(atan(ctp4.X()
            * this->CameraParams()->VirtualViewSize().Height()
            / this->CameraParams()->VirtualViewSize().Width())));
        angle = math::Max(angle, math::Max(math::Max(a1, a2), math::Max(a3, a4)));

        if (this->mode == ZOOM_PAN_ZOOM) {
            // Zoom:

            // set the new aperture angle
            this->CameraParams()->SetApertureAngle(math::AngleRad2Deg(
                angle * 2.0f));

        } else { // ZOOM_PAN_DOLLY
            // Dolly:

            SceneSpaceType dist = this->CameraParams()->FocalDistance() * 
                (tan(angle) / tan(this->CameraParams()->HalfApertureAngle()));

            SceneSpaceType movePos 
                = (this->CameraParams()->FocalDistance() - dist);
            SceneSpaceType moveTar = movePos;

            this->CameraParams()->SetView(
                this->CameraParams()->Position() 
                + this->CameraParams()->Front() * movePos,
                this->CameraParams()->LookAt() 
                + this->CameraParams()->Front() * moveTar,
                this->CameraParams()->Up());

            if (this->fixFocus) {
                this->CameraParams()->SetFocalDistance(dist);
            }
        }
    }
}
