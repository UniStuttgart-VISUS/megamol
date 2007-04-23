/*
 * ZoomToRectangle.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */


#include "vislib/Beholder.h"
#include "vislib/Camera.h"
#include "vislib/ZoomToRectangle.h"
#include "vislib/IllegalStateException.h"
#include "vislib/mathfunctions.h"
#include "vislib/Vector.h"


/*
 * vislib::graphics::ZoomToRectangle::ZoomToRectangle
 */
vislib::graphics::ZoomToRectangle::ZoomToRectangle(void)
        : AbstractBeholderController(), AbstractCameraController(), 
        targetRect(), type(ZOOM_PAN_DOLLY), fixFocus(true), 
        resizeOrthoCams(false) {
}


/*
 * vislib::graphics::ZoomToRectangle::ZoomToRectangle
 */
vislib::graphics::ZoomToRectangle::ZoomToRectangle(
        const ZoomToRectangle& rhs)
        : AbstractBeholderController(), AbstractCameraController(), 
        targetRect(), type(ZOOM_PAN_DOLLY), fixFocus(true), 
        resizeOrthoCams(false) {
    *this = rhs;
}


/*
 * vislib::graphics::ZoomToRectangle::~ZoomToRectangle
 */
vislib::graphics::ZoomToRectangle::~ZoomToRectangle(void) {
}


/*
 * vislib::graphics::ZoomToRectangle::operator=
 */
vislib::graphics::ZoomToRectangle& 
        vislib::graphics::ZoomToRectangle::operator=(
        const ZoomToRectangle& rhs) {
    this->SetBeholder(rhs.GetBeholder());
    this->SetCamera(rhs.GetCamera());
    this->targetRect = rhs.targetRect;
    this->type = rhs.type;
    this->fixFocus = rhs.fixFocus;

    return *this;
}


/*
 * vislib::graphics::ZoomToRectangle::Zoom
 */
void vislib::graphics::ZoomToRectangle::Zoom(void) {
    // check preconditions
    if (this->GetBeholder() == NULL) {
        throw IllegalStateException("Beholder not set", __FILE__, __LINE__);
    }
    if (this->GetCamera() == NULL) {
        throw IllegalStateException("Camera not set", __FILE__, __LINE__);
    }
    if (math::IsEqual(this->targetRect.Width(), 0.0f)
        || math::IsEqual(this->targetRect.Height(), 0.0f)) {
        throw IllegalStateException("Zoom target rectangle is empty", 
            __FILE__, __LINE__);
    }

    if (this->GetCamera()->GetProjectionType() != Camera::MONO_ORTHOGRAPHIC) {
        this->ZoomProjectiveCamera();
    } else {
        this->ZoomOrthographicCamera();
    }
}


/*
 * vislib::graphics::ZoomToRectangle::ZoomOrthographicCamera
 */
void vislib::graphics::ZoomToRectangle::ZoomOrthographicCamera(void) {
    ASSERT(this->GetCamera() != NULL);
    ASSERT(this->GetBeholder() != NULL);
    ASSERT(this->GetCamera()->GetProjectionType() == Camera::MONO_ORTHOGRAPHIC);
    BECAUSE_I_KNOW(this->GetCamera()->GetVirtualWidth() > 0.0f);
    BECAUSE_I_KNOW(this->GetCamera()->GetVirtualHeight() > 0.0f);

    // move the beholder to center the target area
    math::Point<ImageSpaceType, 2> center = this->targetRect.CalcCenter();
    center.Set(center.X() - this->GetCamera()->GetVirtualWidth() * 0.5f, 
             center.Y() - this->GetCamera()->GetVirtualHeight() * 0.5f);
    this->GetBeholder()->SetView(
        this->GetBeholder()->GetPosition() 
        + this->GetBeholder()->GetRightVector() * center.X()
        - this->GetBeholder()->GetUpVector() * center.Y(), 
        this->GetBeholder()->GetLookAt() 
        + this->GetBeholder()->GetRightVector() * center.X()
        - this->GetBeholder()->GetUpVector() * center.Y(), 
        this->GetBeholder()->GetUpVector());

    if ((this->resizeOrthoCams) && (this->GetCamera()->GetVirtualHeight() > 0.0f)) {
        // resize virtual camera image and tile
        float scale = 1.0f;

        if (this->targetRect.Width() <= 0.0f) {
            scale = this->targetRect.Height() / float(this->GetCamera()->GetVirtualHeight());
        } else
        if (this->targetRect.Height() <= 0.0f) {
            scale = this->targetRect.Width() / float(this->GetCamera()->GetVirtualWidth());
        } else {
            double aCam = double(this->GetCamera()->GetVirtualWidth()) / double(this->GetCamera()->GetVirtualHeight());

            if (aCam > this->targetRect.AspectRatio()) {
                scale = this->targetRect.Height() / float(this->GetCamera()->GetVirtualHeight());
            } else {
                scale = this->targetRect.Width() / float(this->GetCamera()->GetVirtualWidth());
            }
        }

        math::Rectangle<ImageSpaceType> r = this->GetCamera()->GetTileRectangle();
        r.Set(r.Left() * scale, r.Bottom() * scale, r.Right() * scale, r.Top() * scale);
        this->GetCamera()->SetTileRectangle(r);
        this->GetCamera()->SetVirtualWidth(this->GetCamera()->GetVirtualWidth() * scale);
        this->GetCamera()->SetVirtualHeight(this->GetCamera()->GetVirtualHeight() * scale);
    }
}


/*
 * vislib::graphics::ZoomToRectangle::ZoomProjectiveCamera
 */
void vislib::graphics::ZoomToRectangle::ZoomProjectiveCamera(void) {
    ASSERT(this->GetCamera() != NULL);
    ASSERT(this->GetBeholder() != NULL);
    ASSERT(this->GetCamera()->GetProjectionType() != Camera::MONO_ORTHOGRAPHIC);
    BECAUSE_I_KNOW(this->GetCamera()->GetVirtualWidth() > 0.0f);
    BECAUSE_I_KNOW(this->GetCamera()->GetVirtualHeight() > 0.0f);

    Camera &cam = *this->GetCamera();
    Beholder &beholder = *this->GetBeholder();

    // calculate distance of the image plane
    SceneSpaceType imgDist = (cam.GetVirtualHeight() * 0.5f) / tanf(cam.GetHalfApertureAngleRad());

    if ((this->type == ZOOM_TRACK_DOLLY) || (this->type == ZOOM_TRACK_ZOOM)) {
        // Track:

        // center of the target rectangle
        math::Point<SceneSpaceType, 3> center = this->targetRect.CalcCenter();
        // make it relative to the point on the view line
        center.Set(center.X() - cam.GetVirtualWidth() * 0.5f, center.Y() - cam.GetVirtualHeight() * 0.5f, 0.0f);
        // bring relative center point on the focus plane
        center.Set(static_cast<ImageSpaceType>(center.X() * cam.GetFocalDistance() / imgDist), 
            static_cast<ImageSpaceType>(center.Y() * cam.GetFocalDistance() / imgDist), 0.0f);

        // calculate the real target height on the image plane
        ImageSpaceType targetHeightHalf;
        // camera aspect ratio
        double aCam = double(cam.GetVirtualWidth()) / double(cam.GetVirtualHeight());
        // ... compared to the target aspect ration
        if (aCam > this->targetRect.AspectRatio()) {
            targetHeightHalf = 0.5f * this->targetRect.Height();
        } else {
            targetHeightHalf = 0.5f * (this->targetRect.Width() / float(aCam));
        }

        if (this->type == ZOOM_TRACK_ZOOM) {
            // Zoom:

            // adopt aperture angle
            cam.SetApertureAngle(math::AngleRad2Deg(
                static_cast<math::AngleRad>(2.0 * atan(targetHeightHalf / imgDist))));

        } else { // ZOOM_TRACK_DOLLY
            // Dolly:

            // calculate needed distance to image plane
            SceneSpaceType dist = targetHeightHalf / tanf(cam.GetHalfApertureAngleRad());
            // transfer relative zu focus plane
            dist = (dist * cam.GetFocalDistance()) / imgDist;
            // setup dolly
            center.SetZ(cam.GetFocalDistance() - dist);
        }

        // move the beholder
        beholder.SetView(
            beholder.GetPosition() 
            + beholder.GetRightVector() * center.X()
            - beholder.GetUpVector() * center.Y()
            + beholder.GetFrontVector() * center.Z(), 
            beholder.GetLookAt() 
            + beholder.GetRightVector() * center.X()
            - beholder.GetUpVector() * center.Y()
            + beholder.GetFrontVector() * center.Z(), 
            beholder.GetUpVector());

        // correct focus distance in case of dolly movement
        if (this->fixFocus) {
            cam.SetFocalDistance(cam.GetFocalDistance() - center.Z());
        }

    } else { // ZOOM_PAN_*
        // Pan:

        // move the image space origin to the middle of the virtual camera image
        this->targetRect.Move(-cam.GetVirtualWidth() * 0.5f, -cam.GetVirtualHeight() * 0.5f);

        // target corner points in world space.
        math::Point<SceneSpaceType, 3> tp1 = beholder.GetPosition()
            + beholder.GetFrontVector() 
            + beholder.GetRightVector() * this->targetRect.Left() / imgDist
            - beholder.GetUpVector() * this->targetRect.Top() / imgDist;
        math::Point<SceneSpaceType, 3> tp2 = beholder.GetPosition()
            + beholder.GetFrontVector() 
            + beholder.GetRightVector() * this->targetRect.Right() / imgDist
            - beholder.GetUpVector() * this->targetRect.Top() / imgDist;
        math::Point<SceneSpaceType, 3> tp3 = beholder.GetPosition()
            + beholder.GetFrontVector() 
            + beholder.GetRightVector() * this->targetRect.Left() / imgDist
            - beholder.GetUpVector() * this->targetRect.Bottom() / imgDist;
        math::Point<SceneSpaceType, 3> tp4 = beholder.GetPosition()
            + beholder.GetFrontVector() 
            + beholder.GetRightVector() * this->targetRect.Right() / imgDist
            - beholder.GetUpVector() * this->targetRect.Bottom() / imgDist;
        
        // horizontal pan
        math::AngleRad angle;
        angle = static_cast<math::AngleRad>((atan2(static_cast<SceneSpaceType>(this->targetRect.Left()), imgDist) 
            + atan2(static_cast<SceneSpaceType>(this->targetRect.Right()), imgDist)) * 0.5f);
        SceneSpaceType lookAtDist = (beholder.GetLookAt() - beholder.GetPosition()).Length();
        math::Vector<SceneSpaceType, 3> lookAtDir = beholder.GetFrontVector() + beholder.GetRightVector() * tan(angle);
        lookAtDir.Normalise();
        beholder.SetLookAt(beholder.GetPosition() + (lookAtDir * lookAtDist)); 

        // target corner vectors in camera space front-up-plane
        math::Vector<SceneSpaceType, 3> ctp1 = tp1 - beholder.GetPosition();
        ctp1 = ctp1 - beholder.GetRightVector() * ctp1.Dot(beholder.GetRightVector());
        math::Vector<SceneSpaceType, 3> ctp2 = tp2 - beholder.GetPosition();
        ctp2 = ctp2 - beholder.GetRightVector() * ctp2.Dot(beholder.GetRightVector());
        math::Vector<SceneSpaceType, 3> ctp3 = tp3 - beholder.GetPosition();
        ctp3 = ctp3 - beholder.GetRightVector() * ctp3.Dot(beholder.GetRightVector());
        math::Vector<SceneSpaceType, 3> ctp4 = tp4 - beholder.GetPosition();
        ctp4 = ctp4 - beholder.GetRightVector() * ctp4.Dot(beholder.GetRightVector());

        // angles between the corner vectors and the view vector
        math::AngleRad a1 = ctp1.Angle(beholder.GetFrontVector());
        math::AngleRad a2 = ctp2.Angle(beholder.GetFrontVector());
        math::AngleRad a3 = ctp3.Angle(beholder.GetFrontVector());
        math::AngleRad a4 = ctp4.Angle(beholder.GetFrontVector());

        // correct signs of the angles
        if (ctp1.Dot(beholder.GetUpVector()) < 0.0f) a1 = -a1;
        if (ctp2.Dot(beholder.GetUpVector()) < 0.0f) a2 = -a2;
        if (ctp3.Dot(beholder.GetUpVector()) < 0.0f) a3 = -a3;
        if (ctp4.Dot(beholder.GetUpVector()) < 0.0f) a4 = -a4;

        // find the centered angle
        angle = (math::Max(math::Max(a1, a2), math::Max(a3, a4)) 
            + math::Min(math::Min(a1, a2), math::Min(a3, a4))) * 0.5f;

        // vertical tilt
        lookAtDir = beholder.GetFrontVector() + beholder.GetUpVector() * tan(angle);
        lookAtDir.Normalise();
        beholder.SetLookAt(beholder.GetPosition() + (lookAtDir * lookAtDist)); 

        // target corner vectors in camera space (right, up, front)
        ctp1 = tp1 - beholder.GetPosition();
        ctp1.Set(beholder.GetRightVector().Dot(ctp1), 
            beholder.GetUpVector().Dot(ctp1), 
            beholder.GetFrontVector().Dot(ctp1));
        ctp2 = tp2 - beholder.GetPosition();
        ctp2.Set(beholder.GetRightVector().Dot(ctp2), 
            beholder.GetUpVector().Dot(ctp2), 
            beholder.GetFrontVector().Dot(ctp2));
        ctp3 = tp3 - beholder.GetPosition();
        ctp3.Set(beholder.GetRightVector().Dot(ctp3), 
            beholder.GetUpVector().Dot(ctp3), 
            beholder.GetFrontVector().Dot(ctp3));
        ctp4 = tp4 - beholder.GetPosition();
        ctp4.Set(beholder.GetRightVector().Dot(ctp4), 
            beholder.GetUpVector().Dot(ctp4), 
            beholder.GetFrontVector().Dot(ctp4));

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
        a1 = static_cast<math::AngleRad>(math::Abs(atan(
            ctp1.X() * cam.GetVirtualHeight() / cam.GetVirtualWidth())));
        a2 = static_cast<math::AngleRad>(math::Abs(atan(
            ctp2.X() * cam.GetVirtualHeight() / cam.GetVirtualWidth())));
        a3 = static_cast<math::AngleRad>(math::Abs(atan(
            ctp3.X() * cam.GetVirtualHeight() / cam.GetVirtualWidth())));
        a4 = static_cast<math::AngleRad>(math::Abs(atan(
            ctp4.X() * cam.GetVirtualHeight() / cam.GetVirtualWidth())));
        angle = math::Max(angle, math::Max(math::Max(a1, a2), math::Max(a3, a4)));

        if (this->type == ZOOM_PAN_ZOOM) {
            // Zoom:

            // set the new aperture angle
            cam.SetApertureAngle(math::AngleRad2Deg(angle * 2.0f));

        } else { // ZOOM_PAN_DOLLY
            // Dolly:

            SceneSpaceType dist = cam.GetFocalDistance() * 
                (tan(angle) / tan(cam.GetHalfApertureAngleRad()));

            SceneSpaceType movePos = (cam.GetFocalDistance() - dist);
            SceneSpaceType moveTar = movePos;
            
            beholder.SetView(beholder.GetPosition() + beholder.GetFrontVector() * movePos,
                beholder.GetLookAt() + beholder.GetFrontVector() * moveTar,
                beholder.GetUpVector());

            if (this->fixFocus) {
                cam.SetFocalDistance(dist);
            }
        }
    }
}
