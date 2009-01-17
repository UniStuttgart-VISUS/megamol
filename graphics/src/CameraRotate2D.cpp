/*
 * CameraRotate2D.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */

#include "vislib/CameraRotate2D.h"
#include <climits>
#include "vislib/Cursor2D.h"
#include "vislib/Trace.h"
#include "vislib/Quaternion.h"


/*
 * vislib::graphics::CameraRotate2D::CameraRotate2D
 */
vislib::graphics::CameraRotate2D::CameraRotate2D(
        const SmartPtr<CameraParameters>& cameraParams) : 
        AbstractCursor2DEvent(), AbstractCameraController(cameraParams), 
        altMod(UINT_MAX), drag(false) {
}


/*
 * vislib::graphics::CameraRotate2D::~CameraRotate2D
 */
vislib::graphics::CameraRotate2D::~CameraRotate2D(void) {
    // Intentionally empty
}


/*
 * vislib::graphics::CameraRotate2D::Trigger
 */
void vislib::graphics::CameraRotate2D::Trigger(
        vislib::graphics::AbstractCursor *caller,
        vislib::graphics::AbstractCursorEvent::TriggerReason reason,
        unsigned int param) {
    Cursor2D *cursor = dynamic_cast<Cursor2D *>(caller);
    ASSERT(cursor != NULL);
    
    // otherwise this would be very strange:
    ASSERT(cursor->CameraParams()->IsSimilar(this->CameraParams()));

    ImageSpaceType curX, curY;
    ImageSpaceType preX, preY;

    if (!this->IsCameraParamsValid()) {
        VLTRACE(vislib::Trace::LEVEL_WARN, 
            "CameraRotate2D::Trigger camera missing.");
        return;
    }

    if ((reason == REASON_BUTTON_DOWN) || (reason == REASON_MOVE)) {
        ImageSpaceType halfWidth = this->CameraParams()
            ->VirtualViewSize().Width() * static_cast<ImageSpaceType>(0.5);
        ImageSpaceType halfHeight = this->CameraParams()
            ->VirtualViewSize().Height() * static_cast<ImageSpaceType>(0.5);

        // calc mouse vector in view space
        curX = cursor->X() - halfWidth;
        curY = cursor->Y() - halfHeight;

        if (reason == REASON_BUTTON_DOWN) {
            // start drag mode
            this->drag = true;

        } else if (this->drag) {
            // alternate camera parameters
            bool alt = false;

            if ((cursor->GetInputModifiers() != NULL) 
                    && (cursor->GetInputModifiers()->GetModifierCount() 
                    > this->altMod)) {
                alt = cursor->GetInputModifiers()->GetModifierState(
                    this->altMod);
            }

            preX = cursor->PreviousX() - halfWidth;
            preY = cursor->PreviousY() - halfHeight;

            if (alt) {
                // roll

                // calc angle between mouse position vectors in image space
                math::AngleRad angle = ::atan2(curY, curX) 
                    - ::atan2(preY, preX);

                // recaluclate the up vector
                math::Vector<SceneSpaceType, 3> up
                    = this->CameraParams()->Right() 
                        * static_cast<SceneSpaceType>(::sin(angle)) 
                    + this->CameraParams()->Up() 
                        * static_cast<SceneSpaceType>(::cos(angle));

                // set the new up vector
                this->CameraParams()->SetUp(up);

            } else {
                // pitch & yaw

                // Big phat rotation crowbar ...
                // TODO: reimplement crowbar
                //  Rotation problem: Moving the mouse in clockwise circles 
                //  around the the mid-point of the upper half window, the 
                //  beholder is rotatet around the view axis. Why???
                //
                // This rotation is similar to the BeholderLookAtRotator2D 
                // rotation. There the "roll"-effect is irrelevant but here
                // the people are getting seasick.

                math::Vector<SceneSpaceType, 3> rot 
                    = (this->CameraParams()->Right() * (curX - preX)) 
                    + (this->CameraParams()->Up() * (curY - preY));

                math::AngleRad angle = static_cast<math::AngleRad>(
                    rot.Normalise() / halfHeight) 
                    * this->CameraParams()->HalfApertureAngle();

                rot = rot.Cross(this->CameraParams()->Front());

                math::Quaternion<SceneSpaceType> quat(angle, rot);

                math::Vector<SceneSpaceType, 3> up 
                    = this->CameraParams()->Up();
                math::Vector<SceneSpaceType, 3> look 
                    = this->CameraParams()->LookAt() 
                    - this->CameraParams()->Position();
                math::Point<SceneSpaceType, 3> pos 
                    = this->CameraParams()->Position();

                up = quat * up;
                look = quat * look;

                this->CameraParams()->SetView(pos, pos + look, up);
            }
        }

    } else if (reason == REASON_BUTTON_UP) {
        // leave drag mode
        this->drag = false;

    }
}
