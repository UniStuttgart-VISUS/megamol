/*
 * CameraRotate2DLookAt.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */

#include "vislib/CameraRotate2DLookAt.h"
#include <climits>
#include "vislib/Cursor2D.h"
#include "vislib/Trace.h"
#include "vislib/Quaternion.h"


/*
 * vislib::graphics::CameraRotate2DLookAt::CameraRotate2DLookAt
 */
vislib::graphics::CameraRotate2DLookAt::CameraRotate2DLookAt(
        const SmartPtr<CameraParameters>& cameraParams) : 
        AbstractCursor2DEvent(), AbstractCameraController(cameraParams), 
        altMod(UINT_MAX), drag(false) {
}


/*
 * vislib::graphics::CameraRotate2DLookAt::~CameraRotate2DLookAt
 */
vislib::graphics::CameraRotate2DLookAt::~CameraRotate2DLookAt(void) {
    // Intentionally empty
}


/*
 * vislib::graphics::CameraRotate2DLookAt::Trigger
 */
void vislib::graphics::CameraRotate2DLookAt::Trigger(
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
            "CameraRotate2DLookAt::Trigger camera missing.");
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
            bool alt = false;
            if (cursor->GetInputModifiers() != NULL) {
                alt = cursor->GetInputModifiers()->GetModifierState(this->altMod);
            }

            // calc mouse vector in view space
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

                // something almost like arc-ball ...

                // mouse-move vector in scene space
                math::Vector<SceneSpaceType, 3> rot 
                    = (this->CameraParams()->Right() * (preX - curX)) 
                    + (this->CameraParams()->Up() * (preY - curY));

                // rotation speed: moving the mouse along the whole window 
                // height yields to an rotation of 360°
                math::AngleRad angle = static_cast<math::AngleRad>(
                    rot.Normalise() * math::PI_DOUBLE / halfHeight);

                // rotation axis is perpendicular to mouse-move vector in 
                // image space
                rot = rot.Cross(this->CameraParams()->Front());

                // setup rotation quaternion.
                math::Quaternion<SceneSpaceType> quat(angle, rot);

                // fetch current view values
                math::Vector<SceneSpaceType, 3> up 
                    = this->CameraParams()->Up();
                math::Vector<SceneSpaceType, 3> antiLook 
                    = this->CameraParams()->Position()
                    - this->CameraParams()->LookAt();
                math::Point<SceneSpaceType, 3> look 
                    = this->CameraParams()->LookAt();

                // rotate current view
                up = quat * up;
                antiLook = quat * antiLook;

                // set new view
                this->CameraParams()->SetView(look + antiLook, look, up);
            }
        }

    } else if (reason == REASON_BUTTON_UP) {
        // leave drag mode
        this->drag = false;

    }
}
