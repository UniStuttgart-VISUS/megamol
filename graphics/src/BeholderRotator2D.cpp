/*
 * BeholderRotator2D.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/BeholderRotator2D.h"
#include "vislib/Camera.h"
#include "vislib/assert.h"
#include "vislib/Cursor2D.h"
#include "vislib/Trace.h"
#include "vislib/Quaternion.h"
#include <stdio.h>


/*
 * vislib::graphics::BeholderRotator2D::BeholderRotator2D
 */
vislib::graphics::BeholderRotator2D::BeholderRotator2D(void)
        : AbstractCursor2DEvent(), AbstractBeholderController(), altMod(UINT_MAX) {
}


/*
 * vislib::graphics::BeholderRotator2D::~BeholderRotator2D
 */
vislib::graphics::BeholderRotator2D::~BeholderRotator2D(void) {
}


/*
 * vislib::graphics::BeholderRotator2D::Trigger
 */
void vislib::graphics::BeholderRotator2D::Trigger(AbstractCursor *caller, TriggerReason reason, unsigned int param) {
    Cursor2D *cursor = dynamic_cast<Cursor2D *>(caller);
    ASSERT(cursor != NULL);
    Camera *cam = cursor->GetCamera();
    Beholder *beh = this->GetBeholder();
    ImageSpaceType curX, curY;

    if ((beh == NULL) || (cam == NULL)) {
        TRACE(vislib::Trace::LEVEL_WARN, "BeholderRotator2D::Trigger beholer or camera missing.");
        return;
    }

    if ((reason == REASON_BUTTON_DOWN) || (reason == REASON_MOVE)) {
        ImageSpaceType halfHeight = cam->GetVirtualHeight() * static_cast<ImageSpaceType>(0.5);

        // calc mouse vector in view space
        curX = cursor->X() - cam->GetVirtualWidth() * static_cast<ImageSpaceType>(0.5);
        curY = cursor->Y() - halfHeight;

        if (reason == REASON_BUTTON_DOWN) {
            this->drag = true;
            this->dragX = curX;
            this->dragY = curY;
        } else if (this->drag) {

            if (cursor->GetModifierState(this->altMod)) {
                // roll

                // calc angle between mouse position vectors in image space
                math::AngleRad angle = ::atan2(curY, curX) - ::atan2(this->dragY, this->dragX);

                // recaluclate the up vector
                math::Vector3D<SceneSpaceType> up
                    = beh->GetRightVector() * static_cast<SceneSpaceType>(::sin(angle)) 
                    + beh->GetUpVector() * static_cast<SceneSpaceType>(::cos(angle));

                // set the new up vector
                beh->SetUpVector(up);

                // keep on dragging
                this->dragX = curX;
                this->dragY = curY;

            } else {
                // pitch & yaw

                // Big phat rotation crowbar ...
                // TODO: 
                //  Rotation problem: Moving the mouse in clockwise circles 
                //  around the the mid-point of the upper half window, the beholder
                //  is rotatet around the view axis. Why???
                //
                // This rotation is similar to the BeholderLookAtRotator2D 
                // rotation. There the "roll"-effect is irrelevant but here
                // the people are getting seasick.

                math::Vector3D<SceneSpaceType> rot = (beh->GetRightVector() * (curX - this->dragX)) 
                    + (beh->GetUpVector() * (curY - this->dragY));

                math::AngleRad angle = rot.Normalise() / halfHeight * cam->GetHalfApertureAngleRad();

                rot = rot.Cross(beh->GetFrontVector());

                math::Quaternion<SceneSpaceType> quat(angle, rot);

                math::Vector3D<SceneSpaceType> up = beh->GetUpVector();
                math::Vector3D<SceneSpaceType> look = beh->GetLookAt() - beh->GetPosition();
                math::Point3D<SceneSpaceType> pos = beh->GetPosition();

                up = quat * up;
                look = quat * look;

                beh->SetView(pos, static_cast<math::Point3D<SceneSpaceType>&>(pos + look), up);

                // keep on dragging
                this->dragX = curX;
                this->dragY = curY;
            }
        }
    } else if (reason == REASON_BUTTON_UP) {
        this->drag = false;
    }
}
