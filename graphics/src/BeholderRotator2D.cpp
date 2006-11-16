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
#include <climits>


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
    ImageSpaceType preX, preY;

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
        } else if (this->drag) {
            preX = cursor->PreviousX() - cam->GetVirtualWidth() * static_cast<ImageSpaceType>(0.5);
            preY = cursor->PreviousY() - halfHeight;

            if (cursor->GetModifierState(this->altMod)) {
                // roll

                // calc angle between mouse position vectors in image space
                math::AngleRad angle = ::atan2(curY, curX) - ::atan2(preY, preX);

                // recaluclate the up vector
                math::Vector<SceneSpaceType, 3> up
                    = beh->GetRightVector() * static_cast<SceneSpaceType>(::sin(angle)) 
                    + beh->GetUpVector() * static_cast<SceneSpaceType>(::cos(angle));

                // set the new up vector
                beh->SetUpVector(up);

            } else {
                // pitch & yaw

                // Big phat rotation crowbar ...
                // TODO: reimplement crowbar
                //  Rotation problem: Moving the mouse in clockwise circles 
                //  around the the mid-point of the upper half window, the beholder
                //  is rotatet around the view axis. Why???
                //
                // This rotation is similar to the BeholderLookAtRotator2D 
                // rotation. There the "roll"-effect is irrelevant but here
                // the people are getting seasick.

                math::Vector<SceneSpaceType, 3> rot = (beh->GetRightVector() * (curX - preX)) 
                    + (beh->GetUpVector() * (curY - preY));

                math::AngleRad angle = rot.Normalise() / halfHeight * cam->GetHalfApertureAngleRad();

                rot = rot.Cross(beh->GetFrontVector());

                math::Quaternion<SceneSpaceType> quat(angle, rot);

                math::Vector<SceneSpaceType, 3> up = beh->GetUpVector();
                math::Vector<SceneSpaceType, 3> look = beh->GetLookAt() - beh->GetPosition();
                math::Point<SceneSpaceType, 3> pos = beh->GetPosition();

                up = quat * up;
                look = quat * look;

                math::Point<SceneSpaceType, 3> lookAt = pos + look;
                beh->SetView(pos, lookAt, up);
            }
        }
    } else if (reason == REASON_BUTTON_UP) {
        this->drag = false;
    }
}
