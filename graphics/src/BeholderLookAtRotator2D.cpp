/*
 * BeholderLookAtRotator2D.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/BeholderLookAtRotator2D.h"
#include "vislib/Camera.h"
#include "vislib/assert.h"
#include "vislib/Cursor2D.h"
#include "vislib/Trace.h"
#include "vislib/Quaternion.h"
#include "vislib/mathtypes.h"
#include <stdio.h>


/*
 * vislib::graphics::BeholderLookAtRotator2D::BeholderLookAtRotator2D
 */
vislib::graphics::BeholderLookAtRotator2D::BeholderLookAtRotator2D(void)
        : AbstractCursor2DEvent(), AbstractBeholderController(), altMod(UINT_MAX) {
}


/*
 * vislib::graphics::BeholderLookAtRotator2D::~BeholderLookAtRotator2D
 */
vislib::graphics::BeholderLookAtRotator2D::~BeholderLookAtRotator2D(void) {
}


/*
 * vislib::graphics::BeholderLookAtRotator2D::Trigger
 */
void vislib::graphics::BeholderLookAtRotator2D::Trigger(AbstractCursor *caller, TriggerReason reason, unsigned int param) {
    Cursor2D *cursor = dynamic_cast<Cursor2D *>(caller);
    ASSERT(cursor != NULL);
    Camera *cam = cursor->GetCamera();
    Beholder *beh = this->GetBeholder();
    ImageSpaceType curX, curY;

    if ((beh == NULL) || (cam == NULL)) {
        TRACE(vislib::Trace::LEVEL_WARN, "BeholderLookAtRotator2D::Trigger beholer or camera missing.");
        return;
    }

    if ((reason == REASON_BUTTON_DOWN) || (reason == REASON_MOVE)) {
        ImageSpaceType halfHeight = cam->GetVirtualHeight() * static_cast<ImageSpaceType>(0.5);
        ImageSpaceType halfWidth = cam->GetVirtualWidth() * static_cast<ImageSpaceType>(0.5);

        // calc mouse vector in view space
        curX = cursor->X() - halfWidth;
        curY = cursor->Y() - halfHeight;

        if (reason == REASON_BUTTON_DOWN) {
            this->drag = true;

            this->dragX = curX;
            this->dragY = curY;

            // initial values.

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

                // something almost like arc-ball ...

                // mouse-move vector in scene space
                math::Vector3D<SceneSpaceType> rot = (beh->GetRightVector() * (this->dragX - curX)) 
                    + (beh->GetUpVector() * (this->dragY - curY));

                // rotation speed: moving the mouse along the whole window 
                // height yields to an rotation of 360°
                math::AngleRad angle = static_cast<math::AngleRad>(rot.Normalise() * math::PI_DOUBLE / halfHeight);

                // rotation axis is perpendicular to mouse-move vector in image space
                rot = rot.Cross(beh->GetFrontVector());

                // setup rotation quaternion.
                math::Quaternion<SceneSpaceType> quat(angle, rot);

                // fetch current view values
                math::Vector3D<SceneSpaceType> up = beh->GetUpVector();
                math::Vector3D<SceneSpaceType> antiLook = beh->GetPosition() - beh->GetLookAt();
                math::Point3D<SceneSpaceType> look = beh->GetLookAt();

                // rotate current view
                up = quat * up;
                antiLook = quat * antiLook;

                // set new view
                beh->SetView(static_cast<math::Point3D<SceneSpaceType>&>(look + antiLook), look, up);

                // keep on dragging
                this->dragX = curX;
                this->dragY = curY;
            }
        }
    } else if (reason == REASON_BUTTON_UP) {
        this->drag = false;
    }
}
