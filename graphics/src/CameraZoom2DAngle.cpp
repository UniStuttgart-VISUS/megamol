/*
 * CameraZoom2DAngle.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */

#include "vislib/CameraZoom2DAngle.h"
#include <cfloat>
#include "vislib/Cursor2D.h"
#include "vislib/Trace.h"


/*
 * vislib::graphics::CameraZoom2DAngle::CameraZoom2DAngle
 */
vislib::graphics::CameraZoom2DAngle::CameraZoom2DAngle(
        const SmartPtr<CameraParameters>& cameraParams) 
        : AbstractCursor2DEvent(), AbstractCameraController(cameraParams), 
        drag(false) {
}


/*
 * vislib::graphics::CameraZoom2DAngle::~CameraZoom2DAngle
 */
vislib::graphics::CameraZoom2DAngle::~CameraZoom2DAngle(void) {
    // Intentionally empty
}


/*
 * vislib::graphics::CameraZoom2DAngle::Trigger
 */
void vislib::graphics::CameraZoom2DAngle::Trigger(
        vislib::graphics::AbstractCursor *caller,
        vislib::graphics::AbstractCursorEvent::TriggerReason reason,
        unsigned int param) {
    if (reason == REASON_BUTTON_DOWN) {
        this->drag = true;

    } else if (reason == REASON_MOVE) {
        if (this->drag) {
            vislib::graphics::Cursor2D *cursor 
                = dynamic_cast<vislib::graphics::Cursor2D*>(caller);
            float minAngle = FLT_MIN;
            float maxAngle = 180.0f - FLT_MIN;

            // otherwise this would be very strange:
            ASSERT(cursor->CameraParams()->IsSimilar(this->CameraParams()));

            if (!this->IsCameraParamsValid()) {
                VLTRACE(vislib::Trace::LEVEL_WARN, 
                    "CameraAngleZoom2D::Trigger camera missing.");
                return;
            }

            if (!this->CameraParams()->Limits().IsNull()) {
                minAngle = math::AngleRad2Deg(
                    this->CameraParams()->Limits()->MinApertureAngle());
                maxAngle = math::AngleRad2Deg(
                    this->CameraParams()->Limits()->MaxApertureAngle());
            }

            vislib::math::AngleDeg a = 
                ((cursor->PreviousY() - cursor->Y()) 
                / this->CameraParams()->VirtualViewSize().Height() 
                * (maxAngle - minAngle))
                + this->CameraParams()->ApertureAngle();

            if (a < minAngle) a = minAngle;
            if (a > maxAngle) a = maxAngle;

            this->CameraParams()->SetApertureAngle(a);
        }

    } else if (reason == REASON_BUTTON_UP) {
        this->drag = false;

    }
}
