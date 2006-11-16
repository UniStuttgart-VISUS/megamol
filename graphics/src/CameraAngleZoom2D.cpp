/*
 * CameraAngleZoom2D.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/CameraAngleZoom2D.h"
#include "vislib/IllegalParamException.h"
#include "vislib/Camera.h"
#include "vislib/Cursor2D.h"
#include "vislib/Trace.h"
#include "vislib/assert.h"


/*
 * vislib::graphics::CameraAngleZoom2D::CameraAngleZoom2D
 */
vislib::graphics::CameraAngleZoom2D::CameraAngleZoom2D(void) 
        : AbstractCursor2DEvent(), AbstractCameraController(), drag(false), minAngle(5.0f), maxAngle(175.0f) {
}


/*
 * vislib::graphics::CameraAngleZoom2D::~CameraAngleZoom2D
 */
vislib::graphics::CameraAngleZoom2D::~CameraAngleZoom2D(void) {
}


/*
 * vislib::graphics::CameraAngleZoom2D::Trigger
 */
void vislib::graphics::CameraAngleZoom2D::Trigger(AbstractCursor *caller, TriggerReason reason, unsigned int param) {
    if (reason == REASON_BUTTON_DOWN) {
        this->drag = true;
    } else if (reason == REASON_MOVE) {
        if (this->drag) {
            vislib::graphics::Cursor2D *cursor = dynamic_cast<vislib::graphics::Cursor2D*>(caller);
            vislib::graphics::Camera *cam = this->GetCamera();

            if (cam == NULL) {
                TRACE(vislib::Trace::LEVEL_WARN, "CameraAngleZoom2D::Trigger camera missing.");
                return;
            }

            ASSERT(cursor != NULL);
            vislib::math::AngleDeg a = 
                ((cursor->PreviousY() - cursor->Y()) / cam->GetVirtualHeight() * (this->maxAngle - this->minAngle))
                + cam->GetApertureAngle();

            if (a < this->minAngle) a = this->minAngle;
            if (a > this->maxAngle) a = this->maxAngle;

            cam->SetApertureAngle(a);
        }
    } else if (reason == REASON_BUTTON_UP) {
        this->drag = false;
    }
}


/*
 * vislib::graphics::CameraAngleZoom2D::SetMinApertureAngle
 */
void vislib::graphics::CameraAngleZoom2D::SetMinApertureAngle(vislib::math::AngleDeg angle) {
    if ((angle <= 0.0f) || (angle >= this->maxAngle)) {
        throw IllegalParamException("angle", __FILE__, __LINE__);
    }
    this->minAngle = angle;
}


/*
 * vislib::graphics::CameraAngleZoom2D::SetMaxApertureAngle
 */
void vislib::graphics::CameraAngleZoom2D::SetMaxApertureAngle(vislib::math::AngleDeg angle) {
    if ((angle <= this->minAngle) || (angle >= 180.0f)) {
        throw IllegalParamException("angle", __FILE__, __LINE__);
    }
    this->maxAngle = angle;
}
