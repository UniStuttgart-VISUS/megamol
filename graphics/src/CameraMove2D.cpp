/*
 * CameraMove2D.cpp
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009, Sebastian Grottel. All rights reserved.
 */

#include "vislib/CameraMove2D.h"
#include "vislib/assert.h"
#include "vislib/Cursor2D.h"
#include "vislib/graphicstypes.h"
#include "vislib/Trace.h"


/*
 * vislib::graphics::CameraMove2D::CameraMove2D
 */
vislib::graphics::CameraMove2D::CameraMove2D(const SmartPtr<CameraParameters>&
        cameraParams) : AbstractCursor2DEvent(),
        AbstractCameraController(cameraParams), drag(false) {
    // Intentionally empty
}


/*
 * vislib::graphics::CameraMove2D::~CameraMove2D
 */
vislib::graphics::CameraMove2D::~CameraMove2D(void) {
    // Intentionally empty
}


/*
 * vislib::graphics::CameraMove2D::Trigger
 */
void vislib::graphics::CameraMove2D::Trigger(
        vislib::graphics::AbstractCursor *caller, 
        vislib::graphics::AbstractCursorEvent::TriggerReason reason, 
        unsigned int param) {

    if (reason == REASON_BUTTON_DOWN) {
        this->drag = true;
    } else if (reason == REASON_BUTTON_UP) {
        this->drag = false;
    } else if ((reason == REASON_MOVE) && this->drag) {
        vislib::graphics::Cursor2D *cursor 
            = dynamic_cast<vislib::graphics::Cursor2D*>(caller);

        // otherwise this would be very strange:
        ASSERT(cursor->CameraParams()->IsSimilar(this->CameraParams()));

        if (!this->IsCameraParamsValid()) {
            VLTRACE(vislib::Trace::LEVEL_WARN, 
                "CameraMove2D::Trigger camera missing.");
            return;
        }

        SceneSpaceVector3D mov(this->CameraParams()->Right());
        mov *= (cursor->PreviousX() - cursor->X());

        SceneSpaceVector3D tmp(this->CameraParams()->Up());
        tmp *= (cursor->PreviousY() - cursor->Y());
        mov += tmp;

        mov *= (this->CameraParams()->FocalDistance()
            * tan(this->CameraParams()->HalfApertureAngle()))
            / (this->CameraParams()->VirtualViewSize().Height() * 0.5f);

        this->CameraParams()->SetView(
            this->CameraParams()->Position() + mov,
            this->CameraParams()->LookAt() + mov,
            this->CameraParams()->Up());
    }
}
