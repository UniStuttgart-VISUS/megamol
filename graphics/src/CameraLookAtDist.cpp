/*
 * CameraLookAtDist.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/CameraLookAtDist.h"
#include "vislib/assert.h"
#include <cfloat>
#include "vislib/CameraParameters.h"
#include "vislib/Cursor2D.h"
#include "vislib/Point.h"
#include "vislib/SmartPtr.h"
#include "vislib/Trace.h"
#include "vislib/Vector.h"


/*
 * vislib::graphics::CameraLookAtDist::CameraLookAtDist
 */
vislib::graphics::CameraLookAtDist::CameraLookAtDist(
        const vislib::SmartPtr<vislib::graphics::CameraParameters>& 
        cameraParams) : AbstractCursor2DEvent(), 
        AbstractCameraController(cameraParams), drag(false), speed(1.0f) {
    // Intentionally empty
}


/*
 * vislib::graphics::CameraLookAtDist::~CameraLookAtDist
 */
vislib::graphics::CameraLookAtDist::~CameraLookAtDist(void) {
    // Intentionally empty
}


/*
 * vislib::graphics::CameraLookAtDist::SetSpeed
 */
void vislib::graphics::CameraLookAtDist::SetSpeed(
        vislib::graphics::SceneSpaceType speed) {
    this->speed = speed;
}


/*
 * vislib::graphics::CameraLookAtDist::Trigger
 */
void vislib::graphics::CameraLookAtDist::Trigger(
        vislib::graphics::AbstractCursor *caller, 
        vislib::graphics::AbstractCursorEvent::TriggerReason reason, 
        unsigned int param) {

    if (reason == REASON_BUTTON_DOWN) {
        this->drag = true;

    } else if (reason == REASON_BUTTON_UP) {
        this->drag = false;

    } else if ((reason == REASON_MOVE) && this->drag) {

        // preconditions
        vislib::graphics::Cursor2D *cursor 
            = dynamic_cast<vislib::graphics::Cursor2D*>(caller);
        // otherwise this would be very strange:
        ASSERT(cursor->CameraParams()->IsSimilar(this->CameraParams()));
        if (!this->IsCameraParamsValid()) {
            VLTRACE(vislib::Trace::LEVEL_WARN, 
                "CameraZoom2DMove::Trigger camera missing.");
            return;
        }

        // geometric informations
        math::Point<SceneSpaceType, 3> pos = this->CameraParams()->Position();
        math::Point<SceneSpaceType, 3> lookAt = this->CameraParams()->LookAt();
        math::Vector<SceneSpaceType, 3> front = this->CameraParams()->Front();

        // distance to move
        SceneSpaceType dist = (lookAt - pos).Length();
        SceneSpaceType delta = SceneSpaceType(cursor->Y() 
            - cursor->PreviousY()) / SceneSpaceType(
            this->CameraParams()->VirtualViewSize().Height())
            * SceneSpaceType(this->speed);

        // minimum distance to keep
        SceneSpaceType minDist = FLT_MIN;
        if (!this->CameraParams()->Limits().IsNull()) {
            minDist = this->CameraParams()->Limits()->MinLookAtDist();
        }
        if ((dist - delta) < minDist) {
            delta = dist - minDist;
        }

        // move
        front *= delta;
        lookAt += front;
        this->CameraParams()->SetLookAt(lookAt);

    }
}
