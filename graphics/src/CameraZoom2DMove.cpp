/*
 * CameraZoom2DMove.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */

#include "vislib/CameraZoom2DMove.h"
#include <cfloat>
#include "vislib/CameraParameters.h"
#include "vislib/Cursor2D.h"
#include "vislib/Point.h"
#include "vislib/SmartPtr.h"
#include "vislib/Trace.h"
#include "vislib/Vector.h"


/*
 * vislib::graphics::CameraZoom2DMove::CameraZoom2DMove
 */
vislib::graphics::CameraZoom2DMove::CameraZoom2DMove(
        const vislib::SmartPtr<vislib::graphics::CameraParameters>& 
        cameraParams) : AbstractCursor2DEvent(), 
        AbstractCameraController(cameraParams), drag(false), speed(1.0f), 
        behaviour(FIX_LOOK_AT) {
}


/*
 * vislib::graphics::CameraZoom2DMove::~CameraZoom2DMove
 */
vislib::graphics::CameraZoom2DMove::~CameraZoom2DMove(void) {
    // Intentionally Empty
}


/*
 * vislib::graphics::CameraZoom2DMove::SetZoomBehaviour
 */
void vislib::graphics::CameraZoom2DMove::SetZoomBehaviour(
        vislib::graphics::CameraZoom2DMove::ZoomBehaviourType behaviour) {
    this->behaviour = behaviour;
}


/*
 * vislib::graphics::CameraZoom2DMove::SetSpeed
 */
void vislib::graphics::CameraZoom2DMove::SetSpeed(
        vislib::graphics::SceneSpaceType speed) {
    this->speed = speed;
}


/*
 * vislib::graphics::CameraZoom2DMove::Trigger
 */
void vislib::graphics::CameraZoom2DMove::Trigger(
        vislib::graphics::AbstractCursor *caller, 
        vislib::graphics::AbstractCursorEvent::TriggerReason reason, 
        unsigned int param) {

    if (reason == REASON_BUTTON_DOWN) {
        this->drag = true;

    } else if (reason == REASON_MOVE) {
        if (this->drag) {
            vislib::graphics::Cursor2D *cursor 
                = dynamic_cast<vislib::graphics::Cursor2D*>(caller);

            // otherwise this would be very strange:
            ASSERT(cursor->CameraParams()->IsSimilar(this->CameraParams()));

            if (!this->IsCameraParamsValid()) {
                VLTRACE(vislib::Trace::LEVEL_WARN, 
                    "CameraZoom2DMove::Trigger camera missing.");
                return;
            }

            math::Point<SceneSpaceType, 3> pos = this->CameraParams()->Position();
            math::Point<SceneSpaceType, 3> lookAt = this->CameraParams()->LookAt();
            math::Vector<SceneSpaceType, 3> front = this->CameraParams()->Front();

            SceneSpaceType dist = (lookAt - pos).Length();
            SceneSpaceType delta = SceneSpaceType(cursor->Y() 
                - cursor->PreviousY()) / SceneSpaceType(
                this->CameraParams()->VirtualViewSize().Height())
                * SceneSpaceType(this->speed);
            bool setLookAt = false;

            SceneSpaceType minDist = FLT_MIN;
            if (!this->CameraParams()->Limits().IsNull()) {
                minDist = this->CameraParams()->Limits()->MinLookAtDist();
            }

            if ((this->behaviour == FIX_DISTANCE) 
                    || (dist - delta < minDist)) {
                // moved in too close
                switch (this->behaviour) {
                    case FIX_LOOK_AT:
                        // keep look-at-point fixed: clamp delta
                        delta = dist - minDist;
                        break;
                    case FIX_DISTANCE:
                        // keep distance fixed: also move look-at-point
                        lookAt += (front * delta);
                        setLookAt = true;
                        break;
                    case MOVE_IF_CLOSE:
                        // move look-at-point to keep the min distance
                        lookAt += (front * (delta - dist + minDist));
                        setLookAt = true;
                        break;
                }
            }

            pos += (front * delta);

            if (setLookAt) {
                this->CameraParams()->SetView(pos, lookAt, 
                    this->CameraParams()->Up());
            } else {
                this->CameraParams()->SetPosition(pos);
            }
        }

    } else if (reason == REASON_BUTTON_UP) {
        this->drag = false;

    }
}
