/*
 * BeholderDistZoom2D.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/BeholderDistZoom2D.h"
#include "vislib/Cursor2D.h"
#include "vislib/Camera.h"
#include "vislib/assert.h"
#include "vislib/Trace.h"
#include "vislib/IllegalParamException.h"
#include <cmath>


/*
 * vislib::graphics::BeholderDistZoom2D::BeholderDistZoom2D
 */
vislib::graphics::BeholderDistZoom2D::BeholderDistZoom2D(void) 
        : AbstractCursor2DEvent(), AbstractBeholderController(), drag(false), squareMinDist(1.0f), scale(1.0f) {
}


/*
 * vislib::graphics::BeholderDistZoom2D::~BeholderDistZoom2D
 */
vislib::graphics::BeholderDistZoom2D::~BeholderDistZoom2D(void) {
}


/*
 * vislib::graphics::BeholderRotator2D::Trigger
 */
void vislib::graphics::BeholderDistZoom2D::Trigger(AbstractCursor *caller, TriggerReason reason, unsigned int param) {
    if (reason == REASON_BUTTON_DOWN) {
        this->drag = true;
    } else if (reason == REASON_MOVE) {
        if (this->drag) {
            vislib::graphics::Cursor2D *cursor = dynamic_cast<vislib::graphics::Cursor2D*>(caller);
            vislib::graphics::Beholder *beholder = this->GetBeholder();
            ASSERT(cursor != NULL);

            if ((beholder == NULL) || (cursor->GetCamera() == NULL)) {
                TRACE(vislib::Trace::LEVEL_WARN, "BeholderDistZoom2D::Trigger beholder or camera missing.");
                return;
            }

            float delta = (cursor->PreviousY() - cursor->Y()) / cursor->GetCamera()->GetVirtualHeight();

            delta *= scale;

            vislib::math::Vector<vislib::graphics::SceneSpaceType, 3> deltaV = beholder->GetFrontVector() * delta;
            vislib::math::Point<vislib::graphics::SceneSpaceType, 3> nPos = beholder->GetPosition() + deltaV;

            if ((nPos - beholder->GetLookAt()).SquareLength() >= this->squareMinDist) {
                beholder->SetPosition(nPos);
            }
        }
    } else if (reason == REASON_BUTTON_UP) {
        this->drag = false;
    }
}


/*
 * vislib::graphics::BeholderDistZoom2D::GetMinDist
 */
vislib::graphics::SceneSpaceType vislib::graphics::BeholderDistZoom2D::GetMinDist(void) {
    return sqrt(this->squareMinDist);
}


/*
 * vislib::graphics::BeholderDistZoom2D::SetMinDist
 */
void vislib::graphics::BeholderDistZoom2D::SetMinDist(vislib::graphics::SceneSpaceType dist) {
    if (dist <= 0.0f) {
        throw IllegalParamException("dist", __FILE__, __LINE__);
    }
    this->squareMinDist = dist * dist;
}


/*
 * vislib::graphics::BeholderDistZoom2D::SetSpeedScaling
 */
void vislib::graphics::BeholderDistZoom2D::SetSpeedScaling(float speed) {
    if (speed <= 0.0f) {
        throw IllegalParamException("speed", __FILE__, __LINE__);
    }
    this->scale = speed;
}
