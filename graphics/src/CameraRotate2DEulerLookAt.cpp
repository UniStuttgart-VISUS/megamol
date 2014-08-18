/*
 * CameraRotate2DEulerLookAt.cpp
 *
 * Copyright (C) 2011 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/CameraRotate2DEulerLookAt.h"
#define _USE_MATH_DEFINES
#include "vislib/assert.h"
#include "vislib/Cursor2D.h"
#include "vislib/graphicstypes.h"
#include "vislib/Quaternion.h"
#include "vislib/Trace.h"
#include <cmath>
#include <climits>


/*
 * vislib::graphics::CameraRotate2DEulerLookAt::CameraRotate2DEulerLookAt
 */
vislib::graphics::CameraRotate2DEulerLookAt::CameraRotate2DEulerLookAt(
        const SmartPtr<CameraParameters>& cameraParams)
        : AbstractCursor2DEvent(), AbstractCameraController(cameraParams),
        altMod(UINT_MAX), drag(false), setBaseOnRoll(true) {
    this->angle[0] = 0.0f;
    this->angle[1] = 0.0f;
    this->angle[2] = 0.0f;
    // Intentionally empty
}


/*
 * vislib::graphics::CameraRotate2DEulerLookAt::~CameraRotate2DEulerLookAt
 */
vislib::graphics::CameraRotate2DEulerLookAt::~CameraRotate2DEulerLookAt(void) {
    // Intentionally empty
}


/*
 * vislib::graphics::CameraRotate2DEulerLookAt::ResetOrientation
 */
void vislib::graphics::CameraRotate2DEulerLookAt::ResetOrientation(void) {

    SceneSpacePoint3D lat = this->CameraParams()->LookAt();
    SceneSpaceVector3D vx = this->CameraParams()->Right();
    SceneSpaceVector3D vy = this->CameraParams()->Front();
    SceneSpaceVector3D vz = this->CameraParams()->Up();
    SceneSpaceType dist = (this->CameraParams()->Position() - lat).Length();

    math::Quaternion<SceneSpaceType> rollBack;
    math::Quaternion<SceneSpaceType> r2;

    rollBack.Set(this->angle[1], vx);
    r2.Set(this->angle[0], vz);
    rollBack = r2 * rollBack;
    r2.Set(this->angle[2], vy);
    rollBack = rollBack * r2;
    rollBack.Invert();

    vx = rollBack * vx;
    vy = rollBack * vy;
    vz = rollBack * vz;

    this->angle[0] = this->angle[1] = this->angle[2] = 0.0f;

    vy *= dist;
    this->CameraParams()->SetView(lat - vy, lat, vz); 

}


/*
 * vislib::graphics::CameraRotate2DEulerLookAt::Trigger
 */
void vislib::graphics::CameraRotate2DEulerLookAt::Trigger(
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

            SceneSpacePoint3D lat = this->CameraParams()->LookAt();
            SceneSpaceVector3D vx = this->CameraParams()->Right();
            SceneSpaceVector3D vy = this->CameraParams()->Front();
            SceneSpaceVector3D vz = this->CameraParams()->Up();
            SceneSpaceType dist = (this->CameraParams()->Position() - lat).Length();

            math::Quaternion<SceneSpaceType> rollBack;
            math::Quaternion<SceneSpaceType> r2;

            rollBack.Set(this->angle[1], vx);
            r2.Set(this->angle[0], vz);
            rollBack = r2 * rollBack;
            r2.Set(this->angle[2], vy);
            rollBack = rollBack * r2;
            rollBack.Invert();

            vx = rollBack * vx;
            vy = rollBack * vy;
            vz = rollBack * vz;

            if (alt) {
                this->angle[2] -=
                    static_cast<float>(M_PI * (preX - curX) / halfWidth)
                    + static_cast<float>(M_PI * (preY - curY) / halfHeight);
                while (this->angle[2] > static_cast<float>(M_PI)) this->angle[2] -= static_cast<float>(2.0 * M_PI);
                while (this->angle[2] < static_cast<float>(M_PI)) this->angle[2] += static_cast<float>(2.0 * M_PI);

            } else {
                this->angle[0] += static_cast<float>(M_PI * (preX - curX) / halfWidth);
                while (this->angle[0] > static_cast<float>(M_PI)) this->angle[0] -= static_cast<float>(2.0 * M_PI);
                while (this->angle[0] < static_cast<float>(M_PI)) this->angle[0] += static_cast<float>(2.0 * M_PI);
                this->angle[1] -= static_cast<float>(M_PI * (preY - curY) / halfHeight);
                if (this->angle[1] > static_cast<float>(0.5 * M_PI)) this->angle[1] = static_cast<float>(0.5 * M_PI);
                if (this->angle[1] < -static_cast<float>(0.5 * M_PI)) this->angle[1] = -static_cast<float>(0.5 * M_PI);
            }

            rollBack.Set(this->angle[1], vx);
            r2.Set(this->angle[0], vz);
            rollBack = r2 * rollBack;
            r2.Set(this->angle[2], vy);
            rollBack = rollBack * r2;

            if (alt && this->setBaseOnRoll) {
                this->angle[0] = this->angle[1] = this->angle[2] = 0.0f;
            }

            vx = rollBack * vx;
            vy = rollBack * vy;
            vz = rollBack * vz;

            vy *= dist;
            this->CameraParams()->SetView(lat - vy, lat, vz); 

        }

    } else if (reason == REASON_BUTTON_UP) {
        // leave drag mode
        this->drag = false;

    }
}


/*
 * vislib::graphics::CameraRotate2DEulerLookAt::UpdateBaseOrientation
 */
void vislib::graphics::CameraRotate2DEulerLookAt::UpdateBaseOrientation(void) {
    this->angle[0] = this->angle[1] = this->angle[2] = 0.0f;
}
