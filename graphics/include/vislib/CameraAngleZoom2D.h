/*
 * CameraAngleZoom2D.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CAMERAANGLEZOOM2D_H_INCLUDED
#define VISLIB_CAMERAANGLEZOOM2D_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/AbstractCameraController.h"
#include "vislib/AbstractCursor2DEvent.h"
#include "vislib/mathtypes.h"


namespace vislib {
namespace graphics {


    /**
     * Cursor 2D event manipulating the camera aperture angle by the Y 
     * coordinate of the 2d cursor to zoom in and out.
     */
    class CameraAngleZoom2D : public AbstractCameraController, public AbstractCursor2DEvent {

    public:

        /** Ctor. */
        CameraAngleZoom2D(void);

        /** Dtor. */
        ~CameraAngleZoom2D(void);

        /**
         * Callback methode called by a Cursor2D this event is registered to 
         * when a cursor event occures. REASON_BUTTON_DOWN, REASON_MOVE and
         * REASON_BUTTON_UP control the mouse position tracking. Changes to
         * the y coordinate of the mouse position will change the aperture 
         * angle of the associated camera. Othere calling reasons have no 
         * effect.
         *
         * @param caller The AbstractCursor object which calles this methode.
         * @param reason The reason why for this call.
         * @param param A reason depending parameter.
         */
        virtual void Trigger(AbstractCursor *caller, TriggerReason reason, unsigned int param);

        /**
         * Return the minimum value for the aperture angle.
         *
         * @return The minimum value for the aperture angle.
         */
        inline vislib::math::AngleDeg GetMinApertureAngle(void) {
            return this->minAngle;
        }

        /**
         * Return the maximum value for the aperture angle.
         *
         * @return The maximum value for the aperture angle.
         */
        inline vislib::math::AngleDeg GetMaxApertureAngle(void) {
            return this->maxAngle;
        }

        /**
         * Sets the minimum value for the aperture angle.
         *
         * @param angle The new minimum value for the aperture angle.
         *
         * @throws IllegalParamException if the angle specified is not more 
         *         then Zero, and less then maximum aperture angle.
         */
        void SetMinApertureAngle(vislib::math::AngleDeg angle);

        /**
         * Sets the maximum value for the aperture angle.
         *
         * @param angle The new maximum value for the aperture angle.
         *
         * @throws IllegalParamException if the angle specified is not more 
         *         then minimum aperture, and less then 180 degree.
         */
        void SetMaxApertureAngle(vislib::math::AngleDeg angle);
        
    private:

        /** Flag whether the mouse draggs. */
        bool drag;

        /** minimum value for the aperture angle */
        vislib::math::AngleDeg minAngle;

        /** maximum value for the aperture angle */
        vislib::math::AngleDeg maxAngle;
    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#endif /* VISLIB_CAMERAANGLEZOOM2D_H_INCLUDED */

