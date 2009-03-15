/*
 * CameraZoom2DMove.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */

#ifndef VISLIB_CAMERAZOOM2DMOVE_H_INCLUDED
#define VISLIB_CAMERAZOOM2DMOVE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/AbstractCameraController.h"
#include "vislib/AbstractCursor2DEvent.h"


namespace vislib {
namespace graphics {


    /**
     * Cursor2DEvent manipulating performing a zoom by moving the camera 
     * position based on the Y coordinate of the 2d cursor.
     */
    class CameraZoom2DMove : public AbstractCursor2DEvent, 
        public AbstractCameraController {
    public:

        /** possible zooming behaviours */
        enum ZoomBehaviourType {
            FIX_LOOK_AT, /**
                          * never move look-at-point. Stop zooming if camera 
                          * too close. (This is the default)
                          */
            FIX_DISTANCE, /**
                           * Always move look-at-point to keep a constant 
                           * distance.
                           */
            MOVE_IF_CLOSE /**
                           * Only move look-at-point if the camera position
                           * would be too close.
                           */
        };

        /** 
         * Ctor. 
         *
         * @param cameraParams The camera parameters object to be rotated.
         */
        CameraZoom2DMove(const SmartPtr<CameraParameters>& cameraParams 
            = SmartPtr<CameraParameters>());

        /** Dtor. */
        virtual ~CameraZoom2DMove(void);

        /**
         * Answer the zooming behaviour.
         *
         * @return The zooming behaviour.
         */
        inline ZoomBehaviourType ZoomBehaviour(void) const {
            return this->behaviour;
        }

        /**
         * Sets the zooming behaviour.
         *
         * @param behaviour The new zooming behaviour.
         */
        void SetZoomBehaviour(ZoomBehaviourType behaviour);

        /**
         * Sets the movement speed factor. This factor should depend on the
         * scene rendered and some of the camera parameters. In addition to
         * this speed factor the mouse movement is normalised using the virtual
         * view height.
         *
         * @param speed The new movement speed.
         */
        void SetSpeed(SceneSpaceType speed);

        /**
         * Answer the movement speed factor.
         *
         * @return The movement speed factor.
         */
        inline SceneSpaceType Speed(void) const {
            return this->speed;
        }

        /**
         * Callback methode called by a Cursor2D this event is registered to 
         * when a cursor event occures. REASON_BUTTON_DOWN, REASON_MOVE and
         * REASON_BUTTON_UP control the mouse position tracking. Changes to
         * the y coordinate of the mouse position will change the position of
         * the associated camera. Othere calling reasons have no effect.
         *
         * @param caller The AbstractCursor object which calles this methode.
         * @param reason The reason why for this call.
         * @param param A reason depending parameter.
         */
        virtual void Trigger(AbstractCursor *caller, TriggerReason reason, unsigned int param);

    private:

        /** Flag whether the mouse draggs. */
        bool drag;

        /** movement speed factor */
        SceneSpaceType speed;

        /** The zooming behaviour */
        ZoomBehaviourType behaviour;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERAZOOM2DMOVE_H_INCLUDED */

