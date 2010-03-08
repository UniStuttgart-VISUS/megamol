/*
 * CameraLookAtDist.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CAMERALOOKATDIST_H_INCLUDED
#define VISLIB_CAMERALOOKATDIST_H_INCLUDED
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
     * Cursor 2d event and camera controller chaning the distance to the look
     * at point interactively.
     */
    class CameraLookAtDist : public AbstractCursor2DEvent, 
        public AbstractCameraController {
    public:

        /** 
         * Ctor. 
         *
         * @param cameraParams The camera parameters object to be rotated.
         */
        CameraLookAtDist(const SmartPtr<CameraParameters>& cameraParams 
            = SmartPtr<CameraParameters>());

        /** Dtor. */
        virtual ~CameraLookAtDist(void);

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

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERALOOKATDIST_H_INCLUDED */

