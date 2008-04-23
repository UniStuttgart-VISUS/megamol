/*
 * CameraZoom2DAngle.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */

#ifndef VISLIB_CAMERAZOOM2DANGLE_H_INCLUDED
#define VISLIB_CAMERAZOOM2DANGLE_H_INCLUDED
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
     * Cursor2DEvent manipulating the camera aperture angle by the Y 
     * coordinate of the 2d cursor to zoom in and out.
     */
    class CameraZoom2DAngle : public AbstractCursor2DEvent, 
        public AbstractCameraController {
    public:

        /** 
         * Ctor. 
         *
         * @param cameraParams The camera parameters object to be rotated.
         */
        CameraZoom2DAngle(const SmartPtr<CameraParameters>& cameraParams 
            = SmartPtr<CameraParameters>());

        /** Dtor. */
        ~CameraZoom2DAngle(void);

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

    private:

        /** Flag whether the mouse draggs. */
        bool drag;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERAZOOM2DANGLE_H_INCLUDED */

