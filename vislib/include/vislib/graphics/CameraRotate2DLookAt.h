/*
 * CameraRotate2DLookAt.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */

#ifndef VISLIB_CAMERAROTATE2DLOOKAT_H_INCLUDED
#define VISLIB_CAMERAROTATE2DLOOKAT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/graphics/AbstractCameraController.h"
#include "vislib/graphics/AbstractCursor2DEvent.h"
#include "vislib/graphics/Camera.h"
#include "vislib/graphics/CameraParameters.h"
#include "vislib/graphics/InputModifiers.h"
#include "vislib/graphics/graphicstypes.h"
#include "vislib/SmartPtr.h"


namespace vislib {
namespace graphics {


    /**
     * Cursor2DEvent rotating the camera around its look-at-point based on the
     * mouse movement.
     *
     * You should set up a button test and you should set the alternative 
     * rotation modifier if you use this class. Normal rotation schema 
     * modelles pitch and yaw rotations, while the alternative rotation schema
     * modelles roll rotation.
     */
    class [[deprecated]] CameraRotate2DLookAt : public AbstractCursor2DEvent, 
        public AbstractCameraController {

    public:

        /** 
         * Ctor. 
         *
         * @param cameraParams The camera parameters object to be rotated.
         */
        CameraRotate2DLookAt(const SmartPtr<CameraParameters>& cameraParams 
            = SmartPtr<CameraParameters>());

        /** Dtor. */
        virtual ~CameraRotate2DLookAt(void);

        /**
         * Callback methode called by a Cursor2D this event is registered to 
         * when a cursor event occures. REASON_BUTTON_DOWN, REASON_MOVE and
         * REASON_BUTTON_UP control the mouse position tracking and the 
         * rotation of the beholder. Othere calling reasons have no effect.
         *
         * @param caller The AbstractCursor object which calles this methode.
         * @param reason The reason why for this call.
         * @param param A reason depending parameter.
         */
        virtual void Trigger(AbstractCursor *caller, TriggerReason reason, unsigned int param);

        /**
         * Sets the modifier for alternative rotation schema. The normal 
         * rotation schema modelles pitch and yaw rotations, while the 
         * alternative rotation schema modelles roll rotations.
         *
         * @param modifier The modifier.
         */
        inline void SetAltModifier(InputModifiers::Modifier modifier) {
            this->altMod = modifier;
        }

        /**
         * Returns the modifier for alternative rotation schema.
         *
         * @return The modifier.
         */
        inline InputModifiers::Modifier GetAltModifier(void) const {
            return this->altMod;
        }

    private:

        /** The modifier for alternate rotation */
        InputModifiers::Modifier altMod;

        /** Flag whether the mouse draggs. */
        bool drag;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERAROTATE2DLOOKAT_H_INCLUDED */

