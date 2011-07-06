/*
 * CameraAdjust3D.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CAMERAADJUST3D_H_INCLUDED
#define VISLIB_CAMERAADJUST3D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/AbstractCameraController.h"
#include "vislib/AbstractCursor3DEvent.h"
#include "vislib/Camera.h"
#include "vislib/CameraParameters.h"
#include "vislib/InputModifiers.h"
#include "vislib/graphicstypes.h"
#include "vislib/SmartPtr.h"


namespace vislib {
namespace graphics {


    /**
     * Cursor3DEvent adjusting the camera pan/zoom/rotate based on either
     * relative or absolute 3d cursor movement.
     *
     * You should set up a button test if you use this class. Normal adjustion
     * schema adjusts the camera. It is possible to invert the axes to rotate
     * the camera around its focal point.
     */
    class CameraAdjust3D : public AbstractCursor3DEvent, 
        public AbstractCameraController {
    public:

        /** 
         * Ctor.
         *
         * @param cameraParams The camera parameters object to be rotated.
         */
        CameraAdjust3D(const SmartPtr<CameraParameters>& cameraParams
            = SmartPtr<CameraParameters>());

        /** Dtor. */
        virtual ~CameraAdjust3D(void);

        /**
         * Callback method called by a Cursor3D this event is registered to
         * when a cursor event occurs. REASON_MOVE controls the mouse position
         * tracking and the rotation of the camera. Other calling reasons have
         * no effect.
         *
         * @param caller The AbstractCursor object which calls this methode.
         * @param reason The reason for this call.
         * @param param A reason-dependent parameter.
         */
        virtual void Trigger(AbstractCursor *caller, TriggerReason reason,
            unsigned int param);

        /**
         * Sets the Single Axis mode value
         *
         * @param True to toggle on Single Axis mode
         */
        inline void SetSingleAxisMode(bool setVal) {
            this->singleAxis = setVal;
        }

        /**
         * Gets the Single Axis mode value
         *
         * @return True if Single Axis mode is on
         */
        inline bool GetSingleAxisMode(void) {
            return this->singleAxis;
        }
        
        /**
         * Sets the Switch YZ mode value
         *
         * @param True to toggle on Switch YZ mode
         */
        inline void SetSwitchYZMode(bool setVal) {
            this->switchYZ = setVal;
        }

        /**
         * Gets the Switch YZ mode value
         *
         * @return True if Switch YZ mode is on
         */
        inline bool GetSwitchYZMode(void) {
            return this->switchYZ;
        }

        /**
         * Sets the No Translation mode value
         *
         * @param True to toggle on No Translation mode
         */
        inline void SetNoTranslationMode(bool setVal) {
            this->noTranslate = setVal;
        }

        /**
         * Gets the No Translation mode value
         *
         * @return True if No Translation mode is on
         */
        inline bool GetNoTranslationMode(void) {
            return this->noTranslate;
        }

        /**
         * Sets the No Rotation mode value
         *
         * @param True to toggle on No Rotation mode
         */
        inline void SetNoRotationMode(bool setVal) {
            this->noRotate = setVal;
        }

        /**
         * Gets the No Rotation mode value
         *
         * @return True if No Rotation mode is on
         */
        inline bool GetNoRotationMode(void) {
            return this->noRotate;
        }

        /**
         * Sets the Invert X value
         *
         * @param True to invert the X axis
         */
        inline void SetInvertX(bool setVal) {
            this->invertX = setVal;
        }

        /**
         * Gets the Invert X value
         *
         * @return True if invert X is on
         */
        inline bool GetInvertX(void) {
            return this->invertX;
        }

        /**
         * Sets the Invert Y value
         *
         * @param True to invert the Y axis
         */
        inline void SetInvertY(bool setVal) {
            this->invertY = setVal;
        }

        /**
         * Gets the Invert Y value
         *
         * @return True if invert Y is on
         */
        inline bool GetInvertY(void) {
            return this->invertY;
        }
        
        /**
         * Sets the Invert Z value
         *
         * @param True to invert the Z axis
         */
        inline void SetInvertZ(bool setVal) {
            this->invertZ = setVal;
        }

        /**
         * Gets the Invert Z value
         *
         * @return True if invert Z is on
         */
        inline bool GetInvertZ(void) {
            return this->invertZ;
        }

    private:

        /**
         * Switches axes so that the plane of the table is corresponds to the
         * plane of the screen
         */
        bool switchYZ;

        /**
         * Single axis domination mode (largest magnitude value of translation
         * or rotation is the only one executed)
         */
        bool singleAxis;

        /** Disables translation */
        bool noTranslate;

        /** Disables rotation */
        bool noRotate;

        /**
         * Invert toggles for the X axis (if the rotation/translation is
         * occurring in an unexpected direction, for instance)
         */
        bool invertX;

        /**
         * Invert toggles for the Y axis (if the rotation/translation is
         * occurring in an unexpected direction, for instance)
         */
        bool invertY;

        /**
         * Invert toggles for the Z axis (if the rotation/translation is
         * occurring in an unexpected direction, for instance)
         */
        bool invertZ;

    };

} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERAADJUST3D_H_INCLUDED */

