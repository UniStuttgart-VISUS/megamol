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

        /** Sets the manner in which the adjustor affects the camera */
        enum CameraControlMode {
            CAMERA_MODE,
            OBJECT_MODE,
            TARGET_CAMERA_MODE
        };

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

        /** 
         * Causes the adjustor to use camera mode control.
         * In this mode, the adjustor directly modifies the camera according
         * to the input (i.e. the camera is rotated and translated).
         * The camera is rotated around the look-at point it has, not its
         * internal axes.
         */
        inline void SetCameraControlMode(void) {
            this->controlMode = CAMERA_MODE;
        }

        /**
         * Causes the adjustor to use object mode control.
         * In this mode, the adjustor appears to manipulate the object position
         * according to the input. That is, the "object" is translated and
         * rotated. If the object is not at the origin, the SetObjectCenter
         * should be called to provide the object center for this mode.
         * This will only work with a single object. Alternatively, this will
         * treat the entire scene as a single object that can be manipulated
         * with the input.
         */
        inline void SetObjectControlMode(void) {
            this->controlMode = OBJECT_MODE;
        }

        /**
         * Causes the adjustor to use target camera mode control.
         * This mode is similar to object mode, but the directions are all
         * reversed. This mode moves the camera but causes it to always pivot
         * around the object position. Thus, SetObjectCenter should be called
         * prior to using this mode if the object is not located at the origin.
         * This mode would make more sense if a visible "pivot point" was
         * rendered, and if this point was set as the new "object center" as it
         * moved around. This point could be attached to a point on the object
         * that is closest to the center of camera, which would allow the user
         * refine his or her movements as the object was zoomed in on.
         */
        inline void SetTargetCameraControlMode(void) {
            this->controlMode = TARGET_CAMERA_MODE;
        }

        /**
         * Saves the current look at point as the local object center variable.
         * This variable is used for the object and target camera modes.
         * If the object is not at the origin (the default), failure to set this
         * will result in bizarre effects while using the object or target
         * camera modes.
         */
        inline void SetObjectCenter(void) {
            this->objectCenter = this->CameraParams()->LookAt();
        }

        /** 
         * Saves the object center as the point provided.
         * This variable is used for the object and target camera modes.
         * If the object is not at the origin (the default), failure to set this
         * will result in bizarre effects while using the object or target
         * camera modes.
         *
         * @param object A vislib point representing the object center.
         */
        inline void SetObjectCenter(
                vislib::math::Point<SceneSpaceType, 3> object) {
            this->objectCenter = object;
        }

    private:

        /** The current camera control mode */
        CameraControlMode controlMode;

        /** 
         * The object center, which is used as the focal point for object and
         * target camera control modes.
         */
        math::Point<SceneSpaceType, 3> objectCenter;

        /** 
         * Switches axes so that the plane of the table is corresponds to the
         * plane of the screen.
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

