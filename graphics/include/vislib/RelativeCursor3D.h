/*
 * RelativeCursor3D.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_RELATIVECURSOR3D_H_INCLUDED
#define VISLIB_RELATIVECURSOR3D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractCursor.h"
#include "vislib/AbstractCursor3DEvent.h"
#include "vislib/CameraParameters.h"
#include "vislib/graphicstypes.h"
#include "vislib/SmartPtr.h"
#include "vislib/Quaternion.h"


namespace vislib {
namespace graphics {


    /**
     * Class modeling a three-dimensional relative cursor, like a 3dConnexion
     * SpaceNavigator.
     */
    class RelativeCursor3D: public AbstractCursor {
    public:

        /** ctor */
        RelativeCursor3D(void);

        /**
         * copy ctor
         *
         * @param rhs Sourc object.
         */
        RelativeCursor3D(const RelativeCursor3D& rhs);

        /** Dtor. */
        virtual ~RelativeCursor3D(void);

        /**
         * Stores the translation and rotation vectors and triggers the motion
         * event for the associated camera. The translation vector passed in
         * as parameters is the translation vector with respect to the camera
         * coordinates (x right, y up, z out of the screen).
         * The rotation vector passed in as parameters is the directed
         * rotation vector, with the direction specifying the axis of rotation
         * (counterclockwise) and the magnitude specifying the amount of
         * rotation in radians.
         *
         * @param tx The x component of the translation vector
         * @param ty The y component of the translation vector
         * @param tz The z component of the translation vector
         * @param rx The x component of the rotation vector
         * @param ry The y component of the rotation vector
         * @param rz The z component of the rotation vector
         */
        void Motion(SceneSpaceType tx, SceneSpaceType ty, SceneSpaceType tz,
                float rx, float ry, float rz);

        /**
         * Assignment operator
         *
         * @param rhs Sourc object.
         *
         * @return Reference to this.
         */
        RelativeCursor3D& operator=(const RelativeCursor3D& rhs);

        /**
         * Behaves like AbstractCursor::RegisterCursorEvent.
         *
         * @param cursorEvent The cursor event to be added.
         */
        virtual void RegisterCursorEvent(AbstractCursor3DEvent *cursorEvent);

        /**
         * Associates a camera parameters object with this cursor.
         *
         * @param cameraParams The camera parameters object.
         */
        void SetCameraParams(SmartPtr<CameraParameters> cameraParams);

        /**
         * Returns the associated translation vector.
         *
         * @return The associated translation vector.
         */
        inline SceneSpaceVector3D getTranslate(void) {
            return this->translate;
        }

        /**
         * Returns the associated rotation vector.
         *
         * @return The associated rotation vector.
         */
        inline vislib::math::Vector<float, 3> getRotate(void) {
            return this->rotate;
        }

        /**
         * Returns the associated camera parameters object.
         *
         * @return The associated camera parameters object.
         */
        inline SmartPtr<CameraParameters> CameraParams(void) {
            return this->camPams;
        }

    private:

        /**
         * The translation vector of the relative cursor
         * X is right, Y is up, Z is out of the screen.
         */
        SceneSpaceVector3D translate;

        /**
         * The rotation vector of the relative cursor. This magnitude of this
         * vector specifies the angle of rotation (in radians) and the
         * normalised vector specifies the axis of rotation (counterclockwise)
         * X is right, Y is up, Z is out of the screen. 
         */
        math::Vector<float, 3> rotate;

        /** The parameters object of the associated camera */
        SmartPtr<CameraParameters> camPams;

    };

} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_RELATIVECURSOR3D_H_INCLUDED */
