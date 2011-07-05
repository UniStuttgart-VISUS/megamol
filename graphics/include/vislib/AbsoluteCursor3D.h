/*
 * AbsoluteCursor3D.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSOLUTECURSOR3D_H_INCLUDED
#define VISLIB_ABSOLUTECURSOR3D_H_INCLUDED
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
#include "vislib/Matrix.h"


namespace vislib {
namespace graphics {


    /**
     * Class modeling a three-dimensional absolute cursor, like a Phantom
     * Desktop pen.
     */
    class AbsoluteCursor3D: public AbstractCursor {
    public:

        /** ctor */
        AbsoluteCursor3D(void);

        /**
         * copy ctor
         *
         * @param rhs Sourc object.
         */
        AbsoluteCursor3D(const AbsoluteCursor3D& rhs);

        /** Dtor. */
        virtual ~AbsoluteCursor3D(void);

        /**
         * Checks if a new position or orientation has occurred that represents
         * a motion larger than the threshold motion value. If so, stores the
         * new position point and orientation vectors and sets the old data as
         * the previous position and orientation. Also triggers any relevant
         * cursor events with the REASON_MOVE parameter. If not, ignores the
         * data and does not trigger events.
         *
         * @param position The new position point in world space coordinates
         * @param orientation The new orientation vector in world space
         *        coordinates
         *
         * @return True if motion was detected above minimum values
         */
        bool SetPosition(vislib::math::Point<double, 3> position,
            vislib::math::Vector<double, 3> orientation);

        /**
         * Checks if a new position or orientation has occurred that represents
         * a motion larger than the threshold motion value. If so, stores the
         * new position point and orientation vectors and sets the old data as
         * the previous position and orientation. Also triggers any relevant
         * cursor events with the REASON_MOVE parameter. If not, ignores the
         * data and does not trigger events.
         *
         * @param position The new position point in world space coordinates
         * @param rotationQuat The rotation quaternion that specifies the new
         *        orientation
         *
         * @return True if motion was detected above minimum values
         */
        bool SetPosition(vislib::math::Point<double, 3> position,
            vislib::math::Quaternion<double> rotationQuat);

        /**
         * Stores the new transform matrix and sets the old data as the
         * previous transform matrix. Transform matrix must be arranged such
         * that axis data lies along the rows (position data is on the
         * rightmost column, not the bottom row).
         *
         * Also updates the position and orientation by calling SetPosition,
         * which triggers on move events in the case of a new position or
         * orientation.
         *
         * @param transformMatrix Matrix to be set as current transform
         */
        void SetTransform(vislib::math::Matrix<double, 4,
            vislib::math::ROW_MAJOR> newTransform);

        /**
         * Assignment operator
         *
         * @param rhs Sourc object.
         *
         * @return Reference to this.
         */
        AbsoluteCursor3D& operator=(const AbsoluteCursor3D& rhs);

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
         * Returns the current cursor position.
         *
         * @return The associated cursor position.
         */
        inline vislib::math::Point<double, 3> GetCurrentPosition(void) {
            return this->currentPosition;
        }

        /**
         * Returns the previous cursor position.
         *
         * @return The previous associated cursor position.
         */
        inline vislib::math::Point<double, 3> GetPreviousPosition(void) {
            return this->previousPosition;
        }

        /**
         * Returns the associated rotation vector. Adding this to the cursor
         * position gives the cursor "look at" point.
         *
         * @return The associated rotation vector, showing the direction the
         *         cursor is pointing.
         */
        inline vislib::math::Vector<double, 3> GetCurrentOrientation(void) {
            return this->currentOrientation;
        }

        /**
         * Returns the previous associated rotation vector.
         *
         * @return The previous associated rotation vector.
         */
        inline vislib::math::Vector<double, 3> GetPreviousOrientation(void) {
            return this->previousOrientation;
        }
        
        /**
         * Returns the associated transform matrix.
         *
         * @return The associated transform matrix.
         */
        inline vislib::math::Matrix<double, 4, vislib::math::ROW_MAJOR>
        GetCurrentTransform(void) {
            return this->currentTransform;
        }

        /**
         * Returns the previous associated transform matrix.
         *
         * @return The previous associated transform matrix.
         */
        inline vislib::math::Matrix <double, 4, vislib::math::ROW_MAJOR>
        GetPreviousTransform(void) {
            return this->previousTransform;
        }

        /**
         * Returns the associated camera parameters object.
         *
         * @return The associated camera parameters object.
         */
        inline SmartPtr<CameraParameters> CameraParams(void) {
            return this->camPams;
        }

        /**
         * Sets the minimum move distance needed to trigger a "motion" event.
         * This is the distance between the current and previous position.
         * This value can be used to reduce 3d device sensitivity, for example
         * to stop "motion" events from occurring due to shaky hands.
         *
         * @param value The minimum move distance
         */
        inline void SetMinimumMoveDistance(double value) {
            this->minMoveDist = value;
        }

        /**
         * Sets the minimum rotate angle needed to trigger a "motion" event.
         * This is the angle between the current orientation vector and the
         * old vector. This value can be used to reduce 3d device sensitivity,
         * for example to stop "motion" events from occurring due to shaky
         * hands.
         *
         * @param value The minimum rotate angle
         */
        inline void SetMinimumRotateAngle(double value) {
            this->minRotateAngle = value;
        }


        /**
         * Sets the initial orientation vector. This is the vector that the
         * rotation portion of the transform matrix will be applied to in
         * order to obtain the final orientation value. Typical values are
         * unit vectors in the y or z direction.
         *
         * @param init A vislib 3-double vector describing the initial
         *             orientation (i.e. the un-rotated orientation).
         */
        inline void SetInitialOrientation(vislib::math::Vector <double, 3> init) {
            this->initOrientation = init;
        }

    private:

        /** The position of cursor in world space coordinates */
        vislib::math::Point <double, 3> currentPosition;

        /**
         * The cursor orientation, expressed as a unit vector from the origin
         * in world space coordinates
         */
        vislib::math::Vector <double, 3> currentOrientation;

        /**
         * Stores the previous position of cursor in world space coordinates
         */
        vislib::math::Point <double, 3> previousPosition;

        /**
         * Stores the previous orientation, expressed as a unit vector from
         * the origin in world space coordinates
         */
        vislib::math::Vector <double, 3> previousOrientation;

        /**
         * The current cursor position/orientation 4x4 transform matrix from
         * world space coordinates
         */
        vislib::math::Matrix<double, 4, vislib::math::ROW_MAJOR>
            currentTransform;

        /**
         * The previous cursor position/orientation 4x4 transform matrix from
         * world space coordinates
         */
        vislib::math::Matrix<double, 4, vislib::math::ROW_MAJOR>
            previousTransform;

        /** The parameters object of the associated camera */
        SmartPtr<CameraParameters> camPams;

        /**
         * Initial rotation vector. The quaternion created from the rotation
         * matrix portion of the transform matrix will be applied to this
         * initial vector to produce the final orientation vector.
         */
        vislib::math::Vector <double, 3> initOrientation;

        /** 
         * The required minimum distance that must be moved in order to
         * trigger a "motion" event Can be used with devices that are
         * sensitive to unintentional hand shaking, for instance
         */
        double minMoveDist;

        /**
         * The required minimum rotate angle needed to trigger a "motion"
         * event, in radians.
         */
        double minRotateAngle;

    };

} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSOLUTECURSOR3D_H_INCLUDED */
