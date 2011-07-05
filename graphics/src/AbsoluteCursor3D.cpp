/*
 * AbsoluteCursor3D.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbsoluteCursor3D.h"
#include "vislib/mathfunctions.h"
#include "vislib/AbstractCursor3DEvent.h"
#include <stdio.h>


/*
 * vislib::graphics::AbsoluteCursor3D::AbsoluteCursor3D
 */
vislib::graphics::AbsoluteCursor3D::AbsoluteCursor3D(void) : AbstractCursor(),
        currentPosition(), currentOrientation(), previousPosition(),
        previousOrientation(), currentTransform(), previousTransform(),
        camPams(NULL), minMoveDist(0.0001), minRotateAngle(0.0001),
        initOrientation(0.0, 1.0, 0.0) { 
    // intentionally empty
}


/*
 * vislib::graphics::AbsoluteCursor3D::AbsoluteCursor3D
 */
vislib::graphics::AbsoluteCursor3D::AbsoluteCursor3D(
        const AbsoluteCursor3D& rhs) : AbstractCursor(rhs),
        currentPosition(rhs.currentPosition),
        previousPosition(rhs.previousPosition),
        currentOrientation(rhs.currentOrientation),
        previousOrientation(rhs.previousOrientation),
        currentTransform(rhs.currentTransform),
        previousTransform(rhs.previousTransform), camPams(rhs.camPams),
        minMoveDist(rhs.minMoveDist), minRotateAngle(rhs.minRotateAngle),
        initOrientation(rhs.initOrientation) {
    // intentionally empty
}


/*
 * vislib::graphics::AbsoluteCursor3D::~AbsoluteCursor3D
 */
vislib::graphics::AbsoluteCursor3D::~AbsoluteCursor3D(void) {
    // Do not delete this->cam
}


/*
 * vislib::graphics::AbsoluteCursor3D::operator=
 */
vislib::graphics::AbsoluteCursor3D&
vislib::graphics::AbsoluteCursor3D::operator=(const AbsoluteCursor3D& rhs) {
    AbstractCursor::operator=(rhs);
    this->currentPosition = rhs.currentPosition;
    this->previousPosition = rhs.previousPosition;
    this->currentOrientation = rhs.currentOrientation;
    this->previousOrientation = rhs.previousOrientation;
    this->currentTransform = rhs.currentTransform;
    this->previousTransform = rhs.previousTransform;
    this->camPams = rhs.camPams;
    this->minMoveDist = rhs.minMoveDist;
    this->minRotateAngle = rhs.minRotateAngle;
    return *this;
}


/*
 * vislib::graphics::AbsoluteCursor3D::SetPosition
 */
bool vislib::graphics::AbsoluteCursor3D::SetPosition(
        vislib::math::Point<double, 3> position,
        vislib::math::Vector<double, 3> orientation) {

    bool motion = false;

    vislib::math::Vector<double, 3> motionVector
        = position - this->currentPosition;
    double distance = motionVector.Length(); // get the distance moved
    if (distance > this->minMoveDist) {
        // set a motion flag if the distance moved is above the threshold
        motion |= true;
    }

    double angle = orientation.Angle(this->currentOrientation);
    // get the angle rotated
    if (angle > this->minRotateAngle) {
        // set a motion flag if the angle rotated is above the threshold
        motion |= true;
    }

    if (motion) {
        // update the values
        this->previousPosition = this->currentPosition;
        this->previousOrientation = this->currentOrientation;
        this->currentPosition = position;
        this->currentOrientation = orientation;

        // trigger move events
        AbstractCursor::TriggerMoved();

        return true; // return true on motion
    } else {

        return false; // return false on no motion and do not update values
    }
}


/*
 * vislib::graphics::AbsoluteCursor3D::SetPosition
 */
bool vislib::graphics::AbsoluteCursor3D::SetPosition(
        vislib::math::Point<double, 3> position,
        vislib::math::Quaternion<double> rotationQuat) {

    bool motion = false;

    vislib::math::Vector<double, 3> motionVector
        = position - this->currentPosition;
    double distance = motionVector.Length(); // get the distance moved
    if (distance > this->minMoveDist) {
        // set a motion flag if the distance moved is above the threshold
        motion |= true;
    }

    vislib::math::Vector<double, 3> orientation;
    // get the new orientation from the quaternion
    orientation = rotationQuat * this->initOrientation;

    // get the angle rotated
    double angle = orientation.Angle(this->currentOrientation);
    if (angle > this->minRotateAngle) {
        // set a motion flag if the angle rotated is above the threshold
        motion |= true;
    }

    if (motion) {
        // update the values
        this->previousPosition = this->currentPosition;
        this->previousOrientation = this->currentOrientation;
        this->currentPosition = position;
        this->currentOrientation = orientation;

        // trigger move events
        AbstractCursor::TriggerMoved();

        return true; // return true on motion
    } else {

        return false; // return false on no motion and do not update values
    }
}


/*
 * vislib::graphics::AbsoluteCursor3D::SetTransform
 */
void vislib::graphics::AbsoluteCursor3D::SetTransform(
        vislib::math::Matrix<double, 4, vislib::math::ROW_MAJOR>
        newTransform) {

    vislib::math::Point<double, 3> newPosition;

    // get the new position from the rightmost column
    newPosition.Set(newTransform(0, 3),
        newTransform(1, 3), newTransform(2, 3));	

    // orientation vector will be the initial vector rotated by the rotation
    // quaternion encapsulated in the matrix
    vislib::math::Vector <double, 3> newOrientation; // create a temp vector
    vislib::math::Quaternion <double> rotQuat;
    // get the rotation quaternion from the matrix
    rotQuat.SetFromRotationMatrix(
        newTransform(0, 0), newTransform(0, 1), newTransform(0, 2),
        newTransform(1, 0), newTransform(1, 1), newTransform(1, 2),
        newTransform(2, 0), newTransform(2, 1), newTransform(2, 2));
    // get the new orientation
    newOrientation = rotQuat * this->initOrientation;

    // normalise the vector
    newOrientation.Normalise();

    // Call the motion function to store the values and test for motion
    bool motion = this->SetPosition(newPosition, newOrientation);

    // Only update the transform matrix if motion was detected
    if (motion) {
        this->previousTransform = this->currentTransform;
        this->currentTransform = newTransform;
    }
}


/*
 * vislib::graphics::AbsoluteCursor3D::RegisterCursorEvent
 */
void vislib::graphics::AbsoluteCursor3D::RegisterCursorEvent(
        AbstractCursor3DEvent *cursorEvent) {
    AbstractCursor::RegisterCursorEvent(cursorEvent);
}


/*
 * vislib::graphics::AbsoluteCursor3D::SetCamera
 */
void vislib::graphics::AbsoluteCursor3D::SetCameraParams(
        vislib::SmartPtr<CameraParameters> cameraParams) {
    this->camPams = cameraParams;
}
