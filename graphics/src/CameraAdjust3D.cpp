/*
 * CameraAdjust3D.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "vislib/CameraAdjust3D.h"
#include "vislib/RelativeCursor3D.h"
#include "vislib/Trace.h"
#include "vislib/Quaternion.h"
#include "vislib/Matrix4.h"
#include <climits>


/*
 * vislib::graphics::CameraAdjust3D::CameraAdjust3D
 */
vislib::graphics::CameraAdjust3D::CameraAdjust3D(
        const SmartPtr<CameraParameters>& cameraParams) : 
        AbstractCursor3DEvent(), AbstractCameraController(cameraParams), 
        switchYZ(false), singleAxis(false),
        invertX(false), invertY(false), invertZ(false),
        noTranslate(false), noRotate(false), 
        objectCenter(0.0f, 0.0f, 0.0f), controlMode(CAMERA_MODE) {
        
    // Intentionally empty
}


/*
 * vislib::graphics::CameraAdjust3D::~CameraAdjust3D
 */
vislib::graphics::CameraAdjust3D::~CameraAdjust3D(void) {
    // Intentionally empty
}


/*
 * vislib::graphics::CameraAdjust3D::Trigger
 */

void vislib::graphics::CameraAdjust3D::Trigger(
        vislib::graphics::AbstractCursor *caller,
        vislib::graphics::AbstractCursorEvent::TriggerReason reason,
        unsigned int param) {

    RelativeCursor3D *cursor = dynamic_cast<RelativeCursor3D *>(caller);
    ASSERT(cursor != NULL);

    // otherwise this would be very strange
    ASSERT(cursor->CameraParams()->IsSimilar(this->CameraParams()));

    if (reason == REASON_MOVE) {

        // Get the current data
        SceneSpaceVector3D translateVector = cursor->getTranslate();
        math::Vector<float, 3> rotateVector = cursor->getRotate();

        /* Tests for various modes */

        // Switch YZ mode: reverse y and z to make plane of desk correspond to
        // plane of screen
        // Only makes sense under CAMERA_MODE
        if (this->switchYZ && controlMode == CAMERA_MODE) {
            float temp = rotateVector.GetY();
            rotateVector.SetY(-1.0f * rotateVector.GetZ());
            rotateVector.SetZ(temp);
            temp = translateVector.GetY();
            translateVector.SetY(-1.0f * translateVector.GetZ());
            translateVector.SetZ(temp);
        }

        // No Translation mode zeroes out translation values
        if (this->noTranslate) {
            translateVector.SetNull();
        }

        // No Rotation mode zeroes out rotation values
        if (this->noRotate) {
            rotateVector.SetNull();
        }

        // Single Axis mode: only respond to largest translation or rotation
        // vector component (zero all others)
        if (this->singleAxis) {
            // determine the largest move value and zero the others
            SceneSpaceType max = translateVector.GetX();
            int axis = 0;
            // get the largest value and set the axis
            if (abs(max) < abs(translateVector.GetY())) {
                max = translateVector.GetY();
                axis = 1;
            }
            if (abs(max) < abs(translateVector.GetZ())) {
                max = translateVector.GetZ();
                axis = 2;
            }
            if (abs(max) < abs(rotateVector.GetX())) {
                max = rotateVector.GetX();
                axis = 3;
            }
            if (abs(max) < abs(rotateVector.GetY())) {
                max = rotateVector.GetY();
                axis = 4;
            }
            if (abs(max) < abs(rotateVector.GetZ())) {
                max = rotateVector.GetZ();
                axis = 5;
            }
            // clear both vectors
            translateVector.Set(0.0f, 0.0f, 0.0f);
            rotateVector.Set(0.0f, 0.0f, 0.0f);
            // insert the largest value
            if (axis == 0) {
                translateVector.SetX(max);
            } else if (axis == 1) {
                translateVector.SetY(max);
            } else if (axis == 2) {
                translateVector.SetZ(max);
            } else if (axis == 3) {
                rotateVector.SetX(max);
            } else if (axis == 4) {
                rotateVector.SetY(max);
            } else if (axis == 5) {
                rotateVector.SetZ(max);
            }
        }

        // Invert X: reverse x axis rotation and translation
        if (this->invertX) {
            translateVector.SetX(-1.0f * translateVector.GetX());
            rotateVector.SetX(-1.0f * rotateVector.GetX());
        }
        
        // Invert Y: reverse y axis rotation and translation
        if (this->invertY) {
            translateVector.SetY(-1.0f * translateVector.GetY());
            rotateVector.SetY(-1.0f * rotateVector.GetY());
        }

        // Invert Z: reverse z axis rotation and translation
        if (this->invertZ) {
            translateVector.SetZ(-1.0f * translateVector.GetZ());
            rotateVector.SetZ(-1.0f * rotateVector.GetZ());
        }
        
        /* End of modes */

        // Since Z will be corresponding to CameraParams()->Front, it should
        // be inverted
        translateVector.SetZ(-1.0f * translateVector.GetZ());
        rotateVector.SetZ(-1.0f * rotateVector.GetZ());

        /* Get the camera control mode and adjust the camera accordingly */
        if (this->controlMode == CAMERA_MODE) {
            // Using camera control mode:
            // In this mode, the camera is translated and rotated directly.
            // The rotation pivot point is always the camera look-at point.
            // This is really quite weird - this mode should instead rotate
            // the camera around an internal pivot.

            // translation
            SceneSpaceVector3D move;
            move = this->CameraParams()->Right() * translateVector.GetX() +
                this->CameraParams()->Up() * translateVector.GetY() +
                this->CameraParams()->Front() * translateVector.GetZ();

            // shift the position and the look at point around
            this->CameraParams()->SetView(
                this->CameraParams()->Position() + move,
                this->CameraParams()->LookAt() + move,
                this->CameraParams()->Up());
        
            // rotation
            math::Vector<float, 3> up;
        
            // up vector is controlled by z-axis rotation - positive z is
            // counterclockwise
            up = this->CameraParams()->Right()
                    * static_cast<SceneSpaceType>(::sin(rotateVector.GetZ()))
                + this->CameraParams()->Up()
                    * static_cast<SceneSpaceType>(::cos(rotateVector.GetZ()));

            // set the new up vector
            this->CameraParams()->SetUp(up);
        
            math::Vector<float, 3> rot; // pitch and yaw vector
            // the vector of rotation will be the x and y components of the
            // camera * the input values
            rot = this->CameraParams()->Right()*rotateVector.GetX() +  
                this->CameraParams()->Up()*rotateVector.GetY(); 
            // get the amount of rotation (the magnitude of the vector) and
            // normalise rot
            math::AngleRad angle = static_cast<math::AngleRad>(rot.Normalise());        

            // set up rotation quaternion
            math::Quaternion<SceneSpaceType> quat(angle, rot);

            // fetch current view values
            up    = this->CameraParams()->Up();
            math::Vector<float, 3> antiLook 
                = this->CameraParams()->Position()
                - this->CameraParams()->LookAt();
            math::Point<float, 3> look 
                = this->CameraParams()->LookAt();

            // rotate current view
            up = quat * up;
            antiLook = quat * antiLook;

            // set new view
            this->CameraParams()->SetView(look + antiLook, look, up);

        } else if (this->controlMode == OBJECT_MODE ||
                this->controlMode == TARGET_CAMERA_MODE) {
            // These modes manipulate the camera around a pivot point at the
            // object center. They are identical save for that in object mode
            // the directions are reversed to make it appear that the object is
            // being manipulated, not the camera.

            if (this->controlMode == OBJECT_MODE) {
                // invert everything cause we're doing object mode
                translateVector *= -1.0f;
                rotateVector *= -1.0f;
            }

            // First step - translation
            // Simple - just translate the camera opposite to the direction of
            // translation. Might need to add compensation for the distance to
            // the object.

            // translation
            SceneSpaceVector3D move;
            move = this->CameraParams()->Right() * translateVector.GetX() +
                this->CameraParams()->Up() * translateVector.GetY() +
                this->CameraParams()->Front() * translateVector.GetZ();

            // shift the position and the look at point around
            this->CameraParams()->SetView(
                this->CameraParams()->Position() + move,
                this->CameraParams()->LookAt() + move,
                this->CameraParams()->Up());

            // Next step - rotation
            // More challenging - need to make the object appear to spin,
            // regardless of whether or not the object is in the center of the
            // view. That is, we can't always assume the look at point is also
            // the pivot point for the camera.

            // Rotations around the Y axis (Up) will cause the object to spin
            // like a top. Basically we need to rotate the vector from the
            // object position to the camera position around a quaternion that
            // is along the up vector of the camera by an amount that is
            // proportional to the device input. We also need to rotate the
            // lookAt point using this same quaternion.

            math::Vector<float, 3> cameraPositionVector;
            // Set the camera position vector to the camera position minus the
            // object center
            cameraPositionVector = this->CameraParams()->Position() -
                this->objectCenter;
            math::Point<float, 3> cameraPositionPoint = this->objectCenter +
                cameraPositionVector;

            // Get a quaternion to represent the rotation
            // make the amount of rotation correspond to the device input
            float rotationAngle = rotateVector.GetY();

            // Build the quaternion using the angle computed and the camera up
            // vector
            math::Quaternion<float> upRotationQuat(rotationAngle,
                this->CameraParams()->Up());

            // Use the quaternion to rotate the camera position vector
            cameraPositionVector = upRotationQuat * cameraPositionVector;

            // The look at position doesn't really matter, as long as it remains
            // in the same place relative to the camera. So let's just compute a 
            // permanent camera-to-lookat vector to get the lookat position from
            // the camera position all the time.
            math::Vector<float, 3> cameraToLookAt = 
                this->CameraParams()->LookAt() - 
                this->CameraParams()->Position();

            // Okay so now the camera is orbiting Up. Now we need to rotate the
            // lookat point around the Up axis of the camera so that it rotates
            // the same amount as the camera position does. Just use the
            // quaternion.
            cameraToLookAt = upRotationQuat * cameraToLookAt;        
        
            // Have to save the view after each rotation so we get a new set of
            // up, right, and front vectors
            cameraPositionPoint = this->objectCenter + cameraPositionVector;
            this->CameraParams()->SetPosition(cameraPositionPoint);
            this->CameraParams()->SetLookAt(cameraPositionPoint +
                cameraToLookAt);

            // Okay so that's the up rotation. Now to repeat the process for the
            // other two


            // Let's tackle the x axis next, or the right vector. Should just be
            // a matter of using the right vector everywhere, instead of the up
            // vector.

            // Get a quaternion to represent the rotation
            // make the amount of rotation correspond to the device input
            rotationAngle = rotateVector.GetX();

            // Build the quaternion using the angle computed and the camera up
            // vector. Could just use the same old quaternion from before but
            // only the constructor allows you to build it from an angle and
            // axis.
            math::Quaternion<float> rightRotationQuat(rotationAngle,
                this->CameraParams()->Right());

            // Use the quaternion to rotate the camera position vector
            cameraPositionVector = rightRotationQuat * cameraPositionVector;

            // Okay so now the camera is orbiting Right. Now we need to rotate
            // the lookat point around the Right axis of the camera so that it
            // rotates the same amount as the camera position does. Just use the
            // quaternion.
            cameraToLookAt = rightRotationQuat * cameraToLookAt;

            // Have to save the view after each rotation so we get a new set of
            // up, right, and front vectors
            cameraPositionPoint = this->objectCenter + cameraPositionVector;
            this->CameraParams()->SetPosition(cameraPositionPoint);
            this->CameraParams()->SetLookAt(cameraPositionPoint +
                cameraToLookAt);


            // Finally, the z axis. Change everything to use the Front vector.
            // Also, at the end, the lookat point doesn't need to change.
            // Instead, the Up vector for the camera needs to be changed. We can
            // just replace the lookAt with the Up vector and it should be fine.

            // Get a quaternion to represent the rotation
            // make the amount of rotation correspond to the device input
            rotationAngle = rotateVector.GetZ();

            // Build the quaternion using the angle computed and the camera up
            // vector. Could just use the same old quaternion from before but
            // only the constructor allows you to build it from an angle and
            // axis.
            math::Quaternion<float> frontRotationQuat(rotationAngle,
                this->CameraParams()->Front());

            // Use the quaternion to rotate the camera position vector
            cameraPositionVector = frontRotationQuat * cameraPositionVector;

            // Get the new camera position
            cameraPositionPoint = this->objectCenter + cameraPositionVector;

            // Okay so now the camera is orbiting Front. Now we need to rotate
            // the Up vector around the Front axis of the camera so that it
            // rotates the same amount as the camera position does. Just use the
            // quaternion.
            math::Vector<float, 3> cameraUpVector = frontRotationQuat *
                this->CameraParams()->Up();

            // Set the new camera position, lookat, and up vectors.
            this->CameraParams()->SetView(cameraPositionPoint,
                cameraPositionPoint + cameraToLookAt,
                cameraUpVector);

            // And that should be a complete object mode camera control
            // It's a bit messy, but it's the best way I can think of to do it
            // without getting into matrices
        }

    } else if (reason == REASON_BUTTON_DOWN) {
        // do nothing yet
    } else if (reason == REASON_BUTTON_UP) {
        // do nothing yet
    } else {
        // handle other reasons than button presses and releases and mouse
        // moves
    }
}