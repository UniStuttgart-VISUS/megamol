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
        /*altMod(UINT_MAX), */switchYZ(false), singleAxis(false), invertX(false),
        invertY(false), invertZ(false), noTranslate(false), noRotate(false) {
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

        // Get the current state of the alt key
        //bool alt = false;
        //if ((cursor->GetInputModifiers() != NULL) 
        //        && (cursor->GetInputModifiers()->GetModifierCount()
        //            > this->altMod)) {
        //    alt = cursor->GetInputModifiers()->GetModifierState(this->altMod);
        //}

        //if (alt) {
        //    // some functionality - maybe locking a direction or something
        //}

        /* Tests for various modes */

        // Switch YZ mode: reverse y and z to make plane of desk correspond to
        // plane of screen
        if (this->switchYZ) {
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

        // Invert X: negate x axis rotation and translation
        if (this->invertX) {
            translateVector.SetX(-1.0f * translateVector.GetX());
            rotateVector.SetX(-1.0f * rotateVector.GetX());
        }
        
        // Invert Y: negate y axis rotation and translation
        if (this->invertY) {
            translateVector.SetY(-1.0f * translateVector.GetY());
            rotateVector.SetY(-1.0f * rotateVector.GetY());
        }

        // Invert Z: negate z axis rotation and translation
        if (this->invertZ) {
            translateVector.SetZ(-1.0f * translateVector.GetZ());
            rotateVector.SetZ(-1.0f * rotateVector.GetZ());
        }
        
        /* End of modes */

        // Since Z will be corresponding to CameraParams()->Front, it should
        // be inverted
        translateVector.SetZ(-1.0f * translateVector.GetZ());
        rotateVector.SetZ(-1.0f * rotateVector.GetZ());

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
        // the vector of rotation will be the x and y components of the camera
        // * the input values
        rot = this->CameraParams()->Right() * rotateVector.GetX() +
            this->CameraParams()->Up() * rotateVector.GetY();
        // get the amount of rotation (the magnitude of the vector) and
        // normalise rot
        math::AngleRad angle = static_cast<math::AngleRad>(rot.Normalise());

        // set up rotation quaternion
        math::Quaternion<SceneSpaceType> quat(angle, rot);

        // fetch current view values
        up	= this->CameraParams()->Up();
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
        

    } else if (reason == REASON_BUTTON_DOWN) {
        // do nothing yet
    } else if (reason == REASON_BUTTON_UP) {
        // do nothing yet
    } else {
        // handle other reasons than button presses and releases and mouse
        // moves
    }
}