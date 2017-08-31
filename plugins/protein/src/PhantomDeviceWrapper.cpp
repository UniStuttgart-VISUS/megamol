/*
 * PhantomDeviceWrapper.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */


#include "stdafx.h"

#include "PhantomDeviceWrapper.h"

#ifdef WITH_OPENHAPTICS

#include <stdio.h>

#include "vislib/math/Matrix.h"
#include "vislib/math/Vector.h"
#include "vislib/sys/Log.h"

/*
 * PhantomDeviceWrapper::PhantomDeviceWrapper
 */
PhantomDeviceWrapper::PhantomDeviceWrapper(void) :
    hDevice(HD_INVALID_HANDLE), hRenderingContext(NULL),
    springForce(0), constantForce(0), motionInterrupts(false),
    maxSpringForce(1.0), springGain(0.25), objectPositions(NULL), 
    touch(false), dragging(false), objectPosSet(false), 
    dragObject(-1), touchObject(-1), 
    modelviewMatrix() {
    // intentionally empty
}


/*
 * PhantomDeviceWrapper::~PhantomDeviceWrapper
 */
PhantomDeviceWrapper::~PhantomDeviceWrapper(void) {
    // free up the device before destruction
    this->Deinitialize();
}


/*
 * PhantomDeviceWrapper::Initialize
 */
bool PhantomDeviceWrapper::Initialize(void) {
    // Initialize the default device
    this->hDevice = hdInitDevice(HD_DEFAULT_DEVICE);

    if (HD_DEVICE_ERROR(hdGetError())) {
        // Return false if device failed to initialize
        return false;
    }
    // Otherwise, make this device context the current one
    this->hRenderingContext = hlCreateContext(this->hDevice);
    hlMakeCurrent(this->hRenderingContext);
    
    // Add all event callbacks except for motion (which must be manually set)
    hlAddEventCallback(HL_EVENT_TOUCH, HL_OBJECT_ANY, HL_COLLISION_THREAD, onTouch, this);
    hlAddEventCallback(HL_EVENT_UNTOUCH, HL_OBJECT_ANY, HL_COLLISION_THREAD, onTouch, this);
    hlAddEventCallback(HL_EVENT_1BUTTONDOWN, HL_OBJECT_ANY, HL_CLIENT_THREAD, onButton, this);
    hlAddEventCallback(HL_EVENT_1BUTTONUP, HL_OBJECT_ANY, HL_CLIENT_THREAD, onButton, this);

    return true;
}


/*
 * PhantomDeviceWrapper::Initialize
 */
bool PhantomDeviceWrapper::Initialize(HDstring pConfigName) {
    // Initialize the device
    this->hDevice = hdInitDevice(pConfigName);

    if (HD_DEVICE_ERROR(hdGetError())) {
        // Return false if device failed to initialize
        return false;
    }
    // Otherwise, make this device context the current one
    this->hRenderingContext = hlCreateContext(this->hDevice);
    hlMakeCurrent(this->hRenderingContext);

    // Add all event callbacks except for motion (which must be manually set)
    hlAddEventCallback(HL_EVENT_TOUCH, HL_OBJECT_ANY, HL_COLLISION_THREAD, onTouch, this);
    hlAddEventCallback(HL_EVENT_UNTOUCH, HL_OBJECT_ANY, HL_COLLISION_THREAD, onTouch, this);
    hlAddEventCallback(HL_EVENT_1BUTTONDOWN, HL_OBJECT_ANY, HL_CLIENT_THREAD, onButton, this);
    hlAddEventCallback(HL_EVENT_1BUTTONUP, HL_OBJECT_ANY, HL_CLIENT_THREAD, onButton, this);
    return true;
}


/*
 * PhantomDeviceWrapper::SetCursor
 */
void PhantomDeviceWrapper::SetCursor(vislib::graphics::AbsoluteCursor3D* cursor3d) {
    this->cursor = cursor3d;
    vislib::math::Vector <double, 3> init(0.0, 0.0, -1.0);
    this->cursor->SetInitialOrientation(init); // set the orientation that the rotation is applied to
    this->hasCursor = true; // flag that the device has a cursor
}


/*
 * PhantomDeviceWrapper::UpdateWorkspace
 */
void PhantomDeviceWrapper::UpdateWorkspace(double* projection) {
    hlMatrixMode(HL_TOUCHWORKSPACE);
    hlLoadIdentity();
    
    // Fit haptic workspace to view volume.
    hluFitWorkspace(projection);
}


/*
 * PhantomDeviceWrapper::UpdateDevice
 */
void PhantomDeviceWrapper::UpdateDevice(void) {
    hlCheckEvents(); // check for button/motion/touch events

    if (this->dragObject != -1 && this->objectPositions == NULL) {
        // currently dragging an object and not using the object position
        if (this->objectPosSet == true) {
            // object position has been set, so continue
            if (dragging == false) {
                // not yet dragging an object, so establish the offset
                this->objectToTouch = this->clickPosition - this->currentObjectPos;
                this->springPoint = this->clickPosition; // store spring attach point
                // put a spring force at the click position in device coordinates
                this->StartSpringForce(this->ModelToWorkspaceTransform(this->springPoint));
                dragging = true; // start dragging
        
            } else {
                // already dragging at object - update spring anchor point based on new object pos + offset
                this->springPoint = this->currentObjectPos + this->objectToTouch;
                // update the spring force
                this->StartSpringForce(this->ModelToWorkspaceTransform(this->springPoint));
            }
        }
    } else if (this->objectPositions != NULL) {
        // object position struct - use this to compute new object pos
        if (this->cursor->GetButtonState(0)) {

            // if button is down, check if currently dragging an object
            if (this->dragObject != -1) {

                // already dragging an object - get the array index
                int arrayIndex = this->dragObject - this->objectPositions->offset;
                if (arrayIndex >= this->objectPositions->numObjects || arrayIndex < 0) {
                    //printf("Object not in array.\n"); // object wasn't in array, so just ignore it
                } else {
                    // get point at object position
                    vislib::math::Point<float, 3> objectPos(&this->objectPositions->positions[arrayIndex*3]);

                    // add offset vector to get new spring attach point
                    this->springPoint = objectPos + this->objectToTouch;
                    this->StartSpringForce(this->ModelToWorkspaceTransform(this->springPoint)); // apply force
                }

            } else if (this->touch) {

                // button down and touching an object - start dragging that object
                this->dragObject = this->touchObject; // set the dragObject

                int arrayIndex = this->dragObject; // index into positions array
                arrayIndex -= this->objectPositions->offset;
                if (arrayIndex >= this->objectPositions->numObjects || arrayIndex < 0) {
                    //printf("Object not in array.\n"); // object wasn't in array, so just ignore it
                } else {
                    // get point at object position
                    vislib::math::Point<float, 3> objectPos(&this->objectPositions->positions[arrayIndex*3]);

                    // set touch offset
                    this->objectToTouch = this->clickPosition - objectPos;
                    this->springPoint = this->clickPosition; // store spring attach point

                    // put a spring force at the touch poisition in device coordinates
                    this->StartSpringForce(this->ModelToWorkspaceTransform(this->springPoint));
                }
            }
        } else {
            // button released - stop dragging
            this->dragObject = -1; // drag object of -1 means no drag object
            dragging = false;
            this->StopSpringForce();
        }
    } else {
        // button released - stop dragging
        this->dragObject = -1;
        this->objectPosSet = false;
        dragging = false;
        this->StopSpringForce();
    }
}


/*
 * PhantomDeviceWrapper::StartSpringForce
 */
void PhantomDeviceWrapper::StartSpringForce (vislib::math::Point<double, 3> force) {
    // Check that the custom force has been assigned an identifier and give it one if not
    if (!hlIsEffect(this->springForce)) {
        this->springForce = hlGenEffects(1);
    }

    // Set the force components
    hlEffectd(HL_EFFECT_PROPERTY_MAGNITUDE, this->maxSpringForce); // maximum spring force that will be applied
    hlEffectd(HL_EFFECT_PROPERTY_GAIN, this->springGain); // spring gain (higher = tighter spring)
    hlEffectdv(HL_EFFECT_PROPERTY_POSITION, force.PeekCoordinates()); // spring "attach point"

    // check if the force is already active - if so, just update it
    HLboolean active = NULL;
    hlGetEffectbv(this->springForce, HL_EFFECT_PROPERTY_ACTIVE, &active);
    if (active == false) {
        // start up the force
        hlStartEffect(HL_EFFECT_SPRING, this->springForce);
    } else {
        // update the already running force
        hlUpdateEffect(this->springForce);
    }
}


/*
 * PhantomDeviceWrapper::StopSpringForce
 */
void PhantomDeviceWrapper::StopSpringForce (void) {
    if (!hlIsEffect(this->springForce)) {
        return; // force hasn't been allocated yet
    }
    HLboolean active = NULL;
    hlGetEffectbv(this->springForce, HL_EFFECT_PROPERTY_ACTIVE, &active);
    if (active == false) {
        return; // force isn't active
    } else {
        hlStopEffect(this->springForce); // stop the force
    }
}


/*
 * PhantomDeviceWrapper::SetSpringAttributes
 */
void PhantomDeviceWrapper::SetSpringAttributes(double maxForce, double gain) {
    // check spring stiffness for safety
    double kLimit;
    hdGetDoublev(HD_NOMINAL_MAX_STIFFNESS, &kLimit);
    if (gain > (kLimit - 0.1)) {
        gain = kLimit - 0.1;
    }
    this->springGain = gain;

    // check max spring force for safety
    double fLimit;
    hdGetDoublev(HD_NOMINAL_MAX_CONTINUOUS_FORCE, &fLimit);
    if (maxForce > (fLimit - 0.1)) {
        maxForce = fLimit - 0.1;
    }
    this->maxSpringForce = maxForce;
}


/*
 * PhantomDeviceWrapper::StartConstantForce
 */
void PhantomDeviceWrapper::StartConstantForce (vislib::math::Vector<double, 3> force) {
    // Check that the custom force has been assigned an identifier and give it one if not
    if (!hlIsEffect(this->constantForce)) {
        this->constantForce = hlGenEffects(1);
    }
        
    // Get the magnitude of the force from the vector magnitude
    double magnitude = force.Normalise();

    // Check if the force is 0 - if so, call the stop force function
    if (magnitude == 0) {
        this->StopConstantForce();
        return;
    }

    // Check that the force does not exceed device limits
    HDdouble nominalMaxContinuousForce;
    hdGetDoublev(HD_NOMINAL_MAX_CONTINUOUS_FORCE, &nominalMaxContinuousForce);
    if (magnitude > nominalMaxContinuousForce) {
        magnitude = nominalMaxContinuousForce;
    } else if (magnitude < -nominalMaxContinuousForce) {
        magnitude = -nominalMaxContinuousForce;
    }

    // Set the force components
    hlEffectd(HL_EFFECT_PROPERTY_MAGNITUDE, magnitude);
    hlEffectdv(HL_EFFECT_PROPERTY_DIRECTION, force.PeekComponents());

    // check if the force is already active - if so, just update it
    HLboolean active = NULL;
    hlGetEffectbv(this->constantForce, HL_EFFECT_PROPERTY_ACTIVE, &active);
    if (active == false) {
        // start up the force
        hlStartEffect(HL_EFFECT_CONSTANT, this->constantForce);
    } else {
        // update the already running force
        hlUpdateEffect(this->constantForce);
    }
}


/*
 * PhantomDeviceWrapper::StopConstantForce
 */
void PhantomDeviceWrapper::StopConstantForce (void) {
    if (!hlIsEffect(this->constantForce)) {
        return; // force hasn't been allocated yet
    }
    HLboolean active = false;
    hlGetEffectbv(this->constantForce, HL_EFFECT_PROPERTY_ACTIVE, &active);
    if (active == false) {
        return; // force isn't active
    } else {
        hlStopEffect(this->constantForce); // stop the force
    }
}


/*
 * PhantomDeviceWrapper::GetCursorScale
 */
double PhantomDeviceWrapper::GetCursorScale (double* modelMatrix,
    double* projMatrix, int* viewport) {
    return hluScreenToModelScale(modelMatrix, projMatrix, viewport);
}

/*
 * PhantomDeviceWrapper::SetLinearMotionTolerance
 */
void PhantomDeviceWrapper::SetLinearMotionTolerance(double distance) {
    hlEventd(HL_EVENT_MOTION_LINEAR_TOLERANCE, distance);
    // reset the motion interrupts to use the new tolerance
    this->DisableMotionInterrupts();
    this->EnableMotionInterrupts();
}

    
/*
 * PhantomDeviceWrapper::SetAngularMotionTolerance
 */
void PhantomDeviceWrapper::SetAngularMotionTolerance(double angle) {
    hlEventd(HL_EVENT_MOTION_ANGULAR_TOLERANCE, angle);
    // reset the motion interrupts to use the new tolerance
    this->DisableMotionInterrupts();
    this->EnableMotionInterrupts();
}


/*
 * PhantomDeviceWrapper::EnableMotionInterrupts
 */
void PhantomDeviceWrapper::EnableMotionInterrupts(void) {
    // check if a cursor is present
    if (this->hasCursor) {
        // check if motion interrupts are already happening
        if (!this->motionInterrupts) {
            // Add them
            hlAddEventCallback(HL_EVENT_MOTION, HL_OBJECT_ANY, HL_CLIENT_THREAD, onMotion, this);
        }
    }
}


/*
 * PhantomDeviceWrapper::DisableMotionInterrupts
 */
void PhantomDeviceWrapper::DisableMotionInterrupts(void) {
    // check if motions interrupts are happening
    if (this->motionInterrupts) {
        // Remove them
        hlRemoveEventCallback(HL_EVENT_MOTION, HL_OBJECT_ANY, HL_CLIENT_THREAD, onMotion);
    }
}


/*
 * PhantomDeviceWrapper::Deinitialize
 */
void PhantomDeviceWrapper::Deinitialize(void) {
    // Deallocate memory used by force effects
    if (hlIsEffect(this->constantForce)) {
        hlDeleteEffects(this->constantForce, 1);
    }
    if (hlIsEffect(this->springForce)) {
        hlDeleteEffects(this->springForce, 1);
    }

    // Free up the haptic rendering context.
    hlMakeCurrent(NULL);
    if (this->hRenderingContext != NULL) {
        hlDeleteContext(this->hRenderingContext);
    }

    // Free up the haptic device.
    if (this->hDevice != HD_INVALID_HANDLE) {
        hdDisableDevice(this->hDevice);
    }
}

/*
 * PhantomDeviceWrapper::SetButtonFunction
 */
void PhantomDeviceWrapper::SetButtonFunction(PhantomButtonDelegate& function) {
    this->buttonCallbackFunction = function;
}

/*
 * PhantomDeviceWrapper::ModelToWorkspaceTransform
 */
vislib::math::Point<double, 3> PhantomDeviceWrapper::ModelToWorkspaceTransform(
    vislib::math::Point<double, 3> position) {

    vislib::math::Point<double, 3> retval(0.0, 0.0, 0.0); // return value
    if (this->modelviewMatrix.IsIdentity()) {
        vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_ERROR, "No Modelview matrix set in phantom device wrapper.");
        return retval;
    }

    double* modelToviewMatrix; // for gl model-to-viewspace matrix
    double viewToTouchMatrix[16]; // for view-to-touchspace matrix
    double touchToWorkspaceMatrix[16]; // for touch-to-workspace matrix
    double modelToWorkspaceMatrix[16]; // to store composed model-to-workspace matrix
    modelToviewMatrix = this->modelviewMatrix.PeekComponents(); // get modelview matrix
    hlGetDoublev(HL_VIEWTOUCH_MATRIX, viewToTouchMatrix); // get view-to-touch matrix
    hlGetDoublev(HL_TOUCHWORKSPACE_MATRIX, touchToWorkspaceMatrix); // get touch-to-workspace matrix
    hluModelToWorkspaceTransform(modelToviewMatrix, viewToTouchMatrix,
        touchToWorkspaceMatrix, modelToWorkspaceMatrix); // get model-to-workspace matrix

    // create vislib form of model-to-workspace matrix
    vislib::math::Matrix<double, 4, vislib::math::COLUMN_MAJOR> mTWMatrix(modelToWorkspaceMatrix);
    // create 4D vector form of position (with 1 for homogeneous coordinate)
    vislib::math::Vector<double, 4> positionVector( position.GetX(),
        position.GetY(), position.GetZ(), 1.0);
    vislib::math::Vector<double, 4> result; // result vector
    result = mTWMatrix * positionVector; // do the matrix math

    // Extract the x,y,z coordinates from the result
    retval.Set(result.GetX(), result.GetY(), result.GetZ());
    return retval;
}


/*
 * PhantomDeviceWrapper::onTouch
 */
void HLCALLBACK PhantomDeviceWrapper::onTouch(HLenum event,
        HLuint object, HLenum thread, HLcache *cache, void *userdata) {

    PhantomDeviceWrapper *This = reinterpret_cast<PhantomDeviceWrapper*>(userdata);

    if (event == HL_EVENT_UNTOUCH) { 
        // store the touch data
        This->touch = false;
    } else if (event == HL_EVENT_TOUCH) {
        //printf("Object number %d\n.", object);
        // store the touch data
        This->touch = true;
        This->touchObject = object;
    } 
}


/*
 * PhantomDeviceWrapper::onMotion
 */
void HLCALLBACK PhantomDeviceWrapper::onMotion(HLenum event,
        HLuint object, HLenum thread, HLcache *cache, void *userdata) {

    double transform[16];
    PhantomDeviceWrapper *This = reinterpret_cast<PhantomDeviceWrapper*>(userdata);

    // Get the transform matrix from the cache
    hlCacheGetDoublev(cache, HL_PROXY_TRANSFORM, transform);
    vislib::math::Matrix<double, 4, vislib::math::COLUMN_MAJOR> transformMat(transform);

    // store the motion data
    This->cursor->SetTransform(transformMat);
}


/*
 * PhantomDeviceWrapper::onButton
 */
void HLCALLBACK PhantomDeviceWrapper::onButton(HLenum event,
    HLuint object, HLenum thread, HLcache *cache, void *userdata) {

    PhantomDeviceWrapper* This = reinterpret_cast<PhantomDeviceWrapper*>(userdata);    
    if (event == HL_EVENT_1BUTTONDOWN) { 
        // get the click position from the cache
        double position[3];
        hlCacheGetDoublev(cache, HL_PROXY_POSITION, position);
        vislib::math::Point<float, 3> pointPosition(static_cast<float>(position[0]), 
            static_cast<float>(position[1]), 
            static_cast<float>(position[2]));

        // store the button push data
        This->cursor->SetButtonState(0, true);
        This->clickPosition = pointPosition;

        This->dragObject = This->buttonCallbackFunction(true, This->touch);
    } else if (event == HL_EVENT_1BUTTONUP) {
        // store the button release data
        This->cursor->SetButtonState(0, false);

        This->dragObject = -1;
        This->buttonCallbackFunction(false, This->touch);
    }
}

#endif // WITH_OPENHAPTICS
