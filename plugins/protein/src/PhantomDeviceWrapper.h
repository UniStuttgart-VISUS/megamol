/*
 * PhantomDeviceWrapper.h
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifdef WITH_OPENHAPTICS

#ifndef PHANTOMDEVICEWRAPPER_H_INCLUDED
#define PHANTOMDEVICEWRAPPER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include <HL/hl.h>
#include <HLU/hlu.h>
#include "vislib/math/Vector.h"
#include "vislib/graphics/AbsoluteCursor3D.h"
#include "vislib/Delegate.h"


/**
* This class serves as a wrapper class for a Phantom pen device.
* The class handles initialization and deinitialization of the
* device and provides simple Get/Set routines to perform cursor
* position/orientation input and force feedback output.
*/

// function pointer class for callback function. First param is button click state
// Second param is whether or not the cursor is touching an object according to the device.
// Return value is object id that was clicked (-1 for no object)
typedef vislib::Delegate<unsigned int, bool, bool> PhantomButtonDelegate; 

class PhantomDeviceWrapper {
public:

    /** 
     * Struct containing a mapping of shape ids (provided by the haptic rendering
     * service) and shape locations for use in relocating spring attach point while
     * moving objects.
     */
    struct ObjectPositionData {
        int offset; // array offset to reach first shape id (equivalently, first id in range)
        int numObjects; // number of objects in array
        /** 
         * Pointer to array of object positions. Object positions should be stored as
         * x0, y0, z0, x1, y1, z1, etc. Array will be 3*numObjects in length.
         */
        float* positions;
    };

    /** 
     * Ctor
     */
    PhantomDeviceWrapper(void);


    /**
     * Dtor
     */
    virtual ~PhantomDeviceWrapper(void);


    /**
     * Initializes the phantom device (uses the first one it finds, if any).
     * Also sets up event callbacks other than motion.
     *
     * @return True if successful.
     */
    bool Initialize(void);


    /**
     * Initializes the phantom device using the config name provided.
     * Device config names are in the control panel "Phantom Configuration".
     * Also sets up event callbacks other than motion.
     *
     * @param pConfigName Device name.
     * @return True if successful.
     */
    bool Initialize(HDstring pConfigName);


    /**
     * Updates the device workspace to fit the projection matrix.
     *
     * @param Projection matrix as array of 16 doubles (such as that obtained by calling
     * glGetDoublev with GL_PROJECTION_MATRIX as the first parameter)
     */
    void UpdateWorkspace(double* projection);


    /**
     * Starts the spring force or updates an already existing spring force to a new
     * location.
     * Spring force is specified by a location to attach the "spring" to. The force
     * feedback will attempt to move the cursor to this position (without exceeding
     * device force feedback limits).
     *
     * @param force Point representing new position for the spring force.
     */
    void StartSpringForce(vislib::math::Point<double, 3> force);


    /** 
     * Stops the spring force.
     */
    void StopSpringForce(void);


    /**
     * Sets spring force attributes. See params:
     *
     * @param maxForce Largest force that will be applied. Value will be checked against
     * device limits, but care should be taken to avoid excessive forces anyway.
     * @param gain Spring coefficient, i.e. measure of spring tightness.
     */
    void SetSpringAttributes(double maxForce, double gain);

    /**
     * Starts a constant force or updates an already existing constant force to a new
     * magnitude and/or direction. 
     * Force is specified by a vector of doubles whose magnitude is the magnitude of
     * the force. The vector should be stated in device coordinates (positive x is
     * right, positive y is up, positive z is out of the screen).
     * Safety checks are run to ensure the force does not exceed device limits
     * but care should be taken to avoid excessively large forces anyway.
     * Starting a force that has a magnitude of 0 is equivalent to calling the
     * StopConstantForce routine.
     *
     * @param force Vector representing new force to be applied (magnitude and direction)
     */
    void StartConstantForce(vislib::math::Vector<double, 3> force);


    /** 
     * Stops a constant force.
     */
    void StopConstantForce(void);


    /**
     * Uses the gl modelMatrix, projMatrix, and viewport parameters to determine an
     * appropriate scaling factor for the cursor and returns this factor. The param
     * values can be found by calling the appropriate glGet* functions.
     *
     * @param modelMatrix Array of 16 doubles representing the 4x4 model matrix.
     * @param projMatrix Array of 16 doubles representing the 4x4 projection matrix.
     * @param viewport Array of 4 ints representing the viewport.
     * @return Double scaling factor for cursor size.
     */
    double PhantomDeviceWrapper::GetCursorScale (double* modelMatrix,
        double* projMatrix, int* viewport);


    /**
     * Return the device position as a point of three doubles.
     *
     * @return Point of doubles specifying device position.
     */
    inline vislib::math::Point<double, 3> GetDevicePosition(void) {
        double position[3];
        hlGetDoublev(HL_DEVICE_POSITION, position);
        vislib::math::Point<double, 3> retval(position);
        return retval;
    }


    /**
     * Return the proxy position as a point of three doubles.
     *
     * @return Point of doubles specifying proxy position.
     */
    inline vislib::math::Point<double, 3> GetProxyPosition(void) {
        double position[3];
        hlGetDoublev(HL_PROXY_POSITION, position);
        vislib::math::Point<double, 3> retval(position);
        return retval;
    }


    /**
     * Return the device rotation as a quaternion of doubles.
     * Rotation should be performed on a unit vector in the z direction.
     *
     * @return Vislib quaternion of doubles representing device rotation from unit z vector.
     */
    inline vislib::math::Quaternion<double> GetDeviceRotation(void) {
        double rotation[4];
        hlGetDoublev(HL_DEVICE_ROTATION, rotation);
        vislib::math::Quaternion<double> retval(rotation[1], rotation[2], rotation[3], rotation[0]);
        return retval;
    }


    /**
     * Return the proxy rotation as a quaternion of doubles.
     * Rotation should be performed on a unit vector in the z direction.
     *
     * @return Vislib quaternion of doubles representing proxy rotation from unit z vector.
     */
    inline vislib::math::Quaternion<double> GetProxyRotation(void) {
        double rotation[4];
        hlGetDoublev(HL_PROXY_ROTATION, rotation);
        vislib::math::Quaternion<double> retval(rotation[1], rotation[2], rotation[3], rotation[0]);
        return retval;
    }


    /**
     * Return the device transform matrix. Matrix generated is column-major.
     *
     * @return Vislib 4x4 column major matrix of doubles representing device position and rotation transform.
     */
    inline vislib::math::Matrix<double, 4, vislib::math::COLUMN_MAJOR> GetDeviceTransform(void) {
        double transform[16];
        hlGetDoublev(HL_DEVICE_TRANSFORM, transform);
        vislib::math::Matrix<double, 4, vislib::math::COLUMN_MAJOR> retval(transform);
        return retval;
    }


    /**
     * Return the proxy transform matrix. Matrix generated is column-major.
     *
     * @return Vislib 4x4 column major matrix of doubles representing proxy position and rotation transform.
     */
    inline vislib::math::Matrix<double, 4, vislib::math::COLUMN_MAJOR> GetProxyTransform(void) {
        double transform[16];
        hlGetDoublev(HL_PROXY_TRANSFORM, transform);
        vislib::math::Matrix<double, 4, vislib::math::COLUMN_MAJOR> retval(transform);
        return retval;
    }


    /**
     * Runs through the device events list and checks for new events.
     * This must be called within the haptic rendering frame in order to process
     * button and motion events.
     */
    void UpdateDevice(void);

    /**
     * Sets the associated absolute 3D cursor, and configures it to device data organization.
     *
     * @param cursor3d Absolute 3D cursor object to be set.
     */
    void SetCursor(vislib::graphics::AbsoluteCursor3D *cursor3d);

    /**
     * Sets the minimum distance (in millimeters) the device must travel to trigger
     * a motion event. (default is 1.0 mm)
     *
     * @param distance Distance in millimeters device must travel to trigger motion event.
     */
    void SetLinearMotionTolerance(double distance);

    
    /**
     * Sets the minimum rotation (in radians) the device must travel to trigger
     * a motion event. (default is 0.02 radians)
     *
     * @param angle Rotation in radians device must travel to trigger motion event.
     */
    void SetAngularMotionTolerance(double angle);


    /** 
     * Sets the associated object position data struct.
     *
     * @param data Pointer to object position data struct.
     */
    inline void SetObjectPositionData(ObjectPositionData* data) {
        this->objectPositions = data;
    }


    /**
     * Updates the object position for single object drag mode.
     *
     * @param position Vislib point representing new object position.
     */
    inline void SetObjectPosition(vislib::math::Point<float, 3> position) {
        this->currentObjectPos = position;
        this->objectPosSet = true;
    }


    /**
     * Sets the GL modelview matrix to be used by this class.
     * Must be called before spring forces are applied in order to apply them in the
     * correct place.
     *
     * @param modelview A 4x4 vislib matrix in column major form of the gl modelview matrix.
     */
    inline void SetModelviewMatrix(vislib::math::Matrix<double, 4, vislib::math::COLUMN_MAJOR> modelview) {
        this->modelviewMatrix = modelview;
    }


    /**
     * Sets the GL modelview matrix to be used by this class.
     * Must be called before spring forces are applied in order to apply them in the
     * correct place.
     *
     * @param modelview A 16 value array of doubles representing the matrix in column major form.
     */
    inline void SetModelviewMatrix(double *modelview) {
        this->modelviewMatrix = modelview;
    }

    
    /**
     * Gets the current spring anchor point. Useful for computing the force being
     * applied to an object. Returns the spring point as a vislib point of floats.
     *
     * @return Spring anchor point as vislib point.
     */
    inline vislib::math::Point<float, 3> GetSpringAnchorPoint(void) {
        return this->springPoint;
    }

    
    /**
     * Enables motion interrupts setting cursor location. Motion interrupts
     * cannot be enabled if there is no associated 3d cursor. Motion interrupts will automatically
     * set cursor location in the 3d cursor.
     */
    void EnableMotionInterrupts(void);
    

    /**
     * Disables motion interrupts setting cursor location. Program will have to poll device and
     * manually set cursor position.
     */
    void DisableMotionInterrupts(void);

    /**
    * Sets the function that is called on a button click or release.
    * 
    * @param Delegate of the function to be called. 
    */
    void SetButtonFunction(PhantomButtonDelegate& function);

    /**
     * Converts position data from the modelview coordinates to device workspace
     * coordinates (needed for proper event position handling).
     *
     * @param position Point to be transformed.
     * @return Point after transformation.
     */
    vislib::math::Point<double, 3> ModelToWorkspaceTransform(vislib::math::Point<double, 3> position);

private:

    /**
     * Frees the phantom device and associated memory.
     * Called in destructor.
     */
    void Deinitialize(void);

    /** Callback function for on object touch */
    static void HLCALLBACK onTouch(HLenum event,
        HLuint object, HLenum thread, HLcache *cache, void *userdata);

    /** Callback function for device motion */
    static void HLCALLBACK onMotion(HLenum event, HLuint object,
        HLenum thread, HLcache *cache, void *userdata);

    /** Callback function for device button click */
    static void HLCALLBACK onButton(HLenum event, HLuint object,
        HLenum thread, HLcache *cache, void *userdata);

    // Private variables //

    /** Haptic device and rendering context handles. */
    HHD hDevice;
    HHLRC hRenderingContext;

    /** If true, motion interrupts are setting cursor position. Otherwise device will have to be polled */
    bool motionInterrupts;

    /** Custom force identifiers */
    HLuint constantForce;
    HLuint springForce;

    /** Spring force attributes */
    double maxSpringForce; // maximum spring force allowed
    double springGain;

    /** Associated absolute cursor 3D */
    vislib::graphics::AbsoluteCursor3D *cursor;
    bool hasCursor; // true if the phantom wrapper has an associated cursor

    /** Associated object position data for hanging on to objects */
    ObjectPositionData* objectPositions;

    /** Current drag object position */
    vislib::math::Point<float, 3> currentObjectPos;

    /** Delegate pointing to on button event callback function */
    PhantomButtonDelegate buttonCallbackFunction;

    /** GL Modelview matrix used for transforms */
    vislib::math::Matrix<double, 4, vislib::math::COLUMN_MAJOR> modelviewMatrix;

    /** Object grabbing variables */
    bool touch; // is touch currently active
    bool dragging; // is drag currently active
    bool objectPosSet; // has a single object position been set 
    unsigned int dragObject; // object currently being dragged
    unsigned int touchObject; // which object is being touched
    vislib::math::Point<float, 3> clickPosition; // point in space where click occurred
    vislib::math::Vector<float, 3> objectToTouch; // vector pointing from object "position" to clickPosition
    vislib::math::Point<float, 3> springPoint; // anchor point of spring

};


#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif // PHANTOMDEVICEWRAPPER_H_INCLUDED

#endif // WITH_OPENHAPTICS
