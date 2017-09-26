/*
 * Raw3DRelativeMouseInput.h
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef RAW3DRELATIVEMOUSEINPUT_H_INCLUDED
#define RAW3DRELATIVEMOUSEINPUT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifdef _WIN32

#include "vislib/Delegate.h"

// Note: Must set _WIN32_IE before including shell api as it goes mad
// otherwise defining it itself in a very uncontrollen manner
#ifndef _WIN32_IE
#define _WIN32_IE 0x0500
#endif /* !_WIN32_IE */

#include <shlwapi.h>
#include <Windows.h>

/**
* This class creates a hidden message window and establishes a connection with a six
* degrees-of-freedom raw input device (such as a 3dConnexion SpaceNavigator). In order
* to use this class, an instance must be created, then the Create function called, and
* finally the interrupt functions for the motion and button functions set.
*/

typedef vislib::Delegate<void, float, float, float, float, float, float> Raw3DMotionDelegate;
typedef vislib::Delegate<void, unsigned long> Raw3DButtonDelegate;


class Raw3DRelativeMouseInput {

public:

    /**
    * Create a new instance using the default instance handle returned by
    * GetModuleHandle(NULL).
    */
    Raw3DRelativeMouseInput(void);


    /**
    * Create a new instance using the specified instance handle.
    *
    * @param hInstance The instance handle.
    */
    Raw3DRelativeMouseInput(HINSTANCE hInstance);


    /**
    * Dtor.
    *
    * The dtor destroys the hidden message window.
    */
    virtual ~Raw3DRelativeMouseInput(void);


    /**
    * Creates a new message window.
    * Sets up RawInput and directs it to the message window.
    *
    * @throws SystemException If the window could not be created
    * or raw input could not be set up.
    */
    void Initialize(void);

    /**
    * Sets the function that is called when new motion data is received.
    * 
    * @param Delegate of the function to be set as the call.
    */
    void SetMotionFunction(Raw3DMotionDelegate& function) {
        this->pMotion = function;
    }

    /**
    * Sets the function that is called when new button data is received.
    *
    * @param Delegate of the function to be set as the call.
    */
    void SetButtonFunction(Raw3DButtonDelegate& function) {
        this->pButton = function;
    }

    /**
     * Sets the translation speed scalar
     *
     * @param speed The new speed scalar value
     */
    inline void SetTranslationSpeed(float speed) {
        this->transSpeed = speed;
    }

    /**
     * Sets the rotation speed scalar
     *
     * @param speed The new speed scalar value
     */
    inline void SetRotationSpeed(float speed) {
        this->rotSpeed = speed;
    }

    /**
     * Returns the translation speed scalar
     *
     * @return The speed scalar value
     */
    inline float GetTranslationSpeed(void) {
        return this->transSpeed;
    }

    /**
     * Returns the rotation speed scalar
     *
     * @return The speed scalar value
     */
    inline float GetRotationSpeed(void) {
        return this->rotSpeed;
    }

    /**
     * Sets the radian scalar for the integer values provided by the device.
     * This should not be used for speed settings - instead, use the
     * SetRotationSpeed and SetTranslationSpeed functions.
     *
     * @param rpt The new radians per tick value.
     */
    inline void SetRadiansPerTick(float rpt) {
        this->radiansPerTick = rpt;
    }

private:

    /** Data type given in first byte of raw input */
    enum RAW_DATA_TYPE {
        POSITION_DATA = 0x01,
        ROTATION_DATA = 0x02,
        BUTTON_DATA = 0x03
    };

    /**
    * Message loop function.
    *
    * Processes Raw Input messages and clears buffer to prevent propogation delay.
    */
    static LRESULT WINAPI wndProc(HWND hWnd, UINT msg, WPARAM wParam,
        LPARAM lParam);

    /**
    * Takes a raw input pointer and extracts the data.
    * If it receives new motion data (even all zeroes) it returns true.
    * If it receives new button data, it calls the button data routine but returns false.
    * In all cases, it runs DefRawInputProc on any valid raw input pointer.
    *
    * @param pRawInput is the pointer to the raw input obtained from GetRawInputData
    * or GetRawInputBuffer.
    *
    * @return True if and only if new motion data is received (rotation or translation,
    * even all zeroes), false in all other cases (including new button data)
    */
    bool ProcessRawData(PRAWINPUT pRawInput);
    
    /**
    * Checks that the data isn't entirely zeroes, and calls the assigned
    * pMotion function (if valid) with the new data.
    */
    void runMotion(void);

    /**
    * Sets the current translation and rotation vector data to zero.
    */
    void clearData(void);

    /**
    * In-ctor initialisations.
    */
    bool init(HINSTANCE hInstance);


    /** The name of the window class used for the controlling window. */
    static const wchar_t *WNDCLASSNAME;


    /**
    * Register the window class for the hidden window. The window class
    * has the name RAW3DRELATIVEMOUSEINPUTWNDCLASS. If this window class has already 
    * been registered, the method does nothing.
    *
    * @return true, if the window class has been registered or has already
    *         been registered before,
    *         false otherwise.
    */
    bool registerWndClass(void);


    /** Instance handle. */
    HINSTANCE hInstance;


    /**
    * The hidden window that processes raw input messages.
    */
    HWND hWnd;

    /** Translation vector */
    float tx, ty, tz;

    /** Rotation vector */
    float rx, ry, rz;

    /** Translation speed scalar */
    float transSpeed;

    /** Rotation speed scalar */
    float rotSpeed;

    /** Radians per tick (possibly device-specific) */
    float radiansPerTick;

    /** Delegate function for the motion function */
    Raw3DMotionDelegate pMotion;

    /** Delegate function for the button function */
    Raw3DButtonDelegate pButton;
};

#endif /* _WIN32 */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* RAW3DRELATIVEMOUSEINPUT_H_INCLUDED */