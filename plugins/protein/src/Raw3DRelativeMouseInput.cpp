/*
 * Raw3DRelativeMouseInput.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Raw3DRelativeMouseInput.h"

#include "vislib/assert.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/StringConverter.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib/sys/SystemException.h"
#include "vislib/Trace.h"

#define INPUT_DEBUG 0 // If nonzero, console will print out data values received.

// Must have NIIF_NODE for default parameters, even if _WIN32_IE is before
// Windows 2000.
#ifndef NIIF_NONE
#define NIIF_NONE 0
#endif /* !NIIF_NONE */

#define LOGITECH_VENDOR_ID 0x46d
#define DEFAULT_RAD_PER_TICK 8.0e-6f // default radians per tick (8.0e-6 from 3dConnexion SDK)

#ifdef _WIN32


/*
 * Raw3DRelativeMouseInput::Raw3DRelativeMouseInput
 */
Raw3DRelativeMouseInput::Raw3DRelativeMouseInput(void) : radiansPerTick(DEFAULT_RAD_PER_TICK) {
    this->init(::GetModuleHandleW(NULL));
}


/*
 * Raw3DRelativeMouseInput::Raw3DRelativeMouseInput
 */
Raw3DRelativeMouseInput::Raw3DRelativeMouseInput(HINSTANCE hInstance) {
    this->init(hInstance);
}


/*
 * Raw3DRelativeMouseInput::~Raw3DRelativeMouseInput
 */
Raw3DRelativeMouseInput::~Raw3DRelativeMouseInput(void) {
    if (this->hWnd != NULL) {
        ::DestroyWindow(this->hWnd);
        this->hWnd = NULL;
    }
}

/* 
 * Raw3DRelativeMouseInput::Initialize
 */
void Raw3DRelativeMouseInput::Initialize(void) {
	using namespace::vislib::sys; // for error handling

    /*
     * Create the message only window.
     */      
    /* Register window class, if necessary. */
    if (!this->registerWndClass()) {
        throw SystemException(__FILE__, __LINE__);
    }

    /* Destroy old window, if any. */
    if (this->hWnd != NULL) {
        ::DestroyWindow(this->hWnd);
    }
        
    /* Create control window. */    
	if ((this->hWnd = ::CreateWindowW(Raw3DRelativeMouseInput::WNDCLASSNAME, L"", WS_POPUP,
            CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, 
            HWND_MESSAGE, 0, this->hInstance, 0)) == NULL) {
        throw SystemException(__FILE__, __LINE__);
    }

    /* Register pointer to this object as user data. */
#pragma warning(disable: 4311)
    ::SetWindowLongPtrW(this->hWnd, GWLP_USERDATA, 
#ifdef _WIN64
        reinterpret_cast<LONG_PTR>
#else  /* _WIN64 */
        reinterpret_cast<LONG>
#endif /* _WIN64 */
        (this));
#pragma warning(default: 4311)

    ASSERT(this->hWnd != NULL);

	const unsigned int numDevices = 2;
	RAWINPUTDEVICE rid[numDevices];

	rid[0].usUsagePage = 0x01;
	rid[0].usUsage = 0x08; // adds 6 degrees-of-freedom devices
	rid[0].dwFlags = 0;
	rid[0].hwndTarget = hWnd; // associates the device with the message window just set up

	/* TODO: FIX: For some reason the spacenavigator won't send/trigger messages
	 * in the hidden window unless the keyboard is included on the raw input
	 * device registry for the window. This is completely baffling.
	 */
	rid[1].usUsagePage = 0x01;
	rid[1].usUsage = 0x06; // adds keyboard
	rid[1].dwFlags = 0;
	rid[1].hwndTarget = hWnd; 

	/* Register the device with the system and throw an exception if failure. */
	if (RegisterRawInputDevices(rid, numDevices, sizeof(rid[0])) == FALSE) {
		throw SystemException(__FILE__, __LINE__);
	}
}


/*
 * Raw3DRelativeMouseInput::wndProc
 */
LRESULT WINAPI Raw3DRelativeMouseInput::wndProc(HWND hWnd, UINT msg, 
        WPARAM wParam, LPARAM lParam) {
#pragma warning(disable: 4312)
    Raw3DRelativeMouseInput *thisPtr = reinterpret_cast<Raw3DRelativeMouseInput *>(
        ::GetWindowLongPtr(hWnd, GWLP_USERDATA));
#pragma warning(default: 4312)

	// Did we receive a message from raw input?
    if (msg == WM_INPUT) {

#if INPUT_DEBUG
		megamol::core::utility::log::Log::DefaultLog.WriteMsg( megamol::core::utility::log::Log::LEVEL_INFO, "Raw Input Received");
#endif
		
		bool newData = false; // set if new motion data was received by ProcessRawInput
		const int bufferSize = 1024; // arbitrary number - possible max buffer size
		char pBuffer[bufferSize]; // buffer array
		PRAWINPUT pRawInput = reinterpret_cast<PRAWINPUT>(pBuffer); // store the first input in the buffer
		HRAWINPUT hRawInput = reinterpret_cast<HRAWINPUT>(lParam); // get the handle to the raw input data package
		UINT inputSize = bufferSize; // will contain size of input package in bytes
		UINT nCount = 0; // will contain the number of raw input packages in the buffer
		// Clear the current vector data
		thisPtr->clearData();

		/* Step 1:
		 * Get the raw input data directly from the handle in lParam.
		 * Check that this data is of an appropriate sort, then process it,
		 * making all the necessary function calls.
		 */
		// Get the data from the raw input - return of -1 is an error
		if (GetRawInputData(hRawInput, RID_INPUT, pRawInput, &inputSize, sizeof(RAWINPUTHEADER)) == static_cast<UINT>(-1)) {
			megamol::core::utility::log::Log::DefaultLog.WriteMsg( megamol::core::utility::log::Log::LEVEL_ERROR,
				"Could not get initial raw input data.");
		}

		// Ignore non-hid data (RIM_TYPEHID)
		if (pRawInput->header.dwType != RIM_TYPEHID) {
			DefRawInputProc(&pRawInput, 1, sizeof(RAWINPUTHEADER));
		} else {
			// Process hid data
			newData = thisPtr->ProcessRawData(pRawInput);
		}

		/* Step 2:
		 * Check the raw input buffer for additional data (e.g. rotation data)
		 * Clear out the buffer, overwriting old data as we go along, so we have
		 * the newest copy of data (even if that is a bunch of 0's).
		 */
		inputSize = bufferSize;
		pRawInput = reinterpret_cast<PRAWINPUT>(pBuffer);
		// Get the data from the raw input buffer - return of -1 is an error
		// TODO: MSDN says there is an alignment error with this function under WOW64 - correct for this
		nCount = GetRawInputBuffer(pRawInput, &inputSize, sizeof(RAWINPUTHEADER));
		if (nCount == static_cast<UINT>(-1)) {
			megamol::core::utility::log::Log::DefaultLog.WriteMsg( megamol::core::utility::log::Log::LEVEL_ERROR,
				"Could not get first raw input data buffer.");
		}

#if INPUT_DEBUG
		megamol::core::utility::log::Log::DefaultLog.WriteMsg( megamol::core::utility::log::Log::LEVEL_INFO, "Initial buffer count: %d", nCount);
#endif

		// Keep checking the buffer and clearing out data until the most recent is obtained
		while (nCount > 0 && nCount != static_cast<UINT>(-1)) {
			PRAWINPUT pCurrentInput = pRawInput; // next raw input package to check
			UINT nInput;

			/* Go through the buffer and toss out keyboard events, as well as old
			 * 3d mouse events. We only want the most recent 3d mouse event; all else
			 * can be ignored.
			 */
			for (nInput = 0; nInput < nCount; nInput += 1) {
				// Filter out non-hid data (RIM_TYPEHID)
				if (pRawInput->header.dwType != RIM_TYPEHID) {
					DefRawInputProc(&pCurrentInput, 1, sizeof(RAWINPUTHEADER));
				} else {
					// Store the new data, if any
					newData |= thisPtr->ProcessRawData(pCurrentInput);
				}

				// Get the next raw data in the buffer
				/* TODO: FIX: The following line really should be 
				pCurrentInput = NEXTRAWINPUTBLOCK(pCurrentInput);				
				 * but for some reason this causes compiler errors because it can't find
				 * the definition for QWORD. The way it is shown now is a de-macroed version
				 * of the NEXTRAWINPUTBLOCK macro using DWORDs.
				 */
				pCurrentInput = (PRAWINPUT)((((ULONG_PTR)((PBYTE)pCurrentInput + pCurrentInput->header.dwSize)) + sizeof(DWORD) - 1) & ~(sizeof(DWORD) - 1));
			}

			inputSize = bufferSize;
			nCount = GetRawInputBuffer(pRawInput, &inputSize, sizeof(RAWINPUTHEADER));
#if INPUT_DEBUG
			megamol::core::utility::log::Log::DefaultLog.WriteMsg( megamol::core::utility::log::Log::LEVEL_INFO, "Next buffer count: %d", nCount);
#endif
		}

		// At this point, the buffer is cleared
		// Run the motion routine if necessary on the new data
		if (newData) {
			thisPtr->runMotion();
		}
    } // done dealing with WM_INPUT

    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}


/*
 * Raw3DRelativeMouseInput::ProcessRawData
 */
bool Raw3DRelativeMouseInput::ProcessRawData(PRAWINPUT pRawInput) {
	bool flag = false; // set if new motion data is received

	if (pRawInput == 0) {
		return false; // Null pointer
	}

	// Time to process pRawInput
	RID_DEVICE_INFO sDeviceInfo; // device information struct
	sDeviceInfo.cbSize = sizeof(RID_DEVICE_INFO);
	UINT cbSize = sizeof(RID_DEVICE_INFO); // for error checking

	// get information about device type
	if (GetRawInputDeviceInfo(pRawInput->header.hDevice, RIDI_DEVICEINFO,
		&sDeviceInfo, &cbSize) != cbSize) {
		megamol::core::utility::log::Log::DefaultLog.WriteMsg( megamol::core::utility::log::Log::LEVEL_ERROR,
			"Error while retrieving raw input device info.");
		// don't return yet - data still received, so run the DefRawInputProc below
	} else {
		// verify device is Logitech brand
		if (sDeviceInfo.hid.dwVendorId == LOGITECH_VENDOR_ID) {
			/* Retrieve the data that was input.
			* bRawData == 0x01 for position data, 0x02 for rotation data,
			* and 0x03 for button data. If bRawData == 0x01 && dwSizeHid >= 13,
			* the device is sending position data and rotation data.
			*/
			if (pRawInput->data.hid.bRawData[0] == 0x01) { // Translation vector
				short* pData = reinterpret_cast<short*>(&pRawInput->data.hid.bRawData[1]);
				// Store the data and set the flag
				this->tx = pData[0];
				this->ty = pData[1];
				this->tz = pData[2];
				flag = true;

#if INPUT_DEBUG
				megamol::core::utility::log::Log::DefaultLog.WriteMsg( megamol::core::utility::log::Log::LEVEL_INFO,
					"Position data received: x=%d, y=%d, z=%d", pData[0], pData[1], pData[2]);
#endif
				// check for rotation data, too
				if (pRawInput->data.hid.dwSizeHid >= 13) {
					// Store the data and set the flag
					this->rx = pData[3];
					this->ry = pData[4];
					this->rz = pData[5];
					flag = true;	

#if INPUT_DEBUG				
				megamol::core::utility::log::Log::DefaultLog.WriteMsg( megamol::core::utility::log::Log::LEVEL_INFO,
					"Rotation data received: rx=%d, ry=%d, rz=%d", pData[3], pData[4], pData[5]);
#endif
				}
			} else if (pRawInput->data.hid.bRawData[0] == 0x02) { // Rotation vector
				short* pData = reinterpret_cast<short*>(&pRawInput->data.hid.bRawData[1]);
				// Store the new data and set the flag
				this->rx = pData[0];
				this->ry = pData[1];
				this->rz = pData[2];
				flag = true;

#if INPUT_DEBUG
				megamol::core::utility::log::Log::DefaultLog.WriteMsg( megamol::core::utility::log::Log::LEVEL_INFO,
					"Rotation data received: rx=%d, ry=%d, rz=%d", pData[0], pData[1], pData[2]);
#endif				
			} else if (pRawInput->data.hid.bRawData[0] == 0x03) { // Button data
				// Get the button data unsigned long
				unsigned long keyState = static_cast<unsigned long>(pRawInput->data.hid.bRawData[1]);
				// Make sure the button function has been set
				if (this->pButton.IsTargetSet()) {
					// Call the button function
					this->pButton(keyState);
				} else {
					// fail silently
				}

#if INPUT_DEBUG
				megamol::core::utility::log::Log::DefaultLog.WriteMsg( megamol::core::utility::log::Log::LEVEL_INFO,
					"Button data received: %x", keyState);
#endif
			} else { // error of unknown nature
				megamol::core::utility::log::Log::DefaultLog.WriteMsg( megamol::core::utility::log::Log::LEVEL_ERROR,
					"Data received was not pos, rot, or button data.");
				// don't return yet - data still received, so run DefRawInputProc below
			}
		} // end vendor id check
	} // done with device info

	// process the data away
	DefRawInputProc(&pRawInput, 1, sizeof(RAWINPUTHEADER));
	return flag; // returns true only if new motion data received
}


/*
 * Raw3DRelativeMouseInput::runMotion
 */
void Raw3DRelativeMouseInput::runMotion(void) {
	// Check for a valid pMotion function
	if (!this->pMotion.IsTargetSet()) {
		return;
	}

	// Check for valid data (not all zeroes)
	bool allZeroes = ((tx == 0) && (ty == 0) && (tz == 0) && (rx == 0) && (ry == 0) && (rz == 0));

	if (!allZeroes) {
		// multiply the values by the appropriate scalars
		tx *= (this->transSpeed * radiansPerTick);
		ty *= (this->transSpeed * radiansPerTick);
		tz *= (this->transSpeed * radiansPerTick);
		rx *= (this->rotSpeed * radiansPerTick);
		ry *= (this->rotSpeed * radiansPerTick);
		rz *= (this->rotSpeed * radiansPerTick);
		// call the motion function
		this->pMotion(tx, ty, tz, rx, ry, rz);
	}
}

/* 
 * Raw3DRelativeMouseInput::clearData 
 */
void Raw3DRelativeMouseInput::clearData(void) {
	this->tx = this->ty = this->tz = this->rx = this->ry = this->rz = 0;
}

/*
 * Raw3DRelativeMouseInput::WNDCLASSNAME
 */
const wchar_t *Raw3DRelativeMouseInput::WNDCLASSNAME = L"RAW3DRELATIVEMOUSEINPUTWNDCLASS";

/*
 * Raw3DRelativeMouseInput::init
 */
bool Raw3DRelativeMouseInput::init(HINSTANCE hInstance) {
    /* Simple initialisations. */
    this->hInstance = hInstance;
    this->hWnd = NULL;
	this->transSpeed = this->rotSpeed = 1;

    return true;
}


/*
 * Raw3DRelativeMouseInput::registerWndClass
 */
bool Raw3DRelativeMouseInput::registerWndClass(void) {
    WNDCLASSEXW wndClass;

    if (!::GetClassInfoExW(this->hInstance, WNDCLASSNAME, &wndClass)) {
	    wndClass.cbSize = sizeof(WNDCLASSEX); 

	    wndClass.style = CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
        wndClass.lpfnWndProc = reinterpret_cast<WNDPROC>(Raw3DRelativeMouseInput::wndProc);
	    wndClass.cbClsExtra = 0;
	    wndClass.cbWndExtra	= 0;
	    wndClass.hInstance = this->hInstance;
	    wndClass.hIcon = 0;
	    wndClass.hCursor = 0;
	    wndClass.hbrBackground = 0;
	    wndClass.lpszMenuName = 0;
	    wndClass.lpszClassName = WNDCLASSNAME;
	    wndClass.hIconSm = 0;

        return (::RegisterClassExW(&wndClass) != FALSE);

    } else {
        /* Window class already registered. */
        return true;
    }
}

#endif /* _WIN32 */
