/*
 * AbstractTest.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "DXUT.h"
#include "AbstractTest.h"


/*
 * AbstractTest::~AbstractTest
 */
AbstractTest::~AbstractTest(void) {
}


/*
 * AbstractTest::OnD3D9CreateDevice
 */
HRESULT AbstractTest::OnD3D9CreateDevice(PDIRECT3DDEVICE9 pd3dDevice,
        const D3DSURFACE_DESC *pBackBufferSurfaceDesc) {
    if (pBackBufferSurfaceDesc->Height != 0) {
        this->aspectRatio = static_cast<float>(pBackBufferSurfaceDesc->Width)
            / static_cast<float>(pBackBufferSurfaceDesc->Height);
    } else {
        this->aspectRatio = 1.0f;
    }
    return D3D_OK;
}


/*
 * AbstractTest::OnD3D9DestroyDevice
 */
void AbstractTest::OnD3D9DestroyDevice(void) {
}


/*
 * AbstractTest::OnD3D9LostDevice
 */
void AbstractTest::OnD3D9LostDevice(void) {
}


/*
 * AbstractTest::OnD3D9ResetDevice
 */
HRESULT AbstractTest::OnD3D9ResetDevice(PDIRECT3DDEVICE9 pd3dDevice,
        const D3DSURFACE_DESC *pBackBufferSurfaceDesc) {
    if (pBackBufferSurfaceDesc->Height != 0) {
        this->aspectRatio = static_cast<float>(pBackBufferSurfaceDesc->Width)
            / static_cast<float>(pBackBufferSurfaceDesc->Height);
    } else {
        this->aspectRatio = 1.0f;
    }
    return D3D_OK;
}


/*
 * AbstractTest::OnFrameMove
 */
void AbstractTest::OnFrameMove(double fTime, float fElapsedTime) {
}


/*
 * AbstractTest::OnKeyboard
 */
void AbstractTest::OnKeyboard(UINT nChar, bool bKeyDown, bool bAltDown) {
}


/*
 * AbstractTest::OnMouse
 */
void AbstractTest::OnMouse(bool bLeftButtonDown, bool bRightButtonDown, 
        bool bMiddleButtonDown, bool bSideButton1Down, bool bSideButton2Down,
        INT nMouseWheelDelta, INT xPos, INT yPos) {
}


/*
 * AbstractTest::OnTestDisable
 */
void AbstractTest::OnTestDisable(void) {
}


/*
 * AbstractTest::OnTestEnable
 */
void AbstractTest::OnTestEnable(void) {
}


/*
 * AbstractTest::AbstractTest
 */
AbstractTest::AbstractTest(const vislib::StringW& name) 
        : aspectRatio(1.0f), name(name){
}
