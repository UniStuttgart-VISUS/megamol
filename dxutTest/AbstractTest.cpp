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
HRESULT CALLBACK AbstractTest::OnD3D9CreateDevice(PDIRECT3DDEVICE9 pd3dDevice, 
        const D3DSURFACE_DESC *pBackBufferSurfaceDesc) {
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
 * AbstractTest::AbstractTest
 */
AbstractTest::AbstractTest(void) {
}