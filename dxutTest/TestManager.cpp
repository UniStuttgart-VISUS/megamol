/*
 * TestManager.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "DXUT.h"
#include "TestManager.h"

#include "DXUTgui.h"


/*
 * TestManager::TestManager
 */
TestManager::TestManager(void) : activeTest(0) {
}


/*
 * TestManager::~TestManager
 */
TestManager::~TestManager(void) {
    this->OnD3D9DestroyDevice();
    this->tests.Resize(0);
}


/*
 * TestManager::OnD3D9CreateDevice
 */
HRESULT TestManager::OnD3D9CreateDevice(PDIRECT3DDEVICE9 pd3dDevice,
        const D3DSURFACE_DESC *pBackBufferSurfaceDesc) {
    ::memcpy(&this->surfaceDesc, pBackBufferSurfaceDesc, 
        sizeof(D3DSURFACE_DESC));

    for (SIZE_T i = 0; i < this->tests.Count(); i++) {
        this->tests[i]->OnD3D9CreateDevice(pd3dDevice, pBackBufferSurfaceDesc);
    }

    return D3D_OK;  // TODO
}


/*
 * TestManager::OnD3D9DestroyDevice
 */
void TestManager::OnD3D9DestroyDevice(void) {
    for (SIZE_T i = 0; i < this->tests.Count(); i++) {
        this->tests[i]->OnD3D9DestroyDevice();
    }
}


/*
 * TestManager::OnD3D9FrameRender
 */
void TestManager::OnD3D9FrameRender(PDIRECT3DDEVICE9 pd3dDevice, double fTime,
        float fElapsedTime) {
    if (this->activeTest < this->tests.Count()) {
        this->tests[this->activeTest]->OnD3D9FrameRender(pd3dDevice, fTime,
            fElapsedTime);
    }
}


/*
 * TestManager::OnD3D9LostDevice
 */
void TestManager::OnD3D9LostDevice(void) {
    for (SIZE_T i = 0; i < this->tests.Count(); i++) {
        this->tests[i]->OnD3D9LostDevice();
    }
}


/*
 * TestManager::OnD3D9ResetDevice
 */
HRESULT TestManager::OnD3D9ResetDevice(PDIRECT3DDEVICE9 pd3dDevice,
        const D3DSURFACE_DESC *pBackBufferSurfaceDesc) {
    ::memcpy(&this->surfaceDesc, pBackBufferSurfaceDesc, 
        sizeof(D3DSURFACE_DESC));

    if (this->activeTest < this->tests.Count()) {
        return this->tests[this->activeTest]->OnD3D9ResetDevice(pd3dDevice,
            pBackBufferSurfaceDesc);
    } else {
        return D3D_OK;
    }
}


/*
 * TestManager::OnFrameMove
 */
void TestManager::OnFrameMove(double fTime, float fElapsedTime) {
    if (this->activeTest < this->tests.Count()) {
        this->tests[this->activeTest]->OnFrameMove(fTime, fElapsedTime);
    }
}


/*
 * TestManager::OnKeyboard
 */
void TestManager::OnKeyboard(UINT nChar, bool bKeyDown, bool bAltDown) {
    if (this->activeTest < this->tests.Count()) {
        this->tests[this->activeTest]->OnKeyboard(nChar, bKeyDown, bAltDown);
    }
}


/*
 * TestManager::OnMouse
 */
void TestManager::OnMouse(bool bLeftButtonDown, bool bRightButtonDown, 
        bool bMiddleButtonDown, bool bSideButton1Down, bool bSideButton2Down,
        INT nMouseWheelDelta, INT xPos, INT yPos) {
    if (this->activeTest < this->tests.Count()) {
        this->tests[this->activeTest]->OnMouse(bLeftButtonDown, 
            bRightButtonDown, bMiddleButtonDown, bSideButton1Down, 
            bSideButton2Down, nMouseWheelDelta, xPos, yPos);
    }
}


/*
 * TestManager::RegisterTests
 */
void TestManager::RegisterTests(CDXUTComboBox *comboBox) {
    if (comboBox != NULL) {
        for (SIZE_T i = 0; i < this->tests.Count(); i++) {
            comboBox->AddItem(this->tests[i]->GetName().PeekBuffer(),
                reinterpret_cast<void *>(i));
        }
    }
}


/*
 * TestManager::SetActiveTest
 */
bool TestManager::SetActiveTest(const SIZE_T activeTest) {
    if (activeTest < this->tests.Count()) {
        this->tests[this->activeTest]->OnD3D9LostDevice();
        this->tests[this->activeTest]->OnTestDisable();
        this->activeTest = activeTest;
        this->tests[this->activeTest]->OnTestEnable();
        this->tests[this->activeTest]->OnD3D9ResetDevice(::DXUTGetD3D9Device(),
            &this->surfaceDesc);

        return true;
    } else {
        return false;
    }
}
