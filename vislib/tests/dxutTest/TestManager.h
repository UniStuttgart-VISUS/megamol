/*
 * TestManager.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_TESTMANAGER_H_INCLUDED
#define VISLIBTEST_TESTMANAGER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/PtrArray.h"

#include "AbstractTest.h"


class CDXUTComboBox;


class TestManager {

public:

    TestManager(void);

    ~TestManager(void);

    inline void AddTest(AbstractTest *test) {
        this->tests.Add(test);
    }

    HRESULT OnD3D9CreateDevice(PDIRECT3DDEVICE9 pd3dDevice,
        const D3DSURFACE_DESC *pBackBufferSurfaceDesc);

    void OnD3D9DestroyDevice(void);

    void OnD3D9FrameRender(PDIRECT3DDEVICE9 pd3dDevice, double fTime,
        float fElapsedTime);

    void OnD3D9LostDevice(void);

    HRESULT OnD3D9ResetDevice(PDIRECT3DDEVICE9 pd3dDevice, 
        const D3DSURFACE_DESC *pBackBufferSurfaceDesc);

    void OnFrameMove(double fTime, float fElapsedTime);

    void OnKeyboard(UINT nChar, bool bKeyDown, bool bAltDown);

    void OnMouse(bool bLeftButtonDown, bool bRightButtonDown, 
        bool bMiddleButtonDown, bool bSideButton1Down, bool bSideButton2Down,
        INT nMouseWheelDelta, INT xPos, INT yPos);

    void RegisterTests(CDXUTComboBox *comboBox);

    bool SetActiveTest(const SIZE_T activeTest);

private:

    SIZE_T activeTest;

    D3DSURFACE_DESC surfaceDesc;

    vislib::PtrArray<AbstractTest> tests;
};

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIBTEST_TESTMANAGER_H_INCLUDED */

