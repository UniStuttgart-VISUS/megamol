/*
 * D3D9SimpleCameraTest.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_D3DSIMPLECAMERATEST_H_INCLUDED
#define VISLIBTEST_D3DSIMPLECAMERATEST_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/D3DCamera.h"
#include "vislib/D3DVISLogo.h"
#include "vislib/MouseInteractionAdapter.h"

#include "AbstractTest.h"
#include "D3D9TestBoxGeometry.h"


class D3D9SimpleCameraTest : public AbstractTest {

public:

    D3D9SimpleCameraTest(void);

    virtual ~D3D9SimpleCameraTest(void);

    virtual HRESULT OnD3D9CreateDevice(PDIRECT3DDEVICE9 pd3dDevice,
        const D3DSURFACE_DESC *pBackBufferSurfaceDesc);

    virtual void OnD3D9DestroyDevice(void);

    virtual void OnD3D9FrameRender(PDIRECT3DDEVICE9 pd3dDevice, double fTime,
        float fElapsedTime);

    virtual HRESULT OnD3D9ResetDevice(PDIRECT3DDEVICE9 pd3dDevice, 
        const D3DSURFACE_DESC *pBackBufferSurfaceDesc);

    virtual void OnKeyboard(UINT nChar, bool bKeyDown, bool bAltDown);

    virtual void OnMouse(bool bLeftButtonDown, bool bRightButtonDown, 
        bool bMiddleButtonDown, bool bSideButton1Down, bool bSideButton2Down,
        INT nMouseWheelDelta, INT xPos, INT yPos);

    virtual void OnTestEnable(void);

private:

    D3D9TestBoxGeometry boxes;

    vislib::graphics::d3d::D3DCamera camera;

    vislib::graphics::d3d::D3DVISLogo *logo;

    vislib::graphics::MouseInteractionAdapter *mia;
};


#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIBTEST_D3DSIMPLECAMERATEST_H_INCLUDED */
