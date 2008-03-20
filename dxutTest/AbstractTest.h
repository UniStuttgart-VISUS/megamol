/*
 * AbstractTest.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_ABSTRACTEST_H_INCLUDED
#define VISLIBTEST_ABSTRACTEST_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


class AbstractTest {

public:

    virtual ~AbstractTest(void);

    virtual HRESULT CALLBACK OnD3D9CreateDevice(PDIRECT3DDEVICE9 pd3dDevice, 
        const D3DSURFACE_DESC *pBackBufferSurfaceDesc);

    virtual void OnD3D9DestroyDevice(void);

    virtual void OnD3D9FrameRender(PDIRECT3DDEVICE9 pd3dDevice, double fTime, 
        float fElapsedTime) = 0;

    virtual void OnD3D9LostDevice(void);

    virtual HRESULT OnD3D9ResetDevice(PDIRECT3DDEVICE9 pd3dDevice, 
        const D3DSURFACE_DESC *pBackBufferSurfaceDesc);

    virtual void OnFrameMove(double fTime, float fElapsedTime);

    virtual void OnKeyboard(UINT nChar, bool bKeyDown, bool bAltDown);

protected:

    AbstractTest(void);
};

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTDIMENSIONIMPL_H_INCLUDED */
