/*
 * D3D9VisLogoTest.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_D3D9VISLOGOTEST_H_INCLUDED
#define VISLIBTEST_D3D9VISLOGOTEST_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/D3DVISLogo.h"

#include "AbstractTest.h"



class D3D9VisLogoTest : public AbstractTest {

public:

    D3D9VisLogoTest(void);

    virtual ~D3D9VisLogoTest(void);

    virtual HRESULT OnD3D9CreateDevice(PDIRECT3DDEVICE9 pd3dDevice,
        const D3DSURFACE_DESC *pBackBufferSurfaceDesc);

    virtual void OnD3D9DestroyDevice(void);

    virtual void OnFrameMove(double fTime, float fElapsedTime);

    virtual void OnD3D9FrameRender(PDIRECT3DDEVICE9 pd3dDevice, double fTime,
        float fElapsedTime);

private:

    float angle;

    vislib::graphics::d3d::D3DVISLogo *logo;
};

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIBTEST_D3D9VISLOGOTEST_H_INCLUDED */
