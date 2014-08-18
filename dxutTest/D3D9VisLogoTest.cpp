/*
 * D3D9VisLogoTest.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "DXUT.h"
#include "D3D9VisLogoTest.h"

#include "vislib/assert.h"
#include "vislib/d3dverify.h"
#include "vislib/memutils.h"


/*
 * D3D9VisLogoTest::D3D9VisLogoTest
 */
D3D9VisLogoTest::D3D9VisLogoTest(void) : AbstractTest(L"VIS-Logo-Test"),
        angle(0.0f), logo(NULL) {
}


/*
 * D3D9VisLogoTest::~D3D9VisLogoTest
 */
D3D9VisLogoTest::~D3D9VisLogoTest(void) {
    SAFE_DELETE(this->logo);
}


/*
 * D3D9VisLogoTest::OnD3D9CreateDevice
 */
HRESULT D3D9VisLogoTest::OnD3D9CreateDevice(
        PDIRECT3DDEVICE9 pd3dDevice,
        const D3DSURFACE_DESC *pBackBufferSurfaceDesc) {
    using namespace vislib::graphics::d3d;
    ASSERT(this->logo == NULL);

    AbstractTest::OnD3D9CreateDevice(pd3dDevice, pBackBufferSurfaceDesc);

    if ((this->logo = new D3DVISLogo(pd3dDevice)) == NULL) {
        return E_OUTOFMEMORY;
    }

    try {
        this->logo->Create();
    } catch (D3DException e) {
        return e.GetResult();
    }

    return D3D_OK;
}


/*
 * D3D9VisLogoTest::OnD3D9DestroyDevice
 */
void D3D9VisLogoTest::OnD3D9DestroyDevice(void) {
    SAFE_DELETE(this->logo);
}


/*
 * D3D9VisLogoTest::OnFrameMove
 */
void D3D9VisLogoTest::OnFrameMove(double fTime, float fElapsedTime) {
    this->angle += fElapsedTime;
    if (this->angle > 2 * D3DX_PI) {
        this->angle -= 2 * D3DX_PI;
    }
}


/*
 * D3D9VisLogoTest::OnD3D9FrameRender
 */
void D3D9VisLogoTest::OnD3D9FrameRender(PDIRECT3DDEVICE9 pd3dDevice, 
        double fTime, float fElapsedTime) {
    USES_D3D_VERIFY;
    ASSERT(this->logo != NULL);

    D3DXMATRIXA16 modelMatrix;                  // Model matrix.
    D3DXMATRIXA16 projMatrix;                   // Projection matrix.
    D3DXMATRIXA16 viewMatrix;                   // View matrix.
    D3DXMATRIXA16 tmpMatrix;                    // For auxiliary computations.
    D3DXVECTOR3 eyeVec(0.0f, 0.0f, -2.0f);      // Position of eye point.
    D3DXVECTOR3 lookAtVec(0.0f, 0.0f, 0.0f);    // Look at vector.
    D3DXVECTOR3 upVec(0.0f, 1.0f, 0.0f);        // Camera up vector.

    D3DXMatrixRotationY(&modelMatrix, this->angle);
    //D3DXMatrixRotationZ(&modelMatrix, -D3DX_PI);
    //D3DXMatrixIdentity(&modelMatrix);
    //modelMatrix *= tmpMatrix;
    D3D_VERIFY_THROW(pd3dDevice->SetTransform(D3DTS_WORLD, &modelMatrix));

    D3DXMatrixLookAtLH(&viewMatrix, &eyeVec, &lookAtVec, &upVec);
    D3D_VERIFY_THROW(pd3dDevice->SetTransform(D3DTS_VIEW, &viewMatrix));

    D3DXMatrixPerspectiveFovLH(&projMatrix, D3DX_PI / 2.0, this->aspectRatio,
        1.0f, 100.0f);
    D3D_VERIFY_THROW(pd3dDevice->SetTransform(D3DTS_PROJECTION, &projMatrix));

    D3D_VERIFY_THROW(pd3dDevice->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE));
    D3D_VERIFY_THROW(pd3dDevice->SetRenderState(D3DRS_ZENABLE, D3DZB_TRUE));
    D3D_VERIFY_THROW(pd3dDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW));
    D3D_VERIFY_THROW(pd3dDevice->SetRenderState(D3DRS_LIGHTING, FALSE));
    D3D_VERIFY_THROW(pd3dDevice->SetRenderState(D3DRS_ZENABLE, D3DZB_TRUE));

    this->logo->Draw();
}
