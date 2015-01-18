/*
 * D3D9SimpleCameraTest.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "DXUT.h"
#include "D3D9SimpleCameraTest.h"

#include "vislib/assert.h"
#include "vislib/d3dverify.h"
#include "vislib/memutils.h"
#include "vislib/Trace.h"


/*
 * D3D9SimpleCameraTest::D3D9SimpleCameraTest
 */
D3D9SimpleCameraTest::D3D9SimpleCameraTest(void)
        : AbstractTest(L"Simple D3DCamera Test"), logo(NULL), mia(NULL) {
    using vislib::graphics::MouseInteractionAdapter;

    this->mia = new MouseInteractionAdapter(this->camera.Parameters());
    //this->mia->ConfigureRotation(MouseInteractionAdapter::ROTATION_FREE);
}


/*
 * D3D9SimpleCameraTest::~D3D9SimpleCameraTest
 */
D3D9SimpleCameraTest::~D3D9SimpleCameraTest(void) {
    SAFE_DELETE(this->mia);
}


/*
 * D3D9SimpleCameraTest::OnD3D9CreateDevice
 */
HRESULT D3D9SimpleCameraTest::OnD3D9CreateDevice(
        PDIRECT3DDEVICE9 pd3dDevice,
        const D3DSURFACE_DESC *pBackBufferSurfaceDesc) {
    using namespace vislib::graphics::d3d;
    USES_D3D_VERIFY;
    ASSERT(this->logo == NULL);

    AbstractTest::OnD3D9CreateDevice(pd3dDevice, pBackBufferSurfaceDesc);

    if ((this->logo = new D3DVISLogo(pd3dDevice)) == NULL) {
        return E_OUTOFMEMORY;
    }

    try {
        this->logo->Create();
        this->boxes.Create(pd3dDevice);
    } catch (D3DException e) {
        return e.GetResult();
    }

    return D3D_OK;
}


/*
 * D3D9SimpleCameraTest::OnD3D9DestroyDevice
 */
void D3D9SimpleCameraTest::OnD3D9DestroyDevice(void) {
    this->boxes.Release();
    SAFE_DELETE(this->logo);
}


/*
 * D3D9SimpleCameraTest::OnD3D9FrameRender
 */
void D3D9SimpleCameraTest::OnD3D9FrameRender(PDIRECT3DDEVICE9 pd3dDevice, 
        double fTime, float fElapsedTime) {
    using vislib::graphics::d3d::D3DMatrix;
    USES_D3D_VERIFY;
    ASSERT(this->logo != NULL);

    D3DXMATRIXA16 modelMatrix;              // Model matrix.
    D3DMatrix projMatrix;                   // Projection matrix.

    D3DXMatrixIdentity(&modelMatrix);
    D3D_VERIFY_THROW(pd3dDevice->SetTransform(D3DTS_WORLD, &modelMatrix));

    D3D_VERIFY_THROW(this->camera.ApplyViewTransform(pd3dDevice));

    D3D_VERIFY_THROW(pd3dDevice->SetTransform(D3DTS_PROJECTION, 
        static_cast<D3DXMATRIX *>(this->camera.CalcProjectionMatrix(
        projMatrix))));

    D3D_VERIFY_THROW(pd3dDevice->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE));
    D3D_VERIFY_THROW(pd3dDevice->SetRenderState(D3DRS_ZENABLE, D3DZB_TRUE));
    D3D_VERIFY_THROW(pd3dDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW));
    D3D_VERIFY_THROW(pd3dDevice->SetRenderState(D3DRS_LIGHTING, FALSE));
    D3D_VERIFY_THROW(pd3dDevice->SetRenderState(D3DRS_ZENABLE, D3DZB_TRUE));

    this->logo->Draw();

    this->boxes.Draw();
}


/*
 * D3D9SimpleCameraTest::OnD3D9ResetDevice
 */
HRESULT D3D9SimpleCameraTest::OnD3D9ResetDevice(PDIRECT3DDEVICE9 pd3dDevice, 
        const D3DSURFACE_DESC *pBackBufferSurfaceDesc) {
    using vislib::graphics::ImageSpaceType;

    this->camera.Parameters()->SetVirtualViewSize(
        static_cast<ImageSpaceType>(pBackBufferSurfaceDesc->Width),
        static_cast<ImageSpaceType>(pBackBufferSurfaceDesc->Height));

    return AbstractTest::OnD3D9ResetDevice(pd3dDevice, pBackBufferSurfaceDesc);
}


/*
 * D3D9SimpleCameraTest::OnKeyboard
 */
void D3D9SimpleCameraTest::OnKeyboard(UINT nChar, bool bKeyDown, 
        bool bAltDown) {
    using vislib::graphics::InputModifiers;

    switch (nChar) {
        case VK_SHIFT:
            this->mia->SetModifierState(InputModifiers::MODIFIER_SHIFT, 
                bKeyDown);
            break;

        case VK_CONTROL:
            this->mia->SetModifierState(InputModifiers::MODIFIER_CTRL, 
                bKeyDown);
            break;
    }

    this->mia->SetModifierState(InputModifiers::MODIFIER_ALT, bAltDown);
}


/*
 * D3D9SimpleCameraTest::OnMouse
 */
void D3D9SimpleCameraTest::OnMouse(bool bLeftButtonDown, bool bRightButtonDown, 
        bool bMiddleButtonDown, bool bSideButton1Down, bool bSideButton2Down,
        INT nMouseWheelDelta, INT xPos, INT yPos) {
    using vislib::graphics::MouseInteractionAdapter;
    VLTRACE(VISLIB_TRCELVL_INFO, "left = %d, right = %d, middle = %d\n", 
        bLeftButtonDown, bRightButtonDown, bMiddleButtonDown);

    this->mia->SetMouseButtonState(MouseInteractionAdapter::BUTTON_LEFT, 
        bLeftButtonDown);
    this->mia->SetMouseButtonState(MouseInteractionAdapter::BUTTON_RIGHT, 
        bRightButtonDown);
    this->mia->SetMouseButtonState(MouseInteractionAdapter::BUTTON_MIDDLE, 
        bMiddleButtonDown);

    this->mia->SetMousePosition(xPos, yPos, true);
}


/*
 * D3D9SimpleCameraTest::OnTestEnable
 */
void D3D9SimpleCameraTest::OnTestEnable(void) {
    using namespace vislib::graphics;

    this->camera.Parameters()->SetClip(0.1f, 20.0f);
    this->camera.Parameters()->SetApertureAngle(D3DX_PI / 2.0f);
    this->camera.Parameters()->SetVirtualViewSize(1.0f, 1.0f);
    this->camera.Parameters()->SetView(
        SceneSpacePoint3D(0.0f, 0.0f, -10.0f),
        SceneSpacePoint3D(0.0f, 0.0f, 0.0f),
        SceneSpaceVector3D(0.0f, 1.0f, 0.0f));
}
