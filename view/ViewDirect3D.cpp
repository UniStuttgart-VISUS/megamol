/*
 * ViewDirect3D.cpp
 *
 * Copyright (C) 2012 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ViewDirect3D.h"

#include "view/CallRender3D.h"
#include "view/CallUpdateDirect3D.h"
#include "view/CameraParamOverride.h"

#include "vislib/COMException.h"
#include "vislib/d3dutils.h"
#include "vislib/ScopedLock.h"
#include "vislib/Trace.h"


/*
 * megamol::core::view::ViewDirect3D::IsAvailable
 */
bool megamol::core::view::ViewDirect3D::IsAvailable(void) {
    VLAUTOSTACKTRACE;
#ifdef MEGAMOLCORE_WITH_DIRECT3D11
    return true;
#else /* MEGAMOLCORE_WITH_DIRECT3D11 */
    return false;
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
}


/*
 * megamol::core::view::ViewDirect3D::ViewDirect3D
 */
megamol::core::view::ViewDirect3D::ViewDirect3D(void) : Base(),
#ifdef MEGAMOLCORE_WITH_DIRECT3D11
        device(NULL), 
        dsv(NULL),
        immediateContext(NULL),
        rtv(NULL),
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
        updateD3D("updated3d", "Propagates Direct3D resources from a view to a renderer.") {
    VLAUTOSTACKTRACE;

    this->updateD3D.SetCompatibleCall<CallUpdateDirect3DDescription>();
    this->MakeSlotAvailable(&this->updateD3D);
}


/*
 * megamol::core::view::ViewDirect3D::~ViewDirect3D
 */
megamol::core::view::ViewDirect3D::~ViewDirect3D(void) {
    VLAUTOSTACKTRACE;
#ifdef MEGAMOLCORE_WITH_DIRECT3D11
    this->finaliseD3D();
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
}


/*
 * megamol::core::view::ViewDirect3D::Render
 */
void megamol::core::view::ViewDirect3D::Render(float time, double instTime) {
    VLAUTOSTACKTRACE;
#ifdef MEGAMOLCORE_WITH_DIRECT3D11
    /* Prepare D3D-specific stuff. */
    if (this->rtv != NULL) {
        this->immediateContext->OMSetRenderTargets(1, &this->rtv, this->dsv);
        this->immediateContext->ClearRenderTargetView(this->rtv, 
            this->bkgndColour());
        this->immediateContext->ClearDepthStencilView(this->dsv, 
            D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

        Base::Render(time, instTime);
    }
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
}


/*
 * megamol::core::view::ViewDirect3D::Render
 */
void megamol::core::view::ViewDirect3D::Render(float tileX, float tileY, 
        float tileW, float tileH, float virtW, float virtH, 
        bool stereo, bool leftEye, double instTime,
        class ::megamol::core::CoreInstance *core) {
    VLAUTOSTACKTRACE;
    ASSERT(false);
}


/*
 * megamol::core::view::ViewDirect3D::Resize
 */
void megamol::core::view::ViewDirect3D::Resize(unsigned int width, 
        unsigned int height) {
    VLAUTOSTACKTRACE;
    Base::Resize(width, height);
#ifdef MEGAMOLCORE_WITH_DIRECT3D11
    this->onResizingD3D(width, height);
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
}


/*
 * megamol::core::view::ViewDirect3D::UpdateFromContext
 */
void megamol::core::view::ViewDirect3D::UpdateFromContext(
        mmcRenderViewContext *context) {
    VLAUTOSTACKTRACE;
#ifdef MEGAMOLCORE_WITH_DIRECT3D11
    if (context != NULL) {
        if (context->Direct3DDevice != this->device) {
            this->finaliseD3D();
            this->initialiseD3D(static_cast<ID3D11Device *>(
                context->Direct3DDevice));
        }
        if ((context->Direct3DRenderTarget != this->rtv) 
                && (context->Direct3DRenderTarget != NULL)) {
            this->onResizedD3D(static_cast<ID3D11RenderTargetView *>(
                context->Direct3DRenderTarget));
        }
    }
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
}


/*
 * megamol::core::view::ViewDirect3D::create
 */
bool megamol::core::view::ViewDirect3D::create(void) {
    VLAUTOSTACKTRACE;
    bool retval = Base::create();

    if (retval) {
    }

    return retval;
}


/*
 * megamol::core::view::ViewDirect3D::release
 */
void megamol::core::view::ViewDirect3D::release(void) {
    VLAUTOSTACKTRACE;
    Base::release();
}


#ifdef MEGAMOLCORE_WITH_DIRECT3D11
/*
 * megamol::core::view::ViewDirect3D::finaliseD3D
 */
void megamol::core::view::ViewDirect3D::finaliseD3D(void) {
    VLAUTOSTACKTRACE;
    VLTRACE(VISLIB_TRCELVL_INFO, "Releasing D3D resources in "
        "ViewDirect3D...\n");
    SAFE_RELEASE(this->device);
    SAFE_RELEASE(this->dsv);
    SAFE_RELEASE(this->immediateContext);
    SAFE_RELEASE(this->rtv);
}


/*
 * megamol::core::view::ViewDirect3D::initialiseD3D
 */
void megamol::core::view::ViewDirect3D::initialiseD3D(ID3D11Device *device) {
    VLAUTOSTACKTRACE;
    ASSERT(device != NULL);
    
    ASSERT(this->device == NULL);
    this->device = device;
    this->device->AddRef();

    ASSERT(this->immediateContext == NULL);
    this->device->GetImmediateContext(&this->immediateContext);

    CallUpdateDirect3D *cud3d = this->updateD3D.CallAs<CallUpdateDirect3D>();
    if (cud3d != NULL) {
        cud3d->SetDevice(this->device);
        (*cud3d)(0);
    }
}


/*
 * megamol::core::view::ViewDirect3D::onResizedD3D
 */
void megamol::core::view::ViewDirect3D::onResizedD3D(
        ID3D11RenderTargetView *rtv) {
    VLAUTOSTACKTRACE;
    ASSERT(this->device != NULL);
    ASSERT(rtv != NULL);
    ASSERT(this->dsv == NULL);
    ASSERT(this->rtv == NULL);

    HRESULT hr = S_OK;
    ID3D11Texture2D *tex = NULL;
    D3D11_TEXTURE2D_DESC texDesc;
    D3D11_VIEWPORT viewport;

    VLTRACE(VISLIB_TRCELVL_INFO, "Storing new render target view in "
        "ViewDirect3D...\n");
    this->rtv = rtv;
    this->rtv->AddRef();

    /* Get back buffer extents. */
    // TODO: because I know... Should query interface instead...
    this->rtv->GetResource(reinterpret_cast<ID3D11Resource **>(&tex));
    tex->GetDesc(&texDesc);
    SAFE_RELEASE(tex);

    /* Create depth buffer. */
    texDesc.ArraySize = 1;
    texDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    texDesc.CPUAccessFlags = 0;
    texDesc.Format = DXGI_FORMAT_D32_FLOAT;
    texDesc.MipLevels = 1;
    texDesc.MiscFlags = 0;
    texDesc.Usage = D3D11_USAGE_DEFAULT;

    hr = this->device->CreateTexture2D(&texDesc, NULL, &tex);
    if (FAILED(hr)) {
        throw vislib::sys::COMException(hr, __FILE__, __LINE__);
    }

    hr = this->device->CreateDepthStencilView(tex, NULL, &this->dsv);
    SAFE_RELEASE(tex);

    /* Update viewport. */
    ::ZeroMemory(&viewport, sizeof(viewport));
    viewport.Width = static_cast<float>(texDesc.Width);
    viewport.Height = static_cast<float>(texDesc.Height);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    viewport.TopLeftX = 0.0f;
    viewport.TopLeftY = 0.0f;

    this->immediateContext->RSSetViewports(1, &viewport);
}


/*
 * megamol::core::view::ViewDirect3D::onResizingD3D
 */
void megamol::core::view::ViewDirect3D::onResizingD3D(const unsigned int width,
        const unsigned int height) {
    VLAUTOSTACKTRACE;
    ASSERT((this->immediateContext != NULL) || (this->rtv == NULL));

    VLTRACE(VISLIB_TRCELVL_INFO, "Releasing render target view in "
        "ViewDirect3D...\n");
    if (this->immediateContext != NULL) {
        this->immediateContext->OMSetRenderTargets(0, NULL, NULL);
    }
    SAFE_RELEASE(this->rtv);
    SAFE_RELEASE(this->dsv);
}
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
