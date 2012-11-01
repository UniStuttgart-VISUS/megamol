/*
 * AbstractD3D11Window.cpp
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractD3D11Window.h"

#include "vislib/d3dutils.h"
#include "vislib/d3dverify.h"



/*
 * vislib::graphics::d3d::AbstractD3D11Window::~AbstractD3D11Window
 */
vislib::graphics::d3d::AbstractD3D11Window::~AbstractD3D11Window(void) {
    VLSTACKTRACE("AbstractD3D11Window::~AbstractD3D11Window", __FILE__, 
        __LINE__);
    this->releaseAllD3D();
}


/*
 * vislib::graphics::d3d::AbstractD3D11Window::Create
 */
void vislib::graphics::d3d::AbstractD3D11Window::Create(
        const vislib::StringA& title, int left, int top, int width, int height,
        ID3D11Device *device) {
    VLSTACKTRACE("AbstractD3D11Window::Create", __FILE__, __LINE__);
    AbstractWindow::Create(title, left, top, width, height);
    this->device = device;
    this->isDeviceShared = (this->device != NULL);
}


/*
 * vislib::graphics::d3d::AbstractD3D11Window::Create
 */
void vislib::graphics::d3d::AbstractD3D11Window::Create(
        const vislib::StringW& title, int left, int top, int width, int height,
        ID3D11Device *device) {
    VLSTACKTRACE("AbstractD3D11Window::Create", __FILE__, __LINE__);
    AbstractWindow::Create(title, left, top, width, height);
    this->device = device;
    this->isDeviceShared = (this->device != NULL);
}


/*
 * vislib::graphics::d3d::AbstractD3D11Window::AbstractD3D11Window
 */
vislib::graphics::d3d::AbstractD3D11Window::AbstractD3D11Window(void)
        : AbstractWindow(), ReferenceCounted(), device(NULL), deviceContext(NULL), dxgiDevice(NULL), dxgiFactory(NULL), isFullscreen(false), 
        renderTargetView(NULL), swapChain(NULL), isDeviceShared(false) {
    VLSTACKTRACE("AbstractD3D11Window::AbstractD3D11Window", __FILE__,
        __LINE__);
    ::ZeroMemory(&this->viewport, sizeof(this->viewport));
    this->viewport.MaxDepth = 1.0f;
}


/*
 * vislib::graphics::d3d::AbstractD3D11Window::onCreated
 */
void vislib::graphics::d3d::AbstractD3D11Window::onCreated(HWND hWnd) {
    VLSTACKTRACE("AbstractD3D11Window::onCreated", __FILE__, __LINE__);
    USES_D3D_VERIFY;

    if (this->device == NULL) {
        D3D_FEATURE_LEVEL featureLevel;
        UINT flags = 0;

#if (defined(DEBUG) || defined(_DEBUG))
        flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif /* (defined(DEBUG) || defined(_DEBUG)) */

        D3D_VERIFY_THROW(::D3D11CreateDevice(NULL,
            D3D_DRIVER_TYPE_HARDWARE,
            NULL,
            flags,
            NULL, 0,
            D3D11_SDK_VERSION,
            &this->device,
            &featureLevel,
            &this->deviceContext));

    } else {
        this->device->GetImmediateContext(&this->deviceContext);
    }
    ASSERT(this->device != NULL);

    /* Retrieve the factory that the device was created from. */
    D3D_VERIFY_THROW(this->device->QueryInterface(IID_IDXGIDevice, 
        reinterpret_cast<void **>(&this->dxgiDevice)));
    IDXGIAdapter *adapter;
    D3D_VERIFY_THROW(this->dxgiDevice->GetParent(IID_IDXGIAdapter,
        reinterpret_cast<void **>(&adapter)));
    D3D_VERIFY_THROW(adapter->GetParent(IID_IDXGIFactory, 
        reinterpret_cast<void **>(&dxgiFactory)));
    adapter->Release(); // TODO

    /* Create the swap chain for the device. */
    DXGI_SWAP_CHAIN_DESC swapChainDesc;
    ::ZeroMemory(&swapChainDesc, sizeof(swapChainDesc));
    swapChainDesc.BufferCount = 1;
    swapChainDesc.BufferDesc.Width = 100;   // TODO
    swapChainDesc.BufferDesc.Height = 100; // TODO
    swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
    swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.OutputWindow = hWnd;
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.SampleDesc.Quality = 0;
    swapChainDesc.Windowed = TRUE;

    D3D_VERIFY_THROW(dxgiFactory->CreateSwapChain(this->dxgiDevice, 
        &swapChainDesc, &this->swapChain));
}


/*
 * vislib::graphics::d3d::AbstractD3D11Window::onMessage
 */
LRESULT vislib::graphics::d3d::AbstractD3D11Window::onMessage(bool& outHandled,
        UINT msg, WPARAM wParam, LPARAM lParam) throw() {
    VLSTACKTRACE("AbstractD3D11Window::onMessage", __FILE__, __LINE__);

    LRESULT retval = AbstractWindow::onMessage(outHandled, msg, wParam, lParam);

    if (!outHandled) {
        switch (msg) {
            case WM_SIZE:
                outHandled = false;
                this->setRenderTargetViewAndViewport();
                retval = 0;
                break;
        }
    }

    return retval;
}


/*
 * vislib::graphics::d3d::AbstractD3D11Window::releaseAllD3D
 */
void vislib::graphics::d3d::AbstractD3D11Window::releaseAllD3D(void) {
    VLSTACKTRACE("vislib::graphics::d3d::AbstractD3D11Window::releaseAllD3D", 
        __FILE__, __LINE__);
    SAFE_RELEASE(this->device);
    SAFE_RELEASE(this->deviceContext);
    SAFE_RELEASE(this->dxgiDevice);
    SAFE_RELEASE(this->dxgiFactory);
    SAFE_RELEASE(this->renderTargetView);
    SAFE_RELEASE(this->swapChain);
}


/*
 * vislib::graphics::d3d::AbstractD3D11Window::setRenderTargetViewAndViewport
 */
void vislib::graphics::d3d::AbstractD3D11Window::setRenderTargetViewAndViewport(
        void) {
    VLSTACKTRACE("AbstractD3D11Window::setRenderTargetViewAndViewport", 
        __FILE__, __LINE__);
    ASSERT(this->device != NULL);
    ASSERT(this->deviceContext != NULL);

    ID3D11Texture2D *backBuffer = NULL;
    D3D11_TEXTURE2D_DESC backBufferDesc;

    /* Release old resources. */
    SAFE_RELEASE(this->renderTargetView);

    //DXGI_SWAP_CHAIN_DESC swapChainDesc;
    //this->swapChain->GetDesc(&swapChainDesc);

    /* Get the (new) back buffer texture and its description. */
    this->swapChain->GetBuffer(0, IID_ID3D11Texture2D, 
        reinterpret_cast<void **>(&backBuffer));
    backBuffer->GetDesc(&backBufferDesc);

    /* Create a new render target view from the back buffer. */
    this->device->CreateRenderTargetView(backBuffer, NULL, 
        &this->renderTargetView);

    /* Release back buffer, we do not need it any more. */
    SAFE_RELEASE(backBuffer);

    // TODO: Create depth buffer.

    /* Set the views as render targets. */
    this->deviceContext->OMSetRenderTargets(1, &this->renderTargetView, NULL);

    /* Adjust the viewport to match the new back buffer size. */
    this->viewport.Width = static_cast<float>(backBufferDesc.Width);
    this->viewport.Height = static_cast<float>(backBufferDesc.Height);

    this->deviceContext->RSSetViewports(1, &this->viewport);
}
