/*
 * AbstractD3D11WindowImpl.cpp
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractD3D11WindowImpl.h"

#include "vislib/D3DException.h"
#include "vislib/d3dutils.h"
#include "vislib/d3dverify.h"
#include "vislib/IllegalParamException.h"
#include "vislib/Trace.h"



/*
 * vislib::graphics::d3d::AbstractD3D11WindowImpl::~AbstractD3D11WindowImpl
 */
vislib::graphics::d3d::AbstractD3D11WindowImpl::~AbstractD3D11WindowImpl(void) {
    VLSTACKTRACE("AbstractD3D11WindowImpl::~AbstractD3D11WindowImpl", 
        __FILE__, __LINE__);
    SAFE_RELEASE(this->device);
    SAFE_RELEASE(this->depthStencilTexture);
    SAFE_RELEASE(this->depthStencilView);
    SAFE_RELEASE(this->deviceContext);
    SAFE_RELEASE(this->dxgiFactory);
    SAFE_RELEASE(this->renderTargetView);
    SAFE_RELEASE(this->swapChain);
}


/*
 * vislib::graphics::d3d::AbstractD3D11WindowImpl::ClearViews
 */
void vislib::graphics::d3d::AbstractD3D11WindowImpl::ClearViews(const float r, 
        const float g, const float b, const float a, const float depth, 
        const BYTE stencil) {
    VLSTACKTRACE("AbstractD3D11WindowImpl::ClearViews", __FILE__, __LINE__);
    ASSERT(this->deviceContext != NULL);
    ASSERT(this->renderTargetView != NULL);
    float colour[] = { r, g, b, a };

    this->deviceContext->ClearRenderTargetView(this->renderTargetView, colour);
    if (this->depthStencilView != NULL) {
        this->deviceContext->ClearDepthStencilView(this->depthStencilView, 0, 
            depth, stencil);
    }
}


/*
 * vislib::graphics::d3d::AbstractD3D11WindowImpl::GetDevice
 */
ID3D11Device *vislib::graphics::d3d::AbstractD3D11WindowImpl::GetDevice(void) {
    VLSTACKTRACE("AbstractD3D11WindowImpl::GetDevice", __FILE__, 
        __LINE__);
    if (this->device != NULL) {
        this->device->AddRef();
    }
    return this->device;
}


/*
 * vislib::graphics::d3d::AbstractD3D11WindowImpl::GetDeviceContext
 */
ID3D11DeviceContext *
vislib::graphics::d3d::AbstractD3D11WindowImpl::GetDeviceContext(void) {
    VLSTACKTRACE("AbstractD3D11WindowImpl::GetDeviceContext", __FILE__, 
        __LINE__);
    if (this->deviceContext != NULL) {
        this->deviceContext->AddRef();
    }
    return this->deviceContext;
}


/*
 * vislib::graphics::d3d::AbstractD3D11WindowImpl::Present
 */
void vislib::graphics::d3d::AbstractD3D11WindowImpl::Present(void) {
    VLSTACKTRACE("AbstractD3D11WindowImpl::Present", __FILE__, __LINE__);
    ASSERT(this->swapChain != NULL);
    this->swapChain->Present(0, 0);
}


/*
 * vislib::graphics::d3d::AbstractD3D11WindowImpl::AbstractD3D11WindowImpl
 */
vislib::graphics::d3d::AbstractD3D11WindowImpl::AbstractD3D11WindowImpl(
        ID3D11Device *device)
        : depthStencilTexture(NULL), depthStencilView(NULL), device(NULL), 
        deviceContext(NULL), dxgiFactory(NULL), renderTargetView(NULL), 
        swapChain(NULL) {
    VLSTACKTRACE("AbstractD3D11WindowImpl::AbstractD3D11WindowImpl", 
        __FILE__, __LINE__);
}


/*
 * vislib::graphics::d3d::AbstractD3D11WindowImpl::findAdapter
 */
IDXGIAdapter *vislib::graphics::d3d::AbstractD3D11WindowImpl::findAdapter(
        HWND hWnd) {
    VLSTACKTRACE("AbstractD3D11WindowImpl::findAdapter", __FILE__, __LINE__);
    IDXGIAdapter *adapter = NULL;
    HMONITOR hMonitor = NULL;
    HRESULT hr = S_OK;
    IDXGIOutput *output = NULL;
    DXGI_OUTPUT_DESC outputDesc;
    IDXGIAdapter *retval = NULL;

    /* Sanity checks. */
    if (hWnd == NULL) {
        throw IllegalParamException("hWnd", __FILE__, __LINE__);
    }

    /* Determine which is the adapter that the window is visible on. */
    if (SUCCEEDED(hr)) {
        VLTRACE(Trace::LEVEL_VL_INFO, "Searching monitor that window is "
            "displayed on...\n");
        hMonitor = ::MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST);
        ASSERT(hMonitor != NULL);
    }

    /* Create DXGI factory if necessary. */
    if (SUCCEEDED(hr) && (this->dxgiFactory == NULL)) {
        VLTRACE(Trace::LEVEL_VL_INFO, "Creating DXGI factory...\n");
        hr = ::CreateDXGIFactory(IID_IDXGIFactory,
            reinterpret_cast<void **>(&this->dxgiFactory));
    }

    /* Search the output that the monitor uses. */
    if (SUCCEEDED(hr)) {
         VLTRACE(Trace::LEVEL_VL_INFO, "Enumerating DXGI adapters...\n");
        for (UINT a = 0; SUCCEEDED(hr); ++a) {
            hr = this->dxgiFactory->EnumAdapters(a, &adapter);

            VLTRACE(Trace::LEVEL_VL_INFO, "Enumerating outputs...\n");
            for (UINT o = 0; SUCCEEDED(hr); ++o) {
                hr = adapter->EnumOutputs(o, &output);
                if (SUCCEEDED(hr)) {
                    hr = output->GetDesc(&outputDesc);
                    SAFE_RELEASE(output);
                    if (SUCCEEDED(hr)) {
                        if (outputDesc.Monitor == hMonitor) {
                            retval = adapter;
                            retval->AddRef();
                            break;
                        }
                    }
                }
            }

            if ((hr == DXGI_ERROR_NOT_FOUND) && (retval == NULL)) {
                hr = S_OK;
            }
            SAFE_RELEASE(adapter);
        }
        if (hr == DXGI_ERROR_NOT_FOUND) {
            hr = S_OK;
        }
    }

    ASSERT(adapter == NULL);
    ASSERT(output == NULL);

    if (FAILED(hr)) {
        throw D3DException(hr, __FILE__, __LINE__);
    }

    return retval;
}


/*
 * vislib::graphics::d3d::AbstractD3D11WindowImpl::initialise
 */
void vislib::graphics::d3d::AbstractD3D11WindowImpl::initialise(HWND hWnd) {
    VLSTACKTRACE("AbstractD3D11WindowImpl::initialise", __FILE__, __LINE__);
    IDXGIAdapter *adapter = NULL;
    IDXGIDevice *dxgiDevice = NULL;
    UINT flags = 0;
    D3D_FEATURE_LEVEL featureLevel;
    vislib::Array<D3D_FEATURE_LEVEL> featureLevels;
    HRESULT hr = S_OK;
    DXGI_SWAP_CHAIN_DESC swapChainDesc;
    RECT wndRect;

    /* Clean old stuff. */
    SAFE_RELEASE(this->depthStencilTexture);
    SAFE_RELEASE(this->depthStencilView);
    //SAFE_RELEASE(this->device);  RE-USE DEVICE FROM CTOR!
    SAFE_RELEASE(this->deviceContext);
    SAFE_RELEASE(this->dxgiFactory);
    SAFE_RELEASE(this->renderTargetView);
    SAFE_RELEASE(this->swapChain);

    /* Update device. */
    if (this->device != NULL) {
        SAFE_RELEASE(this->dxgiFactory);

        if (SUCCEEDED(hr)) {
            this->device->GetImmediateContext(&this->deviceContext);
        }

        if (SUCCEEDED(hr)) {
            hr = this->device->QueryInterface(IID_IDXGIDevice,
                reinterpret_cast<void **>(&dxgiDevice));
        }

        if (SUCCEEDED(hr)) {
            hr = dxgiDevice->GetParent(IID_IDXGIAdapter,
                reinterpret_cast<void **>(&adapter));
        }

        if (SUCCEEDED(hr)) {
            hr = adapter->GetParent(IID_IDXGIFactory, 
                reinterpret_cast<void **>(&this->dxgiFactory));
        }

    } else {
        adapter = this->findAdapter(hWnd);

#if (defined(DEBUG) || defined(_DEBUG))
        flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif /* (defined(DEBUG) || defined(_DEBUG)) */

        this->onCreatingDevice(flags, featureLevels);

        /* Create a device for the window. */
        if (SUCCEEDED(hr)) {
            hr = ::D3D11CreateDevice(adapter, 
                D3D_DRIVER_TYPE_UNKNOWN,
                NULL,
                flags,
                !featureLevels.IsEmpty() ? featureLevels.PeekElements() : NULL,
                featureLevels.Count(),
                D3D11_SDK_VERSION,
                &this->device,
                &featureLevel,
                &this->deviceContext);
        }
    } /* end if (device != NULL) */
    ASSERT(FAILED(hr) || (this->dxgiFactory != NULL));
    ASSERT(FAILED(hr) || (this->device != NULL));

    /* Get windows bounds. */
    if (SUCCEEDED(hr)) {
        if (::GetWindowRect(hWnd, &wndRect) == FALSE) {
            hr = HRESULT_FROM_WIN32(::GetLastError());
        }
    }

    /* Create the swap chain for the device. */
    if (SUCCEEDED(hr)) {
        ::ZeroMemory(&swapChainDesc, sizeof(swapChainDesc));
        swapChainDesc.BufferCount = 1;
        swapChainDesc.BufferDesc.Width = wndRect.right - wndRect.left;
        swapChainDesc.BufferDesc.Height = wndRect.bottom - wndRect.top;
        swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        //swapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
        //swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
        swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swapChainDesc.OutputWindow = hWnd;
        swapChainDesc.SampleDesc.Count = 1;
        swapChainDesc.SampleDesc.Quality = 0;
        swapChainDesc.Windowed = TRUE;

        this->onCreatingSwapChain(swapChainDesc);

        ASSERT(this->dxgiFactory != NULL);
        hr = this->dxgiFactory->CreateSwapChain(this->device, 
            &swapChainDesc, &this->swapChain);
    }

    if (SUCCEEDED(hr)) {
        hr = this->updateViews();
    }

    SAFE_RELEASE(adapter);
    SAFE_RELEASE(dxgiDevice);

    if (FAILED(hr)) {
        throw D3DException(hr, __FILE__, __LINE__);
    }
}


/*
 * vislib::graphics::d3d::AbstractD3D11WindowImpl::onCreatingDepthStencilTexture
 */
bool vislib::graphics::d3d::AbstractD3D11WindowImpl
        ::onCreatingDepthStencilTexture(
        D3D11_TEXTURE2D_DESC& inOutBackBufferDesc) {
    VLSTACKTRACE("AbstractD3D11WindowImpl::inOutBackBufferDesc", __FILE__,
        __LINE__);
    return true;
}


/*
 * vislib::graphics::d3d::AbstractD3D11WindowImpl::onCreatingDevice
 */
void vislib::graphics::d3d::AbstractD3D11WindowImpl::onCreatingDevice(
        UINT& inOutFlags, 
        vislib::Array<D3D_FEATURE_LEVEL>& inOutFeatureLevels) throw() {
    VLSTACKTRACE("AbstractD3D11WindowImpl::onCreatingDevice", __FILE__, 
        __LINE__);
    // Nothing to do.
}


/*
 * vislib::graphics::d3d::AbstractD3D11WindowImpl::onCreatingSwapChain
 */
void vislib::graphics::d3d::AbstractD3D11WindowImpl::onCreatingSwapChain(
        DXGI_SWAP_CHAIN_DESC& inOutSwapChainDesc) throw() {
    VLSTACKTRACE("AbstractD3D11WindowImpl::onCreatingSwapChain", __FILE__, 
        __LINE__);
    // Nothing to do.
}


/*
 * vislib::graphics::d3d::AbstractD3D11WindowImpl::resizeSwapChain
 */
void vislib::graphics::d3d::AbstractD3D11WindowImpl::resizeSwapChain(
        const int width, const int height) {
    VLSTACKTRACE("AbstractD3D11WindowImpl::resizeSwapChain", __FILE__, 
        __LINE__);
    USES_D3D_VERIFY;
    ASSERT(this->swapChain != NULL);

    DXGI_SWAP_CHAIN_DESC swapChainDesc;

    D3D_VERIFY_THROW(this->swapChain->GetDesc(&swapChainDesc));
    D3D_VERIFY_THROW(this->swapChain->ResizeBuffers(swapChainDesc.BufferCount,
        width, height, swapChainDesc.BufferDesc.Format, 0));
    D3D_VERIFY_THROW(this->updateViews());
}


/*
 * vislib::graphics::d3d::AbstractD3D11WindowImpl::updateViews(void)
 */
HRESULT vislib::graphics::d3d::AbstractD3D11WindowImpl::updateViews(void) {
    VLSTACKTRACE("AbstractD3D11WindowImpl::updateBackBuffers", __FILE__, 
        __LINE__);
    ASSERT(this->device != NULL);
    ASSERT(this->swapChain != NULL);

    ID3D11Texture2D *backBuffer = NULL;
    D3D11_TEXTURE2D_DESC backBufferDesc;
    D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc;
    HRESULT hr = S_OK;
    D3D11_RENDER_TARGET_VIEW_DESC rtvDesc;

    /* Clean all old resources. */
    SAFE_RELEASE(this->depthStencilTexture);
    SAFE_RELEASE(this->depthStencilView);
    SAFE_RELEASE(this->renderTargetView);

    /* Get render target texture. */
    if (SUCCEEDED(hr)) {
        hr = this->swapChain->GetBuffer(0, IID_ID3D11Texture2D, 
            reinterpret_cast<void **>(&backBuffer));
    }

    if (SUCCEEDED(hr)) {
        backBuffer->GetDesc(&backBufferDesc);
    }

    /* Create render target view. */
    if (SUCCEEDED(hr)) {
        ::ZeroMemory(&rtvDesc, sizeof(rtvDesc));
        rtvDesc.Format = backBufferDesc.Format;
        rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        rtvDesc.Texture2D.MipSlice = 0;
        hr = this->device->CreateRenderTargetView(backBuffer, &rtvDesc,
            &this->renderTargetView);
    }

    /* Create depth buffer. */
    if (SUCCEEDED(hr)) {
        backBufferDesc.MipLevels = 1;
        backBufferDesc.ArraySize = 1;
        backBufferDesc.SampleDesc.Count = 1;
        backBufferDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        backBufferDesc.Usage = D3D11_USAGE_DEFAULT;
        backBufferDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;

        if (this->onCreatingDepthStencilTexture(backBufferDesc)) {
            hr = device->CreateTexture2D(&backBufferDesc, NULL,
                &this->depthStencilTexture);
        }
    }

    /* Create depth buffer view. */
    if (SUCCEEDED(hr) && (this->depthStencilTexture != NULL)) {
        ::ZeroMemory(&dsvDesc, sizeof(dsvDesc));
        dsvDesc.Format = backBufferDesc.Format;
        dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;

        hr = this->device->CreateDepthStencilView(this->depthStencilTexture,
            &dsvDesc, &this->depthStencilView);
    }

    /* Update viewport. */
    if (SUCCEEDED(hr)) {
        ::ZeroMemory(&this->viewport, sizeof(this->viewport));
        this->viewport.TopLeftX = 0.0f;
        this->viewport.TopLeftY = 0.0f;
        this->viewport.Width = static_cast<FLOAT>(backBufferDesc.Width);
        this->viewport.Height= static_cast<FLOAT>(backBufferDesc.Height);
        this->viewport.MinDepth = 0.0f;
        this->viewport.MaxDepth = 1.0f;
    }

    /* Enable all the new stuff. */
    if (SUCCEEDED(hr)) {
        this->deviceContext->OMSetRenderTargets(1, &this->renderTargetView,
            this->depthStencilView);
        this->deviceContext->RSSetViewports(1, &this->viewport);
    }

    SAFE_RELEASE(backBuffer);
    return hr;
}
