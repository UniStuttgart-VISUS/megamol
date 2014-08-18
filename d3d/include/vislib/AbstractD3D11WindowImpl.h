/*
 * AbstractD3D11WindowImpl.h
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTD3D11WINDOWIMPL_H_INCLUDED
#define VISLIB_ABSTRACTD3D11WINDOWIMPL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <D3D11.h>
#include <DXGI.h>

#include "vislib/Array.H"
#include "vislib/StackTrace.h"


namespace vislib {
namespace graphics {
namespace d3d {


    /**
     * This class provides the functionality for creating D3D 11 devices and
     * swap chains.
     */
    class AbstractD3D11WindowImpl {

    public:

        /** Dtor. */
        ~AbstractD3D11WindowImpl(void);

        /**
         * Clear the render target and depth-stencil views.
         *
         * @param r       The value to be set for the red channel.
         * @param g       The value to be set for the green channel.
         * @param b       The value to be set for the blue channel.
         * @param a       The value to be set for the alpha channel.
         * @param depth   The depth value to be set.
         * @param stencil The stencil value to be set.
         */
        void ClearViews(const float r = 0.0f, const float g = 0.0f, 
            const float b = 0.0f, const float a = 0.0f, const float depth = 0.0f, 
            const BYTE stencil = 0);

        /**
         * Return the immediate that is used for rendering to the  window.
         *
         * The caller must release the resource returned once it is no longer
         * used.
         *
         * @return The Direct3D device.
         */
        ID3D11Device *GetDevice(void);

        /**
         * Return the immediate context that is used for rendering to the 
         * window.
         *
         * The caller must release the resource returned once it is no longer
         * used.
         *
         * @return The immediate context.
         */
        ID3D11DeviceContext *GetDeviceContext(void);

        /**
         * Present the swap chain.
         */
        void Present(void);

    protected:

        /** 
         * Initialises a new instance.
         *
         * @param device The D3D device to be used for rendering to the window.
         *               If this parameter is NULL (the default), the 
         *               initialise() method will create the device lazily on
         *               the adapter that shows the window.
         */
        AbstractD3D11WindowImpl(ID3D11Device *device = NULL);

        /**
         * Find the DXGI adapter that displays (most) of the given window.
         *
         * @param hWnd
         * @param device
         *
         * @return
         *
         * @throws D3DException
         */
        IDXGIAdapter *findAdapter(HWND hWnd);

        /**
         * Allocate D3D resources.
         *
         * If 'device' has not yet been set, find the adapter that currently
         * displays 'hWnd' and create the device using this adapter.
         *
         * @param hWnd
         *
         * @throws D3DException
         */
        void initialise(HWND hWnd);

        bool onCreatingDepthStencilTexture(
            D3D11_TEXTURE2D_DESC& depthStencilDesc);

        void onCreatingDevice(UINT& inOutFlags,
            vislib::Array<D3D_FEATURE_LEVEL>& inOutFeatureLevels) throw();

        void onCreatingSwapChain(
            DXGI_SWAP_CHAIN_DESC& inOutSwapChainDesc) throw();

        inline ID3D11DepthStencilView *peekDepthStencilView(void) {
            VLSTACKTRACE("AbstractD3D11WindowImpl::peekDepthStencilView", 
                __FILE__, __LINE__);
            return this->depthStencilView;
        }

        /**
         * Return the D3D device without incrementing the reference counter.
         *
         * @return The D3D device.
         */
        inline ID3D11Device *peekDevice(void) {
            VLSTACKTRACE("AbstractD3D11WindowImpl::peekDevice", __FILE__, __LINE__);
            return this->device;
        }

        /**
         * Return the D3D immediate context without incrementing the reference
         * counter.
         *
         * @reutnr THe D3D immediate context.
         */
        inline ID3D11DeviceContext *peekDeviceContext(void) {
            VLSTACKTRACE("AbstractD3D11WindowImpl::peekDeviceContext", __FILE__, 
                __LINE__);
            return this->deviceContext;
        }

        inline ID3D11RenderTargetView *peekRenderTargetView(void) {
            VLSTACKTRACE("AbstractD3D11WindowImpl::peekRenderTargetView",
                __FILE__, __LINE__);
            return this->renderTargetView;
        }

        inline IDXGISwapChain *peekSwapChain(void) {
            VLSTACKTRACE("AbstractD3D11WindowImpl::peekSwapChain",__FILE__, 
                __LINE__);
            return this->swapChain;
        }

        void resizeSwapChain(const int width, const int height);

    private:

        /**
         * Updates the render target view and the depth-stencil view.
         *
         * @return S_OK in case of success, an error code otherwise.
         */
        HRESULT updateViews(void);

        /** The depth and stencil buffer. */
        ID3D11Texture2D *depthStencilTexture;

        /** Depth and stencil buffer view. */
        ID3D11DepthStencilView *depthStencilView;

        /** The Direct3D device used for this window. */
        ID3D11Device *device;

        /** The immediate context for 'device'. */
        ID3D11DeviceContext *deviceContext;

         /** The DXGI factory that 'dxgiDevice' was created by. */
        IDXGIFactory *dxgiFactory;

        ///** 
        // * Remembers whether the window should be created in fullscreen mode, if
        // * possible (i.e. if it is not requested to create multiple windows on
        // * different screens), using the exclusive mode.
        // */
        //bool isFullscreen;

        /** View of the back buffer of the window. */
        ID3D11RenderTargetView *renderTargetView;

        /** The swap chain of the window. */
        IDXGISwapChain *swapChain;

        /** The viewport used for this window. */
        D3D11_VIEWPORT viewport;

    };
    
} /* end namespace d3d */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTD3D11WINDOWIMPL_H_INCLUDED */
