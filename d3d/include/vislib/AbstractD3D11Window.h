/*
 * AbstractD3D11Window.h
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2012 Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTD3D11WINDOW_H_INCLUDED
#define VISLIB_ABSTRACTD3D11WINDOW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <D3D11.h>
#include <DXGI.h>

#include "vislib/AbstractWindow.h"
#include "vislib/ReferenceCounted.h"


namespace vislib {
namespace graphics {
namespace d3d {


    /**
     * This is a superclass for D3D11-based windows.
     */
    class AbstractD3D11Window : public vislib::graphics::AbstractWindow, 
            public vislib::ReferenceCounted {

    public:

        /** Dtor. */
        virtual ~AbstractD3D11Window(void);

        void Create(const vislib::StringA& title, int left, int top, 
            int width, int height, ID3D11Device *device);

        void Create(const vislib::StringW& title, int left, int top, 
            int width, int height, ID3D11Device *device);

        /**
         * Answer whether the window is intended to be a fullscreen window. If
         * this flag is set, the window will use exclusive mode in case only
         * one window is created. Exclusive mode is not possible in a 
         * multi-monitor setup. In this case, a decoration-less window will be
         * created.
         */
        inline bool GetIsFullscreen(void) const {
            VLSTACKTRACE("AbstractD3D11Window::GetIsFullscreen", __FILE__, 
                __LINE__);
            return this->isFullscreen;
        }

    protected:

        /** Ctor. */
        AbstractD3D11Window(void);

        /**
         * Creates the D3D resources once the window was created.
         */
        virtual void onCreated(HWND hWnd);

        virtual LRESULT onMessage(bool& outHandled, UINT msg, WPARAM wParam, 
            LPARAM lParam) throw();

        /**
         * Releases all D3D resources and resets the pointers to NULL.
         */
        void releaseAllD3D(void);

        /**
         * Update the render target view and the viewport of the D3D device to
         * match the current window client area.
         */
        void setRenderTargetViewAndViewport(void);

        /** The Direct3D device used for this window. */
        ID3D11Device *device;

        /** The immediate context for 'device'. */
        ID3D11DeviceContext *deviceContext;

        /** The DXGI device interface of 'device'. */
        IDXGIDevice *dxgiDevice;

         /** The DXGI factory that 'dxgiDevice' was created by. */
        IDXGIFactory *dxgiFactory;

        /** 
         * Remembers whether the window should be created in fullscreen mode, if
         * possible (i.e. if it is not requested to create multiple windows on
         * different screens), using the exclusive mode.
         */
        bool isFullscreen;

        /** View of the back buffer of the window. */
        ID3D11RenderTargetView *renderTargetView;

        /** The swap chain of the window. */
        IDXGISwapChain *swapChain;

        /** The viewport used for this window. */
        D3D11_VIEWPORT viewport;

    private:

        /** 
         * Remembers whether the 'device' is shared, i.e. that there are 
         * multiple windows using the same device object.
         */
        bool isDeviceShared;

    };
    
} /* end namespace d3d */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTD3D11WINDOW_H_INCLUDED */
