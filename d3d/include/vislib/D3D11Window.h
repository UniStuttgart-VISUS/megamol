/*
 * D3D11Window.h
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2012 Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_D3D11WINDOW_H_INCLUDED
#define VISLIB_ABSTRACTD3D11WINDOW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <D3D11.h>
#include <DXGI.h>

#include "vislib/AbstractD3D11WindowImpl.h"
#include "vislib/AbstractWindow.h"
#include "vislib/ReferenceCounted.h"
#include "vislib/SmartRef.h"


namespace vislib {
namespace graphics {
namespace d3d {


    /**
     * This is a superclass for D3D11-based windows.
     */
    class D3D11Window : public AbstractWindow,
            public AbstractD3D11WindowImpl,
            public ReferenceCounted {

    public:

        /** Ctor. */
        inline D3D11Window(ID3D11Device *device = NULL) : AbstractWindow(),
                AbstractD3D11WindowImpl(device), ReferenceCounted() {
            VLSTACKTRACE("D3D11Window::D3D11Window", __FILE__, __LINE__);
        }

        /** Dtor. */
        virtual ~D3D11Window(void);

        ///**
        // * Answer whether the window is intended to be a fullscreen window. If
        // * this flag is set, the window will use exclusive mode in case only
        // * one window is created. Exclusive mode is not possible in a 
        // * multi-monitor setup. In this case, a decoration-less window will be
        // * created.
        // */
        //inline bool GetIsFullscreen(void) const {
        //    VLSTACKTRACE("AbstractD3D11Window::GetIsFullscreen", __FILE__, 
        //        __LINE__);
        //    return this->isFullscreen;
        //}

    protected:

        /**
         * Allocates D3D resources once the window has been created.
         *
         * @param hWnd The handle of this window
         *
         * @throws D3DException In case fo an error.
         */
        virtual void onCreated(HWND hWnd);

        /**
         * Resizes the swap chain to match the new window dimensions.
         *
         * @param width
         * @param height
         *
         * @throws D3DException In case fo an error.
         */
        virtual void onResized(const int width, const int height);
    };

} /* end namespace d3d */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_D3D11WINDOW_H_INCLUDED */
