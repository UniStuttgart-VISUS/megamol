/*
 * D3D11Window.cpp
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/D3D11Window.h"

#include "vislib/D3DException.h"
#include "vislib/Trace.h"


/*
 * vislib::graphics::d3d::D3D11Window::~D3D11Window
 */
vislib::graphics::d3d::D3D11Window::~D3D11Window(void) {
    VLSTACKTRACE("D3D11Window::~D3D11Window", __FILE__, __LINE__);
}


/*
 * vislib::graphics::d3d::D3D11Window::onCreated
 */
void vislib::graphics::d3d::D3D11Window::onCreated(HWND hWnd) {
    VLSTACKTRACE("D3D11Window::onCreated", __FILE__, __LINE__);
    AbstractWindow::onCreated(hWnd);
    this->initialise(hWnd);
}


/*
 * vislib::graphics::d3d::D3D11Window::onResized
 */
void vislib::graphics::d3d::D3D11Window::onResized(const int width, 
        const int height) {
    VLSTACKTRACE("D3D11Window::onResized", __FILE__, __LINE__);
    this->resizeSwapChain(width, height);
}
