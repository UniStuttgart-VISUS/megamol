/*
 * D3D11Window.cpp
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/D3D11Window.h"

#include "vislib/D3DException.h"
#include "the/trace.h"


/*
 * vislib::graphics::d3d::D3D11Window::~D3D11Window
 */
vislib::graphics::d3d::D3D11Window::~D3D11Window(void) {
    THE_STACK_TRACE;
}


/*
 * vislib::graphics::d3d::D3D11Window::onCreated
 */
void vislib::graphics::d3d::D3D11Window::onCreated(HWND hWnd) {
    THE_STACK_TRACE;
    AbstractWindow::onCreated(hWnd);
    this->initialise(hWnd);
}


/*
 * vislib::graphics::d3d::D3D11Window::onResized
 */
void vislib::graphics::d3d::D3D11Window::onResized(const int width, 
        const int height) {
    THE_STACK_TRACE;
    this->resizeSwapChain(width, height);
}
