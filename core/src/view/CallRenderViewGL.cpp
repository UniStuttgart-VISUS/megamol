/*
 * CallRenderViewGL.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/CallRenderViewGL.h"

using namespace megamol::core;


/*
 * view::CallRenderViewGL::CallRenderViewGL
 */
view::CallRenderViewGL::CallRenderViewGL(void) : CallRenderView(), RenderOutputOpenGL() {
    // intentionally empty
}


/*
 * view::CallRenderViewGL::CallRenderViewGL
 */
view::CallRenderViewGL::CallRenderViewGL(const CallRenderViewGL& src)
        : CallRenderView(), RenderOutputOpenGL() {
    *this = src;
}


/*
 * view::CallRenderViewGL::~CallRenderViewGL
 */
view::CallRenderViewGL::~CallRenderViewGL(void) {
    // intentionally empty
}


/*
 * view::CallRenderViewGL::operator=
 */
view::CallRenderViewGL& view::CallRenderViewGL::operator=(const view::CallRenderViewGL& rhs) {
    view::CallRenderView::operator=(rhs);
    view::RenderOutputOpenGL::operator=(rhs);
    view::GPUAffinity::operator=(rhs);
    return *this;
}
