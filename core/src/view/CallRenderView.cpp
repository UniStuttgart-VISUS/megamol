/*
 * CallRenderView.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/CallRenderView.h"

using namespace megamol::core;


/*
 * view::CallRenderViewGL::CallRenderViewGL
 */
view::CallRenderView::CallRenderView(void) : AbstractCallRenderView() {
    // intentionally empty
}


/*
 * view::CallRenderViewGL::CallRenderViewGL
 */
view::CallRenderView::CallRenderView(const CallRenderView& src)
        : AbstractCallRenderView() {
    *this = src;
}


/*
 * view::CallRenderViewGL::~CallRenderViewGL
 */
view::CallRenderView::~CallRenderView(void) {
    // intentionally empty
}


/*
 * view::CallRenderViewGL::operator=
 */
view::CallRenderView& view::CallRenderView::operator=(const view::CallRenderView& rhs) {
    view::AbstractCallRenderView::operator=(rhs);
    _framebuffer = rhs._framebuffer;
    return *this;
}
