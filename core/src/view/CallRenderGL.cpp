/*
 * CallRenderGL.cpp
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/CallRender3DGL.h"

using namespace megamol::core;

/*
 * view::CallRender3DGL::CallRender3DGL
 */
view::CallRenderGL::CallRenderGL(void) : AbstractCallRender() {
    // intentionally empty
}


/*
 * view::CallRender3DGL::~CallRender3DGL
 */
view::CallRenderGL::~CallRenderGL(void) {
    // intentionally empty
}


/*
 * view::CallRender3DGL::operator=
 */
view::CallRenderGL& view::CallRenderGL::operator=(const view::CallRenderGL& rhs) {
    view::AbstractCallRender::operator=(rhs);
    _framebuffer = rhs._framebuffer;
    return *this;
}
