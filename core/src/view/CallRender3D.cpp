/*
 * CallRender3D.cpp
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/CallRender3D.h"

using namespace megamol::core;

/*
 * view::CallRender3D::CallRender3D
 */
view::CallRender3D::CallRender3D(void) : AbstractCallRender() {
    // intentionally empty
}


/*
 * view::CallRender3D::~CallRender3D
 */
view::CallRender3D::~CallRender3D(void) {
    // intentionally empty
}

/*
 * view::CallRender3D::operator=
 */
view::CallRender3D& view::CallRender3D::operator=(const view::CallRender3D& rhs) {
    view::AbstractCallRender::operator=(rhs);
    _framebuffer = rhs._framebuffer;
    return *this;
}
