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
    _backgroundCol = rhs._backgroundCol;
    _minCamState = rhs._minCamState;
    _bboxs = rhs._bboxs;
    _backgroundCol = rhs._backgroundCol;
    _framebuffer = rhs._framebuffer;
    return *this;
}
