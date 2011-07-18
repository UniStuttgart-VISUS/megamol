/*
 * CallRender3D.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CallRender3D.h"

using namespace megamol::core;


/*
 * view::CallRender3D::CallRender3D
 */
view::CallRender3D::CallRender3D(void) : AbstractCallRender3D(), RenderOutput() {
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
view::CallRender3D& view::CallRender3D::operator=(
        const view::CallRender3D& rhs) {
    view::AbstractCallRender3D::operator=(rhs);
    view::RenderOutput::operator=(rhs);

    return *this;
}
