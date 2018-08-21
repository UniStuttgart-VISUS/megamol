/*
 * CallRender3D2000GT.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/CallRender3D2000GT.h"

using namespace megamol::core;

/*
 * view::CallRender3D2000GT::CallRender3D2000GT
 */
view::CallRender3D2000GT::CallRender3D2000GT(void) : AbstractCallRender3D2000GT(), RenderOutput() {
    // intentionally empty
}


/*
 * view::CallRender3D2000GT::~CallRender3D2000GT
 */
view::CallRender3D2000GT::~CallRender3D2000GT(void) {
    // intentionally empty
}


/*
 * view::CallRender3D2000GT::operator=
 */
view::CallRender3D2000GT& view::CallRender3D2000GT::operator=(const view::CallRender3D2000GT& rhs) {
    view::AbstractCallRender3D2000GT::operator=(rhs);
    view::RenderOutput::operator=(rhs);

    return *this;
}
