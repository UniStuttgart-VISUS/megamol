/*
 * CallRender3D_2.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/nextgen/CallRender3D_2.h"

using namespace megamol::core;

/*
 * nextgen::CallRender3D_2::CallRender3D_2
 */
nextgen::CallRender3D_2::CallRender3D_2(void) : nextgen::AbstractCallRender3D_2(), RenderOutput() {
    // intentionally empty
}


/*
 * nextgen::CallRender3D_2::~CallRender3D_2
 */
nextgen::CallRender3D_2::~CallRender3D_2(void) {
    // intentionally empty
}


/*
 * view::CallRender3D_2::operator=
 */
nextgen::CallRender3D_2& nextgen::CallRender3D_2::operator=(const nextgen::CallRender3D_2& rhs) {
    nextgen::AbstractCallRender3D_2::operator=(rhs);
    view::RenderOutput::operator=(rhs);

    return *this;
}
