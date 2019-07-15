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
nextgen::CallRender3D_2::CallRender3D_2(void) : nextgen::AbstractCallRender3D_2(), RenderOutputOpenGL() {
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
    view::RenderOutputOpenGL::operator=(rhs);
    this->mouseX = rhs.mouseX;
    this->mouseY = rhs.mouseY;
    this->mouseFlags = rhs.mouseFlags;
    this->backgroundCol = rhs.backgroundCol;
    this->mouseSelection = rhs.mouseSelection;
    return *this;
}
