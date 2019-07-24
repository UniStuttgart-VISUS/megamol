/*
 * CallRender3D_2.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/CallRender3D_2.h"

using namespace megamol::core;

/*
 * view::CallRender3D_2::CallRender3D_2
 */
view::CallRender3D_2::CallRender3D_2(void) : view::AbstractCallRender3D_2(), RenderOutputOpenGL() {
    // intentionally empty
}


/*
 * view::CallRender3D_2::~CallRender3D_2
 */
view::CallRender3D_2::~CallRender3D_2(void) {
    // intentionally empty
}


/*
 * view::CallRender3D_2::operator=
 */
view::CallRender3D_2& view::CallRender3D_2::operator=(const view::CallRender3D_2& rhs) {
    view::AbstractCallRender3D_2::operator=(rhs);
    view::RenderOutputOpenGL::operator=(rhs);
    this->mouseX = rhs.mouseX;
    this->mouseY = rhs.mouseY;
    this->mouseFlags = rhs.mouseFlags;
    this->backgroundCol = rhs.backgroundCol;
    this->mouseSelection = rhs.mouseSelection;
    return *this;
}
