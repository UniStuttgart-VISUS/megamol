/*
 * CallRender3DGL.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/CallRender3DGL.h"

using namespace megamol::core;

/*
 * view::CallRender3DGL::CallRender3DGL
 */
view::CallRender3DGL::CallRender3DGL(void) : view::AbstractCallRender3D_2(), RenderOutputOpenGL() {
    // intentionally empty
}


/*
 * view::CallRender3DGL::~CallRender3DGL
 */
view::CallRender3DGL::~CallRender3DGL(void) {
    // intentionally empty
}


/*
 * view::CallRender3DGL::operator=
 */
view::CallRender3DGL& view::CallRender3DGL::operator=(const view::CallRender3DGL& rhs) {
    view::AbstractCallRender3D_2::operator=(rhs);
    view::RenderOutputOpenGL::operator=(rhs);
    this->mouseX = rhs.mouseX;
    this->mouseY = rhs.mouseY;
    this->mouseFlags = rhs.mouseFlags;
    this->backgroundCol = rhs.backgroundCol;
    this->mouseSelection = rhs.mouseSelection;
    return *this;
}
