/*
 * CallRender2D.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/CallRender2D.h"

using namespace megamol::core;


/*
 * view::CallRender2D::CallRender2D
 */
view::CallRender2D::CallRender2D(void) : AbstractCallRender(), RenderOutput(),
        bbox(), mouseX(0.0f), mouseY(0.0f), mouseFlags(0) {
    // intentionally empty
}


/*
 * view::CallRender2D::~CallRender2D
 */
view::CallRender2D::~CallRender2D(void) {
    // intentionally empty
}


/*
 * view::CallRender2D::operator=
 */
view::CallRender2D& view::CallRender2D::operator=(
        const view::CallRender2D& rhs) {
    view::AbstractCallRender::operator=(rhs);
    view::RenderOutput::operator=(rhs);

    this->bbox = rhs.bbox;
    this->bkgndCol[0] = rhs.bkgndCol[0];
    this->mouseX = rhs.mouseX;
    this->mouseY = rhs.mouseY;
    this->mouseFlags = rhs.mouseFlags;

    return *this;
}
