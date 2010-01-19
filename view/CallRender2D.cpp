/*
 * CallRender2D.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CallRender2D.h"

using namespace megamol::core;


/*
 * view::CallRender2D::CallRender2D
 */
view::CallRender2D::CallRender2D(void) : Call(), bbox(), height(1), width(1) {
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

    this->bbox = rhs.bbox;
    this->bkgndCol[0] = rhs.bkgndCol[0];
    this->height = rhs.height;
    this->width = rhs.width;

    return *this;
}
