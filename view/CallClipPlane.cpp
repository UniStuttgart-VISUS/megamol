/*
 * CallClipPlane.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CallClipPlane.h"

using namespace megamol::core;


/*
 * view::CallClipPlane::CallClipPlane
 */
view::CallClipPlane::CallClipPlane(void) : Call(), plane() {
    this->col[0] = 192;
    this->col[1] = 192;
    this->col[2] = 192;
}


/*
 * view::CallClipPlane::~CallClipPlane
 */
view::CallClipPlane::~CallClipPlane(void) {
    // intentionally empty
}


/*
 * view::CallClipPlane::SetPlane
 */
void view::CallClipPlane::SetPlane(const vislib::math::Plane<float>& plane) {
    this->plane = plane;
}
