/*
 * ClusterDisplayTile.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ClusterDisplayTile.h"
#include "vislib/mathfunctions.h"

using namespace megamol::core;


/*
 * special::ClusterDisplayTile::ClusterDisplayTile
 */
special::ClusterDisplayTile::ClusterDisplayTile(void) : h(1.0f), plane(0),
        w(1.0f), x(0.0f), y(0.0f) {
    // intentionally empty
}


/*
 * special::ClusterDisplayTile::ClusterDisplayTile
 */
special::ClusterDisplayTile::ClusterDisplayTile(
        const special::ClusterDisplayTile& src) : h(src.h), plane(src.plane),
        w(src.w), x(src.x), y(src.y) {
    // intentionally empty
}


/*
 * special::ClusterDisplayTile::~ClusterDisplayTile
 */
special::ClusterDisplayTile::~ClusterDisplayTile(void) {
    // intentionally empty
}


/*
 * special::ClusterDisplayTile::operator=
 */
special::ClusterDisplayTile& special::ClusterDisplayTile::operator=(
        const special::ClusterDisplayTile& rhs) {
    this->h = rhs.h;
    this->plane = rhs.plane;
    this->w = rhs.w;
    this->x = rhs.x;
    this->y = rhs.y;
    return *this;
}


/*
 * special::ClusterDisplayTile::operator==
 */
bool special::ClusterDisplayTile::operator==(
        const special::ClusterDisplayTile& rhs) const {
    return vislib::math::IsEqual(this->h, rhs.h)
        && (this->plane == rhs.plane)
        && vislib::math::IsEqual(this->w, rhs.w)
        && vislib::math::IsEqual(this->x, rhs.x)
        && vislib::math::IsEqual(this->y, rhs.y);
}
