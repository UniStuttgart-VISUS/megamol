/*
 * CallClipPlane.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "mmcore/view/CallClipPlane.h"

using namespace megamol::core;


/*
 * view::CallClipPlane::CallClipPlane
 */
view::CallClipPlane::CallClipPlane(void) : Call(), plane() {
    this->col[0] = 192;
    this->col[1] = 192;
    this->col[2] = 192;
    this->col[3] = 255;
}


/*
 * view::CallClipPlane::~CallClipPlane
 */
view::CallClipPlane::~CallClipPlane(void) {
    // intentionally empty
}


/*
 * view::CallClipPlane::CalcPlaneSystem
 */
void view::CallClipPlane::CalcPlaneSystem(vislib::math::Vector<float, 3>& outX, vislib::math::Vector<float, 3>& outY,
    vislib::math::Vector<float, 3>& outZ) const {
    outZ = this->plane.Normal();
    outX.Set(1.0f, 0.0f, 0.0f);
    outY.Set(0.0f, 1.0f, 0.0f);

    if (outZ.IsParallel(outX)) {
        // project cy
        outY -= (outZ * outZ.Dot(outY));
        outY.Normalise();
        outX = outY.Cross(outZ);
        outX.Normalise();
    } else {
        // project cx
        outX -= (outZ * outZ.Dot(outX));
        outX.Normalise();
        outY = outZ.Cross(outX);
        outY.Normalise();
    }
}


/*
 * view::CallClipPlane::SetPlane
 */
void view::CallClipPlane::SetPlane(const vislib::math::Plane<float>& plane) {
    this->plane = plane;
}
