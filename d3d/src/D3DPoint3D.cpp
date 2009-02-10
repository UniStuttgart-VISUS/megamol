/*
 * D3DPoint3D.cpp
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/D3DPoint3D.h"


/*
 * vislib::graphics::d3d::D3DPoint3D::D3DPoint3D
 */
vislib::graphics::d3d::D3DPoint3D::D3DPoint3D(void) : Super() {
    for (unsigned int i = 0; i < D; i++) {
        this->coordinates[i] = static_cast<T>(0);
    }
}


/*
 * vislib::graphics::d3d::D3DPoint3D::~D3DPoint3D
 */
vislib::graphics::d3d::D3DPoint3D::~D3DPoint3D(void) {
}


/*
 * vislib::graphics::d3d::D3DPoint3D::D
 */
const unsigned int vislib::graphics::d3d::D3DPoint3D::D = 3;
