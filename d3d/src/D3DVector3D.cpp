/*
 * D3DVector3D.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/D3DVector3D.h"


/*
 * vislib::graphics::d3d::D3DVector3D::D3DVector3D
 */
vislib::graphics::d3d::D3DVector3D::D3DVector3D(void) : Super() {
    for (unsigned int i = 0; i < D; i++) {
        this->components[i] = static_cast<T>(0);
    }
}


/*
 * vislib::graphics::d3d::D3DVector3D::~D3DVector3D
 */
vislib::graphics::d3d::D3DVector3D::~D3DVector3D(void) {
}


/*
 * vislib::graphics::d3d::D3DVector3D::D
 */
const unsigned int vislib::graphics::d3d::D3DVector3D::D = 3;
