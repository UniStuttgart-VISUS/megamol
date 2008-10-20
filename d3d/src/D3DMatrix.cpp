/*
 * D3DMatrix.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/D3DMatrix.h"



/*
 * vislib::graphics::d3d::D3DMatrix::D3DMatrix
 */
vislib::graphics::d3d::D3DMatrix::D3DMatrix(const T& value) : Super() {
    for (unsigned int i = 0; i < D; i++) {
        for (unsigned int j = 0; j < D; j++) {
            this->components.m[i][j] = value;
        }
    }
}


/*
 * vislib::graphics::d3d::D3DMatrix::D3DMatrix
 */
vislib::graphics::d3d::D3DMatrix::D3DMatrix(
        const T& m11, const T& m12, const T& m13, const T& m14, 
        const T& m21, const T& m22, const T& m23, const T& m24, 
        const T& m31, const T& m32, const T& m33, const T& m34, 
        const T& m41, const T& m42, const T& m43, const T& m44) {
    this->components._11 = m11;
    this->components._12 = m12;
    this->components._13 = m13;
    this->components._14 = m14;

    this->components._21 = m21;
    this->components._22 = m22;
    this->components._23 = m23;
    this->components._24 = m24;

    this->components._31 = m31;
    this->components._32 = m32;
    this->components._33 = m33;
    this->components._34 = m34;

    this->components._41 = m41;
    this->components._42 = m42;
    this->components._43 = m43;
    this->components._44 = m44;
}


/*
 * vislib::graphics::d3d::D3DMatrix::~D3DMatrix
 */
vislib::graphics::d3d::D3DMatrix::~D3DMatrix(void) {
}


/*
 * vislib::graphics::d3d::D3DMatrix::D
 */
const unsigned int vislib::graphics::d3d::D3DMatrix::D = 4;
