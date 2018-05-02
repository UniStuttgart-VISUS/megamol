/*
 * MatrixTransform.cpp
 *
 * Copyright (C) 2018 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/graphics/gl/MatrixTransform.h"


using namespace vislib::graphics::gl;


/*
 * MatrixTransform::MatrixTransform
 */
MatrixTransform::MatrixTransform(): modelViewMatrix(), projectionMatrix(), modelViewProjMatrix(), isMVPset(false) {

}


/*
 * MatrixTransform::~MatrixTransform
 */
MatrixTransform::~MatrixTransform(void) {

}


/*
* MatrixTransform::Quat2RotMat
*/
MatrixTransform::MatrixType MatrixTransform::Quat2RotMat(vislib::math::Quaternion<float> q) const {

    // q = a(=R) + b * I + c * J + d * K
    // see: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Conversion_to_and_from_the_matrix_representation

    MatrixTransform::MatrixType rotMat;

    rotMat.SetIdentity();

    float aa = q.R() * q.R();
    float bb = q.I() * q.I();
    float cc = q.J() * q.J();
    float dd = q.K() * q.K();
    float ab2 = 2.0f * q.R() * q.I();
    float ac2 = 2.0f * q.R() * q.J();
    float ad2 = 2.0f * q.R() * q.K();
    float bc2 = 2.0f * q.I() * q.J();
    float bd2 = 2.0f * q.I() * q.K();
    float cd2 = 2.0f * q.J() * q.K();

    rotMat.SetAt(0, 0, aa + bb - cc - dd);
    rotMat.SetAt(1, 0, bc2 + ad2);
    rotMat.SetAt(2, 0, bd2 - ac2);

    rotMat.SetAt(0, 1, bc2 - ad2);
    rotMat.SetAt(1, 1, aa - bb + cc - dd);
    rotMat.SetAt(2, 1, cd2 + ab2);

    rotMat.SetAt(0, 2, bd2 + ac2);
    rotMat.SetAt(1, 2, cd2 - ab2);
    rotMat.SetAt(2, 2, aa - bb - cc + dd);

    return rotMat;
}
