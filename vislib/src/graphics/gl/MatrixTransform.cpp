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
MatrixTransform::MatrixTransform(): viewMatrix(), projectionMatrix(), viewProjMatrix(), isMVPset(false) {

}


/*
 * MatrixTransform::~MatrixTransform
 */
MatrixTransform::~MatrixTransform(void) {

}
