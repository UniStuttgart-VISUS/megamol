/*
 * CallGetTransferFunction.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/CallGetTransferFunction.h"

using namespace megamol::core;


/*
 * view::CallGetTransferFunction::CallGetTransferFunction
 */
view::CallGetTransferFunction::CallGetTransferFunction(void) : Call(),
    texID(0), texSize(1), texData(NULL), range({0.0f, 1.0f}) {
    // intentionally empty
}


/*
 * view::CallGetTransferFunction::~CallGetTransferFunction
 */
view::CallGetTransferFunction::~CallGetTransferFunction(void) {
    // intentionally empty
}

void view::CallGetTransferFunction::BindConvenience(vislib::graphics::gl::GLSLShader& shader, GLenum activeTexture, int textureUniform) {
    glEnable(GL_TEXTURE_1D);
    glActiveTexture(activeTexture);
    glBindTexture(GL_TEXTURE_1D, this->texID);
    glUniform1i(shader.ParameterLocation("tfTexture"), textureUniform);
    glUniform2fv(shader.ParameterLocation("tfRange"), 1, this->range.data());
}

void view::CallGetTransferFunction::UnbindConvenience() {
    glDisable(GL_TEXTURE_1D);
    glBindTexture(GL_TEXTURE_1D, 0);
}
