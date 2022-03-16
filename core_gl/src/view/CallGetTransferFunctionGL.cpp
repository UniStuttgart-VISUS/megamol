/*
 * CallGetTransferFunctionGL.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "mmcore_gl/view/CallGetTransferFunctionGL.h"
#include "stdafx.h"

using namespace megamol::core_gl;


/*
 * view::CallGetTransferFunctionGL::CallGetTransferFunctionGL
 */
view::CallGetTransferFunctionGL::CallGetTransferFunctionGL(void) : AbstractCallGetTransferFunction(), texID(0) {
    // intentionally empty
}

/*
 * view::CallGetTransferFunctionGL::~CallGetTransferFunctionGL
 */
view::CallGetTransferFunctionGL::~CallGetTransferFunctionGL(void) {
    // intentionally empty
}

void view::CallGetTransferFunctionGL::BindConvenience(
    vislib_gl::graphics::gl::GLSLShader& shader, GLenum activeTexture, int textureUniform) {
    glEnable(GL_TEXTURE_1D);
    glActiveTexture(activeTexture);
    glBindTexture(GL_TEXTURE_1D, this->texID);
    glUniform1i(shader.ParameterLocation("tfTexture"), textureUniform);
    glUniform2fv(shader.ParameterLocation("tfRange"), 1, this->range.data());
}

void view::CallGetTransferFunctionGL::BindConvenience(
    std::unique_ptr<glowl::GLSLProgram>& shader, GLenum activeTexture, int textureUniform) {
    glEnable(GL_TEXTURE_1D);
    glActiveTexture(activeTexture);
    glBindTexture(GL_TEXTURE_1D, this->texID);
    shader->setUniform("tfTexture", textureUniform);
    glUniform2fv(shader->getUniformLocation("tfRange"), 1, this->range.data());
}

void view::CallGetTransferFunctionGL::UnbindConvenience() {
    glDisable(GL_TEXTURE_1D);
    glBindTexture(GL_TEXTURE_1D, 0);
}
