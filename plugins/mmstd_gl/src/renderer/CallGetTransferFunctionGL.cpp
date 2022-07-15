/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd_gl/renderer/CallGetTransferFunctionGL.h"

using namespace megamol::mmstd_gl;


/*
 * view::CallGetTransferFunctionGL::CallGetTransferFunctionGL
 */
CallGetTransferFunctionGL::CallGetTransferFunctionGL(void) : AbstractCallGetTransferFunction(), texID(0) {
    // intentionally empty
}

/*
 * view::CallGetTransferFunctionGL::~CallGetTransferFunctionGL
 */
CallGetTransferFunctionGL::~CallGetTransferFunctionGL(void) {
    // intentionally empty
}

void CallGetTransferFunctionGL::BindConvenience(
    vislib_gl::graphics::gl::GLSLShader& shader, GLenum activeTexture, int textureUniform) {
    glEnable(GL_TEXTURE_1D);
    glActiveTexture(activeTexture);
    glBindTexture(GL_TEXTURE_1D, this->texID);
    glUniform1i(shader.ParameterLocation("tfTexture"), textureUniform);
    glUniform2fv(shader.ParameterLocation("tfRange"), 1, this->range.data());
}

void CallGetTransferFunctionGL::BindConvenience(
    std::unique_ptr<glowl::GLSLProgram>& shader, GLenum activeTexture, int textureUniform) {
    glEnable(GL_TEXTURE_1D);
    glActiveTexture(activeTexture);
    glBindTexture(GL_TEXTURE_1D, this->texID);
    shader->setUniform("tfTexture", textureUniform);
    glUniform2fv(shader->getUniformLocation("tfRange"), 1, this->range.data());
}

void CallGetTransferFunctionGL::UnbindConvenience() {
    glDisable(GL_TEXTURE_1D);
    glBindTexture(GL_TEXTURE_1D, 0);
}
