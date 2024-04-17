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
CallGetTransferFunctionGL::CallGetTransferFunctionGL() : AbstractCallGetTransferFunction(), texID(0) {
    this->caps.RequireOpenGL();
}

/*
 * view::CallGetTransferFunctionGL::~CallGetTransferFunctionGL
 */
CallGetTransferFunctionGL::~CallGetTransferFunctionGL() {
    // intentionally empty
}

void CallGetTransferFunctionGL::BindConvenience(
    std::unique_ptr<glowl::GLSLProgram>& shader, GLenum activeTexture, int textureUniform) {
    BindConvenience(*shader, activeTexture, textureUniform);
}

void CallGetTransferFunctionGL::BindConvenience(
    std::shared_ptr<glowl::GLSLProgram>& shader, GLenum activeTexture, int textureUniform) {
    BindConvenience(*shader, activeTexture, textureUniform);
}

void CallGetTransferFunctionGL::BindConvenience(glowl::GLSLProgram& shader, GLenum activeTexture, int textureUniform) {
    glEnable(GL_TEXTURE_1D);
    glActiveTexture(activeTexture);
    glBindTexture(GL_TEXTURE_1D, this->texID);
    shader.setUniform("tfTexture", textureUniform);
    shader.setUniform("tfRange", range[0], range[1]);
}

void CallGetTransferFunctionGL::UnbindConvenience() {
    glDisable(GL_TEXTURE_1D);
    glBindTexture(GL_TEXTURE_1D, 0);
}
