/*
 * AbstractCallRender.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "AbstractCallRender.h"
#include "vislib/assert.h"

using namespace megamol::core;


/*
 * view::AbstractCallRender::~AbstractCallRender
 */
view::AbstractCallRender::~AbstractCallRender(void) {
    this->outputFBO = NULL; // DO NOT DELETE
}


/*
 * view::AbstractCallRender::DisableOutputBuffer
 */
void view::AbstractCallRender::DisableOutputBuffer(void) {
    if (this->outputFBO) {
        this->outputFBO->Disable();
    }
}


/*
 * view::AbstractCallRender::EnableOutputBuffer
 */
void view::AbstractCallRender::EnableOutputBuffer(void) {
    if (this->outputFBO) {
        this->outputFBO->Enable(); // TODO: need more control here! (multiple-render-targets, multiple-colour-attachments)
    } else {
        ::glDrawBuffer(this->outputBuffer);
        ::glReadBuffer(this->outputBuffer);
    }
}


/*
 * view::AbstractCallRender::FrameBufferObject
 */
vislib::graphics::gl::FramebufferObject *view::AbstractCallRender::FrameBufferObject(void) const {
    return this->outputFBO;
}


/*
 * view::AbstractCallRender::GetViewport
 */
const vislib::math::Rectangle<int>& view::AbstractCallRender::GetViewport(void) const {
    if (this->outputViewport.IsEmpty()) {
        // sort of lazy evaluation
        if (this->outputFBO != NULL) {
            this->outputViewport.Set(0, 0,
                this->outputFBO->GetWidth(), this->outputFBO->GetHeight());
        } else {
            GLint vp[4];
            ::glGetIntegerv(GL_VIEWPORT, vp);
            this->outputViewport.Set(vp[0], vp[1], vp[2], vp[3]);
        }
    }
    return this->outputViewport;
}


/*
 * view::AbstractCallRender::OutputBuffer
 */
GLenum view::AbstractCallRender::OutputBuffer(void) const {
    return this->outputBuffer;
}


/*
 * view::AbstractCallRender::ResetOutputBuffer
 */
void view::AbstractCallRender::ResetOutputBuffer(void) {
    this->outputBuffer = GL_BACK;
    this->outputFBO = NULL; // DO NOT DELETE
    this->outputViewport.Set(0, 0, 0, 0);
}


/*
 * view::AbstractCallRender::SetOutputBuffer
 */
void view::AbstractCallRender::SetOutputBuffer(GLenum buffer) {
    this->outputBuffer = buffer;
    this->outputFBO = NULL; // DO NOT DELETE
    GLint vp[4];
    ::glGetIntegerv(GL_VIEWPORT, vp);
    this->outputViewport.Set(vp[0], vp[1], vp[2], vp[3]);
}


/*
 * view::AbstractCallRender::SetOutputBuffer
 */
void view::AbstractCallRender::SetOutputBuffer(
        vislib::graphics::gl::FramebufferObject *fbo) {
    ASSERT(fbo != NULL);
    this->outputFBO = fbo;
    this->outputViewport.Set(0, 0,
        this->outputFBO->GetWidth(), this->outputFBO->GetHeight());
}


/*
 * view::AbstractCallRender::SetOutputBuffer
 */
void view::AbstractCallRender::SetOutputBuffer(GLenum buffer,
        const vislib::math::Rectangle<int>& viewport) {
    this->outputBuffer = buffer;
    this->outputFBO = NULL; // DO NOT DELETE
    this->outputViewport = viewport;
}


/*
 * view::AbstractCallRender::SetOutputBuffer
 */
void view::AbstractCallRender::SetOutputBuffer(
        vislib::graphics::gl::FramebufferObject *fbo,
        const vislib::math::Rectangle<int>& viewport) {
    ASSERT(fbo != NULL);
    this->outputFBO = fbo;
    this->outputViewport = viewport;
}


/*
 * view::AbstractCallRender::SetOutputBuffer
 */
void view::AbstractCallRender::SetOutputBuffer(GLenum buffer, int w, int h) {
    this->outputBuffer = buffer;
    this->outputFBO = NULL; // DO NOT DELETE
    this->outputViewport.Set(0, 0, w, h);
}


/*
 * view::AbstractCallRender::SetOutputBuffer
 */
void view::AbstractCallRender::SetOutputBuffer(
        vislib::graphics::gl::FramebufferObject *fbo,
        int w, int h) {
    ASSERT(fbo != NULL);
    this->outputFBO = fbo;
    this->outputViewport.Set(0, 0, w, h);
}


/*
 * view::AbstractCallRender::SetOutputBuffer
 */
void view::AbstractCallRender::SetOutputBuffer(GLenum buffer,
        int x, int y, int w, int h) {
    this->outputBuffer = buffer;
    this->outputFBO = NULL; // DO NOT DELETE
    this->outputViewport.Set(x, y, w, h);
}


/*
 * view::AbstractCallRender::SetOutputBuffer
 */
void view::AbstractCallRender::SetOutputBuffer(
        vislib::graphics::gl::FramebufferObject *fbo,
        int x, int y, int w, int h) {
    ASSERT(fbo != NULL);
    this->outputFBO = fbo;
    this->outputViewport.Set(x, y, w, h);
}


/*
 * view::AbstractCallRender::operator=
 */
view::AbstractCallRender& view::AbstractCallRender::operator=(
        const view::AbstractCallRender& rhs) {
    this->outputBuffer = rhs.outputBuffer;
    this->outputFBO = rhs.outputFBO;
    this->outputViewport = rhs.outputViewport;
    return *this;
}


/*
 * view::AbstractCallRender::AbstractCallRender
 */
view::AbstractCallRender::AbstractCallRender(void) : Call(),
        outputBuffer(GL_BACK), outputFBO(NULL), outputViewport(0, 0, 0, 0) {
    // intentionally empty
}
