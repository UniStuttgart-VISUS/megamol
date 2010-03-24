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
        if (this->outputFBOTargets.Count() == 0) {
            this->outputFBO->Enable(); 
        } else if (this->outputFBOTargets.Count() == 1) {
            this->outputFBO->Enable(this->outputFBOTargets[0]);
        } else {
            this->outputFBO->EnableMultipleV(
                static_cast<UINT>(this->outputFBOTargets.Count()),
                this->outputFBOTargets.PeekElements());
        }
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
    this->outputFBOTargets.Clear();
    this->outputViewport.Set(0, 0, 0, 0);
}


/*
 * view::AbstractCallRender::SetOutputBuffer
 */
void view::AbstractCallRender::SetOutputBuffer(GLenum buffer,
        int x, int y, int w, int h) {
    this->outputBuffer = buffer;
    this->outputFBO = NULL; // DO NOT DELETE
    this->outputFBOTargets.Clear();
    this->outputViewport.Set(x, y, w, h);
}


/*
 * view::AbstractCallRender::SetOutputBuffer
 */
void view::AbstractCallRender::SetOutputBuffer(
        vislib::graphics::gl::FramebufferObject *fbo,
        UINT cntTargets, UINT* targets, int x, int y, int w, int h) {
    ASSERT(fbo != NULL);
    this->outputFBO = fbo;
    if ((cntTargets == 0) || (targets == NULL)) {
        this->outputFBOTargets.Clear();
    } else {
        this->outputFBOTargets.SetCount(cntTargets);
        for (UINT i = 0; i < cntTargets; i++) {
            this->outputFBOTargets[i] = targets[i];
        }
    }
    this->outputViewport.Set(x, y, w, h);
}


/*
 * view::AbstractCallRender::operator=
 */
view::AbstractCallRender& view::AbstractCallRender::operator=(
        const view::AbstractCallRender& rhs) {
    this->outputBuffer = rhs.outputBuffer;
    this->outputFBO = rhs.outputFBO;
    this->outputFBOTargets = rhs.outputFBOTargets;
    this->outputViewport = rhs.outputViewport;
    return *this;
}


/*
 * view::AbstractCallRender::AbstractCallRender
 */
view::AbstractCallRender::AbstractCallRender(void) : Call(),
        outputBuffer(GL_BACK), outputFBO(NULL), outputFBOTargets(),
        outputViewport(0, 0, 0, 0) {
    // intentionally empty
}
