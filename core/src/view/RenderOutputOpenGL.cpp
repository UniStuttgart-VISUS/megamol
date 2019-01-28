/*
 * RenderOutputOpenGL.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/RenderOutputOpenGL.h"
#include "vislib/assert.h"
#include "vislib/Trace.h"

using namespace megamol::core;


/*
 * view::RenderOutputOpenGL::DisableOutputBuffer
 */
void view::RenderOutputOpenGL::DisableOutputBuffer(void) {
    if (this->outputFBO) {
        this->outputFBO->Disable();
    }
}


/*
 * view::RenderOutputOpenGL::EnableOutputBuffer
 */
void view::RenderOutputOpenGL::EnableOutputBuffer(void) {
#if defined(DEBUG) || defined(_DEBUG)
    unsigned int otl = vislib::Trace::GetInstance().GetLevel();
    vislib::Trace::GetInstance().SetLevel(0);
#endif /* DEBUG || _DEBUG */
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
    ::glViewport(this->outputViewport.Left(),
        this->outputViewport.Bottom(),
        this->outputViewport.Width(),
        this->outputViewport.Height());
#if defined(DEBUG) || defined(_DEBUG)
    vislib::Trace::GetInstance().SetLevel(otl);
#endif /* DEBUG || _DEBUG */
}


/*
 * view::RenderOutputOpenGL::FrameBufferObject
 */
vislib::graphics::gl::FramebufferObject *view::RenderOutputOpenGL::FrameBufferObject(void) const {
    return this->outputFBO;
}


/*
 * view::RenderOutputOpenGL::GetViewport
 */
const vislib::math::Rectangle<int>& view::RenderOutputOpenGL::GetViewport(void) const {
    if (this->outputViewport.IsEmpty()) {
        // sort of lazy evaluation
        if (this->outputFBO != NULL) {
            this->outputViewport.SetFromSize(0, 0,
                this->outputFBO->GetWidth(), this->outputFBO->GetHeight());
        } else {
            GLint vp[4];
            ::glGetIntegerv(GL_VIEWPORT, vp);
            this->outputViewport.SetFromSize(vp[0], vp[1], vp[2], vp[3]);
        }
    }
    return this->outputViewport;
}


/*
 * view::RenderOutputOpenGL::OutputBuffer
 */
GLenum view::RenderOutputOpenGL::OutputBuffer(void) const {
    return this->outputBuffer;
}


/*
 * view::RenderOutputOpenGL::ResetOutputBuffer
 */
void view::RenderOutputOpenGL::ResetOutputBuffer(void) {
    this->outputBuffer = GL_BACK;
    this->outputFBO = NULL; // DO NOT DELETE
    this->outputFBOTargets.Clear();
    this->outputViewport.Set(0, 0, 0, 0);
}


/*
 * view::RenderOutputOpenGL::SetOutputBuffer
 */
void view::RenderOutputOpenGL::SetOutputBuffer(GLenum buffer,
        int x, int y, int w, int h) {
    this->outputBuffer = buffer;
    this->outputFBO = NULL; // DO NOT DELETE
    this->outputFBOTargets.Clear();
    this->outputViewport.SetFromSize(x, y, w, h);
}


/*
 * view::RenderOutputOpenGL::SetOutputBuffer
 */
void view::RenderOutputOpenGL::SetOutputBuffer(
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
    this->outputViewport.SetFromSize(x, y, w, h);
}


/*
 * view::RenderOutputOpenGL::operator=
 */
view::RenderOutputOpenGL& view::RenderOutputOpenGL::operator=(
        const view::RenderOutputOpenGL& rhs) {
    this->outputBuffer = rhs.outputBuffer;
    this->outputFBO = rhs.outputFBO;
    this->outputFBOTargets = rhs.outputFBOTargets;
    this->outputViewport = rhs.outputViewport;
    return *this;
}


/*
 * view::RenderOutputOpenGL::RenderOutput
 */
view::RenderOutputOpenGL::RenderOutputOpenGL(void) : AbstractRenderOutput(),
        outputBuffer(GL_BACK), outputFBO(NULL), outputFBOTargets(),
        outputViewport(0, 0, 0, 0) {
    // intentionally empty
}


/*
 * view::RenderOutputOpenGL::~RenderOutput
 */
view::RenderOutputOpenGL::~RenderOutputOpenGL(void) {
    this->outputFBO = NULL; // DO NOT DELETE
}
