/*
 * RenderOutput.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/RenderOutput.h"
#include "vislib/assert.h"
#include "vislib/Trace.h"

using namespace megamol::core;


/*
 * view::RenderOutput::DisableOutputBuffer
 */
void view::RenderOutput::DisableOutputBuffer(void) {
    if (this->outputFBO) {
        this->outputFBO->Disable();
    }
}


/*
 * view::RenderOutput::EnableOutputBuffer
 */
void view::RenderOutput::EnableOutputBuffer(void) {
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
 * view::RenderOutput::FrameBufferObject
 */
vislib::graphics::gl::FramebufferObject *view::RenderOutput::FrameBufferObject(void) const {
    return this->outputFBO;
}


/*
 * view::RenderOutput::GetViewport
 */
const vislib::math::Rectangle<int>& view::RenderOutput::GetViewport(void) const {
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
 * view::RenderOutput::OutputBuffer
 */
GLenum view::RenderOutput::OutputBuffer(void) const {
    return this->outputBuffer;
}


/*
 * view::RenderOutput::ResetOutputBuffer
 */
void view::RenderOutput::ResetOutputBuffer(void) {
    this->outputBuffer = GL_BACK;
    this->outputFBO = NULL; // DO NOT DELETE
    this->outputFBOTargets.Clear();
    this->outputViewport.Set(0, 0, 0, 0);
}


/*
 * view::RenderOutput::SetOutputBuffer
 */
void view::RenderOutput::SetOutputBuffer(GLenum buffer,
        int x, int y, int w, int h) {
    this->outputBuffer = buffer;
    this->outputFBO = NULL; // DO NOT DELETE
    this->outputFBOTargets.Clear();
    this->outputViewport.SetFromSize(x, y, w, h);
}


/*
 * view::RenderOutput::SetOutputBuffer
 */
void view::RenderOutput::SetOutputBuffer(
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
 * view::RenderOutput::operator=
 */
view::RenderOutput& view::RenderOutput::operator=(
        const view::RenderOutput& rhs) {
    this->outputBuffer = rhs.outputBuffer;
    this->outputFBO = rhs.outputFBO;
    this->outputFBOTargets = rhs.outputFBOTargets;
    this->outputViewport = rhs.outputViewport;
    return *this;
}


/*
 * view::RenderOutput::RenderOutput
 */
view::RenderOutput::RenderOutput(void) : AbstractRenderOutput(),
        outputBuffer(GL_BACK), outputFBO(NULL), outputFBOTargets(),
        outputViewport(0, 0, 0, 0) {
    // intentionally empty
}


/*
 * view::RenderOutput::~RenderOutput
 */
view::RenderOutput::~RenderOutput(void) {
    this->outputFBO = NULL; // DO NOT DELETE
}
