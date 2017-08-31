/*
 * RenderDeferredOutput.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/RenderDeferredOutput.h"
#include "vislib/assert.h"

using namespace megamol::core;


/*
 * view::RenderDeferredOutput::DisableOutputBuffer
 */
void view::RenderDeferredOutput::DisableOutputBuffer(void) {
    if (this->outputFBO) {
        this->outputFBO->Disable();
    }
}


/*
 * view::RenderDeferredOutput::EnableOutputBuffer
 */
void view::RenderDeferredOutput::EnableOutputBuffer(void) {
    if (this->outputFBO) {
        this->outputFBO->EnableMultiple(2, GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT);
    }
}


/*
 * view::RenderDeferredOutput::SetOutputBuffer
 */
void view::RenderDeferredOutput::SetOutputBuffer(
        vislib::graphics::gl::FramebufferObject *fbo) {
    this->outputFBO = fbo;
#if defined(DEBUG) || defined(_DEBUG)
    if (this->outputFBO != NULL) {
        ASSERT(this->outputFBO->GetCntColourAttachments() == 2);
        ASSERT(this->outputFBO->GetDepthTextureID() != 0);
    }
#endif
}


/*
 * view::RenderDeferredOutput::RenderDeferredOutput
 */
view::RenderDeferredOutput::RenderDeferredOutput(void) : AbstractRenderOutput(),
        outputFBO(NULL) {
    // intentionally empty
}


/*
 * view::RenderDeferredOutput::~RenderDeferredOutput
 */
view::RenderDeferredOutput::~RenderDeferredOutput(void) {
    this->outputFBO = NULL; // DO NOT DELETE
}
