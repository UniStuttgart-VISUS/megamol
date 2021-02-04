/*
 * RenderOutput.cpp
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/RenderOutput.h"

namespace megamol::core::view {

/*
 * view::RenderOutput::DisableOutputBuffer
 */
void view::RenderOutput::DisableOutputBuffer(void) {
    // intentionally empty
}


/*
 * view::RenderOutput::EnableOutputBuffer
 */
void view::RenderOutput::EnableOutputBuffer(void) {
    // intentionally empty
}


/*
 * view::RenderOutput::FrameBufferObject
 */
std::shared_ptr<CPUFramebuffer> view::RenderOutput::getGenericFramebuffer(void) const {
    return _framebuffer;
}

/*
 * view::RenderOutput::SetFrameBufferObject
 */
void RenderOutput::setGenericFramebuffer(std::shared_ptr<CPUFramebuffer> fbo) {
    _framebuffer = fbo;
}

/*
 * view::RenderOutput::operator=
 */
view::RenderOutput& view::RenderOutput::operator=(
        const view::RenderOutput& rhs) {
    _framebuffer = rhs._framebuffer;
    return *this;
}


/*
 * view::RenderOutput::RenderOutput
 */
view::RenderOutput::RenderOutput(void) : AbstractRenderOutput() {
    // intentionally empty
}


/*
 * view::RenderOutput::~RenderOutput
 */
view::RenderOutput::~RenderOutput(void) {
    // intentionally empty
}
}
