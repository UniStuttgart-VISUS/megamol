/*
 * CallRenderDeferred3D.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/CallRenderDeferred3D.h"

using namespace megamol::core;


/*
 * view::CallRenderDeferred3D::CallRenderDeferred3D
 */
view::CallRenderDeferred3D::CallRenderDeferred3D(void) : AbstractCallRender3D(), RenderDeferredOutput() {
    // intentionally empty
}


/*
 * view::CallRenderDeferred3D::~CallRenderDeferred3D
 */
view::CallRenderDeferred3D::~CallRenderDeferred3D(void) {
    // intentionally empty
}


/*
 * view::CallRenderDeferred3D::operator=
 */
view::CallRenderDeferred3D& view::CallRenderDeferred3D::operator=(
        const view::CallRenderDeferred3D& rhs) {
    view::AbstractCallRender3D::operator=(rhs);
    view::RenderDeferredOutput::operator=(rhs);

    return *this;
}
