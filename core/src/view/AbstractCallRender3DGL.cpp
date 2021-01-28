/*
 * AbstractCallRender3DGL.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/AbstractCallRender3DGL.h"

using namespace megamol::core;

/*
 * view::AbstractCallRender3D::~AbstractCallRender3DGL
 */
view::AbstractCallRender3DGL::~AbstractCallRender3DGL(void) {
    // intentionally empty
}

/*
 * view::AbstractCallRender3D::operator=
 */
view::AbstractCallRender3DGL& view::AbstractCallRender3DGL::operator=(const view::AbstractCallRender3DGL& rhs) {
	view::AbstractCallRenderGL::operator=(rhs);
    this->minCamState = rhs.minCamState;
    this->bboxs = rhs.bboxs;
    this->lastFrameTime = rhs.lastFrameTime;
    return *this;
}

/*
 * view::AbstractCallRender3D::AbstractCallRender3DGL
 */
view::AbstractCallRender3DGL::AbstractCallRender3DGL(void) : AbstractCallRenderGL(), bboxs(), lastFrameTime(0.0) {
    // intentionally empty
    // TODO init camera parameters
}
