/*
 * AbstractCallRender3D.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/AbstractCallRender3D.h"

using namespace megamol::core;


/*
 * view::AbstractCallRender3D::~AbstractCallRender3D
 */
view::AbstractCallRender3D::~AbstractCallRender3D(void) {
    // intentionally empty
}


/*
 * view::AbstractCallRender3D::operator=
 */
view::AbstractCallRender3D& view::AbstractCallRender3D::operator=(
        const view::AbstractCallRender3D& rhs) {
    view::AbstractCallRender::operator=(rhs);

    this->camParams = rhs.camParams;
    this->bboxs = rhs.bboxs;
    this->lastFrameTime = rhs.lastFrameTime;

    return *this;
}


/*
 * view::AbstractCallRender3D::AbstractCallRender3D
 */
view::AbstractCallRender3D::AbstractCallRender3D(void) : AbstractCallRender(),
        camParams(), bboxs(), lastFrameTime(0.0) {
    // intentionally empty
}
