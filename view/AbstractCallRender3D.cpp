/*
 * AbstractCallRender3D.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "AbstractCallRender3D.h"

using namespace megamol::core;


/*
 * view::AbstractCallRender3D::CAP_RENDER
 */
const UINT64 view::AbstractCallRender3D::CAP_RENDER = 0x1;


/*
 * view::AbstractCallRender3D::CAP_LIGHTING
 */
const UINT64 view::AbstractCallRender3D::CAP_LIGHTING = 0x2;


/*
 * view::AbstractCallRender3D::CAP_ANIMATION
 */
const UINT64 view::AbstractCallRender3D::CAP_ANIMATION = 0x4;


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
    this->capabilities = rhs.capabilities;
    this->lastFrameTime = rhs.lastFrameTime;

    return *this;
}


/*
 * view::AbstractCallRender3D::AbstractCallRender3D
 */
view::AbstractCallRender3D::AbstractCallRender3D(void) : AbstractCallRender(),
        camParams(), bboxs(), capabilities(0), lastFrameTime(0.0) {
    // intentionally empty
}
