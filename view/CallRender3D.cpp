/*
 * CallRender3D.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CallRender3D.h"

using namespace megamol::core;


/*
 * view::CallRender3D::CAP_RENDER
 */
const UINT64 view::CallRender3D::CAP_RENDER = 0x1;


/*
 * view::CallRender3D::CAP_LIGHTING
 */
const UINT64 view::CallRender3D::CAP_LIGHTING = 0x2;


/*
 * view::CallRender3D::CAP_ANIMATION
 */
const UINT64 view::CallRender3D::CAP_ANIMATION = 0x4;


/*
 * view::CallRender3D::CallRender3D
 */
view::CallRender3D::CallRender3D(void) : AbstractCallRender(), camParams(),
        bboxs(), cntTimeFrames(0), capabilities(0), lastFrameTime(0.0) {
    // intentionally empty
}


/*
 * view::CallRender3D::~CallRender3D
 */
view::CallRender3D::~CallRender3D(void) {
    // intentionally empty
}


/*
 * view::CallRender3D::operator=
 */
view::CallRender3D& view::CallRender3D::operator=(
        const view::CallRender3D& rhs) {
    view::AbstractCallRender::operator=(rhs);

    this->camParams = rhs.camParams;
    this->bboxs = rhs.bboxs;
    this->cntTimeFrames = rhs.cntTimeFrames;
    this->capabilities = rhs.capabilities;
    this->time = rhs.time;
    this->lastFrameTime = rhs.lastFrameTime;

    return *this;
}
