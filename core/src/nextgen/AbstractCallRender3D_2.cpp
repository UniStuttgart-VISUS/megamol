/*
 * AbstractCallRender3D_2.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/nextgen/AbstractCallRender3D_2.h"

using namespace megamol::core;

/*
 * nextgen::AbstractCallRender3D::~AbstractCallRender3D_2
 */
nextgen::AbstractCallRender3D_2::~AbstractCallRender3D_2(void) {
    // intentionally empty
}

/*
 * nextgen::AbstractCallRender3D::operator=
 */
nextgen::AbstractCallRender3D_2& nextgen::AbstractCallRender3D_2::operator=(const nextgen::AbstractCallRender3D_2& rhs) {
    this->minCamState = rhs.minCamState;
    this->bboxs = rhs.bboxs;
    this->lastFrameTime = rhs.lastFrameTime;
    return *this;
}

/*
 * nextgen::AbstractCallRender3D::AbstractCallRender3D_2
 */
nextgen::AbstractCallRender3D_2::AbstractCallRender3D_2(void) : AbstractCallRender(), bboxs(), lastFrameTime(0.0) {
    // intentionally empty
    // TODO init camera parameters
}
