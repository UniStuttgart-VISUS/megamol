/*
 * AbstractCallRender3D_2.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/AbstractCallRender3D_2.h"

using namespace megamol::core;

/*
 * view::AbstractCallRender3D::CAP_RENDER
 */
const UINT64 view::AbstractCallRender3D_2::CAP_RENDER = 0x1;

/*
 * view::AbstractCallRender3D::CAP_LIGHTING
 */
const UINT64 view::AbstractCallRender3D_2::CAP_LIGHTING = 0x2;

/*
 * view::AbstractCallRender3D::CAP_ANIMATION
 */
const UINT64 view::AbstractCallRender3D_2::CAP_ANIMATION = 0x4;

/*
 * view::AbstractCallRender3D::~AbstractCallRender3D_2
 */
view::AbstractCallRender3D_2::~AbstractCallRender3D_2(void) {
    // intentionally empty
}

/*
 * view::AbstractCallRender3D::operator=
 */
view::AbstractCallRender3D_2& view::AbstractCallRender3D_2::operator=(
    const view::AbstractCallRender3D_2& rhs) {
    // TODO implement
    return *this;
}

/*
 * view::AbstractCallRender3D::AbstractCallRender3D_2
 */
view::AbstractCallRender3D_2::AbstractCallRender3D_2(void)
    : AbstractCallRender(), bboxs(), capabilities(0), lastFrameTime(0.0) {
    // intentionally empty
    // TODO init camera parameters
}
