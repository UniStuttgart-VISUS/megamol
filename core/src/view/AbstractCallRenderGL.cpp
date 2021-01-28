/*
 * AbstractCallRenderGL.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/AbstractCallRenderGL.h"
#include "vislib/assert.h"

using namespace megamol::core;


/*
 * view::AbstractCallRenderGL::NO_GPU_AFFINITY
 */
const view::AbstractCallRenderGL::GpuHandleType
view::AbstractCallRenderGL::NO_GPU_AFFINITY = nullptr;

/*
 * view::AbstractCallRenderGL::operator=
 */
view::AbstractCallRenderGL& view::AbstractCallRenderGL::operator=(
        const view::AbstractCallRenderGL& rhs) {
    this->cntTimeFrames = rhs.cntTimeFrames;
    this->gpuAffinity = rhs.gpuAffinity;
    this->time = rhs.time;
    this->instTime = rhs.instTime;
    this->isInSituTime = rhs.isInSituTime;
    this->lastFrameTime = rhs.lastFrameTime;

    return *this;
}


/*
 * view::AbstractCallRenderGL::AbstractCallRenderGL
 */
view::AbstractCallRenderGL::AbstractCallRenderGL(void) : InputCall(), cntTimeFrames(1),
        gpuAffinity(NO_GPU_AFFINITY), time(0.0f), instTime(0.0f), isInSituTime(false), lastFrameTime(0.0) {
    // intentionally empty
}
