/*
 * AbstractCallRender.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/AbstractCallRender.h"
#include "vislib/assert.h"

using namespace megamol::core;


/*
 * view::AbstractCallRender::NO_GPU_AFFINITY
 */
const view::AbstractCallRender::GpuHandleType
view::AbstractCallRender::NO_GPU_AFFINITY = nullptr;


/*
 * view::AbstractCallRender::~AbstractCallRender
 */
view::AbstractCallRender::~AbstractCallRender(void) {
    // intentionally empty
}


/*
 * view::AbstractCallRender::operator=
 */
view::AbstractCallRender& view::AbstractCallRender::operator=(
        const view::AbstractCallRender& rhs) {
    this->cntTimeFrames = rhs.cntTimeFrames;
    this->gpuAffinity = rhs.gpuAffinity;
    this->time = rhs.time;
    this->instTime = rhs.instTime;
    this->isInSituTime = rhs.isInSituTime;
    return *this;
}


/*
 * view::AbstractCallRender::AbstractCallRender
 */
view::AbstractCallRender::AbstractCallRender(void) : Call(), cntTimeFrames(1),
        gpuAffinity(NO_GPU_AFFINITY), time(0.0f), instTime(0.0f), isInSituTime(false) {
    // intentionally empty
}
