/*
 * AbstractCallRender.cpp
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/AbstractCallRender.h"

namespace megamol::core::view {


AbstractCallRender& view::AbstractCallRender::operator=(
        const view::AbstractCallRender& rhs) {
    this->cntTimeFrames = rhs.cntTimeFrames;
    this->time = rhs.time;
    this->instTime = rhs.instTime;
    this->lastFrameTime = rhs.lastFrameTime;

    return *this;
}


AbstractCallRender::AbstractCallRender(void) : InputCall(), cntTimeFrames(1),
        time(0.0f), instTime(0.0f), lastFrameTime(0.0) {
    // intentionally empty
}

}
