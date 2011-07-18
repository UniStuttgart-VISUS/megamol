/*
 * AbstractCallRender.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "AbstractCallRender.h"
#include "vislib/assert.h"

using namespace megamol::core;


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
    this->time = rhs.time;
    this->instTime = rhs.instTime;
    return *this;
}


/*
 * view::AbstractCallRender::AbstractCallRender
 */
view::AbstractCallRender::AbstractCallRender(void) : Call(), cntTimeFrames(1),
        time(0.0f), instTime(0.0f) {
    // intentionally empty
}
