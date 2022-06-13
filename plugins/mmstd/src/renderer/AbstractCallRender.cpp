/*
 * AbstractCallRender.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "mmstd/renderer/AbstractCallRender.h"
#include "vislib/assert.h"

using namespace megamol::core;


/*
 * view::AbstractCallRender::NO_GPU_AFFINITY
 */


/*
 * view::AbstractCallRender::operator=
 */
view::AbstractCallRender& view::AbstractCallRender::operator=(const view::AbstractCallRender& rhs) {
    this->_cntTimeFrames = rhs._cntTimeFrames;
    this->_camera = rhs._camera;
    this->_bboxs = rhs._bboxs;
    this->_lastFrameTime = rhs._lastFrameTime;
    this->_time = rhs._time;
    this->_instTime = rhs._instTime;
    this->_isInSituTime = rhs._isInSituTime;
    this->_backgroundCol = rhs._backgroundCol;
    this->_viewResoltion = rhs._viewResoltion;

    return *this;
}


/*
 * view::AbstractCallRender::AbstractCallRender
 */
view::AbstractCallRender::AbstractCallRender(void)
        : InputCall()
        , _cntTimeFrames(1)
        , _time(0.0f)
        , _instTime(0.0f)
        , _isInSituTime(false)
        , _lastFrameTime(0.0)
        , _bboxs() {
    // intentionally empty
}
