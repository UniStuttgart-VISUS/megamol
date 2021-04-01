/*
 * AbstractCallRenderView.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/CallRenderView.h"

using namespace megamol::core;


/*
 * view::CallRenderView::CallRenderView
 */
view::AbstractCallRenderView::AbstractCallRenderView(void) : AbstractCallRender(), _height(1), _width(1) {
    // intentionally empty
}


/*
 * view::CallRenderView::operator=
 */
view::AbstractCallRenderView& view::AbstractCallRenderView::operator=(const view::AbstractCallRenderView& rhs) {
    view::AbstractCallRender::operator=(rhs);
    _height = rhs._height;
    _width = rhs._width;
    return *this;
}
