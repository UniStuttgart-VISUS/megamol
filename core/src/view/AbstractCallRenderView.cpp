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
view::AbstractCallRenderView::AbstractCallRenderView(void)
        : AbstractCallRender()
        ,
        flagBkgnd(false),
        height(1.0f), width(1.0f),
        btn(0), down(false), x(0.0f), y(0.0f), mod(Modifier::SHIFT) {
    // intentionally empty
}


/*
 * view::CallRenderView::operator=
 */
view::AbstractCallRenderView& view::AbstractCallRenderView::operator=(const view::AbstractCallRenderView& rhs) {
    view::AbstractCallRender::operator=(rhs);
    this->flagBkgnd = rhs.flagBkgnd;
    this->height = rhs.height;
    this->width = rhs.width;
    this->btn = rhs.btn;
    this->down = rhs.down;
    this->x = rhs.x;
    this->y = rhs.y;
    this->mod = rhs.mod;
    return *this;
}
