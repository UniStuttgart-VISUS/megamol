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
        eye(thecam::Eye::right), flagBkgnd(false),
        flagProj(false), flagTile(false), height(1.0f),
        projType(thecam::Projection_type::perspective),
        tileH(1.0f), tileW(1.0f), tileX(0.0f), tileY(0.0f), width(1.0f),
        btn(0), down(false), x(0.0f), y(0.0f), mod(Modifier::SHIFT) {
    // intentionally empty
}


/*
 * view::CallRenderView::operator=
 */
view::AbstractCallRenderView& view::AbstractCallRenderView::operator=(const view::AbstractCallRenderView& rhs) {
    view::AbstractCallRender::operator=(rhs);
    this->eye = rhs.eye;
    this->flagBkgnd = rhs.flagBkgnd;
    this->flagProj = rhs.flagProj;
    this->flagTile = rhs.flagTile;
    this->height = rhs.height;
    this->projType = rhs.projType;
    this->tileH = rhs.tileH;
    this->tileW = rhs.tileW;
    this->tileX = rhs.tileX;
    this->tileY = rhs.tileY;
    this->width = rhs.width;
    this->btn = rhs.btn;
    this->down = rhs.down;
    this->x = rhs.x;
    this->y = rhs.y;
    this->mod = rhs.mod;
    return *this;
}
