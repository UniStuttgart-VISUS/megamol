/*
 * CallRenderViewGL.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/CallRenderViewGL.h"

using namespace megamol::core;


/*
 * view::CallRenderViewGL::CallRenderViewGL
 */
view::CallRenderViewGL::CallRenderViewGL(void) : AbstractCallRenderGL(), RenderOutputOpenGL(),
        bkgndB(0), bkgndG(0), bkgndR(0),
        eye(thecam::Eye::right), flagBkgnd(false),
        flagProj(false), flagTile(false), height(1.0f),
        projType(thecam::Projection_type::perspective),
        tileH(1.0f), tileW(1.0f), tileX(0.0f), tileY(0.0f), width(1.0f),
        btn(0), down(false), x(0.0f), y(0.0f), mod(Modifier::SHIFT) {
    // intentionally empty
}


/*
 * view::CallRenderViewGL::CallRenderViewGL
 */
view::CallRenderViewGL::CallRenderViewGL(const CallRenderViewGL& src)
        : AbstractCallRenderGL(), RenderOutputOpenGL() {
    *this = src;
}


/*
 * view::CallRenderViewGL::~CallRenderViewGL
 */
view::CallRenderViewGL::~CallRenderViewGL(void) {
    // intentionally empty
}


/*
 * view::CallRenderViewGL::operator=
 */
view::CallRenderViewGL& view::CallRenderViewGL::operator=(const view::CallRenderViewGL& rhs) {
    view::AbstractCallRenderGL::operator=(rhs);
    view::RenderOutputOpenGL::operator=(rhs);
    this->bkgndB = rhs.bkgndB;
    this->bkgndG = rhs.bkgndG;
    this->bkgndR = rhs.bkgndR;
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
