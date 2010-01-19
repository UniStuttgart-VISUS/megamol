/*
 * CallRenderView.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CallRenderView.h"

using namespace megamol::core;


/*
 * view::CallRenderView::CallRenderView
 */
view::CallRenderView::CallRenderView(void) : Call(), bkgndB(0), bkgndG(0),
        bkgndR(0), eye(vislib::graphics::CameraParameters::RIGHT_EYE),
        flagBkgnd(false), flagProj(false), flagTile(false), flagVP(false),
        height(1.0f),
        projType(vislib::graphics::CameraParameters::MONO_PERSPECTIVE),
        tileH(1.0f), tileW(1.0f), tileX(0.0f), tileY(0.0f), vpHeight(1),
        vpWidth(1), width(1.0f) {
    // intentionally empty
}


/*
 * view::CallRenderView::CallRenderView
 */
view::CallRenderView::CallRenderView(const CallRenderView& src) : Call() {
    *this = src;
}


/*
 * view::CallRenderView::~CallRenderView
 */
view::CallRenderView::~CallRenderView(void) {
    // intentionally empty
}


/*
 * view::CallRenderView::operator=
 */
view::CallRenderView& view::CallRenderView::operator=(const view::CallRenderView& rhs) {
    this->bkgndB = rhs.bkgndB;
    this->bkgndG = rhs.bkgndG;
    this->bkgndR = rhs.bkgndR;
    this->eye = rhs.eye;
    this->flagBkgnd = rhs.flagBkgnd;
    this->flagProj = rhs.flagProj;
    this->flagTile = rhs.flagTile;
    this->flagVP = rhs.flagVP;
    this->height = rhs.height;
    this->projType = rhs.projType;
    this->tileH = rhs.tileH;
    this->tileW = rhs.tileW;
    this->tileX = rhs.tileX;
    this->tileY = rhs.tileY;
    this->vpHeight = rhs.vpHeight;
    this->vpWidth = rhs.vpWidth;
    this->width = rhs.width;
    return *this;
}
