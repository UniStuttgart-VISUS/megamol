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
 * view::CallRenderView::CALL_RENDER
 */
const unsigned int view::CallRenderView::CALL_RENDER = 0;


/*
 * view::CallRenderView::CALL_FREEZE
 */
const unsigned int view::CallRenderView::CALL_FREEZE = 1;


/*
 * view::CallRenderView::CALL_UNFREEZE
 */
const unsigned int view::CallRenderView::CALL_UNFREEZE = 2;


/*
 * view::CallRenderView::CALL_SETCURSOR2DBUTTONSTATE
 */
const unsigned int view::CallRenderView::CALL_SETCURSOR2DBUTTONSTATE = 3;


/*
 * view::CallRenderView::CALL_SETCURSOR2DPOSITION
 */
const unsigned int view::CallRenderView::CALL_SETCURSOR2DPOSITION = 4;


/*
 * view::CallRenderView::CALL_SETINPUTMODIFIER
 */
const unsigned int view::CallRenderView::CALL_SETINPUTMODIFIER = 5;


/*
 * view::CallRenderView::CALL_RESETVIEW
 */
const unsigned int view::CallRenderView::CALL_RESETVIEW = 6;


/*
 * view::CallRenderView::FunctionName
 */
const char * view::CallRenderView::FunctionName(unsigned int idx) {
    switch (idx) {
        case 0: return "render";
        case 1: return "freeze";
        case 2: return "unfreeze";
        case 3: return "SetCursor2DButtonState";
        case 4: return "SetCursor2DPosition";
        case 5: return "SetInputModifier";
        case 6: return "ResetView";
        default: return NULL;
    }
}

/*
 * view::CallRenderView::CallRenderView
 */
view::CallRenderView::CallRenderView(void) : AbstractCallRender(), RenderOutput(),
        bkgndB(0), bkgndG(0), bkgndR(0),
        eye(vislib::graphics::CameraParameters::RIGHT_EYE), flagBkgnd(false),
        flagProj(false), flagTile(false), height(1.0f),
        projType(vislib::graphics::CameraParameters::MONO_PERSPECTIVE),
        tileH(1.0f), tileW(1.0f), tileX(0.0f), tileY(0.0f), width(1.0f),
        btn(0), down(false), x(0.0f), y(0.0f), mod(MMC_INMOD_SHIFT) {
    // intentionally empty
}


/*
 * view::CallRenderView::CallRenderView
 */
view::CallRenderView::CallRenderView(const CallRenderView& src)
        : AbstractCallRender(), RenderOutput() {
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
    view::AbstractCallRender::operator=(rhs);
    view::RenderOutput::operator=(rhs);
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
