/*
 * TileView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "TileView.h"

using namespace megamol::core;
using vislib::graphics::CameraParameters;


/*
 * view::TileView::TileView
 */
view::TileView::TileView(void) : AbstractTileView() {

}


/*
 * view::TileView::~TileView
 */
view::TileView::~TileView(void) {
    this->Release();
}


/*
 * view::TileView::Render
 */
void view::TileView::Render(void) {
    view::CallRenderView *crv = this->getCallRenderView();
    if (crv == NULL) return; // false ?

    this->checkParameters();

    crv->ResetAll();
    crv->SetProjection(this->getProjType(), this->getEye());
    if ((this->getVirtWidth() != 0) && (this->getVirtHeight() != 0)
            && (this->getTileW() != 0) && (this->getTileH() != 0)) {
        crv->SetTile(this->getVirtWidth(), this->getVirtHeight(),
            this->getTileX(), this->getTileY(), this->getTileW(), this->getTileH());
    }
    crv->SetViewportSize(this->getViewportWidth(), this->getViewportHeight());
    (*crv)(view::CallRenderView::CALL_RENDER);
}


/*
 * view::TileView::create
 */
bool view::TileView::create(void) {
    // intentionally empty
    return true;
}


/*
 * view::TileView::release
 */
void view::TileView::release(void) {
    // intentionally empty
}
