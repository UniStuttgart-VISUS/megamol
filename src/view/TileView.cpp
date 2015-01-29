/*
 * TileView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "view/TileView.h"
#include "vislib/memutils.h"

using namespace megamol::core;
using vislib::graphics::CameraParameters;


/*
 * view::TileView::TileView
 */
view::TileView::TileView(void) : AbstractTileView(), firstFrame(false), outCtrl(NULL) {

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
void view::TileView::Render(const mmcRenderViewContext& context) {
    view::CallRenderView *crv = this->getCallRenderView();
    if (crv == NULL) return; // false ?
    if (this->firstFrame) {
        this->initTileViewParameters();
        this->firstFrame = false;
    }
    this->checkParameters();

    crv->ResetAll();
    crv->SetTime(context.Time);
    crv->SetInstanceTime(context.InstanceTime);
    crv->SetProjection(this->getProjType(), this->getEye());
    crv->SetGpuAffinity(context.GpuAffinity);
    if ((this->getVirtWidth() != 0) && (this->getVirtHeight() != 0)
            && (this->getTileW() != 0) && (this->getTileH() != 0)) {
        crv->SetTile(this->getVirtWidth(), this->getVirtHeight(),
            this->getTileX(), this->getTileY(), this->getTileW(), this->getTileH());
    }
    if (this->outCtrl == NULL) {
        crv->SetOutputBuffer(GL_BACK, this->getViewportWidth(), this->getViewportHeight()); // TODO: Fix me!
    } else {
        crv->SetOutputBuffer(*this->outCtrl);
    }
    (*crv)(view::CallRenderView::CALL_RENDER);
}


/*
 * view::TileView::create
 */
bool view::TileView::create(void) {
    this->firstFrame = true;
    return true;
}


/*
 * view::TileView::release
 */
void view::TileView::release(void) {
    // intentionally empty
}


/*
 * view::TileView::OnRenderView
 */
bool view::TileView::OnRenderView(Call& call) {
    view::CallRenderView *crv = dynamic_cast<view::CallRenderView *>(&call);
    if (crv == NULL) return false;

    this->outCtrl = crv;

    mmcRenderViewContext c;
    ::ZeroMemory(&c, sizeof(mmcRenderViewContext));
    c.Size = sizeof(mmcRenderViewContext);
    c.Time = crv->Time();
    if (c.Time < 0.0f) c.Time = this->DefaultTime(crv->InstanceTime());
    c.InstanceTime = crv->InstanceTime();
    // TODO: Affinity
    this->Render(c);

    // TODO: Fix me!

    this->outCtrl = NULL;

    return true;
}


/*
 * view::TileView::unpackMouseCoordinates
 */
void view::TileView::unpackMouseCoordinates(float &x, float &y) {
    x *= this->getTileW();
    y *= this->getTileH();
}
