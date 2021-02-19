/*
 * TileView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/TileView.h"
#include "vislib/memutils.h"

using namespace megamol::core;


/*
 * view::TileView::TileView
 */
view::TileView::TileView(void) : AbstractTileView(), firstFrame(false) {

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
void view::TileView::Render(const mmcRenderViewContext& context, Call* call) {
    view::CallRenderViewGL *crv = this->getCallRenderView();
    if (crv == NULL) return; // false ?
    if (this->firstFrame) {
        this->initTileViewParameters();
        this->firstFrame = false;
    }
    this->checkParameters();

    crv->ResetAll();
    crv->SetTime(static_cast<float>(context.Time));
    crv->SetInstanceTime(context.InstanceTime);
    crv->SetProjection(this->getProjType(), this->getEye());
    if ((this->getVirtWidth() != 0) && (this->getVirtHeight() != 0)
            && (this->getTileW() != 0) && (this->getTileH() != 0)) {
        crv->SetTile(this->getVirtWidth(), this->getVirtHeight(),
            this->getTileX(), this->getTileY(), this->getTileW(), this->getTileH());
    }
    if (this->_fbo == NULL) {
        if (!this->_fbo->Create(this->getViewportWidth(), this->getViewportHeight(), GL_RGBA8, GL_RGBA,
                GL_UNSIGNED_BYTE,
                vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE, GL_DEPTH_COMPONENT24)) {
            throw vislib::Exception("[TILEVIEW] Unable to create image framebuffer object.", __FILE__, __LINE__);
            return;
        }
        crv->SetFramebufferObject(this->_fbo);
    } else {
        if (this->_fbo->IsValid()) {
            if ((this->_fbo->GetWidth() != this->getViewportWidth()) ||
                (this->_fbo->GetHeight() != this->getViewportHeight())) {
                this->_fbo->Release();
                if (!this->_fbo->Create(this->getViewportWidth(), this->getViewportHeight(), GL_RGBA8, GL_RGBA,
                        GL_UNSIGNED_BYTE, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE,
                        GL_DEPTH_COMPONENT24)) {
                    throw vislib::Exception(
                        "[TILEVIEW] Unable to create image framebuffer object.", __FILE__, __LINE__);
                    return;
                }
            }
        }
        crv->SetFramebufferObject(this->_fbo);
    }
    (*crv)(view::CallRenderViewGL::CALL_RENDER);

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
    view::CallRenderViewGL *crv = dynamic_cast<view::CallRenderViewGL *>(&call);
    if (crv == NULL) return false;

    this->_fbo = crv->GetFramebufferObject();

    mmcRenderViewContext c;
    ::ZeroMemory(&c, sizeof(mmcRenderViewContext));
    c.Size = sizeof(mmcRenderViewContext);
    c.Time = crv->Time();
    if (c.Time < 0.0f) c.Time = this->DefaultTime(crv->InstanceTime());
    c.InstanceTime = crv->InstanceTime();
    // TODO: Affinity
    this->Render(c, &call);

    return true;
}


/*
 * view::TileView::unpackMouseCoordinates
 */
void view::TileView::unpackMouseCoordinates(float &x, float &y) {
    x *= this->getTileW();
    y *= this->getTileH();
}
