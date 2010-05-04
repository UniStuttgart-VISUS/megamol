/*
 * PowerwallView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "PowerwallView.h"

using namespace megamol::core;


/*
 * cluster::PowerwallView::PowerwallView
 */
cluster::PowerwallView::PowerwallView(void) : AbstractClusterView() {

    // TODO: Implement

}


/*
 * cluster::PowerwallView::~PowerwallView
 */
cluster::PowerwallView::~PowerwallView(void) {
    this->Release();

    // TODO: Implement

}


/*
 * cluster::PowerwallView::Render
 */
void cluster::PowerwallView::Render(void) {
    view::CallRenderView *crv = this->getCallRenderView();

    this->checkParameters();

    if (crv != NULL) {

        crv->ResetAll();
        crv->SetProjection(this->getProjType(), this->getEye());
        if ((this->getVirtWidth() != 0) && (this->getVirtHeight() != 0)
                && (this->getTileW() != 0) && (this->getTileH() != 0)) {
            crv->SetTile(this->getVirtWidth(), this->getVirtHeight(),
                this->getTileX(), this->getTileY(), this->getTileW(), this->getTileH());
        }
        crv->SetOutputBuffer(GL_BACK, this->getViewportWidth(), this->getViewportHeight());

        if ((*crv)(view::CallRenderView::CALL_RENDER)) {
            // successfully rendered client view
            return;
        }
    }

    this->renderFallbackView();
}


/*
 * cluster::PowerwallView::create
 */
bool cluster::PowerwallView::create(void) {

    // TODO: Implement

    return true;
}


/*
 * cluster::PowerwallView::release
 */
void cluster::PowerwallView::release(void) {

    // TODO: Implement

}


/*
 * cluster::PowerwallView::getFallbackMessageInfo
 */
void cluster::PowerwallView::getFallbackMessageInfo(vislib::TString& outMsg,
        InfoIconRenderer::IconState& outState) {

    // TODO: Implement

    outState = InfoIconRenderer::ICONSTATE_WAIT;
    outMsg = _T("Still waiting for an implementation");
}
