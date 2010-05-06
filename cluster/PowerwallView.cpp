/*
 * PowerwallView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "PowerwallView.h"
#include "CallRegisterAtController.h"

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

    cluster::CallRegisterAtController *crac
        = this->registerSlot.CallAs<cluster::CallRegisterAtController>();
    if (crac == NULL) {
        outState = InfoIconRenderer::ICONSTATE_ERROR;
        outMsg = _T("Not connected to the cluster controller");
        return;
    }

    if (!(*crac)(cluster::CallRegisterAtController::CALL_GETSTATUS)) {
        outState = InfoIconRenderer::ICONSTATE_ERROR;
        outMsg = _T("Unable to contact cluster controller");
        return;
    }

    if (!crac->GetStatusRunning()) {
        outState = InfoIconRenderer::ICONSTATE_ERROR;
        outMsg = _T("Cluster discovery service is not running");
        return;
    }

    outState = InfoIconRenderer::ICONSTATE_WORK;
    outMsg.Format(_T("Discovering cluster with %d nodes"), crac->GetStatusPeerCount());


    // TODO: Implement

    //outState = InfoIconRenderer::ICONSTATE_WAIT;
    //outMsg = _T("Still waiting for an implementation");
}
