/*
 * AbstractTileView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/AbstractClusterView.h"
#include "cluster/InfoIconRenderer.h"
#include <GL/gl.h>
#include "vislib/AutoLock.h"
#include "vislib/Log.h"
#include "vislib/sysfunctions.h"


using namespace megamol::core;


/*
 * cluster::AbstractClusterView::AbstractClusterView
 */
cluster::AbstractClusterView::AbstractClusterView(void)
        : view::AbstractTileView(), ClusterControllerClient(),
        setupThread(&AbstractClusterView::setupProcedure),
        setupState(SETUPSTATE_PRECONNECT), setupStateLock(),
        commChnlCtrl(), commChnlCam(), ctrlMsgDispatch(), camMsgDispatch() {

    // slot initialized in 'ClusterControllerClient::ctor'
    this->MakeSlotAvailable(&this->registerSlot);

    this->ctrlMsgDispatch.AddListener(this);
    this->camMsgDispatch.AddListener(this);

    // TODO: Implement

}


/*
 * cluster::AbstractClusterView::~AbstractClusterView
 */
cluster::AbstractClusterView::~AbstractClusterView(void) {
    this->setupStateLock.Lock();
    this->setupState = SETUPSTATE_DISCONNECTED;
    if (this->setupThread.IsRunning()) {
        this->setupStateLock.Unlock();
        this->setupThread.Join();
    } else {
        this->setupStateLock.Unlock();
    }

    // TODO: Implement

}


/*
 * cluster::AbstractClusterView::ResetView
 */
void cluster::AbstractClusterView::ResetView(void) {
    // intentionally empty to disallow local user input
}


/*
 * cluster::AbstractClusterView::SetCursor2DButtonState
 */
void cluster::AbstractClusterView::SetCursor2DButtonState(unsigned int btn, bool down) {
    // intentionally empty to disallow local user input
}


/*
 * cluster::AbstractClusterView::SetCursor2DPosition
 */
void cluster::AbstractClusterView::SetCursor2DPosition(float x, float y) {
    // intentionally empty to disallow local user input
}


/*
 * cluster::AbstractClusterView::SetInputModifier
 */
void cluster::AbstractClusterView::SetInputModifier(mmcInputModifier mod, bool down) {
    // intentionally empty to disallow local user input
}


/*
 * cluster::AbstractClusterView::OnClusterAvailable
 */
void cluster::AbstractClusterView::OnClusterAvailable(void) {
    ClusterControllerClient::OnClusterAvailable();

    vislib::sys::AutoLock(this->setupStateLock);

    if (((this->setupState == SETUPSTATE_ERROR)
            || (this->setupState == SETUPSTATE_PRECONNECT)
            || (this->setupState == SETUPSTATE_DISCONNECTED))
            && !this->setupThread.IsRunning()) {
        this->setupState = SETUPSTATE_PRECONNECT;
        this->setupThread.Start(static_cast<AbstractClusterView*>(this));
    }

}


/*
 * cluster::ClusterControllerClient::OnClusterUnavailable
 */
void cluster::AbstractClusterView::OnClusterUnavailable(void) {
    ClusterControllerClient::OnClusterUnavailable();

    this->setupStateLock.Lock();
    this->setupState = SETUPSTATE_DISCONNECTED;
    if (this->setupThread.IsRunning()) {
        this->setupStateLock.Unlock();
        this->setupThread.Join();
    } else {
        this->setupStateLock.Unlock();
    }

}


/*
 * cluster::AbstractClusterView::OnUserMsg
 */
void cluster::AbstractClusterView::OnUserMsg(
        const cluster::ClusterController::PeerHandle& hPeer,
        const UINT32 msgType, const BYTE *msgBody) {

    if (msgType == ClusterControllerClient::USRMSG_QUERYHEAD) {
        printf("QUERYHEAD message received :-)\n");
    }

    //TODO: Implement

}


/*
 * cluster::AbstractClusterView::OnMessageReceived
 */
bool cluster::AbstractClusterView::OnMessageReceived(
        vislib::net::SimpleMessageDispatcher& src,
        const vislib::net::AbstractSimpleMessage& msg) throw() {

    //TODO: Implement

    return true;
}


/*
 * cluster::AbstractClusterView::renderFallbackView
 */
void cluster::AbstractClusterView::renderFallbackView(void) {

    ::glViewport(0, 0, this->getViewportWidth(), this->getViewportHeight());
    ::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    ::glClear(GL_COLOR_BUFFER_BIT);

    if ((this->getViewportHeight() <= 1) || (this->getViewportWidth() <= 1)) return;

    ::glMatrixMode(GL_PROJECTION);
    ::glLoadIdentity();
    float aspect = static_cast<float>(this->getViewportWidth())
        / static_cast<float>(this->getViewportHeight());
    if ((this->getProjType() == vislib::graphics::CameraParameters::MONO_PERSPECTIVE)
            || (this->getProjType() == vislib::graphics::CameraParameters::MONO_ORTHOGRAPHIC)) {
        if (aspect > 1.0f) {
            ::glScalef(2.0f / aspect, -2.0f, 1.0f);
        } else {
            ::glScalef(2.0f, -2.0f * aspect, 1.0f);
        }
        ::glTranslatef(-0.5f, -0.5f, 0.0f);
    } else {
        if (this->getEye() == vislib::graphics::CameraParameters::RIGHT_EYE) {
            ::glTranslatef(0.5f, 0.0f, 0.0f);
        } else {
            ::glTranslatef(-0.5f, 0.0f, 0.0f);
        }
        if (aspect > 2.0f) {
            ::glScalef(2.0f / aspect, -2.0f, 1.0f);
        } else {
            ::glScalef(1.0f, -1.0f * aspect, 1.0f);
        }
        ::glTranslatef(-0.5f, -0.5f, 0.0f);
    }
    const float border = 0.05f;
    ::glTranslatef(border, border, 0.0f);
    ::glScalef(1.0f - 2.0f * border, 1.0f - 2.0f * border, 0.0f);

    ::glMatrixMode(GL_MODELVIEW);
    ::glLoadIdentity();

    InfoIconRenderer::IconState icon = InfoIconRenderer::ICONSTATE_UNKNOWN;
    vislib::TString msg;
    this->getFallbackMessageInfo(msg, icon);
    InfoIconRenderer::RenderInfoIcon(icon, msg);

}


/*
 * cluster::AbstractClusterView::getFallbackMessageInfo
 */
void cluster::AbstractClusterView::getFallbackMessageInfo(vislib::TString& outMsg,
        InfoIconRenderer::IconState& outState) {
    outState = InfoIconRenderer::ICONSTATE_UNKNOWN;
    outMsg = _T("State unknown");
}


/*
 * cluster::AbstractClusterView::isConnectedToHead
 */
bool cluster::AbstractClusterView::isConnectedToHead(void) const {
    vislib::sys::AutoLock(this->setupStateLock);
    return (this->setupState == SETUPSTATE_CONNECTED);
}


/*
 * cluster::AbstractClusterView::setupProcedure
 */
DWORD cluster::AbstractClusterView::setupProcedure(void *userData) {
    using vislib::sys::Log;
    bool loop = true;

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Cluster setup procedure started");

    cluster::AbstractClusterView* This = static_cast<cluster::AbstractClusterView*>(userData);
    if (This == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Cluster setup procedure failed: 'This' invalid");
        return -1;
    }

    while (loop) {
        vislib::sys::Thread::Sleep(250);

        This->setupStateLock.Lock();
        switch (This->setupState) {
            case SETUPSTATE_ERROR: // fall through
            case SETUPSTATE_DISCONNECTED:
                loop = false; // abort thread
                break;
            case SETUPSTATE_PRECONNECT:
                This->setupStateLock.Unlock();
                // searching for a head node to connect
                This->SendUserMsg(ClusterControllerClient::USRMSG_QUERYHEAD, reinterpret_cast<BYTE*>(&loop), 1);

                // don't be that aggressive
                vislib::sys::Thread::Sleep(750);
                break;
            case SETUPSTATE_CONNECTED:
                // successfully finished thread
                loop = false;
                break;
        }
        This->setupStateLock.Unlock();
    }

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Cluster setup procedure finished");

    return 0;
}
