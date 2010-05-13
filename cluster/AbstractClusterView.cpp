/*
 * AbstractTileView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/AbstractClusterView.h"
#include "cluster/InfoIconRenderer.h"
#include "param/StringParam.h"
#include <GL/gl.h>
#include "vislib/AutoLock.h"
#include "vislib/IPEndPoint.h"
#include "vislib/Log.h"
#include "vislib/sysfunctions.h"
#include "vislib/NetworkInformation.h"


using namespace megamol::core;


/*
 * cluster::AbstractClusterView::AbstractClusterView
 */
cluster::AbstractClusterView::AbstractClusterView(void)
        : view::AbstractTileView(), ClusterControllerClient(),
        setupThread(&AbstractClusterView::setupProcedure),
        ctrlMsgDisp(), setupState(SETUPSTATE_PRECONNECT), setupStateLock() {

    // slot initialized in 'ClusterControllerClient::ctor'
    this->MakeSlotAvailable(&this->registerSlot);
    this->MakeSlotAvailable(&this->ctrlCommAddressSlot);

    this->ctrlMsgDisp.AddListener(this);

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
    if (this->ctrlMsgDisp.IsRunning()) {
        this->ctrlMsgDisp.Terminate(false); // waits for thread to finish
    }
    this->stopCtrlComm();

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
    switch(msgType) {
        case ClusterControllerClient::USRMSG_HEADHERE: {
            const char *address = reinterpret_cast<const char *>(msgBody);
            if (this->isConnectedToHead()) {
                if (!this->isConnectedToHead(address)) {
                    if (this->ctrlMsgDisp.IsRunning()) {
                        this->ctrlMsgDisp.Terminate(false); // waits for thread to finish
                    }
                    this->stopCtrlComm();
                } else break;
            }
            this->ctrlCommAddressSlot.Param<param::StringParam>()->SetValue(address);
        } break;
    }

}


/*
 * cluster::AbstractClusterView::OnCommunicationError
 */
bool cluster::AbstractClusterView::OnCommunicationError(vislib::net::SimpleMessageDispatcher& src,
        const vislib::Exception& exception) throw() {
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO + 100,
        "OnCommunicationError: %s\n", exception.GetMsgA());
    return true;
}


/*
 * cluster::AbstractClusterView::OnDispatcherExited
 */
void cluster::AbstractClusterView::OnDispatcherExited(vislib::net::SimpleMessageDispatcher& src) throw() {
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO + 10, "OnDispatcherExited\n");
    vislib::sys::AutoLock(this->setupStateLock);

    this->setupState = SETUPSTATE_DISCONNECTED;
    if (!this->setupThread.IsRunning()) {
        this->setupThread.Start(static_cast<AbstractClusterView*>(this));
    }
}


/*
 * cluster::AbstractClusterView::OnMessageReceived
 */
bool cluster::AbstractClusterView::OnMessageReceived(
        vislib::net::SimpleMessageDispatcher& src,
        const vislib::net::AbstractSimpleMessage& msg) throw() {

    if (this->setupState != SETUPSTATE_CONNECTED) {
        vislib::sys::AutoLock(this->setupStateLock);
        this->setupState = SETUPSTATE_CONNECTED;
    }

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
bool cluster::AbstractClusterView::isConnectedToHead(const char *address) const {
    vislib::sys::AutoLock(this->setupStateLock);
    if (this->setupState == SETUPSTATE_CONNECTED) {
        return (address == NULL) || this->isCtrlCommConnectedTo(address);
    }
    return false;
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
                This->SendUserMsg(ClusterControllerClient::USRMSG_QUERYHEAD, NULL, 0);

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


/*
 * cluster::AbstractClusterView::OnCtrlCommAddressChanged
 */
void cluster::AbstractClusterView::OnCtrlCommAddressChanged(const vislib::TString& address) {
    if (this->isConnectedToHead()) {
        if (!this->isConnectedToHead(vislib::StringA(address))) {
            this->stopCtrlComm();
        } else return;
    }

    vislib::net::IPEndPoint ep;
    vislib::net::NetworkInformation::GuessRemoteEndPoint(ep, address);
    vislib::net::AbstractCommChannel *cc = this->startCtrlCommClient(ep);
    if (cc != NULL) {
        this->ctrlMsgDisp.Start(cc);

        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
            "Connect to server at \"%s\"\n",
            vislib::StringA(address).PeekBuffer());

    } else {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to connect to server at \"%s\"\n",
            vislib::StringA(address).PeekBuffer());
    }
}
