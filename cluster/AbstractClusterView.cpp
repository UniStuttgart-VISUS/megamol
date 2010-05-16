/*
 * AbstractTileView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/AbstractClusterView.h"
#include "cluster/InfoIconRenderer.h"
#include "CoreInstance.h"
#include "param/StringParam.h"
#include <GL/gl.h>
#include "vislib/AutoLock.h"
#include "vislib/IPEndPoint.h"
#include "vislib/Log.h"
#include "vislib/NetworkInformation.h"
#include "vislib/sysfunctions.h"
#include "vislib/TcpCommChannel.h"


using namespace megamol::core;


/*
 * cluster::AbstractClusterView::AbstractClusterView
 */
cluster::AbstractClusterView::AbstractClusterView(void)
        : view::AbstractTileView(), ClusterControllerClient::Listener(), ControlChannel::Listener(),
        ccc(), ctrlChannel(), lastPingTime(0),
        serverAddressSlot("serverAddress", "The TCP/IP address of the server including the port") {

    this->ccc.AddListener(this);
    this->MakeSlotAvailable(&this->ccc.RegisterSlot());
    this->ctrlChannel.AddListener(this);

    this->serverAddressSlot << new param::StringParam("");
    this->serverAddressSlot.SetUpdateCallback(&AbstractClusterView::onServerAddressChanged);
    this->MakeSlotAvailable(&this->serverAddressSlot);

    // TODO: Implement

}


/*
 * cluster::AbstractClusterView::~AbstractClusterView
 */
cluster::AbstractClusterView::~AbstractClusterView(void) {
    try {
        if (this->ctrlChannel.IsOpen()) {
            this->ctrlChannel.Close();
        }
    } catch(...) {
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
 * cluster::AbstractClusterView::commPing
 */
void cluster::AbstractClusterView::commPing(void) {
    unsigned int ping = vislib::sys::GetTicksOfDay() / 1000;
    if (ping == this->lastPingTime) return;
    this->lastPingTime = ping;

    if (!this->ctrlChannel.IsOpen()) {
        this->ccc.SendUserMsg(ClusterControllerClient::USRMSG_QUERYHEAD, NULL, 0);
    }
    // blublub
    // TODO: Implement

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
 * cluster::AbstractClusterView::OnClusterUserMessage
 */
void cluster::AbstractClusterView::OnClusterUserMessage(cluster::ClusterControllerClient& sender,
        const cluster::ClusterController::PeerHandle& hPeer, bool isClusterMember,
        const UINT32 msgType, const BYTE *msgBody) {

    switch (msgType) {
        case ClusterControllerClient::USRMSG_HEADHERE:
            this->serverAddressSlot.Param<param::StringParam>()->SetValue(reinterpret_cast<const char *>(msgBody));
            break;
        case ClusterControllerClient::USRMSG_SHUTDOWN:
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                "Cluster Shutdown message received. Terminating application.\n");
            this->GetCoreInstance()->Shutdown();
            break;
    }

}


/*
 * cluster::AbstractClusterView::OnControlChannelConnect
 */
void cluster::AbstractClusterView::OnControlChannelConnect(ControlChannel& sender) {
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Connected to head node\n");
}


/*
 * cluster::AbstractClusterView::OnControlChannelDisconnect
 */
void cluster::AbstractClusterView::OnControlChannelDisconnect(ControlChannel& sender) {
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Disconnected from head node\n");
    this->serverAddressSlot.Param<param::StringParam>()->SetValue("", false);
}


/*
 * cluster::AbstractClusterView::onServerAddressChanged
 */
bool cluster::AbstractClusterView::onServerAddressChanged(param::ParamSlot& slot) {
    ASSERT(&slot == &this->serverAddressSlot);
    vislib::StringA address(this->serverAddressSlot.Param<param::StringParam>()->Value());

    if (address.IsEmpty()) {
        try {
            if (this->ctrlChannel.IsOpen()) {
                this->ctrlChannel.Close();
            }
        } catch(...) {
        }
        return true;
    }

    vislib::net::IPEndPoint ep;
    float wildness = vislib::net::NetworkInformation::GuessRemoteEndPoint(ep, address);

    try {
        if (this->ctrlChannel.IsOpen()) {
            this->ctrlChannel.Close();
        }
    } catch(...) {
    }

    if (wildness > 0.8) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Guessed server end point \"%s\" from \"%s\" with too high wildness: %f\n",
            ep.ToStringA().PeekBuffer(), address.PeekBuffer(), wildness);
        this->serverAddressSlot.Param<param::StringParam>()->SetValue("", false);
        return true;
    }

    vislib::sys::Log::DefaultLog.WriteMsg((wildness > 0.3) ? vislib::sys::Log::LEVEL_WARN : vislib::sys::Log::LEVEL_INFO,
        "Starting server on \"%s\" guessed from \"%s\" with wildness: %f\n",
        ep.ToStringA().PeekBuffer(), address.PeekBuffer(), wildness);

    vislib::SmartRef<vislib::net::TcpCommChannel> channel = new vislib::net::TcpCommChannel(vislib::net::TcpCommChannel::FLAG_NODELAY);
    try {
        channel->Connect(ep);
        this->ctrlChannel.Open(channel.DynamicCast<vislib::net::AbstractBidiCommChannel>());
    } catch(vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to connect to server: %s\n", ex.GetMsgA());
        this->serverAddressSlot.Param<param::StringParam>()->SetValue("", false);
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to connect to server: unexpected exception\n");
        this->serverAddressSlot.Param<param::StringParam>()->SetValue("", false);
    }

    return true;
}
