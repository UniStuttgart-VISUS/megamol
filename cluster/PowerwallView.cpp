/*
 * PowerwallView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "PowerwallView.h"
#include "CallRegisterAtController.h"
#include "cluster/NetMessages.h"
#include "param/BoolParam.h"
#include "vislib/Log.h"

using namespace megamol::core;


/*
 * cluster::PowerwallView::PowerwallView
 */
cluster::PowerwallView::PowerwallView(void) : AbstractClusterView(),
        pauseView(false), pauseFbo(NULL),
        netVSyncSlot("netVSync", "Activates or deactivates the network v-sync"),
        netVSyncBarrier(NULL) {

    this->netVSyncSlot << new param::BoolParam(false);
    this->netVSyncSlot.SetUpdateCallback(&PowerwallView::onNetVSyncChanged);
    this->MakeSlotAvailable(&this->netVSyncSlot);

}


/*
 * cluster::PowerwallView::~PowerwallView
 */
cluster::PowerwallView::~PowerwallView(void) {
    this->Release();
    ASSERT(this->pauseFbo == NULL);
    ASSERT(this->netVSyncBarrier == NULL);
}


/*
 * cluster::PowerwallView::Render
 */
void cluster::PowerwallView::Render(void) {
    view::CallRenderView *crv = this->getCallRenderView();

    if (this->netVSyncBarrier != NULL) {
        this->netVSyncBarrier->Cross(1);
    }

    this->commPing();

    this->checkParameters();

    if (this->pauseFbo != NULL) {
        ::glViewport(0, 0, this->getViewportWidth(), this->getViewportHeight());
        ::glDisable(GL_DEPTH_TEST);
        ::glDisable(GL_COLOR);
        ::glDisable(GL_LIGHTING);
        ::glDisable(GL_CULL_FACE);
        ::glDisable(GL_BLEND);
        ::glEnable(GL_TEXTURE_2D);

        ::glMatrixMode(GL_PROJECTION);
        ::glLoadIdentity();
        ::glMatrixMode(GL_MODELVIEW);
        ::glLoadIdentity();

        this->pauseFbo->BindColourTexture();
        ::glColor3ub(255, 0, 0);
        ::glBegin(GL_QUADS);
        ::glTexCoord2i(0, 0);
        ::glVertex2i(-1, -1);
        ::glTexCoord2i(1, 0);
        ::glVertex2i( 1, -1);
        ::glTexCoord2i(1, 1);
        ::glVertex2i( 1,  1);
        ::glTexCoord2i(0, 1);
        ::glVertex2i(-1,  1);
        ::glEnd();

        ::glBindTexture(GL_TEXTURE_2D, 0);
        ::glDisable(GL_TEXTURE_2D);

        float a = 128.0f / 255.0f;
        ::glColor4f(1.0f, 1.0f, 1.0f, a);

        ::glBegin(GL_QUADS); // TODO: FIX THIS
        ::glVertex2f(0.01f, 0.01f);
        ::glVertex2f(0.02f, 0.01f);
        ::glVertex2f(0.02f, 0.04f);
        ::glVertex2f(0.01f, 0.04f);
        ::glVertex2f(0.03f, 0.01f);
        ::glVertex2f(0.04f, 0.01f);
        ::glVertex2f(0.04f, 0.04f);
        ::glVertex2f(0.03f, 0.04f);
        ::glEnd();

        if (!this->pauseView) {
            this->pauseFbo->Release();
            SAFE_DELETE(this->pauseFbo);
            this->pauseView = false;
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                "Remote view unpaused\n");
        }

    } else if (this->pauseView) {
        try {
            if (!vislib::graphics::gl::FramebufferObject::InitialiseExtensions()) {
                throw vislib::Exception("Unable to initialize frame buffer graphics extensions", __FILE__, __LINE__);
            }
            this->pauseFbo = new vislib::graphics::gl::FramebufferObject();
            if (!this->pauseFbo->Create(this->getViewportWidth(), this->getViewportHeight())) {
                throw vislib::Exception("Unable to create frame buffer object", __FILE__, __LINE__);
            }
            crv->ResetAll();
            crv->SetProjection(this->getProjType(), this->getEye());
            if ((this->getVirtWidth() != 0) && (this->getVirtHeight() != 0)
                    && (this->getTileW() != 0) && (this->getTileH() != 0)) {
                crv->SetTile(this->getVirtWidth(), this->getVirtHeight(),
                    this->getTileX(), this->getTileY(), this->getTileW(), this->getTileH());
            }
            crv->SetOutputBuffer(this->pauseFbo);

            GLint vp[4];
            ::glGetIntegerv(GL_VIEWPORT, vp);
            this->pauseFbo->Enable();
            ::glViewport(vp[0], vp[1], vp[2], vp[3]);

            if (!(*crv)(view::CallRenderView::CALL_RENDER)) {
                throw vislib::Exception("Unable to render current screen", __FILE__, __LINE__);
            }
            this->pauseFbo->Disable();

            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                "Remote view paused\n");

        } catch(vislib::Exception ex) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to pause view: %s\n", ex.GetMsgA());
            this->pauseView = false;
        } catch(...) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to pause view: Unexpected exception\n");
            this->pauseView = false;
        }
        if ((!this->pauseView) && (this->pauseFbo != NULL)) {
            try {
                this->pauseFbo->Release();
            } catch(...) {
            }
            SAFE_DELETE(this->pauseFbo);
        }

        ::glViewport(0, 0, this->getViewportWidth(), this->getViewportHeight());
        ::glDisable(GL_DEPTH_TEST);
        ::glDisable(GL_COLOR);
        ::glDisable(GL_LIGHTING);
        ::glDisable(GL_CULL_FACE);
        ::glDisable(GL_BLEND);
        ::glEnable(GL_TEXTURE_2D);

        ::glMatrixMode(GL_PROJECTION);
        ::glLoadIdentity();
        ::glMatrixMode(GL_MODELVIEW);
        ::glLoadIdentity();

        this->pauseFbo->BindColourTexture();
        ::glColor3ub(255, 255, 255);
        ::glBegin(GL_QUADS);
        ::glTexCoord2i(0, 0);
        ::glVertex2i(-1, -1);
        ::glTexCoord2i(1, 0);
        ::glVertex2i( 1, -1);
        ::glTexCoord2i(1, 1);
        ::glVertex2i( 1,  1);
        ::glTexCoord2i(0, 1);
        ::glVertex2i(-1,  1);
        ::glEnd();

        ::glBindTexture(GL_TEXTURE_2D, 0);
        ::glDisable(GL_TEXTURE_2D);

    } else if (crv != NULL) {
        crv->ResetAll();
        crv->SetProjection(this->getProjType(), this->getEye());
        if ((this->getVirtWidth() != 0) && (this->getVirtHeight() != 0)
                && (this->getTileW() != 0) && (this->getTileH() != 0)) {
            crv->SetTile(this->getVirtWidth(), this->getVirtHeight(),
                this->getTileX(), this->getTileY(), this->getTileW(), this->getTileH());
        }
        crv->SetOutputBuffer(GL_BACK, this->getViewportWidth(), this->getViewportHeight());

        if (!(*crv)(view::CallRenderView::CALL_RENDER)) {
            this->renderFallbackView();
        }

    } else {
        this->renderFallbackView();

    }

    if (this->netVSyncBarrier != NULL) {
        ::glFlush();
        this->netVSyncBarrier->Cross(2);
    }

}


/*
 * cluster::PowerwallView::create
 */
bool cluster::PowerwallView::create(void) {
    this->initTileViewParameters();
    return true;
}


/*
 * cluster::PowerwallView::release
 */
void cluster::PowerwallView::release(void) {
    if (this->pauseFbo != NULL) {
        this->pauseFbo->Release();
        SAFE_DELETE(this->pauseFbo);
        this->pauseView = false;
    }
    if (this->netVSyncBarrier != NULL) {
        NetVSyncBarrier *b = this->netVSyncBarrier;
        this->netVSyncBarrier = NULL;
        b->Disconnect();
        delete b;
    }
}


/*
 * cluster::PowerwallView::getFallbackMessageInfo
 */
void cluster::PowerwallView::getFallbackMessageInfo(vislib::TString& outMsg,
        InfoIconRenderer::IconState& outState) {

    if (this->ctrlChannel.IsOpen()) {
        outState = InfoIconRenderer::ICONSTATE_OK;
        outMsg = _T("Connected to master node");
        return;
    }

    cluster::CallRegisterAtController *crac
        = this->ccc.RegisterSlot().CallAs<cluster::CallRegisterAtController>();
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

    unsigned int nodeCnt = crac->GetStatusPeerCount();

    outState = InfoIconRenderer::ICONSTATE_WORK;
    outMsg.Format(_T("Discovering cluster %s:\nfound %u node%s"),
        vislib::TString(crac->GetStatusClusterName()).PeekBuffer(),
        nodeCnt, (nodeCnt == 1) ? _T("") : _T("s"));
}


/*
 * cluster::PowerwallView::OnCommChannelMessage
 */
void cluster::PowerwallView::OnCommChannelMessage(cluster::CommChannel& sender,
        const vislib::net::AbstractSimpleMessage& msg) {
    using vislib::sys::Log;
    vislib::net::SimpleMessage outMsg;

    switch (msg.GetHeader().GetMessageID()) {
        case cluster::netmessages::MSG_REMOTEVIEW_PAUSE: {
            ASSERT(msg.GetHeader().GetBodySize() == 1);
            this->pauseView = (msg.GetBodyAs<char>()[0] != 0);
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                "Remote view pause %s\n", this->pauseView ? "set" : "unset");
        } break;

        case cluster::netmessages::MSG_FORCENETVSYNC: {
            ASSERT(msg.GetHeader().GetBodySize() == 1);
            this->netVSyncSlot.Param<param::BoolParam>()->SetValue(
                (msg.GetBodyAs<char>()[0] != 0));
        } break;

        case cluster::netmessages::MSG_NETVSYNC_JOIN: {
            vislib::StringA vsyncServerAddress(msg.GetBodyAs<char>());
            if (this->netVSyncBarrier == NULL) {
                NetVSyncBarrier *b = new NetVSyncBarrier();
                try {
                    if (b->Connect(vsyncServerAddress)) {
                        this->netVSyncBarrier = b;
                    } else {
                        delete b;
                        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                            "Unable to connect to network v-sync server \"%s\"\n",
                            vsyncServerAddress.PeekBuffer());
                    }
                } catch(vislib::Exception ex) {
                } catch(...) {
                }
            } else {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
                    "Already connected to network v-sync server\n");
            }
        } break;

        case cluster::netmessages::MSG_SET_CLUSTERVIEW: {
            cluster::AbstractClusterView::OnCommChannelMessage(sender, msg);
            if (this->netVSyncSlot.Param<param::BoolParam>()->Value()) {
                outMsg.GetHeader().SetMessageID(cluster::netmessages::MSG_NETVSYNC_JOIN);
                outMsg.GetHeader().SetBodySize(0);
                outMsg.AssertBodySize();
                sender.SendMessage(outMsg);
            }
        } break;

        default:
            cluster::AbstractClusterView::OnCommChannelMessage(sender, msg);
            break;
    }
}


/*
 * cluster::PowerwallView::onNetVSyncChanged
 */
bool cluster::PowerwallView::onNetVSyncChanged(param::ParamSlot& slot) {
    using vislib::sys::Log;
    ASSERT(&this->netVSyncSlot == &slot);
    bool v = this->netVSyncSlot.Param<param::BoolParam>()->Value();
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Network V-Sync requested %s\n", v ? "on" : "off");

    if (v) {
        // use v-sync barrier
        if (this->netVSyncBarrier == NULL) {
            vislib::net::SimpleMessage msg;
            msg.GetHeader().SetMessageID(cluster::netmessages::MSG_NETVSYNC_JOIN);
            msg.GetHeader().SetBodySize(0);
            msg.AssertBodySize();
            this->ctrlChannel.SendMessage(msg);

        } else {
            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Network V-Sync barrier already on");
        }

    } else {
        // stop using v-sync barrier
        if (this->netVSyncBarrier != NULL) {
            NetVSyncBarrier *b = this->netVSyncBarrier;
            this->netVSyncBarrier = NULL;
            b->Disconnect();
            delete b;

        } else {
            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Network V-Sync barrier already off");
        }
    }

    return true;
}
