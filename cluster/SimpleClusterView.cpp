/*
 * SimpleClusterView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/SimpleClusterView.h"
#include "cluster/SimpleClusterClientViewRegistration.h"
#include "cluster/SimpleClusterClient.h"
#include "cluster/SimpleClusterCommUtil.h"
#include "AbstractNamedObject.h"
#include "CoreInstance.h"
#include <GL/gl.h>
#include "vislib/assert.h"
#include "vislib/Thread.h"


using namespace megamol::core;


/*
 * cluster::SimpleClusterView::SimpleClusterView
 */
cluster::SimpleClusterView::SimpleClusterView(void) : view::AbstractTileView(),
        firstFrame(false), frozen(false), frozenTime(0.0), frozenCam(NULL),
        registerSlot("register", "The slot registering this view"), client(NULL), initMsg(NULL) {

    this->registerSlot.SetCompatibleCall<SimpleClusterClientViewRegistrationDescription>();
    this->MakeSlotAvailable(&this->registerSlot);
}


/*
 * cluster::SimpleClusterView::~SimpleClusterView
 */
cluster::SimpleClusterView::~SimpleClusterView(void) {
    this->Release();
    this->frozenCam = NULL; // DO NOT DELETE
    ASSERT(this->client == NULL);
}


/*
 * cluster::SimpleClusterView::Render
 */
void cluster::SimpleClusterView::Render(void) {
    if (this->firstFrame) {
        this->firstFrame = false;
        this->initTileViewParameters();
        AbstractNamedObject *ano = this;
        while (ano != NULL) {
            if (this->loadConfiguration(ano->Name())) break;
            ano = ano->Parent();
        }
    }

    if (this->initMsg != NULL) {
        if (this->initMsg->GetHeader().GetMessageID() == MSG_MODULGRAPH) {
            this->GetCoreInstance()->SetupGraphFromNetwork(this->initMsg);
            this->client->ContinueSetup();
        } else if (this->initMsg->GetHeader().GetMessageID() == MSG_CAMERAUPDATE) {
            this->client->ContinueSetup(2);
        }
        SAFE_DELETE(this->initMsg);
    }

    if (this->client == NULL) {
        SimpleClusterClientViewRegistration *sccvr
            = this->registerSlot.CallAs<SimpleClusterClientViewRegistration>();
        if (sccvr != NULL) {
            sccvr->SetView(this);
            if ((*sccvr)()) {
                this->client = sccvr->GetClient();
            }
        }
    }

    view::CallRenderView *crv = this->getCallRenderView();
    this->checkParameters();

    /* *HAZARD* problem here !!! */
    if (!this->frozen) {
        this->frozenTime = this->instance()->GetInstanceTime();
    }

    if (crv != NULL) {
        crv->ResetAll();
        crv->SetProjection(this->getProjType(), this->getEye());
        if ((this->getVirtWidth() != 0) && (this->getVirtHeight() != 0)
                && (this->getTileW() != 0) && (this->getTileH() != 0)) {
            crv->SetTile(this->getVirtWidth(), this->getVirtHeight(),
                this->getTileX(), this->getTileY(), this->getTileW(), this->getTileH());
        }
        crv->SetOutputBuffer(GL_BACK, this->getViewportWidth(), this->getViewportHeight());

        //if ((this->netVSyncBarrier != NULL) && (this->netVSyncBarrier->GetDataSize() > 0)) {
        //    //printf("Barrier with %u bytes data\n", this->netVSyncBarrier->GetDataSize());
        //    vislib::RawStorageSerialiser camera(
        //        this->netVSyncBarrier->GetData() + 4,
        //        this->netVSyncBarrier->GetDataSize() - 4);
        //}
        view::AbstractView *view = NULL;
        if (crv->PeekCalleeSlot() != NULL) view = dynamic_cast<view::AbstractView*>(
                const_cast<AbstractNamedObject*>(crv->PeekCalleeSlot()->Parent()));
        if (view != NULL){
            if (this->frozenCam != NULL) view->DeserialiseCamera(*this->frozenCam);
            /* this forces to use this time */
            //view->SetFrameTime(static_cast<float>(this->frozenTime));
        }

        if (!(*crv)(view::CallRenderView::CALL_RENDER)) {
            this->renderFallbackView();
        }

    } else {
        this->renderFallbackView();

    }

    ::glFlush();
}


/*
 * cluster::SimpleClusterView::Unregister
 */
void cluster::SimpleClusterView::Unregister(cluster::SimpleClusterClient *client) {
    if (this->client == client) {
        if (this->client != NULL) {
            this->client->Unregister(this);
        }
        this->client = NULL;
    }
}


/*
 * cluster::SimpleClusterView::DisconnectViewCall
 */
void cluster::SimpleClusterView::DisconnectViewCall(void) {
    this->disconnectOutgoingRenderCall();
}


/*
 * cluster::SimpleClusterView::SetSetupMessage
 */
void cluster::SimpleClusterView::SetSetupMessage(const vislib::net::AbstractSimpleMessage& msg) {
    if (this->initMsg != NULL) {
        SAFE_DELETE(this->initMsg);
    }
    this->initMsg = new vislib::net::SimpleMessage(msg);
}


/*
 * cluster::SimpleClusterView::SetCamIniMessage
 */
void cluster::SimpleClusterView::SetCamIniMessage(void) {
    if (this->initMsg != NULL) {
        SAFE_DELETE(this->initMsg);
    }
    vislib::net::SimpleMessage *m = new vislib::net::SimpleMessage();
    m->GetHeader().SetMessageID(MSG_CAMERAUPDATE);
    this->initMsg = m;
}


/*
 * cluster::SimpleClusterView::ConnectView
 */
void cluster::SimpleClusterView::ConnectView(const vislib::StringA toName) {
    this->GetCoreInstance()->InstantiateCall(this->FullName() + "::renderView", toName,
        CallDescriptionManager::Instance()->Find("CallRenderView"));
}


/*
 * cluster::SimpleClusterView::create
 */
bool cluster::SimpleClusterView::create(void) {
    this->firstFrame = true;
    return true;
}


/*
 * cluster::SimpleClusterView::release
 */
void cluster::SimpleClusterView::release(void) {
    this->frozenCam = NULL; // DO NOT DELETE
    if (this->client != NULL) {
        this->client->Unregister(this);
        this->client = NULL;
    }
    if (this->initMsg != NULL) {
        SAFE_DELETE(this->initMsg);
    }
}


/*
 * cluster::SimpleClusterView::renderFallbackView
 */
void cluster::SimpleClusterView::renderFallbackView(void) {
    ::glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
    ::glClear(GL_COLOR_BUFFER_BIT);
    ::glMatrixMode(GL_PROJECTION);
    ::glLoadIdentity();
    ::glMatrixMode(GL_MODELVIEW);
    ::glLoadIdentity();
    ::glRotated(this->frozenTime * 36.0, 0.0, 0.0, 1.0);
    ::glColor3ub(200, 200, 255);
    ::glBegin(GL_LINES);
    ::glVertex2i(0, 0);
    ::glVertex2i(1, 0);
    ::glEnd();
}


/*
 * cluster::SimpleClusterView::UpdateFreeze
 */
void cluster::SimpleClusterView::UpdateFreeze(bool freeze) {
    this->frozen = freeze;
    this->frozenTime = this->instance()->GetInstanceTime();
}


/*
 * cluster::SimpleClusterView::loadConfiguration
 */
bool cluster::SimpleClusterView::loadConfiguration(const vislib::StringA& name) {
    vislib::StringA vname(name);
    vname.Append("-tvtile");
    if (this->instance()->Configuration().IsConfigValueSet(vname)) {
        return this->setTile(this->instance()->Configuration().ConfigValue(vname));
    }
    return false;
}
