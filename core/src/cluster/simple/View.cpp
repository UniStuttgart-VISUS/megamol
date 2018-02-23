/*
 * View.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/cluster/simple/View.h"
#include "mmcore/cluster/simple/ClientViewRegistration.h"
#include "mmcore/cluster/simple/Client.h"
#include "mmcore/cluster/simple/CommUtil.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/AbstractNamedObject.h"
#include "mmcore/CoreInstance.h"
#include "vislib/assert.h"
#include "vislib/sys/AutoLock.h"
#include "vislib/net/DNS.h"
#include "vislib/net/IPHostEntry.h"
#include "vislib/net/NetworkInformation.h"
#include "vislib/RawStorageSerialiser.h"
#include "vislib/sys/Thread.h"
#include <climits>


using namespace megamol::core;


/*
 * cluster::simple::View::View
 */
cluster::simple::View::View(void) : view::AbstractTileView(),
        firstFrame(false), frozen(false), frozenTime(0.0), frozenCam(NULL),
        registerSlot("register", "The slot registering this view"), client(NULL), initMsg(NULL),
        isFirstInitMsg(false),
        heartBeatPortSlot("heartbeat::port", "The port the heartbeat server communicates on"),
        heartBeatServerSlot("heartbeat::server", "The machine the heartbeat server runs on"),
        directCamSyncSlot("directCamSyn", "Flag controlling whether or not this view directly syncs it's camera without using the heartbeat server. It is not recommended to change this setting!"),
        heartbeat(), heartbeatPayload() {

    this->registerSlot.SetCompatibleCall<ClientViewRegistrationDescription>();
    this->MakeSlotAvailable(&this->registerSlot);

    this->heartBeatPortSlot << new param::IntParam(0, 0, USHRT_MAX);
    this->MakeSlotAvailable(&this->heartBeatPortSlot);

    this->heartBeatServerSlot << new param::StringParam("");
    this->MakeSlotAvailable(&this->heartBeatServerSlot);
    this->heartBeatServerSlot.ForceSetDirty();

    this->directCamSyncSlot << new param::BoolParam(true);
    this->directCamSyncSlot.SetUpdateCallback(&View::directCamSyncUpdated);
    this->MakeSlotAvailable(&this->directCamSyncSlot);

}


/*
 * cluster::simple::View::~View
 */
cluster::simple::View::~View(void) {
    this->Release();
    this->frozenCam = NULL; // DO NOT DELETE
    ASSERT(this->client == NULL);
}


/*
 * cluster::simple::View::Render
 */
void cluster::simple::View::Render(const mmcRenderViewContext& context) {
    double instTime = context.InstanceTime;
    float time = static_cast<float>(context.Time);

    if (this->firstFrame) {
        this->firstFrame = false;
        this->initTileViewParameters();
        AbstractNamedObject::ptr_type ano = this->shared_from_this();
        while (ano != NULL) {
            if (this->loadConfiguration(ano->Name())) break;
            ano = ano->Parent();
        }
        if (this->GetCoreInstance()->Configuration().IsConfigValueSet("scv-heartbeat-port")) {
            try {
                this->heartBeatPortSlot.Param<param::IntParam>()->SetValue(
                    vislib::CharTraitsW::ParseInt(
                        this->GetCoreInstance()->Configuration().ConfigValue("scv-heartbeat-port")));
            } catch(vislib::Exception e) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "Failed to load heartbeat port configuration: %s [%s, %d]\n",
                    e.GetMsgA(), e.GetFile(), e.GetLine());
            } catch(...) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "Failed to load heartbeat port configuration: Unknown exception\n");
            }
        }
        if (this->GetCoreInstance()->Configuration().IsConfigValueSet("scv-heartbeat-server")) {
            this->heartBeatServerSlot.Param<param::StringParam>()->SetValue(
                this->GetCoreInstance()->Configuration().ConfigValue("scv-heartbeat-server"));
        }
    }

    this->processInitialisationMessage();
    this->registerClient();

    if (this->heartBeatPortSlot.IsDirty() || this->heartBeatServerSlot.IsDirty()) {
        this->heartBeatPortSlot.ResetDirty();
        this->heartBeatServerSlot.ResetDirty();

        try {
            this->heartbeat.Connect(
                this->heartBeatServerSlot.Param<param::StringParam>()->Value(),
                static_cast<unsigned int>(this->heartBeatPortSlot.Param<param::IntParam>()->Value()));

        } catch(vislib::Exception e) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Failed to configure heartbeat: %s [%s, %d]\n",
                e.GetMsgA(), e.GetFile(), e.GetLine());
        } catch(...) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Failed to configure heartbeat: Unknown exception\n");
        }
    }

    bool heartbeatOn = false;
    bool doSecondHeartbeat = false;
    try {
        heartbeatOn = this->heartbeat.Sync(1, this->heartbeatPayload);
    } catch(...) {
        heartbeatOn = false;
        doSecondHeartbeat = true;
    }
    if (heartbeatOn) {
        ASSERT(this->heartbeatPayload.GetSize() >= 13);
        unsigned char c = *this->heartbeatPayload.As<unsigned char>();
        doSecondHeartbeat = ((c & 0x01) == 0x01);
        instTime = *this->heartbeatPayload.AsAt<double>(1);
        time = *this->heartbeatPayload.AsAt<float>(1 + sizeof(double));
        view::AbstractView *view = this->GetConnectedView();
        if ((this->heartbeatPayload.GetSize() > 13) && (view != NULL)) {
            vislib::RawStorageSerialiser ser(&this->heartbeatPayload, 1 + sizeof(double) + sizeof(float));
            view->DeserialiseCamera(ser);
        }
    } else {
        doSecondHeartbeat = true;
    }

    view::CallRenderView *crv = this->getCallRenderView();
    this->checkParameters();

    if (!this->frozen) {
        this->frozenTime = instTime;
    }

    if (crv != NULL) {
        crv->ResetAll();
        crv->SetTime(time);
        crv->SetInstanceTime(instTime);
        crv->SetGpuAffinity(context.GpuAffinity);
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
        AbstractNamedObject::const_ptr_type view_p;
        view::AbstractView *view = NULL;
        if (crv->PeekCalleeSlot() != NULL) {
            view_p = crv->PeekCalleeSlot()->Parent();
            view = dynamic_cast<view::AbstractView*>(
                const_cast<AbstractNamedObject*>(view_p.get()));
        }
        if (view != NULL){
            if (this->frozenCam != NULL) view->DeserialiseCamera(*this->frozenCam);
            /* this forces to use this time */
            //view->SetFrameTime(static_cast<float>(this->frozenTime));
        }

        {
            vislib::sys::AutoLock lock(renderLock);

            if (!(*crv)(view::CallRenderView::CALL_RENDER)) {
                this->renderFallbackView();
            }

        }

    } else {
        this->renderFallbackView();

    }

    ::glFlush();

    if (doSecondHeartbeat) {
        try {
            this->heartbeat.Sync(2, this->heartbeatPayload);
        } catch(...) {
        }
    }

#if 0 // TODO: activate with something else
    // HAZARD: requires a second message to ensure all nodes synchronize at the same point!!!
    // sync with second heartbeat ping 
    heartbeatOn = false;
    try {
        heartbeatOn = this->heartbeat.Sync(this->heartbeatPayload);
    } catch(...) {
        heartbeatOn = false;
    }
    if (heartbeatOn) {
        ASSERT(this->heartbeatPayload.GetSize() >= 12);
        instTime = *this->heartbeatPayload.As<double>();
        time = *this->heartbeatPayload.AsAt<float>(sizeof(double));
        view::AbstractView *view = this->GetConnectedView();
        if ((this->heartbeatPayload.GetSize() > 12) && (view != NULL)) {
            vislib::RawStorageSerialiser ser(&this->heartbeatPayload, sizeof(double) + sizeof(float));
            view->DeserialiseCamera(ser);
        }
    }
#endif

}


/*
 * cluster::simple::View::Unregister
 */
void cluster::simple::View::Unregister(cluster::simple::Client *client) {
    if (this->client == client) {
        if (this->client != NULL) {
            this->client->Unregister(this);
        }
        this->client = NULL;
    }
}


/*
 * cluster::simple::View::DisconnectViewCall
 */
void cluster::simple::View::DisconnectViewCall(void) {
    this->disconnectOutgoingRenderCall();
}


/*
 * cluster::simple::View::SetSetupMessage
 */
void cluster::simple::View::SetSetupMessage(const vislib::net::AbstractSimpleMessage& msg) {
    if (this->initMsg != NULL) {
        SAFE_DELETE(this->initMsg);
    }
    this->isFirstInitMsg = true;
    this->initMsg = new vislib::net::SimpleMessage(msg);
}


/*
 * cluster::simple::View::SetCamIniMessage
 */
void cluster::simple::View::SetCamIniMessage(void) {
    if (this->initMsg != NULL) {
        SAFE_DELETE(this->initMsg);
    }
    vislib::net::SimpleMessage *m = new vislib::net::SimpleMessage();
    m->GetHeader().SetMessageID(MSG_CAMERAUPDATE);
    this->initMsg = m;
}


/*
 * cluster::simple::View::ConnectView
 */
void cluster::simple::View::ConnectView(const vislib::StringA& toName) {
    this->GetCoreInstance()->InstantiateCall(this->FullName() + "::renderView", toName,
        this->GetCoreInstance()->GetCallDescriptionManager().Find("CallRenderView"));
}


/*
 * cluster::simple::View::create
 */
bool cluster::simple::View::create(void) {
    this->firstFrame = true;
    return true;
}


/*
 * cluster::simple::View::processInitialisationMessage
 */
void cluster::simple::View::processInitialisationMessage(void) {
    bool deleteInitMessage = true;
    if (this->initMsg != NULL) {
        if (this->initMsg->GetHeader().GetMessageID() == MSG_MODULGRAPH) {
            this->GetCoreInstance()->SetupGraphFromNetwork(this->initMsg);
            this->client->ContinueSetup();
        } else if (this->initMsg->GetHeader().GetMessageID() == MSG_MODULGRAPH_LUA) {
            std::string result;
            char *mg = this->initMsg->GetBodyAs<char>();
            if (this->isFirstInitMsg) {
                // queue graph changes. they will be executed at the start of the next frame.
                if (this->GetCoreInstance()->GetLuaState()->RunString(mg, result)) {
                    this->isFirstInitMsg = false;
                    deleteInitMessage = false;
                } else {
                    vislib::sys::Log::DefaultLog.WriteError("processInitialisationMessage: %s", result.c_str());
                }
            } else {
                // this needs to be delayed until the 'next' frame, otherwise the graph does not exist yet!!!
                this->client->ContinueSetup();
            }

        } else {
            this->directCamSyncUpdated(this->directCamSyncSlot);
            if (this->initMsg->GetHeader().GetMessageID() == MSG_CAMERAUPDATE) {
                this->client->ContinueSetup(2);
            }
        }
        if (deleteInitMessage) {
            SAFE_DELETE(this->initMsg);
        }
    }
}


/*
 * cluster::simple::View::registerClient
 */
bool cluster::simple::View::registerClient(const bool isRawMessageDispatching) {
    if (this->client == NULL) {
        ClientViewRegistration *sccvr = this->registerSlot.CallAs<ClientViewRegistration>();
        if (sccvr != NULL) {
            sccvr->SetView(this);
            sccvr->SetIsRawMessageDispatching(isRawMessageDispatching);
            if ((*sccvr)()) {
                this->client = sccvr->GetClient();
            }
        }
    }

    return (this->client != nullptr);
}


/*
 * cluster::simple::View::release
 */
void cluster::simple::View::release(void) {
    this->frozenCam = NULL; // DO NOT DELETE
    this->heartbeat.Shutdown();
    if (this->client != NULL) {
        this->client->Unregister(this);
        this->client = NULL;
    }
    if (this->initMsg != NULL) {
        SAFE_DELETE(this->initMsg);
    }
}


/*
 * cluster::simple::View::renderFallbackView
 */
void cluster::simple::View::renderFallbackView(void) {
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
 * cluster::simple::View::UpdateFreeze
 */
void cluster::simple::View::UpdateFreeze(bool freeze) {
    this->frozen = freeze;
    this->frozenTime = this->instance()->GetCoreInstanceTime(); // HAZARD
}


/*
 * cluster::simple::View::renderLock
 */
vislib::sys::CriticalSection cluster::simple::View::renderLock;


/*
 * cluster::simple::View::loadConfiguration
 */
bool cluster::simple::View::loadConfiguration(const vislib::StringA& name) {
    vislib::StringA vname(name);
    vname.Append("-tvtile");
    if (this->instance()->Configuration().IsConfigValueSet(vname)) {
        return this->setTile(this->instance()->Configuration().ConfigValue(vname));
    }
    return false;
}


/*
 * cluster::simple::View::directCamSyncUpdated
 */
bool cluster::simple::View::directCamSyncUpdated(param::ParamSlot& slot) {
    ASSERT(&slot == &this->directCamSyncSlot);
    if (this->client != NULL) {
        this->client->SetDirectCamSync(this->directCamSyncSlot.Param<param::BoolParam>()->Value());
    }
    return true;
}
