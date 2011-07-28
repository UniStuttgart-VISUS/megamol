/*
 * Heartbeat.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/simple/Heartbeat.h"
#include "cluster/simple/ClientViewRegistration.h"
#include "cluster/simple/Client.h"
//#include "cluster/simple/CommUtil.h"
//#include "param/BoolParam.h"
#include "param/IntParam.h"
//#include "param/StringParam.h"
//#include "AbstractNamedObject.h"
#include "CoreInstance.h"
//#include <GL/gl.h>
#include "vislib/assert.h"
#include "vislib/AutoLock.h"
#include "vislib/Log.h"
//#include "vislib/DNS.h"
//#include "vislib/IPHostEntry.h"
//#include "vislib/NetworkInformation.h"
//#include "vislib/Thread.h"
//#include <climits>


using namespace megamol::core;


/*
 * cluster::simple::Heartbeat::Heartbeat
 */
cluster::simple::Heartbeat::Heartbeat(void)
        : job::AbstractThreadedJob(), Module(),
        registerSlot("register", "The slot registering this view"), client(NULL), run(false), mainlock(),
        heartBeatPortSlot("heartbeat::port", "The port the heartbeat server communicates on"),
        /* tcBuf, */ tcBufIdx(0)
    /* : view::AbstractTileView(),
        firstFrame(false), frozen(false), frozenTime(0.0), frozenCam(NULL), initMsg(NULL),
        heartBeatServerSlot("heartbeat::server", "The machine the heartbeat server runs on"),
        directCamSyncSlot("directCamSyn", "Flag controlling whether or not this view directly syncs it's camera without using the heartbeat server. It is not recommended to change this setting!")*/
{

    this->registerSlot.SetCompatibleCall<ClientViewRegistrationDescription>();
    this->MakeSlotAvailable(&this->registerSlot);

    this->heartBeatPortSlot << new param::IntParam(0, 0, USHRT_MAX);
    this->MakeSlotAvailable(&this->heartBeatPortSlot);
/*
    this->heartBeatServerSlot << new param::StringParam("");
    this->MakeSlotAvailable(&this->heartBeatServerSlot);
    this->heartBeatServerSlot.ForceSetDirty();

    this->directCamSyncSlot << new param::BoolParam(true);
    this->directCamSyncSlot.SetUpdateCallback(&SimpleClusterView::directCamSyncUpdated);
    this->MakeSlotAvailable(&this->directCamSyncSlot);
*/
}


/*
 * cluster::simple::Heartbeat::~Heartbeat
 */
cluster::simple::Heartbeat::~Heartbeat(void) {
    this->Release();
    //this->frozenCam = NULL; // DO NOT DELETE
    //ASSERT(this->client == NULL);
}


/*
 * cluster::simple::Heartbeat::Terminate
 */
bool cluster::simple::Heartbeat::Terminate(void) {
    this->run = false;
    this->mainlock.Set();
    return true; // will terminate as soon as possible
}


/*
 * cluster::simple::Heartbeat::Unregister
 */
void cluster::simple::Heartbeat::Unregister(cluster::simple::Client *client) {
    if (this->client == client) {
        if (this->client != NULL) {
            this->client->Unregister(this);
        }
        this->client = NULL;
    }
}


/*
 * cluster::simple::Heartbeat::SetTCData
 */
void cluster::simple::Heartbeat::SetTCData(const void *data, SIZE_T size) {
    const unsigned char *dat = static_cast<const unsigned char*>(data);

    double instTime = this->GetCoreInstance()->GetCoreInstanceTime();
    float time = 0.0f;

    if (size >= sizeof(double)) {
        instTime = *static_cast<const double*>(data);
        dat += sizeof(double);
        size -= sizeof(double);
        data = dat;

        if (size >= sizeof(float)) {
            time = *static_cast<const float*>(data);
            dat += sizeof(float);
            size -= sizeof(float);
            data = dat;

        } else {
            size = 0;
        }

        this->GetCoreInstance()->OffsetInstanceTime(instTime - this->GetCoreInstance()->GetCoreInstanceTime());

    } else {
        size = 0;
    }

    // remaining data is camera serialization data

    unsigned int bi = this->tcBufIdx;
    TCBuffer& abuf = this->tcBuf[bi];
    TCBuffer& buf = this->tcBuf[1 - bi];
    {
        vislib::sys::AutoLock(buf.lock);
        if (size > 0) {
            buf.camera.EnforceSize(size);
            ::memcpy(buf.camera, data, size);
        } else {
            buf.camera = abuf.camera;
        }
        buf.instTime = instTime;
        buf.time = time;
    }
    this->tcBufIdx = 1 - this->tcBufIdx;
    this->mainlock.Set();
}


/*
 * cluster::simple::Heartbeat::create
 */
bool cluster::simple::Heartbeat::create(void) {

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

    return true;
}


/*
 * cluster::simple::Heartbeat::release
 */
void cluster::simple::Heartbeat::release(void) {
    if (this->client != NULL) {
        this->client->Unregister(this);
        this->client = NULL;
    }
    this->mainlock.Set();
    // TODO: Implement
}


/*
 * cluster::simple::Heartbeat::Run
 */
DWORD cluster::simple::Heartbeat::Run(void *userData) {
    using vislib::sys::Log;
    this->run = true;

    if (this->client == NULL) {
        ClientViewRegistration *sccvr = this->registerSlot.CallAs<ClientViewRegistration>();
        if (sccvr != NULL) {
            sccvr->SetView(NULL);
            sccvr->SetHeartbeat(this);
            if ((*sccvr)()) {
                this->client = sccvr->GetClient();
                if (this->client != NULL) {
                    Log::DefaultLog.WriteInfo("Connected to SimpleClusterController");
                }
            }
        }
    }
    this->mainlock.Set();

    while (this->run) {
        if (this->client == NULL) break;

        if (this->heartBeatPortSlot.IsDirty()) {
            this->heartBeatPortSlot.ResetDirty();

            // TODO: Implement server restart

        }

        if (!this->client->RequestTCUpdate()) {
            // no connection yet
            this->mainlock.Wait(1000 / 4); // retry 4 times a second
            continue;
        }

        // request was successful
        if (!this->mainlock.Wait(100)) {
            // timed out ... re-request?
            continue;
        }

        // new data or parameter changed

        vislib::sys::Thread::Reschedule();

    }

    if (this->client != NULL) {
        this->client->Unregister(this);
        this->client = NULL;
    }

    return 0;
}


//namespace intern {
//
//    /**
//     * Selects adapters which match this predicate
//     */
//    bool iamSelectAdapterCallback(const vislib::net::NetworkInformation::Adapter& adapter, void *userContext) {
//        vislib::net::TIPHostEntry *he = static_cast<vislib::net::TIPHostEntry*>(userContext);
//        vislib::Array<vislib::net::IPAgnosticAddress> a1 = he->GetAddresses();
//        vislib::net::NetworkInformation::UnicastAddressList a2 = adapter.GetUnicastAddresses();
//        for (SIZE_T i2 = 0; i2 < a2.Count(); i2++) {
//            if (a1.Contains(a2[i2].GetAddress())) {
//                return true;
//            }
//        }
//        return false;
//    }
//
//}
//
//
///*
// * cluster::simple::View::Render
// */
//void cluster::simple::View::Render(float time, double instTime) {
//    if (this->firstFrame) {
//        this->firstFrame = false;
//        this->initTileViewParameters();
//        AbstractNamedObject *ano = this;
//        while (ano != NULL) {
//            if (this->loadConfiguration(ano->Name())) break;
//            ano = ano->Parent();
//        }
//        if (this->GetCoreInstance()->Configuration().IsConfigValueSet("scv-heartbeat-port")) {
//            try {
//                this->heartBeatPortSlot.Param<param::IntParam>()->SetValue(
//                    vislib::CharTraitsW::ParseInt(
//                        this->GetCoreInstance()->Configuration().ConfigValue("scv-heartbeat-port")));
//            } catch(vislib::Exception e) {
//                vislib::sys::Log::DefaultLog.WriteError(
//                    "Failed to load heartbeat port configuration: %s [%s, %d]\n",
//                    e.GetMsgA(), e.GetFile(), e.GetLine());
//            } catch(...) {
//                vislib::sys::Log::DefaultLog.WriteError(
//                    "Failed to load heartbeat port configuration: Unknown exception\n");
//            }
//        }
//        if (this->GetCoreInstance()->Configuration().IsConfigValueSet("scv-heartbeat-server")) {
//            this->heartBeatServerSlot.Param<param::StringParam>()->SetValue(
//                this->GetCoreInstance()->Configuration().ConfigValue("scv-heartbeat-server"));
//        }
//    }
//
//    if (this->initMsg != NULL) {
//        if (this->initMsg->GetHeader().GetMessageID() == MSG_MODULGRAPH) {
//            this->GetCoreInstance()->SetupGraphFromNetwork(this->initMsg);
//            this->client->ContinueSetup();
//        } else {
//            this->directCamSyncUpdated(this->directCamSyncSlot);
//            if (this->initMsg->GetHeader().GetMessageID() == MSG_CAMERAUPDATE) {
//                this->client->ContinueSetup(2);
//            }
//        }
//        SAFE_DELETE(this->initMsg);
//    }
//
//    if (this->client == NULL) {
//        SimpleClusterClientViewRegistration *sccvr
//            = this->registerSlot.CallAs<SimpleClusterClientViewRegistration>();
//        if (sccvr != NULL) {
//            sccvr->SetView(this);
//            if ((*sccvr)()) {
//                this->client = sccvr->GetClient();
//            }
//        }
//    }
//
//    if (this->heartBeatPortSlot.IsDirty() || this->heartBeatServerSlot.IsDirty()) {
//        this->heartBeatPortSlot.ResetDirty();
//        this->heartBeatServerSlot.ResetDirty();
//
//        try {
//
//            //vislib::net::TIPHostEntry he;
//            //vislib::net::DNS::GetHostEntry(he, this->heartBeatServerSlot.Param<param::StringParam>()->Value());
//            //vislib::net::NetworkInformation::AdapterList adapters;
//            //vislib::net::NetworkInformation::GetAdaptersForPredicate(adapters,
//            //    intern::iamSelectAdapterCallback,
//            //    static_cast<void*>(&he));
//
//            //if (!adapters.IsEmpty()) {
//            //    // TODO: Implement Server
//            //}
//            //// TODO: Implement Client
//
//        } catch(vislib::Exception e) {
//            vislib::sys::Log::DefaultLog.WriteError(
//                "Failed to configure heartbeat: %s [%s, %d]\n",
//                e.GetMsgA(), e.GetFile(), e.GetLine());
//        } catch(...) {
//            vislib::sys::Log::DefaultLog.WriteError(
//                "Failed to configure heartbeat: Unknown exception\n");
//        }
//    }
//
//    view::CallRenderView *crv = this->getCallRenderView();
//    this->checkParameters();
//
//    if (!this->frozen) {
//        this->frozenTime = instTime;
//    }
//
//    if (crv != NULL) {
//        crv->ResetAll();
//        crv->SetTime(time);
//        crv->SetInstanceTime(instTime);
//        crv->SetProjection(this->getProjType(), this->getEye());
//        if ((this->getVirtWidth() != 0) && (this->getVirtHeight() != 0)
//                && (this->getTileW() != 0) && (this->getTileH() != 0)) {
//            crv->SetTile(this->getVirtWidth(), this->getVirtHeight(),
//                this->getTileX(), this->getTileY(), this->getTileW(), this->getTileH());
//        }
//        crv->SetOutputBuffer(GL_BACK, this->getViewportWidth(), this->getViewportHeight());
//
//        //if ((this->netVSyncBarrier != NULL) && (this->netVSyncBarrier->GetDataSize() > 0)) {
//        //    //printf("Barrier with %u bytes data\n", this->netVSyncBarrier->GetDataSize());
//        //    vislib::RawStorageSerialiser camera(
//        //        this->netVSyncBarrier->GetData() + 4,
//        //        this->netVSyncBarrier->GetDataSize() - 4);
//        //}
//        view::AbstractView *view = NULL;
//        if (crv->PeekCalleeSlot() != NULL) view = dynamic_cast<view::AbstractView*>(
//                const_cast<AbstractNamedObject*>(crv->PeekCalleeSlot()->Parent()));
//        if (view != NULL){
//            if (this->frozenCam != NULL) view->DeserialiseCamera(*this->frozenCam);
//            /* this forces to use this time */
//            //view->SetFrameTime(static_cast<float>(this->frozenTime));
//        }
//
//        if (!(*crv)(view::CallRenderView::CALL_RENDER)) {
//            this->renderFallbackView();
//        }
//
//    } else {
//        this->renderFallbackView();
//
//    }
//
//    ::glFlush();
//}
//
//
///*
// * cluster::simple::View::DisconnectViewCall
// */
//void cluster::simple::View::DisconnectViewCall(void) {
//    this->disconnectOutgoingRenderCall();
//}
//
//
///*
// * cluster::simple::View::SetSetupMessage
// */
//void cluster::simple::View::SetSetupMessage(const vislib::net::AbstractSimpleMessage& msg) {
//    if (this->initMsg != NULL) {
//        SAFE_DELETE(this->initMsg);
//    }
//    this->initMsg = new vislib::net::SimpleMessage(msg);
//}
//
//
///*
// * cluster::simple::View::SetCamIniMessage
// */
//void cluster::simple::View::SetCamIniMessage(void) {
//    if (this->initMsg != NULL) {
//        SAFE_DELETE(this->initMsg);
//    }
//    vislib::net::SimpleMessage *m = new vislib::net::SimpleMessage();
//    m->GetHeader().SetMessageID(MSG_CAMERAUPDATE);
//    this->initMsg = m;
//}
//
//
///*
// * cluster::simple::View::ConnectView
// */
//void cluster::simple::View::ConnectView(const vislib::StringA toName) {
//    this->GetCoreInstance()->InstantiateCall(this->FullName() + "::renderView", toName,
//        CallDescriptionManager::Instance()->Find("CallRenderView"));
//}
//
//
///*
// * cluster::simple::View::create
// */
//bool cluster::simple::View::create(void) {
//    this->firstFrame = true;
//    return true;
//}
//
//
///*
// * cluster::simple::View::release
// */
//void cluster::simple::View::release(void) {
//    this->frozenCam = NULL; // DO NOT DELETE
//    if (this->client != NULL) {
//        this->client->Unregister(this);
//        this->client = NULL;
//    }
//    if (this->initMsg != NULL) {
//        SAFE_DELETE(this->initMsg);
//    }
//}
//
//
///*
// * cluster::simple::View::renderFallbackView
// */
//void cluster::simple::View::renderFallbackView(void) {
//    ::glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
//    ::glClear(GL_COLOR_BUFFER_BIT);
//    ::glMatrixMode(GL_PROJECTION);
//    ::glLoadIdentity();
//    ::glMatrixMode(GL_MODELVIEW);
//    ::glLoadIdentity();
//    ::glRotated(this->frozenTime * 36.0, 0.0, 0.0, 1.0);
//    ::glColor3ub(200, 200, 255);
//    ::glBegin(GL_LINES);
//    ::glVertex2i(0, 0);
//    ::glVertex2i(1, 0);
//    ::glEnd();
//}
//
//
///*
// * cluster::simple::View::UpdateFreeze
// */
//void cluster::simple::View::UpdateFreeze(bool freeze) {
//    this->frozen = freeze;
//    this->frozenTime = this->instance()->GetCoreInstanceTime(); // HAZARD
//}
//
//
///*
// * cluster::simple::View::loadConfiguration
// */
//bool cluster::simple::View::loadConfiguration(const vislib::StringA& name) {
//    vislib::StringA vname(name);
//    vname.Append("-tvtile");
//    if (this->instance()->Configuration().IsConfigValueSet(vname)) {
//        return this->setTile(this->instance()->Configuration().ConfigValue(vname));
//    }
//    return false;
//}
//
//
///*
// * cluster::simple::View::directCamSyncUpdated
// */
//bool cluster::simple::View::directCamSyncUpdated(param::ParamSlot& slot) {
//    ASSERT(&slot == &this->directCamSyncSlot);
//    if (this->client != NULL) {
//        this->client->SetDirectCamSync(this->directCamSyncSlot.Param<param::BoolParam>()->Value());
//    }
//    return true;
//}
