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
#include "param/IntParam.h"
#include "CoreInstance.h"
#include "vislib/assert.h"
#include "vislib/AutoLock.h"
#include "vislib/Log.h"
#include <climits>


using namespace megamol::core;


/*
 * cluster::simple::Heartbeat::Heartbeat
 */
cluster::simple::Heartbeat::Heartbeat(void)
        : job::AbstractThreadedJob(), Module(),
        registerSlot("register", "The slot registering this view"), client(NULL), run(false), mainlock(),
        heartBeatPortSlot("heartbeat::port", "The port the heartbeat server communicates on"),
        tcBuf(), tcBufIdx(0), server() {
    vislib::net::Socket::Startup();

    this->registerSlot.SetCompatibleCall<ClientViewRegistrationDescription>();
    this->MakeSlotAvailable(&this->registerSlot);

    this->heartBeatPortSlot << new param::IntParam(0, 0, USHRT_MAX);
    this->MakeSlotAvailable(&this->heartBeatPortSlot);

}


/*
 * cluster::simple::Heartbeat::~Heartbeat
 */
cluster::simple::Heartbeat::~Heartbeat(void) {
    this->Release();
    vislib::net::Socket::Cleanup();
}


/*
 * cluster::simple::Heartbeat::Terminate
 */
bool cluster::simple::Heartbeat::Terminate(void) {
    this->run = false;
    if (this->server.IsRunning()) {
        this->server.Terminate();
        this->server.Join();
    }
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
 * cluster::simple::Heartbeat::OnServerError
 */
bool cluster::simple::Heartbeat::OnServerError(const vislib::net::CommServer& src, const vislib::Exception& exception) throw() {
    vislib::sys::Log::DefaultLog.WriteError("Heartbeat server error: %s [%s, %d]",
        exception.GetMsgA(), exception.GetFile(), exception.GetLine());
    return false;
}


/*
 * cluster::simple::Heartbeat::OnNewConnection
 */
bool cluster::simple::Heartbeat::OnNewConnection(const vislib::net::CommServer& src, vislib::SmartRef<vislib::net::AbstractCommChannel> channel) throw() {
    vislib::sys::Log::DefaultLog.WriteInfo("New heartbeat connection");

    // TODO: Implement

    return false;
}


/*
 * cluster::simple::Heartbeat::OnServerExited
 */
void cluster::simple::Heartbeat::OnServerExited(const vislib::net::CommServer& src) throw() {
    vislib::sys::Log::DefaultLog.WriteInfo("Heartbeat server exited");
}


/*
 * cluster::simple::Heartbeat::OnServerStarted
 */
void cluster::simple::Heartbeat::OnServerStarted(const vislib::net::CommServer& src) throw() {
    vislib::sys::Log::DefaultLog.WriteInfo("Heartbeat server started");
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
    if (this->server.IsRunning()) {
        this->server.Terminate();
        this->server.Join();
    }

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

            if (this->server.IsRunning()) {
                this->server.Terminate();
                this->server.Join();
            }

            //vislib::net::CommServer::Configuration cfg(
            //this->server

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
