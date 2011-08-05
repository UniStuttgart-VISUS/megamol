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
#include "vislib/IPCommEndPoint.h"
#include "vislib/TcpCommChannel.h"
#include "vislib/Trace.h"
#include <climits>


using namespace megamol::core;


/****************************************************************************/

/*
 * cluster::simple::Heartbeat::Connection::Connection
 */
cluster::simple::Heartbeat::Connection::Connection(cluster::simple::Heartbeat& parent,
        vislib::SmartRef<vislib::net::AbstractCommChannel> chan) : parent(parent), chan(chan),
        waiting(false), kun(&Connection::receive), wait(), data() {
    this->kun.Start(static_cast<void*>(this));
    vislib::sys::Thread::Sleep(10);
}


/*
 * cluster::simple::Heartbeat::Connection::Close
 */
void cluster::simple::Heartbeat::Connection::Close(void) {
    if (!this->chan.IsNull()) {
        vislib::SmartRef<vislib::net::AbstractCommChannel> c = this->chan;
        this->chan = NULL;
        c->Close();
    }
    if (this->kun.IsRunning()) {
        this->kun.Join();
    }
}


/*
 * cluster::simple::Heartbeat::Connection::receive
 */
DWORD cluster::simple::Heartbeat::Connection::receive(void *userData) {
    Connection *that = static_cast<Connection *>(userData);
    vislib::SmartRef<vislib::net::AbstractCommChannel> chan = that->chan;
    Heartbeat& parent = that->parent;
    bool& waiting = that->waiting;
    vislib::sys::Event& wait = that->wait;
    vislib::RawStorage& data = that->data;
    unsigned char& serverTier = parent.tier;

    unsigned char clientTier;
    unsigned int len;

    const SIZE_T inDataSize = 4;
    char inData[inDataSize];

    waiting = false;

    try {
        SIZE_T size = inDataSize;
        while (size == inDataSize) {

            size = chan->Receive(inData, inDataSize,
                vislib::net::TcpCommChannel::TIMEOUT_INFINITE, true);

            if (::memcmp(inData, "MMB", 3) != 0) {
                throw vislib::Exception("cardiac seizure", __FILE__, __LINE__);
            }
            clientTier = static_cast<unsigned char>(inData[3]);

            if (clientTier != serverTier) {
                // wrong tier fall through!

                printf("Client %u rejected because I am waiting on %u\n", clientTier, serverTier);

                len = 1;
                inData[0] = static_cast<char>(0x0f);
                chan->Send(&len, 4,
                    vislib::net::TcpCommChannel::TIMEOUT_INFINITE, true);
                chan->Send(inData, 1,
                    vislib::net::TcpCommChannel::TIMEOUT_INFINITE, true);
                continue;
            }

            waiting = true;
            parent.connWaiting(that);

            wait.Wait();
            waiting = false;

            len = static_cast<unsigned int>(data.GetSize());
            chan->Send(&len, 4,
                vislib::net::TcpCommChannel::TIMEOUT_INFINITE, true);
            chan->Send(data, len,
                vislib::net::TcpCommChannel::TIMEOUT_INFINITE, true);

        }

    } catch(vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteError("Heartbeat Connection: %s [%s, %d]",
            ex.GetMsgA(), ex.GetFile(), ex.GetLine());
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteError("Heartbeat Connection: Unexpected Exception");
    }

    parent.removeConn(that); // HAZARD: might already delete this object ... thread object may behave strange

    return 0;
}


/****************************************************************************/

/*
 * cluster::simple::Heartbeat::Heartbeat
 */
cluster::simple::Heartbeat::Heartbeat(void)
        : job::AbstractThreadedJob(), Module(),
        registerSlot("register", "The slot registering this view"), client(NULL), run(false), mainlock(),
        heartBeatPortSlot("heartbeat::port", "The port the heartbeat server communicates on"),
        tcBuf(), tcBufIdx(0), server(), connLock(), connList(), tier(1) {
    vislib::net::Socket::Startup();

    this->registerSlot.SetCompatibleCall<ClientViewRegistrationDescription>();
    this->MakeSlotAvailable(&this->registerSlot);

    this->heartBeatPortSlot << new param::IntParam(0, 0, USHRT_MAX);
    this->MakeSlotAvailable(&this->heartBeatPortSlot);

    this->server.AddListener(this);

    this->tcBuf[0].isValid = false;
    this->tcBuf[1].isValid = false;

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
        buf.isValid = true;
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

    vislib::SmartPtr<Connection> con = new Connection(*this, channel);
    this->addConn(con);

    return true;
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

    vislib::Array<vislib::SmartPtr<Connection> > c;
    {
        vislib::sys::AutoLock lock(this->connLock);
        vislib::SingleLinkedList<vislib::SmartPtr<Connection> >::Iterator iter = this->connList.GetIterator();
        while (iter.HasNext()) {
            c.Add(iter.Next());
        }
    }
    for (SIZE_T i = 0; i < c.Count(); i++) {
        c[i]->Close();
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

            vislib::net::IPEndPoint ep;
            ep.SetIPAddress(vislib::net::IPAddress::ANY);
            ep.SetPort(this->heartBeatPortSlot.Param<param::IntParam>()->Value());

            vislib::net::CommServer::Configuration cfg(
                vislib::net::TcpCommChannel::Create(
                    vislib::net::TcpCommChannel::FLAG_NODELAY
                    | vislib::net::TcpCommChannel::FLAG_REUSE_ADDRESS),
                vislib::net::IPCommEndPoint::Create(ep));
            this->server.Start(&cfg);
            vislib::sys::Thread::Sleep(100);

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


/*
 * cluster::simple::Heartbeat::addConn
 */
void cluster::simple::Heartbeat::addConn(vislib::SmartPtr<Connection> con) {
    vislib::sys::AutoLock lock(this->connLock);

    if (this->connList.Contains(con)) return;

    this->connList.Add(con);
    this->connWaiting(NULL);

}


/*
 * cluster::simple::Heartbeat::removeConn
 */
void cluster::simple::Heartbeat::removeConn(Connection *con) {
    vislib::sys::AutoLock lock(this->connLock);

    vislib::SingleLinkedList<vislib::SmartPtr<Connection> >::Iterator iter = this->connList.GetIterator();
    while (iter.HasNext()) {
        vislib::SmartPtr<Connection> c = iter.Next();
        if (c.operator->() == con) {
            this->connList.Remove(c);
            break;
        }
    }
    this->connWaiting(NULL);

}


/*
 * cluster::simple::Heartbeat::connWaiting
 */
void cluster::simple::Heartbeat::connWaiting(Connection *con) {
    vislib::sys::AutoLock lock(this->connLock);
    SIZE_T w = 0, a = 0;

    vislib::SingleLinkedList<vislib::SmartPtr<Connection> >::Iterator iter = this->connList.GetIterator();
    while (iter.HasNext()) {
        a++;
        if (iter.Next()->IsWaiting()) w++;
    }

    VLTRACE(VISLIB_TRCELVL_INFO, "Heartbeat connections waiting: %u/%u\n", w, a);

    if ((w >= a) && (a > 0)) {

        // two-tier sync
        this->tier = 3 - this->tier;

        iter = this->connList.GetIterator();

        {
            TCBuffer& buf = this->tcBuf[this->tcBufIdx];
            vislib::sys::AutoLock(buf.lock);

            while (iter.HasNext()) {
                vislib::RawStorage& data = iter.Next()->Data();
                if (buf.isValid) {
                    data.AssertSize(1 + sizeof(double) + sizeof(float) + buf.camera.GetSize());
                    *data.As<unsigned char>() = 1; // Do a two-tier sync
                    *data.AsAt<double>(1) = buf.instTime;
                    *data.AsAt<float>(1 + sizeof(double)) = buf.time;
                    ::memcpy(data.At(1 + sizeof(double) + sizeof(float)), buf.camera, buf.camera.GetSize());
                } else {
                    data.AssertSize(1 + sizeof(double) + sizeof(float));
                    *data.As<unsigned char>() = 1; // Do a two-tier sync
                    *data.AsAt<double>(1) = this->GetCoreInstance()->GetCoreInstanceTime();
                    *data.AsAt<float>(1 + sizeof(double)) = 0.0f;
                }
            }

        }

        iter = this->connList.GetIterator();
        while (iter.HasNext()) {
            iter.Next()->Continue();
        }

    }

}
