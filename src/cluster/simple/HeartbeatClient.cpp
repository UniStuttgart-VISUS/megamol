/*
 * HeartbeatClient.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/cluster/simple/HeartbeatClient.h"
#include "vislib/assert.h"
#include "vislib/net/IPCommEndPoint.h"
#include "vislib/sys/Log.h"
#include "vislib/net/Socket.h"

using namespace megamol::core;


/*
 * cluster::simple::HeartbeatClient::HeartbeatClient
 */
cluster::simple::HeartbeatClient::HeartbeatClient(void) : chan(),
        conn(&HeartbeatClient::connector), port(0), server() {
    vislib::net::Socket::Startup();
}


/*
 * cluster::simple::HeartbeatClient::~HeartbeatClient
 */
cluster::simple::HeartbeatClient::~HeartbeatClient(void) {
    this->Shutdown();
    vislib::net::Socket::Cleanup();
}


/*
 * cluster::simple::HeartbeatClient::Connect
 */
void cluster::simple::HeartbeatClient::Connect(vislib::StringW server, unsigned int port) {
    this->Shutdown();
    this->server = server;
    this->port = port;
    this->conn.Start(static_cast<void*>(this));
}


/*
 * cluster::simple::HeartbeatClient::Shutdown
 */
void cluster::simple::HeartbeatClient::Shutdown(void) {
    if (!this->chan.IsNull()) {
        this->chan->Close();
        if (this->conn.IsRunning()) {
            this->conn.Join();
        }
    }
}


/*
 * cluster::simple::HeartbeatClient::Sync
 */
bool cluster::simple::HeartbeatClient::Sync(unsigned char tier, vislib::RawStorage& outPayload) {
    const char outData[] = "MMBx";
    unsigned int inSize;
    const_cast<char *>(outData)[3] = static_cast<char>(tier);

    try {
        if (!this->chan.IsNull()) {

            if (this->chan->Send(outData, 4) != 4) throw vislib::Exception("heart attack", __FILE__, __LINE__);
            if (this->chan->Receive(&inSize, 4) != 4) throw vislib::Exception("heart attack", __FILE__, __LINE__);

            outPayload.EnforceSize(inSize);
            if (inSize > 0) {
                if (this->chan->Receive(outPayload, inSize) != inSize) throw vislib::Exception("heart attack", __FILE__, __LINE__);
            }

            if (inSize <= 12) {
                return false; // wrong tier reject received

            }

            return true;
        }
    } catch(vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteError("HeartbeatClient: %s [%s, %u]\n",
            ex.GetMsgA(), ex.GetFile(), ex.GetLine());

    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteError("HeartbeatClient: Unexpected Exception\n");
    }

    return false;
}


/*
 * cluster::simple::HeartbeatClient::connector
 */
DWORD cluster::simple::HeartbeatClient::connector(void *userData) {
    using vislib::sys::Log;
    HeartbeatClient *that = static_cast<HeartbeatClient*>(userData);
    ASSERT(that != NULL);

    vislib::StringW server = that->server;
    if (server.IsEmpty()) return 0;
    unsigned short port = static_cast<unsigned short>(that->port);
    if (port == 0) return 0;

    Log::DefaultLog.WriteInfo(L"Establishing connection to heartbeat server %s:%u\n",
        server.PeekBuffer(), port);

    try {
        vislib::SmartRef<vislib::net::TcpCommChannel> c = vislib::net::TcpCommChannel::Create(vislib::net::TcpCommChannel::FLAG_NODELAY);
        vislib::SmartRef<vislib::net::AbstractCommEndPoint> endPoint
            = vislib::net::IPCommEndPoint::Create(vislib::net::IPCommEndPoint::IPV4, server, port);

        vislib::sys::Thread::Sleep(100 + ::rand() % 2000);

        c->Connect(endPoint);

        Log::DefaultLog.WriteInfo("Connection to heartbeat server established\n");
        that->chan = c;

    } catch(vislib::Exception ex) {
        Log::DefaultLog.WriteError("Failed to connect to heartbeat server: %s [%s; %d]\n",
            ex.GetMsgA(), ex.GetFile(), ex.GetLine());
        that->chan = NULL;

    } catch(...) {
        Log::DefaultLog.WriteError("Failed to connect to heartbeat server: unexpected exception\n");
        that->chan = NULL;

    }

    return 0;
}
