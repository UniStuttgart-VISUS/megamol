/*
 * HeartbeatClient.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/simple/HeartbeatClient.h"
#include "vislib/assert.h"
#include "vislib/IPCommEndPoint.h"
#include "vislib/Log.h"

using namespace megamol::core;


/*
 * cluster::simple::HeartbeatClient::HeartbeatClient
 */
cluster::simple::HeartbeatClient::HeartbeatClient(void) : chan(),
        conn(&HeartbeatClient::connector), port(0), server() {
    // Intentionally empty
}


/*
 * cluster::simple::HeartbeatClient::~HeartbeatClient
 */
cluster::simple::HeartbeatClient::~HeartbeatClient(void) {
    this->Shutdown();
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
    // TODO: Implement
}


/*
 * cluster::simple::HeartbeatClient::Sync
 */
bool cluster::simple::HeartbeatClient::Sync(vislib::RawStorage& outPayload) {
    // TODO: Implement
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
        that->chan = vislib::net::TcpCommChannel::Create(vislib::net::TcpCommChannel::FLAG_NODELAY);
        vislib::SmartRef<vislib::net::AbstractCommEndPoint> endPoint
            = vislib::net::IPCommEndPoint::Create(vislib::net::IPCommEndPoint::IPV4, server, port);
        that->chan->Connect(endPoint);

        Log::DefaultLog.WriteInfo("Connection to heartbeat server established\n");

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
