/*
 * NetVSyncBarrierServer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/NetVSyncBarrierServer.h"
#include "cluster/NetMessages.h"
#include "vislib/assert.h"
#include "vislib/NetworkInformation.h"
#include "vislib/Log.h"

using namespace megamol::core;


/*
 * cluster::NetVSyncBarrierServer::NetVSyncBarrierServer
 */
cluster::NetVSyncBarrierServer::NetVSyncBarrierServer(void)
        : server(), serverEndpoint(), peers(), currentBarrier(0),
        waitingPeerCount(0), lock() {
    this->server.AddListener(this);
}


/*
 * cluster::NetVSyncBarrierServer::~NetVSyncBarrierServer
 */
cluster::NetVSyncBarrierServer::~NetVSyncBarrierServer(void) {
    this->server.RemoveListener(this);
    this->Stop();
    ASSERT(!this->server.IsRunning());
}


/*
 * cluster::NetVSyncBarrierServer::Start
 */
bool cluster::NetVSyncBarrierServer::Start(vislib::StringA lep) {
    this->Stop();
    float wildness = vislib::net::NetworkInformation::GuessLocalEndPoint(this->serverEndpoint, lep);
    if (wildness > 0.8) {
        throw vislib::Exception("Unable to guess local end point from input string", __FILE__, __LINE__);
    }
    vislib::sys::Log::DefaultLog.WriteMsg(
        ((wildness > 0.4) ? vislib::sys::Log::LEVEL_WARN : vislib::sys::Log::LEVEL_INFO),
        "Guessed local end point %s from input %s with wildness %f\n",
        this->serverEndpoint.ToStringA().PeekBuffer(), lep.PeekBuffer(), wildness);

    this->lock.Lock();
    this->peers.Clear();
    this->currentBarrier = 0;
    this->waitingPeerCount = 0;
    this->lock.Unlock();

    this->server.Start(this->serverEndpoint);

    return true;
}


/*
 * cluster::NetVSyncBarrierServer::Stop
 */
void cluster::NetVSyncBarrierServer::Stop(void) throw() {
    if (this->server.IsRunning()) {
        vislib::net::SimpleMessage outMsg;
        outMsg.GetHeader().SetMessageID(cluster::netmessages::MSG_NETVSYNC_CROSS);
        outMsg.GetHeader().SetBodySize(0);
        outMsg.AssertBodySize();
        try {
            this->server.MultiSendMessage(outMsg);
        } catch(...) {
        }
        this->server.Stop();
    }
}


/*
 * cluster::NetVSyncBarrierServer::OnCommChannelServerStarted
 */
void cluster::NetVSyncBarrierServer::OnCommChannelServerStarted(
        cluster::CommChannelServer& server) {
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "Network V-Sync Barrier Server started");
}


/*
 * cluster::NetVSyncBarrierServer::OnCommChannelServerStopped
 */
void cluster::NetVSyncBarrierServer::OnCommChannelServerStopped(
        cluster::CommChannelServer& server) {
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "Network V-Sync Barrier Server stopped");
}


/*
 * cluster::NetVSyncBarrierServer::OnCommChannelConnect
 */
void cluster::NetVSyncBarrierServer::OnCommChannelConnect(
        cluster::CommChannelServer& server, cluster::CommChannel& channel) {
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "Network V-Sync Barrier peer connected");
    void *pid = static_cast<void*>(&channel);
    this->lock.Lock();
    if (!this->peers.Contains(pid)) {
        this->peers.Add(pid);
    }
    this->lock.Unlock();
}


/*
 * cluster::NetVSyncBarrierServer::OnCommChannelDisconnect
 */
void cluster::NetVSyncBarrierServer::OnCommChannelDisconnect(
        cluster::CommChannelServer& server, cluster::CommChannel& channel) {
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "Network V-Sync Barrier peer disconnected");
    this->lock.Lock();
    void *pid = static_cast<void*>(&channel);
    this->peers.RemoveAll(pid);
    this->lock.Unlock();
    this->checkBarrier();
}


/*
 * cluster::NetVSyncBarrierServer::OnCommChannelMessage
 */
void cluster::NetVSyncBarrierServer::OnCommChannelMessage(
        cluster::CommChannelServer& server, cluster::CommChannel& channel,
        const vislib::net::AbstractSimpleMessage& msg) {
    void *pid = static_cast<void*>(&channel);

    switch(msg.GetHeader().GetMessageID()) {
        case cluster::netmessages::MSG_NETVSYNC_JOIN:
            break;
        case cluster::netmessages::MSG_NETVSYNC_LEAVE:
            break;
        case cluster::netmessages::MSG_NETVSYNC_CROSS: {
            unsigned char bid = *msg.GetBodyAs<unsigned char>();
            this->lock.Lock();
            if (this->currentBarrier == 0) {
                this->currentBarrier = bid;
                this->waitingPeerCount = 1;
                this->lock.Unlock();
                this->checkBarrier();
            } else if (this->currentBarrier == bid) {
                this->waitingPeerCount++;
                this->lock.Unlock();
                this->checkBarrier();
            } else {
                this->lock.Unlock();
                // reject nodes waiting for the wrong barrier
                vislib::net::SimpleMessage outMsg;
                outMsg.GetHeader().SetMessageID(cluster::netmessages::MSG_NETVSYNC_CROSS);
                outMsg.GetHeader().SetBodySize(0);
                outMsg.AssertBodySize();
                channel.SendMessage(outMsg);
            }
        } break;
        default:
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                "Network V-Sync Barrier unexpected message %d received",
                static_cast<int>(msg.GetHeader().GetMessageID()));
    }

}


/*
 * cluster::NetVSyncBarrierServer::checkBarrier
 */
void cluster::NetVSyncBarrierServer::checkBarrier(void) {
    this->lock.Lock();
    if ((this->currentBarrier != 0) && (this->peers.Count() <= this->waitingPeerCount)) {
        this->currentBarrier = 0;
        this->waitingPeerCount = 0;

        vislib::net::SimpleMessage outMsg;
        outMsg.GetHeader().SetMessageID(cluster::netmessages::MSG_NETVSYNC_CROSS);
        outMsg.GetHeader().SetBodySize(0);
        outMsg.AssertBodySize();

        this->server.MultiSendMessage(outMsg);
    }
    this->lock.Unlock();
}
