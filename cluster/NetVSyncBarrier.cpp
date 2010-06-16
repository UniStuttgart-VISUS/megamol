/*
 * NetVSyncBarrier.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/NetVSyncBarrier.h"
#include "cluster/NetMessages.h"
#include "vislib/IPEndPoint.h"
#include "vislib/Log.h"
#include "vislib/NetworkInformation.h"
#include "vislib/SimpleMessage.h"
#include "vislib/TCPCommChannel.h"
#include "vislib/Trace.h"

using namespace megamol::core;


/*
 * cluster::NetVSyncBarrier::NetVSyncBarrier
 */
cluster::NetVSyncBarrier::NetVSyncBarrier(void) : channel(NULL), data(), dataSize(0) {
    // Intentionally empty
}


/*
 * cluster::NetVSyncBarrier::~NetVSyncBarrier
 */
cluster::NetVSyncBarrier::~NetVSyncBarrier(void) {
    this->Disconnect();
    this->dataSize = 0;
    this->data.EnforceSize(0);
}


/*
 * cluster::NetVSyncBarrier::Connect
 */
bool cluster::NetVSyncBarrier::Connect(const vislib::StringA& address) {
    this->Disconnect();

    vislib::net::IPEndPoint ep;
    float wildness = vislib::net::NetworkInformation::GuessRemoteEndPoint(ep, address);
    if (wildness > 0.8) {
        throw vislib::Exception("Wildness too high when guessing remote end point", __FILE__, __LINE__);
    }
    vislib::sys::Log::DefaultLog.WriteMsg(
        (wildness > 0.4) ? vislib::sys::Log::LEVEL_WARN : vislib::sys::Log::LEVEL_INFO,
        "Guessed remote end-point \"%s\" from input \"%s\" with wildness %f\n",
        ep.ToStringA().PeekBuffer(), address.PeekBuffer(), wildness);

    vislib::net::TcpCommChannel *tcp
        = new vislib::net::TcpCommChannel(vislib::net::TcpCommChannel::FLAG_NODELAY
            | vislib::net::TcpCommChannel::FLAG_REUSE_ADDRESS);
    this->channel = tcp;
    tcp->Connect(ep);

    return true;
}


/*
 * cluster::NetVSyncBarrier::Disconnect
 */
void cluster::NetVSyncBarrier::Disconnect(void) {
    if (!this->channel.IsNull()) {
        try {
            vislib::net::SimpleMessage msg;
            msg.GetHeader().SetMessageID(cluster::netmessages::MSG_NETVSYNC_LEAVE);
            msg.GetHeader().SetBodySize(0);
            msg.AssertBodySize();
            this->channel->Send(msg, msg.GetMessageSize(), 0, true);
        } catch(...) {
        }
        try {
            this->channel->Close();
        } catch(...) {
        }
        this->channel.Release();
    }
}


/*
 * cluster::NetVSyncBarrier::Cross
 */
void cluster::NetVSyncBarrier::Cross(unsigned char id) {
    if (this->channel.IsNull()) return;

    try {
        vislib::net::SimpleMessage msg;
        this->dataSize = 0;

        //VLTRACE(VISLIB_TRCELVL_INFO, "Sending barrier request");
        // request
        msg.GetHeader().SetMessageID(cluster::netmessages::MSG_NETVSYNC_CROSS);
        msg.GetHeader().SetBodySize(1);
        msg.AssertBodySize();
        *msg.GetBodyAs<unsigned char>() = id;
        this->channel->Send(msg, msg.GetMessageSize(), 0, true);

        // receive
        msg.GetHeader().SetBodySize(0);
        msg.AssertBodySize();
        SIZE_T r = this->channel->Receive(msg, sizeof(vislib::net::SimpleMessageHeaderData), 0, true);
        if (r != sizeof(vislib::net::SimpleMessageHeaderData)) {
            throw vislib::Exception("Only partial header received", __FILE__, __LINE__);
        }
        if (msg.GetHeader().GetBodySize() > 0) {
            this->data.AssertSize(msg.GetHeader().GetBodySize());
            r = this->channel->Receive(this->data, msg.GetHeader().GetBodySize(), 0, true);
            if (r != msg.GetHeader().GetBodySize()) {
                throw vislib::Exception("Body not completely received", __FILE__, __LINE__);
            } else {
                this->dataSize = msg.GetHeader().GetBodySize();
            }
        }
        //VLTRACE(VISLIB_TRCELVL_INFO, "Barrier complete");

    } catch(...) {
    }
}
