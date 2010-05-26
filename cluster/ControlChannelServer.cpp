/*
 * ControlChannelServer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/ControlChannelServer.h"
#include "cluster/NetMessages.h"
#include "vislib/assert.h"
#include "vislib/AutoLock.h"
#include "vislib/Log.h"
#include "vislib/String.h"
#include "vislib/SystemInformation.h"

using namespace megamol::core;


/*
 * cluster::ControlChannelServer::ControlChannelServer
 */
cluster::ControlChannelServer::ControlChannelServer(void)
        : vislib::Listenable<ControlChannelServer>(), clientsLock(),
        clients(), commChannel(), server() {
    this->server.AddListener(this);
}


/*
 * cluster::ControlChannelServer::~ControlChannelServer
 */
cluster::ControlChannelServer::~ControlChannelServer(void) {
    this->server.RemoveListener(this);
    this->Stop();
}


/*
 * cluster::ControlChannelServer::IsRunning
 */
bool cluster::ControlChannelServer::IsRunning(void) const {
    return this->server.IsRunning();
}


/*
 * cluster::ControlChannelServer::Start
 */
void cluster::ControlChannelServer::Start(vislib::net::IPEndPoint& ep) {
    this->Stop();
    if (this->commChannel.IsNull()) {
        this->commChannel = new vislib::net::TcpCommChannel(
            vislib::net::TcpCommChannel::FLAG_NODELAY);
    }
    vislib::SmartRef<vislib::net::AbstractServerEndPoint> endpoint = 
        this->commChannel.DynamicCast<vislib::net::AbstractServerEndPoint>();
    vislib::StringA address = ep.ToStringA();
    this->server.Configure(endpoint, address);
    this->server.Start(NULL);
}


/*
 * cluster::ControlChannelServer::Stop
 */
void cluster::ControlChannelServer::Stop(void) {
    if (this->server.IsRunning()) {
        this->server.Terminate(false);
    }
    vislib::sys::AutoLock(this->clientsLock);
    vislib::SingleLinkedList<cluster::CommChannel>::Iterator iter = this->clients.GetIterator();
    while (iter.HasNext()) {
        iter.Next().RemoveListener(this);
    }
    this->clients.Clear();
}


/*
 * cluster::ControlChannelServer::MultiSendMessage
 */
void cluster::ControlChannelServer::MultiSendMessage(const vislib::net::AbstractSimpleMessage& msg) {
    vislib::sys::AutoLock(this->clientsLock);
    vislib::SingleLinkedList<cluster::CommChannel>::Iterator iter = this->clients.GetIterator();
    while (iter.HasNext()) {
        cluster::CommChannel& channel = iter.Next();
        try {
            channel.SendMessage(msg);
        } catch(...) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to send message to %s", channel.CounterpartName().PeekBuffer());
        }
    }
}


/*
 * cluster::ControlChannelServer::OnCommChannelDisconnect
 */
void cluster::ControlChannelServer::OnCommChannelDisconnect(cluster::CommChannel& sender) {
    vislib::sys::AutoLock(this->clientsLock);
    if (this->clients.Contains(sender)) {
        sender.RemoveListener(this);
        vislib::Listenable<ControlChannelServer>::ListenerIterator iter = this->GetListeners();
        while (iter.HasNext()) {
            Listener *l = dynamic_cast<Listener*>(iter.Next());
            if (l == NULL) continue;
            l->OnControlChannelDisconnect(*this, sender);
        }
        this->clients.RemoveAll(sender);
    }
}


/*
 * cluster::ControlChannelServer::OnCommChannelMessage
 */
void cluster::ControlChannelServer::OnCommChannelMessage(cluster::CommChannel& sender, const vislib::net::AbstractSimpleMessage& msg) {
    vislib::Listenable<ControlChannelServer>::ListenerIterator iter = this->GetListeners();
    while (iter.HasNext()) {
        Listener *l = dynamic_cast<Listener*>(iter.Next());
        if (l == NULL) continue;
        l->OnControlChannelMessage(*this, sender, msg);
    }
}


/*
 * cluster::ControlChannelServer::OnServerError
 */
bool cluster::ControlChannelServer::OnServerError(const vislib::net::CommServer& src, const vislib::Exception& exception) throw() {
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN, "Control Channel Server: %s\n", exception.GetMsgA());
    return true; // keep server running
}


/*
 * cluster::ControlChannelServer::OnNewConnection
 */
bool cluster::ControlChannelServer::OnNewConnection(const vislib::net::CommServer& src, vislib::SmartRef<vislib::net::AbstractCommChannel> channel) throw() {
    vislib::SmartRef<vislib::net::AbstractBidiCommChannel> bidiChannel = channel.DynamicCast<vislib::net::AbstractBidiCommChannel>();
    ASSERT(!bidiChannel.IsNull()); // internal error like problem (should never happen)
    try {
        vislib::sys::AutoLock(this->clientsLock);
        this->clients.Append(cluster::CommChannel());
        this->clients.Last().Open(bidiChannel);
        this->clients.Last().AddListener(this);

        vislib::Listenable<ControlChannelServer>::ListenerIterator iter = this->GetListeners();
        while (iter.HasNext()) {
            Listener *l = dynamic_cast<Listener*>(iter.Next());
            if (l == NULL) continue;
            l->OnControlChannelConnect(*this, this->clients.Last());
        }

        vislib::net::SimpleMessage simsg;
        simsg.GetHeader().SetMessageID(cluster::netmessages::MSG_WHATSYOURNAME);
        simsg.GetHeader().SetBodySize(0);
        simsg.AssertBodySize();
        this->clients.Last().SendMessage(simsg);
        vislib::StringA myname;
        vislib::sys::SystemInformation::ComputerName(myname);
        simsg.GetHeader().SetMessageID(cluster::netmessages::MSG_MYNAMEIS);
        simsg.GetHeader().SetBodySize(myname.Length() + 1);
        simsg.AssertBodySize();
        ::memcpy(simsg.GetBody(), myname.PeekBuffer(), myname.Length() + 1);
        this->clients.Last().SendMessage(simsg);

        return true; // connection accepted
    } catch(vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Exception on accepting connection: %s\n", ex.GetMsgA());
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Exception on accepting connection: unexpected exception\n");
    }
    return false;
}


/*
 * cluster::ControlChannelServer::OnServerExited
 */
void cluster::ControlChannelServer::OnServerExited(const vislib::net::CommServer& src) throw() {
    vislib::Listenable<ControlChannelServer>::ListenerIterator iter = this->GetListeners();
    while (iter.HasNext()) {
        Listener *l = dynamic_cast<Listener*>(iter.Next());
        if (l == NULL) continue;
        l->OnControlChannelServerStopped(*this);
    }
    if (!this->commChannel.IsNull()) {
        try {
            this->commChannel->Close();
        } catch(...) {
        }
        this->commChannel.Release();
    }
}


/*
 * cluster::ControlChannelServer::OnServerStarted
 */
void cluster::ControlChannelServer::OnServerStarted(const vislib::net::CommServer& src) throw() {
    vislib::Listenable<ControlChannelServer>::ListenerIterator iter = this->GetListeners();
    while (iter.HasNext()) {
        Listener *l = dynamic_cast<Listener*>(iter.Next());
        if (l == NULL) continue;
        l->OnControlChannelServerStopped(*this);
    }
}
