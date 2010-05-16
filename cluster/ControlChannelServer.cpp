/*
 * ControlChannelServer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/ControlChannelServer.h"
#include "vislib/assert.h"
#include "vislib/Log.h"

using namespace megamol::core;


/*
 * cluster::ControlChannelServer::ControlChannelServer
 */
cluster::ControlChannelServer::ControlChannelServer(void)
        : vislib::Listenable<ControlChannelServer>(), clients(), server(),
        commChannel() {
    this->server.AddListener(this);
    // TODO: Implement
}


/*
 * cluster::ControlChannelServer::~ControlChannelServer
 */
cluster::ControlChannelServer::~ControlChannelServer(void) {
    this->clients.Clear();
    // TODO: Implement
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
    this->server.Configure(this->commChannel.DynamicCast<vislib::net::AbstractServerEndPoint>(), ep.ToStringA());
    this->server.Start(NULL);
}


/*
 * cluster::ControlChannelServer::Stop
 */
void cluster::ControlChannelServer::Stop(void) {
    if (this->server.IsRunning()) {
        this->server.Terminate(false);
    }
    this->clients.Clear();
}


/*
 * cluster::ControlChannelServer::OnControlChannelDisconnect
 */
void cluster::ControlChannelServer::OnControlChannelDisconnect(cluster::ControlChannel& sender) {
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
 * cluster::ControlChannelServer::OnControlChannelMessage
 */
void cluster::ControlChannelServer::OnControlChannelMessage(cluster::ControlChannel& sender, const vislib::net::AbstractSimpleMessage& msg) {
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
        this->clients.Append(cluster::ControlChannel());
        this->clients.Last().Open(bidiChannel);
        this->clients.Last().AddListener(this);

        vislib::Listenable<ControlChannelServer>::ListenerIterator iter = this->GetListeners();
        while (iter.HasNext()) {
            Listener *l = dynamic_cast<Listener*>(iter.Next());
            if (l == NULL) continue;
            l->OnControlChannelConnect(*this, this->clients.Last());
        }

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
