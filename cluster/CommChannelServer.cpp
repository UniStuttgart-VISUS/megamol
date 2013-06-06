/*
 * CommChannelServer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/CommChannelServer.h"
#include "cluster/NetMessages.h"
#include "vislib/assert.h"
#include "vislib/AutoLock.h"
#include "vislib/IPCommEndPoint.h"
#include "vislib/Log.h"
#include "vislib/SocketException.h"
#include "vislib/String.h"
#include "vislib/SystemInformation.h"

using namespace megamol::core;


/*
 * cluster::CommChannelServer::CommChannelServer
 */
cluster::CommChannelServer::CommChannelServer(void)
        : vislib::Listenable<CommChannelServer>(), clientsLock(),
        clients(), commChannel(), server() {
    this->server.AddListener(this);
}


/*
 * cluster::CommChannelServer::~CommChannelServer
 */
cluster::CommChannelServer::~CommChannelServer(void) {
    this->server.RemoveListener(this);
    this->Stop();
}


/*
 * cluster::CommChannelServer::IsRunning
 */
bool cluster::CommChannelServer::IsRunning(void) const {
    return this->server.IsRunning();
}


/*
 * cluster::CommChannelServer::Start
 */
void cluster::CommChannelServer::Start(vislib::net::IPEndPoint& ep) {
    this->Stop();
    if (this->commChannel.IsNull()) {
        this->commChannel = vislib::net::TcpCommChannel::Create(
            vislib::net::TcpCommChannel::FLAG_NODELAY
            | vislib::net::TcpCommChannel::FLAG_REUSE_ADDRESS);
    }
    vislib::net::CommServer::Configuration cfg(this->commChannel, vislib::net::IPCommEndPoint::Create(ep));
    this->server.Start(&cfg);
}


/*
 * cluster::CommChannelServer::Stop
 */
void cluster::CommChannelServer::Stop(void) {
    if (this->server.IsRunning()) {
        this->server.Terminate(false);
    }
    vislib::sys::AutoLock(this->clientsLock);
    vislib::SingleLinkedList<cluster::CommChannel>::Iterator iter = this->clients.GetIterator();
    while (iter.HasNext()) {
        cluster::CommChannel &c = iter.Next();
        c.RemoveListener(this);
        try {
            c.Close();
        } catch(...) {
        }
    }
    this->clients.Clear();
}


/*
 * cluster::CommChannelServer::MultiSendMessage
 */
void cluster::CommChannelServer::MultiSendMessage(const vislib::net::AbstractSimpleMessage& msg) {
    vislib::sys::AutoLock(this->clientsLock);
    vislib::SingleLinkedList<cluster::CommChannel>::Iterator iter = this->clients.GetIterator();
    while (iter.HasNext()) {
        cluster::CommChannel& channel = iter.Next();
        try {
            channel.SendMessage(msg);
        } catch(vislib::net::SocketException skex) {
            if ((static_cast<int>(skex.GetErrorCode()) != 10054) 
                    && (static_cast<int>(skex.GetErrorCode()) != 10053)) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    "Unable to send message to %s: [%d] %s", channel.CounterpartName().PeekBuffer(),
                    static_cast<int>(skex.GetErrorCode()), skex.GetMsgA());
            }
        } catch(vislib::Exception ex) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to send message to %s: %s", channel.CounterpartName().PeekBuffer(), ex.GetMsgA());
        } catch(...) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to send message to %s", channel.CounterpartName().PeekBuffer());
        }
    }
}

/*
 * cluster::CommChannelServer::MultiSendMessage
 */
void cluster::CommChannelServer::SingleSendMessage(const vislib::net::AbstractSimpleMessage& msg, unsigned int node) {
    vislib::sys::AutoLock(this->clientsLock);
    vislib::SingleLinkedList<cluster::CommChannel>::Iterator iter = this->clients.GetIterator();
	
	unsigned int current = 0;

    while (iter.HasNext()) {
        cluster::CommChannel& channel = iter.Next();
		if ( current == node ) {
			try {
				channel.SendMessage(msg);
			} catch(vislib::net::SocketException skex) {
				if ((static_cast<int>(skex.GetErrorCode()) != 10054) 
						&& (static_cast<int>(skex.GetErrorCode()) != 10053)) {
					vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
						"Unable to send message to %s: [%d] %s", channel.CounterpartName().PeekBuffer(),
						static_cast<int>(skex.GetErrorCode()), skex.GetMsgA());
				}
			} catch(vislib::Exception ex) {
				vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
					"Unable to send message to %s: %s", channel.CounterpartName().PeekBuffer(), ex.GetMsgA());
			} catch(...) {
				vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
					"Unable to send message to %s", channel.CounterpartName().PeekBuffer());
			}
			break;
		} 
		++ current;
    }
}

/**
 * Answers the number of clients currently connected.
 */
unsigned int cluster::CommChannelServer::ClientsCount()
{
	return this->clients.Count();
}

/*
 * cluster::CommChannelServer::OnCommChannelDisconnect
 */
void cluster::CommChannelServer::OnCommChannelDisconnect(cluster::CommChannel& sender) {
    vislib::sys::AutoLock(this->clientsLock);
    if (this->clients.Contains(sender)) {
        sender.RemoveListener(this);
        vislib::Listenable<CommChannelServer>::ListenerIterator iter = this->GetListeners();
        while (iter.HasNext()) {
            Listener *l = dynamic_cast<Listener*>(iter.Next());
            if (l == NULL) continue;
            l->OnCommChannelDisconnect(*this, sender);
        }
        this->clients.RemoveAll(sender);
    }
}


/*
 * cluster::CommChannelServer::OnCommChannelMessage
 */
void cluster::CommChannelServer::OnCommChannelMessage(cluster::CommChannel& sender, const vislib::net::AbstractSimpleMessage& msg) {
    vislib::Listenable<CommChannelServer>::ListenerIterator iter = this->GetListeners();
    while (iter.HasNext()) {
        Listener *l = dynamic_cast<Listener*>(iter.Next());
        if (l == NULL) continue;
        l->OnCommChannelMessage(*this, sender, msg);
    }
}


/*
 * cluster::CommChannelServer::OnServerError
 */
bool cluster::CommChannelServer::OnServerError(const vislib::net::CommServer& src, const vislib::Exception& exception) throw() {
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN, "Control Channel Server: %s\n", exception.GetMsgA());
    return true; // keep server running
}


/*
 * cluster::CommChannelServer::OnNewConnection
 */
bool cluster::CommChannelServer::OnNewConnection(const vislib::net::CommServer& src, vislib::SmartRef<vislib::net::AbstractCommClientChannel> channel) throw() {
    vislib::SmartRef<vislib::net::AbstractCommClientChannel> bidiChannel = channel;//.DynamicCast<vislib::net::AbstractCommChannel>();
    ASSERT(!bidiChannel.IsNull()); // internal error like problem (should never happen)
    try {
        vislib::sys::AutoLock(this->clientsLock);
        this->clients.Append(cluster::CommChannel());
        this->clients.Last().Open(bidiChannel);
        this->clients.Last().AddListener(this);

        vislib::Listenable<CommChannelServer>::ListenerIterator iter = this->GetListeners();
        while (iter.HasNext()) {
            Listener *l = dynamic_cast<Listener*>(iter.Next());
            if (l == NULL) continue;
            l->OnCommChannelConnect(*this, this->clients.Last());
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
 * cluster::CommChannelServer::OnServerExited
 */
void cluster::CommChannelServer::OnServerExited(const vislib::net::CommServer& src) throw() {
    vislib::Listenable<CommChannelServer>::ListenerIterator iter = this->GetListeners();
    while (iter.HasNext()) {
        Listener *l = dynamic_cast<Listener*>(iter.Next());
        if (l == NULL) continue;
        l->OnCommChannelServerStopped(*this);
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
 * cluster::CommChannelServer::OnServerStarted
 */
void cluster::CommChannelServer::OnServerStarted(const vislib::net::CommServer& src) throw() {
    vislib::Listenable<CommChannelServer>::ListenerIterator iter = this->GetListeners();
    while (iter.HasNext()) {
        Listener *l = dynamic_cast<Listener*>(iter.Next());
        if (l == NULL) continue;
        l->OnCommChannelServerStopped(*this);
    }
}
