/*
 * ControlChannel.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/ControlChannel.h"
#include "vislib/IllegalStateException.h"
#include "vislib/Log.h"
#include "vislib/UnsupportedOperationException.h"

using namespace megamol::core;


/*
 * cluster::ControlChannel::ControlChannel
 */
cluster::ControlChannel::ControlChannel(void)
        : vislib::Listenable<ControlChannel>(),
        vislib::net::SimpleMessageDispatchListener(),
        channel() {
    // Intentionally empty
    this->receiver.AddListener(this);
}


/*
 * cluster::ControlChannel::~ControlChannel
 */
cluster::ControlChannel::~ControlChannel(void) {
    this->Close();
}


/*
 * cluster::ControlChannel::Close
 */
void cluster::ControlChannel::Close(void) {
    if (this->receiver.IsRunning()) {
        this->receiver.Terminate(false);
    }
    if (!this->channel.IsNull()) {
        try {
            this->channel->Close();
        } catch(...) {
        }
        this->channel = NULL;
    }
}


/*
 * cluster::ControlChannel::IsOpen
 */
bool cluster::ControlChannel::IsOpen(void) const {
    return !this->channel.IsNull() && this->receiver.IsRunning();
}


/*
 * cluster::ControlChannel::Open
 */
void cluster::ControlChannel::Open(vislib::SmartRef<vislib::net::AbstractBidiCommChannel> channel) {
    this->Close();
    this->channel = channel;
    vislib::net::AbstractCommChannel *cc = dynamic_cast<vislib::net::AbstractCommChannel *>(this->channel.operator ->());
    this->receiver.Start(cc);
    if ((!this->receiver.IsRunning()) && (!this->channel.IsNull())) {
        try {
            this->channel->Close();
        } catch(...) {
        }
        this->channel = NULL;
    }
}


/*
 * cluster::ControlChannel::SendMessage
 */
void cluster::ControlChannel::SendMessage(const vislib::net::AbstractSimpleMessage& msg) {
    if (this->channel.IsNull()) {
        throw vislib::IllegalStateException("Channel not open", __FILE__, __LINE__);
    }
    if (this->channel->Send(msg, msg.GetMessageSize(), 0, true) != msg.GetMessageSize()) {
        throw vislib::Exception("Unable to send the whole message", __FILE__, __LINE__);
    }
}


/*
 * cluster::ControlChannel::operator==
 */
bool cluster::ControlChannel::operator==(const cluster::ControlChannel& rhs) const {
    return (this->channel == rhs.channel);
}


/*
 * cluster::ControlChannel::operator=
 */
cluster::ControlChannel& cluster::ControlChannel::operator=(const cluster::ControlChannel& rhs) {
    if (this->IsOpen() || rhs.IsOpen()) {
        throw vislib::IllegalStateException("operator= is illegal on open channels", __FILE__, __LINE__);
    }
    this->channel = rhs.channel;
    return *this;
}


/*
 * cluster::ControlChannel::OnCommunicationError
 */
bool cluster::ControlChannel::OnCommunicationError(vislib::net::SimpleMessageDispatcher& src, const vislib::Exception& exception) throw() {
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN, "Control Channel: %s\n", exception.GetMsgA());
    return true; // keep receiver running
}


/*
 * cluster::ControlChannel::OnDispatcherExited
 */
void cluster::ControlChannel::OnDispatcherExited(vislib::net::SimpleMessageDispatcher& src) throw() {
    vislib::Listenable<ControlChannel>::ListenerIterator iter = this->GetListeners();
    while (iter.HasNext()) {
        Listener *l = dynamic_cast<Listener *>(iter.Next());
        if (l == NULL) continue;
        l->OnControlChannelDisconnect(*this);
    }
    if (!this->channel.IsNull()) {
        try {
            this->channel->Close();
        } catch(...) {
        }
        this->channel = NULL;
    }
}


/*
 * cluster::ControlChannel::OnDispatcherStarted
 */
void cluster::ControlChannel::OnDispatcherStarted(vislib::net::SimpleMessageDispatcher& src) throw() {
    vislib::Listenable<ControlChannel>::ListenerIterator iter = this->GetListeners();
    while (iter.HasNext()) {
        Listener *l = dynamic_cast<Listener *>(iter.Next());
        if (l == NULL) continue;
        l->OnControlChannelConnect(*this);
    }
}


/*
 * cluster::ControlChannel::OnMessageReceived
 */
bool cluster::ControlChannel::OnMessageReceived(vislib::net::SimpleMessageDispatcher& src, const vislib::net::AbstractSimpleMessage& msg) throw() {
    vislib::Listenable<ControlChannel>::ListenerIterator iter = this->GetListeners();
    while (iter.HasNext()) {
        Listener *l = dynamic_cast<Listener *>(iter.Next());
        if (l == NULL) continue;
        l->OnControlChannelMessage(*this, msg);
    }
    return true; // keep receiver running
}


/*
 * cluster::ControlChannel::ControlChannel
 */
cluster::ControlChannel::ControlChannel(const cluster::ControlChannel& src) {
    throw vislib::UnsupportedOperationException("copy ctor", __FILE__, __LINE__);
}
