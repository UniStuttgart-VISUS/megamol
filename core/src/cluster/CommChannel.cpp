/*
 * CommChannel.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/cluster/CommChannel.h"
#include "mmcore/utility/log/Log.h"
#include "stdafx.h"
#include "vislib/IllegalStateException.h"
#include "vislib/UnsupportedOperationException.h"

using namespace megamol::core;


/*
 * cluster::CommChannel::CommChannel
 */
cluster::CommChannel::CommChannel(void)
        : vislib::Listenable<CommChannel>()
        , vislib::net::SimpleMessageDispatchListener()
        , channel()
        , counterpartName("Unknown") {
    // Intentionally empty
    this->receiver.AddListener(this);
}


/*
 * cluster::CommChannel::CommChannel
 */
cluster::CommChannel::CommChannel(const cluster::CommChannel& src)
        : vislib::Listenable<CommChannel>()
        , vislib::net::SimpleMessageDispatchListener()
        , channel()
        , counterpartName("Unknown") {
    if (!(src == *this)) {
        throw vislib::UnsupportedOperationException("copy ctor", __FILE__, __LINE__);
    }
}


/*
 * cluster::CommChannel::~CommChannel
 */
cluster::CommChannel::~CommChannel(void) {
    this->Close();
}


/*
 * cluster::CommChannel::Close
 */
void cluster::CommChannel::Close(void) {
    if (this->receiver.IsRunning()) {
        this->receiver.Terminate(false);
    }
    if (!this->channel.IsNull()) {
        try {
            this->channel->Close();
        } catch (...) {}
        this->channel = NULL;
    }
}


/*
 * cluster::CommChannel::IsOpen
 */
bool cluster::CommChannel::IsOpen(void) const {
    return !this->channel.IsNull() && this->receiver.IsRunning();
}


/*
 * cluster::CommChannel::Open
 */
void cluster::CommChannel::Open(vislib::SmartRef<vislib::net::AbstractCommClientChannel> channel) {
    this->Close();
    this->channel = channel;
    vislib::net::SimpleMessageDispatcher::Configuration cfg(channel);
    this->receiver.Start(&cfg);
    if ((!this->receiver.IsRunning()) && (!this->channel.IsNull())) {
        try {
            this->channel->Close();
        } catch (...) {}
        this->channel = NULL;
    }
}


/*
 * cluster::CommChannel::SendMessage
 */
void cluster::CommChannel::SendMessage(const vislib::net::AbstractSimpleMessage& msg) {
    if (this->channel.IsNull()) {
        throw vislib::IllegalStateException("Channel not open", __FILE__, __LINE__);
    }
    if (this->channel->Send(msg, msg.GetMessageSize(), 0, true) != msg.GetMessageSize()) {
        throw vislib::Exception("Unable to send the whole message", __FILE__, __LINE__);
    }
}


/*
 * cluster::CommChannel::operator==
 */
bool cluster::CommChannel::operator==(const cluster::CommChannel& rhs) const {
    return (this->channel == rhs.channel);
}


/*
 * cluster::CommChannel::operator=
 */
cluster::CommChannel& cluster::CommChannel::operator=(const cluster::CommChannel& rhs) {
    if (this->IsOpen() || rhs.IsOpen()) {
        throw vislib::IllegalStateException("operator= is illegal on open channels", __FILE__, __LINE__);
    }
    this->channel = rhs.channel;
    return *this;
}


/*
 * cluster::CommChannel::OnCommunicationError
 */
bool cluster::CommChannel::OnCommunicationError(
    vislib::net::SimpleMessageDispatcher& src, const vislib::Exception& exception) throw() {
    megamol::core::utility::log::Log::DefaultLog.WriteMsg(
        megamol::core::utility::log::Log::LEVEL_WARN, "Communication Channel: %s\n", exception.GetMsgA());
    return true; // keep receiver running
}


/*
 * cluster::CommChannel::OnDispatcherExited
 */
void cluster::CommChannel::OnDispatcherExited(vislib::net::SimpleMessageDispatcher& src) throw() {
    vislib::Listenable<CommChannel>::ListenerIterator iter = this->GetListeners();
    while (iter.HasNext()) {
        Listener* l = dynamic_cast<Listener*>(iter.Next());
        if (l == NULL)
            continue;
        l->OnCommChannelDisconnect(*this);
    }
    if (!this->channel.IsNull()) {
        try {
            this->channel->Close();
        } catch (...) {}
        this->channel = NULL;
    }
}


/*
 * cluster::CommChannel::OnDispatcherStarted
 */
void cluster::CommChannel::OnDispatcherStarted(vislib::net::SimpleMessageDispatcher& src) throw() {
    vislib::Listenable<CommChannel>::ListenerIterator iter = this->GetListeners();
    while (iter.HasNext()) {
        Listener* l = dynamic_cast<Listener*>(iter.Next());
        if (l == NULL)
            continue;
        l->OnCommChannelConnect(*this);
    }
}


/*
 * cluster::CommChannel::OnMessageReceived
 */
bool cluster::CommChannel::OnMessageReceived(
    vislib::net::SimpleMessageDispatcher& src, const vislib::net::AbstractSimpleMessage& msg) throw() {
    vislib::Listenable<CommChannel>::ListenerIterator iter = this->GetListeners();
    while (iter.HasNext()) {
        Listener* l = dynamic_cast<Listener*>(iter.Next());
        if (l == NULL)
            continue;
        l->OnCommChannelMessage(*this, msg);
    }
    return true; // keep receiver running
}
