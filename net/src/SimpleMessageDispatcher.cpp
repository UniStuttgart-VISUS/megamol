/*
 * SimpleMessageDispatcher.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/Socket.h"
#include "vislib/SimpleMessageDispatcher.h"

#include "vislib/AbstractInboundCommChannel.h"
#include "vislib/SimpleMessageDispatchListener.h"
#include "vislib/Trace.h"


/*
 * vislib::net::SimpleMessageDispatcher::SimpleMessageDispatcher
 */
vislib::net::SimpleMessageDispatcher::SimpleMessageDispatcher(void) 
        : channel(NULL) {
    VLSTACKTRACE("SimpleMessageDispatcher::SimpleMessageDispatcher", 
        __FILE__, __LINE__);
}


/*
 * vislib::net::SimpleMessageDispatcher::~SimpleMessageDispatcher
 */
vislib::net::SimpleMessageDispatcher::~SimpleMessageDispatcher(void) {
    VLSTACKTRACE("SimpleMessageDispatcher::~SimpleMessageDispatcher", 
        __FILE__, __LINE__);
}


/*
 * vislib::net::SimpleMessageDispatcher::AddListener
 */
void vislib::net::SimpleMessageDispatcher::AddListener(
        SimpleMessageDispatchListener *listener) {
    VLSTACKTRACE("SimpleMessageDispatcher::AddListener", __FILE__, __LINE__);
    ASSERT(listener != NULL);

    this->listeners.Lock();
    if ((listener != NULL) && !this->listeners.Contains(listener)) {
        this->listeners.Append(listener);
    }
    this->listeners.Unlock();
}


/*
 * vislib::net::SimpleMessageDispatcher::RemoveListener
 */
void vislib::net::SimpleMessageDispatcher::RemoveListener(
        SimpleMessageDispatchListener *listener) {
    VLSTACKTRACE("SimpleMessageDispatcher::AddListener", __FILE__, __LINE__);
    ASSERT(listener != NULL);
    this->listeners.RemoveAll(listener);
}


/*
 * vislib::net::SimpleMessageDispatcher::Run
 */
DWORD vislib::net::SimpleMessageDispatcher::Run(void *channel) {
    VLSTACKTRACE("SimpleMessageDispatcher::Run", __FILE__, __LINE__);
    ASSERT(channel != NULL);

    bool exitRequested = false;
    this->channel = static_cast<AbstractInboundCommChannel *>(channel);

    Socket::Startup();

    VLTRACE(VISLIB_TRCELVL_INFO, "The SimpleMessageDispatcher is entering the "
        "message loop ...\n");
    this->fireDispatcherStarted();

    try {
        while (!exitRequested) {
            VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "Waiting for "
                "message header ...\n");
            this->channel->Receive(static_cast<void *>(this->msg), 
                this->msg.GetHeader().GetHeaderSize(), 
                AbstractCommChannel::TIMEOUT_INFINITE, 
                true);
            VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "Received message "
                "header with { MessageID = %u, BodySize = %u }\n", 
                this->msg.GetHeader().GetMessageID(),
                this->msg.GetHeader().GetBodySize());
            this->msg.AssertBodySize();

            VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "Waiting for "
                "message body ...\n");
            this->channel->Receive(this->msg.GetBody(), 
                this->msg.GetHeader().GetBodySize(),
                AbstractCommChannel::TIMEOUT_INFINITE, 
                true);

            VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "Message body "
                "received.\n");
            this->fireMessageReceived(this->msg);
        }
    } catch (Exception e) {
        VLTRACE(VISLIB_TRCELVL_ERROR, "The SimpleMessageDispatcher encountered "
            " an error: %s\n", e.GetMsgA());
        this->fireCommunicationError(e);
    }

    Socket::Cleanup();

    VLTRACE(VISLIB_TRCELVL_INFO, "The SimpleMessageDispatcher has left the "
        "message loop ...\n");
    this->fireDispatcherExited();
    return 0;
}


/* 
 * vislib::net::SimpleMessageDispatcher::Terminate
 */
bool vislib::net::SimpleMessageDispatcher::Terminate(void) {
    VLSTACKTRACE("SimpleMessageDispatcher::Terminate", __FILE__, __LINE__);

    if (!this->channel.IsNull()) {
        try {
            this->channel->Close();
        } catch (Exception e) {
            VLTRACE(VISLIB_TRCELVL_WARN, "An error occurred while trying to "
                "terminate a SimpleMessageDispatcher: %s\n", e.GetMsgA());
        }
    }

    return true;
}


/*
 * vislib::net::SimpleMessageDispatcher::fireCommunicationError
 */
void vislib::net::SimpleMessageDispatcher::fireCommunicationError(
        const vislib::Exception& exception) {
    VLSTACKTRACE("SimpleMessageDispatcher::fireCommunicationError", 
        __FILE__, __LINE__);

    this->listeners.Lock();
    ListenerList::Iterator it = this->listeners.GetIterator();
    while (it.HasNext()) {
        it.Next()->OnCommunicationError(*this, exception);
    }
    this->listeners.Unlock();

}


/*
 * vislib::net::SimpleMessageDispatcher::fireDispatcherExited
 */
void vislib::net::SimpleMessageDispatcher::fireDispatcherExited(void) {
    VLSTACKTRACE("SimpleMessageDispatcher::fireDispatcherExited", 
        __FILE__, __LINE__);

    this->listeners.Lock();
    ListenerList::Iterator it = this->listeners.GetIterator();
    while (it.HasNext()) {
        it.Next()->OnDispatcherExited(*this);
    }
    this->listeners.Unlock();
}


/*
 * vislib::net::SimpleMessageDispatcher::fireDispatcherStarted
 */
void vislib::net::SimpleMessageDispatcher::fireDispatcherStarted(void) {
    VLSTACKTRACE("SimpleMessageDispatcher::fireDispatcherStarted", 
        __FILE__, __LINE__);

    this->listeners.Lock();
    ListenerList::Iterator it = this->listeners.GetIterator();
    while (it.HasNext()) {
        it.Next()->OnDispatcherStarted(*this);
    }
    this->listeners.Unlock();
}


/*
 * vislib::net::SimpleMessageDispatcher::fireMessageReceived
 */
bool vislib::net::SimpleMessageDispatcher::fireMessageReceived(
        const AbstractSimpleMessage& msg) {
    VLSTACKTRACE("SimpleMessageDispatcher::fireMessageReceived", 
        __FILE__, __LINE__);
    bool retval = true;

    this->listeners.Lock();
    ListenerList::Iterator it = this->listeners.GetIterator();
    while (it.HasNext()) {
        retval = it.Next()->OnMessageReceived(*this, msg) && retval;
    }
    this->listeners.Unlock();
    return retval;
}
