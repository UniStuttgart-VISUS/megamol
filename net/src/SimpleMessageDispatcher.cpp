/*
 * SimpleMessageDispatcher.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/SimpleMessageDispatcher.h"

#include "vislib/AbstractCommChannel.h"
#include "vislib/SimpleMessageDispatchListener.h"
#include "vislib/SocketException.h"
#include "vislib/Trace.h"
#include "vislib/Thread.h"


/*
 * vislib::net::SimpleMessageDispatcher::SimpleMessageDispatcher
 */
vislib::net::SimpleMessageDispatcher::SimpleMessageDispatcher(void) {
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
 * vislib::net::SimpleMessageDispatcher::OnThreadStarting
 */
void vislib::net::SimpleMessageDispatcher::OnThreadStarting(void *config) {
    VLSTACKTRACE("SimpleMessageDispatcher::OnThreadStarting", __FILE__, 
        __LINE__);
    ASSERT(config != NULL);
    Configuration *c = static_cast<Configuration *>(config);

    ASSERT(!c->Channel.IsNull());
    this->configuration.Channel = c->Channel;
}


/*
 * vislib::net::SimpleMessageDispatcher::RemoveListener
 */
void vislib::net::SimpleMessageDispatcher::RemoveListener(
        SimpleMessageDispatchListener *listener) {
    VLSTACKTRACE("SimpleMessageDispatcher::RemoveListener", __FILE__, __LINE__);
    ASSERT(listener != NULL);
    this->listeners.RemoveAll(listener);
}


/*
 * vislib::net::SimpleMessageDispatcher::Run
 */
DWORD vislib::net::SimpleMessageDispatcher::Run(void *config) {
    VLSTACKTRACE("SimpleMessageDispatcher::Run", __FILE__, __LINE__);
    ASSERT(!this->configuration.Channel.IsNull());

    bool doReceive = true;
    
    try {
        Socket::Startup();
    } catch (SocketException e) {
        VLTRACE(VISLIB_TRCELVL_ERROR, "Socket::Startup failed in "
            "SimpleMessageDispatcher::Run: %s\n", e.GetMsgA());
        this->fireCommunicationError(e);
        return e.GetErrorCode();
    }

    VLTRACE(VISLIB_TRCELVL_INFO, "The SimpleMessageDispatcher [%u] is entering "
        "the message loop ...\n", vislib::sys::Thread::CurrentID());
    this->fireDispatcherStarted();

    try {
        while (doReceive) {
            VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "[%u] is waiting for "
                "message header ...\n", vislib::sys::Thread::CurrentID());
            this->configuration.Channel->Receive(static_cast<void *>(this->msg),
                this->msg.GetHeader().GetHeaderSize(), 
                AbstractCommChannel::TIMEOUT_INFINITE, 
                true);
            VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "[%u] received message "
                "header with { MessageID = %u, BodySize = %u }\n", 
                vislib::sys::Thread::CurrentID(),
                this->msg.GetHeader().GetMessageID(),
                this->msg.GetHeader().GetBodySize());
            this->msg.AssertBodySize();

            if (this->msg.GetHeader().GetBodySize() > 0) {
                VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "[%u] is waiting "
                    "for message body ...\n", vislib::sys::Thread::CurrentID());
                this->configuration.Channel->Receive(this->msg.GetBody(), 
                    this->msg.GetHeader().GetBodySize(),
                    AbstractCommChannel::TIMEOUT_INFINITE, 
                    true);
                VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "[%u] received "
                    "message body.\n", vislib::sys::Thread::CurrentID());
            }

            VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "[%u] is dispatching "
                "message to registered listeners ...\n", 
                vislib::sys::Thread::CurrentID());
            doReceive = this->fireMessageReceived(this->msg);
        }
    } catch (Exception e) {
        VLTRACE(VISLIB_TRCELVL_ERROR, "The SimpleMessageDispatcher [%u] "
            "encountered an error: %s\n", vislib::sys::Thread::CurrentID(),
            e.GetMsgA());
        doReceive = this->fireCommunicationError(e);
    }

    try {
        this->configuration.Channel->Close();
    } catch (...) {
        VLTRACE(VISLIB_TRCELVL_WARN, "The SimpleMessageDispatcher tried to "
            "close a channel which was probably already closed due to an "
            "error before.\n");
    }

    try {
        Socket::Cleanup();
    } catch (SocketException e) {
        VLTRACE(VISLIB_TRCELVL_WARN, "Socket::Cleanup failed in "
            "SimpleMessageDispatcher::Run: %s\n", e.GetMsgA());
    }

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

    if (!this->configuration.Channel.IsNull()) {
        try {
            this->configuration.Channel->Close();
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
bool vislib::net::SimpleMessageDispatcher::fireCommunicationError(
        const vislib::Exception& exception) {
    VLSTACKTRACE("SimpleMessageDispatcher::fireCommunicationError", 
        __FILE__, __LINE__);
    bool retval = true;

    this->listeners.Lock();
    ListenerList::Iterator it = this->listeners.GetIterator();
    while (it.HasNext()) {
        retval = it.Next()->OnCommunicationError(*this, exception) && retval;
    }
    this->listeners.Unlock();

    VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "SimpleMessageDispatcher "
        "received %sexit request from registered error listener.\n", 
        (retval ? "no ": ""));
    return retval;
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

    VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "SimpleMessageDispatcher "
        "received %sexit request from registered message listener.\n", 
        (retval ? "no ": ""));
    return retval;
}
