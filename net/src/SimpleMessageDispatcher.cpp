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
#include "the/trace.h"
#include "vislib/Thread.h"


/*
 * vislib::net::SimpleMessageDispatcher::SimpleMessageDispatcher
 */
vislib::net::SimpleMessageDispatcher::SimpleMessageDispatcher(void) {
    THE_STACK_TRACE;
}


/*
 * vislib::net::SimpleMessageDispatcher::~SimpleMessageDispatcher
 */
vislib::net::SimpleMessageDispatcher::~SimpleMessageDispatcher(void) {
    THE_STACK_TRACE;
}


/*
 * vislib::net::SimpleMessageDispatcher::AddListener
 */
void vislib::net::SimpleMessageDispatcher::AddListener(
        SimpleMessageDispatchListener *listener) {
    THE_STACK_TRACE;
    THE_ASSERT(listener != NULL);

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
    THE_STACK_TRACE;
    THE_ASSERT(config != NULL);
    Configuration *c = static_cast<Configuration *>(config);

    THE_ASSERT(!c->Channel.IsNull());
    this->configuration.Channel = c->Channel;
}


/*
 * vislib::net::SimpleMessageDispatcher::RemoveListener
 */
void vislib::net::SimpleMessageDispatcher::RemoveListener(
        SimpleMessageDispatchListener *listener) {
    THE_STACK_TRACE;
    THE_ASSERT(listener != NULL);
    this->listeners.RemoveAll(listener);
}


/*
 * vislib::net::SimpleMessageDispatcher::Run
 */
unsigned int vislib::net::SimpleMessageDispatcher::Run(void *config) {
    THE_STACK_TRACE;
    THE_ASSERT(!this->configuration.Channel.IsNull());

    bool doReceive = true;
    
    try {
        Socket::Startup();
    } catch (SocketException e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Socket::Startup failed in "
            "SimpleMessageDispatcher::Run: %s\n", e.what());
        this->fireCommunicationError(e);
        return e.get_error().native_error();
    }

    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "The SimpleMessageDispatcher [%u] is entering "
        "the message loop ...\n", vislib::sys::Thread::CurrentID());
    this->fireDispatcherStarted();

    try {
        while (doReceive) {
            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "[%u] is waiting for "
                "message header ...\n", vislib::sys::Thread::CurrentID());
            this->configuration.Channel->Receive(static_cast<void *>(this->msg),
                this->msg.GetHeader().GetHeaderSize(), 
                AbstractCommChannel::TIMEOUT_INFINITE, 
                true);
            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "[%u] received message "
                "header with { MessageID = %u, BodySize = %u }\n", 
                vislib::sys::Thread::CurrentID(),
                this->msg.GetHeader().GetMessageID(),
                this->msg.GetHeader().GetBodySize());
            this->msg.AssertBodySize();

            if (this->msg.GetHeader().GetBodySize() > 0) {
                THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "[%u] is waiting "
                    "for message body ...\n", vislib::sys::Thread::CurrentID());
                this->configuration.Channel->Receive(this->msg.GetBody(), 
                    this->msg.GetHeader().GetBodySize(),
                    AbstractCommChannel::TIMEOUT_INFINITE, 
                    true);
                THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "[%u] received "
                    "message body.\n", vislib::sys::Thread::CurrentID());
            }

            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "[%u] is dispatching "
                "message to registered listeners ...\n", 
                vislib::sys::Thread::CurrentID());
            doReceive = this->fireMessageReceived(this->msg);
        }
    } catch (the::exception e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "The SimpleMessageDispatcher [%u] "
            "encountered an error: %s\n", vislib::sys::Thread::CurrentID(),
            e.what());
        doReceive = this->fireCommunicationError(e);
    }

    try {
        this->configuration.Channel->Close();
    } catch (...) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "The SimpleMessageDispatcher tried to "
            "close a channel which was probably already closed due to an "
            "error before.\n");
    }

    try {
        Socket::Cleanup();
    } catch (SocketException e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "Socket::Cleanup failed in "
            "SimpleMessageDispatcher::Run: %s\n", e.what());
    }

    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "The SimpleMessageDispatcher has left the "
        "message loop ...\n");
    this->fireDispatcherExited();

    return 0;
}


/* 
 * vislib::net::SimpleMessageDispatcher::Terminate
 */
bool vislib::net::SimpleMessageDispatcher::Terminate(void) {
    THE_STACK_TRACE;

    if (!this->configuration.Channel.IsNull()) {
        try {
            this->configuration.Channel->Close();
        } catch (the::exception e) {
            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "An error occurred while trying to "
                "terminate a SimpleMessageDispatcher: %s\n", e.what());
        }
    }

    return true;
}


/*
 * vislib::net::SimpleMessageDispatcher::fireCommunicationError
 */
bool vislib::net::SimpleMessageDispatcher::fireCommunicationError(
        const the::exception& exception) {
    THE_STACK_TRACE;
    bool retval = true;

    this->listeners.Lock();
    ListenerList::Iterator it = this->listeners.GetIterator();
    while (it.HasNext()) {
        retval = it.Next()->OnCommunicationError(*this, exception) && retval;
    }
    this->listeners.Unlock();

    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "SimpleMessageDispatcher "
        "received %sexit request from registered error listener.\n", 
        (retval ? "no ": ""));
    return retval;
}


/*
 * vislib::net::SimpleMessageDispatcher::fireDispatcherExited
 */
void vislib::net::SimpleMessageDispatcher::fireDispatcherExited(void) {
    THE_STACK_TRACE;

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
    THE_STACK_TRACE;

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
    THE_STACK_TRACE;
    bool retval = true;

    this->listeners.Lock();
    ListenerList::Iterator it = this->listeners.GetIterator();
    while (it.HasNext()) {
        retval = it.Next()->OnMessageReceived(*this, msg) && retval;
    }
    this->listeners.Unlock();

    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "SimpleMessageDispatcher "
        "received %sexit request from registered message listener.\n", 
        (retval ? "no ": ""));
    return retval;
}
