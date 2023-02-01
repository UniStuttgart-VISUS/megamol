/*
 * CommServer.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/net/CommServer.h"
#include "vislib/net/Socket.h"

#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/Trace.h"
#include "vislib/net/SocketException.h"
#include "vislib/net/TcpCommChannel.h"
#include "vislib/sys/Interlocked.h"
#include "vislib/sys/SystemException.h"


/*
 * vislib::net::CommServer::CommServer
 */
vislib::net::CommServer::CommServer() {}


/*
 * vislib::net::CommServer::~CommServer
 */
vislib::net::CommServer::~CommServer() {}


/*
 * vislib::net::CommServer::AddListener
 */
void vislib::net::CommServer::AddListener(CommServerListener* listener) {
    ASSERT(listener != NULL);

    this->listeners.Lock();
    if ((listener != NULL) && !this->listeners.Contains(listener)) {
        this->listeners.Append(listener);
    }
    this->listeners.Unlock();
}


/*
 * vislib::net::CommServer::OnThreadStarting
 */
void vislib::net::CommServer::OnThreadStarting(void* config) {
    ASSERT(config != NULL);
    Configuration* c = static_cast<Configuration*>(config);

    ASSERT(!c->Channel.IsNull());
    this->configuration.Channel = c->Channel;

    ASSERT(!c->EndPoint.IsNull());
    this->configuration.EndPoint = c->EndPoint;

    this->doServe = 1;
}


/*
 * vislib::net::CommServer::RemoveListener
 */
void vislib::net::CommServer::RemoveListener(CommServerListener* listener) {
    ASSERT(listener != NULL);
    this->listeners.RemoveAll(listener);
}


/*
 * vislib::net::CommServer::Run
 */
DWORD vislib::net::CommServer::Run(void* config) {
    DWORD retval = 0;

    /* Prepare the socket subsystem. */
    try {
        Socket::Startup();
        VLTRACE(Trace::LEVEL_VL_VERBOSE, "Socket::Startup succeeded in "
                                         "CommServer::Run\n.");
    } catch (SocketException e) {
        VLTRACE(VISLIB_TRCELVL_ERROR,
            "Socket::Startup in CommServer::Run "
            "failed: %s\n",
            e.GetMsgA());
        retval = e.GetErrorCode();
        this->fireServerError(e);
        return retval;
    }

    /* Bind the end point. */
    try {
        this->configuration.Channel->Bind(this->configuration.EndPoint);
        VLTRACE(Trace::LEVEL_VL_VERBOSE, "CommServer bound to %s.\n",
            this->configuration.EndPoint->ToStringA().PeekBuffer());
    } catch (sys::SystemException se) {
        VLTRACE(VISLIB_TRCELVL_ERROR,
            "Binding server end point to specified "
            "address failed: %s\n",
            se.GetMsgA());
        retval = se.GetErrorCode();
        this->fireServerError(se);
    } catch (Exception e) {
        VLTRACE(VISLIB_TRCELVL_ERROR,
            "Binding server end point to specified "
            "address failed: %s\n",
            e.GetMsgA());
        retval = -1;
        this->fireServerError(e);
    }

    /* Put the connection in listening state. */
    try {
        this->configuration.Channel->Listen(SOMAXCONN);
        VLTRACE(Trace::LEVEL_VL_VERBOSE, "CommServer is now in listen "
                                         "state.\n");
    } catch (sys::SystemException se) {
        VLTRACE(VISLIB_TRCELVL_ERROR,
            "Putting server end point in listen "
            "state failed: %s\n",
            se.GetMsgA());
        retval = se.GetErrorCode();
        this->fireServerError(se);
    } catch (Exception e) {
        VLTRACE(VISLIB_TRCELVL_ERROR,
            "Putting server end point in listen "
            "state failed: %s\n",
            e.GetMsgA());
        retval = -1;
        this->fireServerError(e);
    }

    /* Enter server loop if no error so far. */
    if (retval == 0) {
        VLTRACE(Trace::LEVEL_VL_VERBOSE, "CommServer is entering the server "
                                         "loop ...\n");
        this->fireServerStarted();

        while (doServe != 0) {
            try {
                SmartRef<AbstractCommClientChannel> channel = this->configuration.Channel->Accept();
                VLTRACE(Trace::LEVEL_VL_INFO, "CommServer accepted new "
                                              "connection.\n");

                if (!this->fireNewConnection(channel)) {
                    VLTRACE(Trace::LEVEL_VL_INFO, "CommServer is closing "
                                                  "connection, because none of the registered listeners "
                                                  "is interested in the client.\n");
                    try {
                        channel->Close();
                        channel.Release();
                    } catch (Exception e) {
                        VLTRACE(Trace::LEVEL_VL_WARN,
                            "Closing unused peer "
                            "connection caused an error: %s\n",
                            e.GetMsgA());
                    }
                }
            } catch (sys::SystemException e) {
                VLTRACE(VISLIB_TRCELVL_WARN,
                    "Communication error in "
                    "CommServer: %s\n",
                    e.GetMsgA());
                retval = e.GetErrorCode();
                VL_INT32 ds = this->fireServerError(e);
                vislib::sys::Interlocked::CompareExchange(&this->doServe, ds, static_cast<VL_INT32>(1));
            } catch (Exception e) {
                VLTRACE(VISLIB_TRCELVL_WARN,
                    "Communication error in "
                    "CommServer: %s\n",
                    e.GetMsgA());
                retval = -1;
                VL_INT32 ds = this->fireServerError(e);
                vislib::sys::Interlocked::CompareExchange(&this->doServe, ds, static_cast<VL_INT32>(1));
            }
        }
    }
    VLTRACE(Trace::LEVEL_VL_VERBOSE, "CommServer has left the server "
                                     "loop.\n");

    /* Clean up connection and socket library. */
    this->Terminate();

    try {
        Socket::Cleanup();
    } catch (SocketException e) {
        VLTRACE(VISLIB_TRCELVL_WARN,
            "Socket::Cleanup in CommServer::Run "
            "failed: %s\n",
            e.GetMsgA());
        retval = e.GetErrorCode();
        this->fireServerError(e);
    }

    /* Inform listener that server exits. */
    this->fireServerExited();

    return retval;
}


/*
 * vislib::net::CommServer::Terminate
 */
bool vislib::net::CommServer::Terminate() {
    try {
        vislib::sys::Interlocked::Exchange(&this->doServe, static_cast<VL_INT32>(0));
        if (!this->configuration.Channel.IsNull()) {
            this->configuration.Channel->Close();
        }
    } catch (Exception e) {
        VLTRACE(Trace::LEVEL_VL_WARN,
            "Exception when shutting down "
            "CommServer: %s. This is usually no problem.",
            e.GetMsgA());
    }
    return true;
}


/*
 * vislib::net::CommServer::fireNewConnection
 */
bool vislib::net::CommServer::fireNewConnection(SmartRef<AbstractCommClientChannel>& channel) {
    bool retval = false;

    this->listeners.Lock();
    ListenerList::Iterator it = this->listeners.GetIterator();
    while (it.HasNext()) {
        if (it.Next()->OnNewConnection(*this, channel)) {
            retval = true;
            break;
        }
    }
    this->listeners.Unlock();

    VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE,
        "CommServer informed "
        "listeners about new connection. Was accepted: %s\n",
        retval ? "yes" : "no");
    return retval;
}


/*
 * vislib::net::CommServer::fireServerError
 */
bool vislib::net::CommServer::fireServerError(const vislib::Exception& exception) {
    bool retval = true;

    this->listeners.Lock();
    ListenerList::Iterator it = this->listeners.GetIterator();
    while (it.HasNext()) {
        retval = it.Next()->OnServerError(*this, exception) && retval;
    }
    this->listeners.Unlock();

    VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE,
        "CommServer "
        "received exit request from registered error listener: %s\n",
        !retval ? "yes" : "no");
    return retval;
}


/*
 * vislib::net::CommServer::fireServerExited
 */
void vislib::net::CommServer::fireServerExited() {
    this->listeners.Lock();
    ListenerList::Iterator it = this->listeners.GetIterator();
    while (it.HasNext()) {
        it.Next()->OnServerExited(*this);
    }
    this->listeners.Unlock();
}


/*
 * vislib::net::CommServer::fireServerStarted
 */
void vislib::net::CommServer::fireServerStarted() {
    this->listeners.Lock();
    ListenerList::Iterator it = this->listeners.GetIterator();
    while (it.HasNext()) {
        it.Next()->OnServerStarted(*this);
    }
    this->listeners.Unlock();
}
