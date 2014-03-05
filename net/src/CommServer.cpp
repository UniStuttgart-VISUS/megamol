/*
 * CommServer.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/Socket.h"  // Must be first.
#include "vislib/CommServer.h"

#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/Interlocked.h"
#include "vislib/SocketException.h"
#include "vislib/SystemException.h"
#include "vislib/TcpCommChannel.h"
#include "the/trace.h"


/*
 * vislib::net::CommServer::CommServer
 */
vislib::net::CommServer::CommServer(void) {
    THE_STACK_TRACE;
}


/*
 * vislib::net::CommServer::~CommServer
 */
vislib::net::CommServer::~CommServer(void) {
    THE_STACK_TRACE;
}


/*
 * vislib::net::CommServer::AddListener
 */
void vislib::net::CommServer::AddListener(CommServerListener *listener) {
    THE_STACK_TRACE;
    THE_ASSERT(listener != NULL);

    this->listeners.Lock();
    if ((listener != NULL) && !this->listeners.Contains(listener)) {
        this->listeners.Append(listener);
    }
    this->listeners.Unlock();
}


/*
 * vislib::net::CommServer::OnThreadStarting
 */
void vislib::net::CommServer::OnThreadStarting(void *config) {
    THE_STACK_TRACE;
    THE_ASSERT(config != NULL);
    Configuration *c = static_cast<Configuration *>(config);

    THE_ASSERT(!c->Channel.IsNull());
    this->configuration.Channel = c->Channel;

    THE_ASSERT(!c->EndPoint.IsNull());
    this->configuration.EndPoint = c->EndPoint;

    this->doServe = 1;
}


/*
 * vislib::net::CommServer::RemoveListener
 */
void vislib::net::CommServer::RemoveListener(CommServerListener *listener) {
    THE_STACK_TRACE;
    THE_ASSERT(listener != NULL);
    this->listeners.RemoveAll(listener);
}


/*
 * vislib::net::CommServer::Run
 */
DWORD vislib::net::CommServer::Run(void *config) {
    THE_STACK_TRACE;
    DWORD retval = 0;

    /* Prepare the socket subsystem. */
    try {
        Socket::Startup();
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Socket::Startup succeeded in "
            "CommServer::Run\n.");
    } catch (SocketException e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Socket::Startup in CommServer::Run "
            "failed: %s\n", e.GetMsgA());
        retval = e.GetErrorCode();
        this->fireServerError(e);
        return retval;
    }

    /* Bind the end point. */
    try {
        this->configuration.Channel->Bind(this->configuration.EndPoint);
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "CommServer bound to %s.\n",
            this->configuration.EndPoint->ToStringA().PeekBuffer());
    } catch (sys::SystemException se) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Binding server end point to specified "
            "address failed: %s\n", se.GetMsgA());
        retval = se.GetErrorCode();
        this->fireServerError(se);
    } catch (Exception e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Binding server end point to specified "
            "address failed: %s\n", e.GetMsgA());
        retval = -1;
        this->fireServerError(e);
    }

    /* Put the connection in listening state. */
    try {
        this->configuration.Channel->Listen(SOMAXCONN);
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "CommServer is now in listen "
            "state.\n");
    } catch (sys::SystemException se) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Putting server end point in listen "
            "state failed: %s\n", se.GetMsgA());
        retval = se.GetErrorCode();
        this->fireServerError(se);
    } catch (Exception e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Putting server end point in listen "
            "state failed: %s\n", e.GetMsgA());
        retval = -1;
        this->fireServerError(e);
    }

    /* Enter server loop if no error so far. */
    if (retval == 0) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "CommServer is entering the server "
            "loop ...\n");
        this->fireServerStarted();

        while (doServe != 0) {
            try {
                SmartRef<AbstractCommClientChannel> channel 
                    = this->configuration.Channel->Accept();
                THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "CommServer accepted new "
                    "connection.\n");

                if (!this->fireNewConnection(channel)) {
                    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "CommServer is closing "
                        "connection, because none of the registered listeners "
                        "is interested in the client.\n");
                    try {
                        channel->Close();
                        channel.Release();
                    } catch (Exception e) {
                        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "Closing unused peer "
                            "connection caused an error: %s\n", e.GetMsgA());
                    }
                }
           } catch (sys::SystemException e) {
                THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "Communication error in "
                    "CommServer: %s\n", e.GetMsgA());
                retval = e.GetErrorCode();
                INT32 ds = this->fireServerError(e); 
                vislib::sys::Interlocked::CompareExchange(&this->doServe, ds, 
                    static_cast<INT32>(1));
            } catch (Exception e) {
                THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "Communication error in "
                    "CommServer: %s\n", e.GetMsgA());
                retval = -1;
                INT32 ds = this->fireServerError(e);
                vislib::sys::Interlocked::CompareExchange(&this->doServe, ds, 
                    static_cast<INT32>(1));
            }
        }
    }
    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "CommServer has left the server "
        "loop.\n");

    /* Clean up connection and socket library. */
    this->Terminate();

    try {
        Socket::Cleanup();
    } catch (SocketException e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "Socket::Cleanup in CommServer::Run "
            "failed: %s\n", e.GetMsgA());
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
bool vislib::net::CommServer::Terminate(void) {
    THE_STACK_TRACE;
    try {
        vislib::sys::Interlocked::Exchange(&this->doServe, 
            static_cast<INT32>(0));
        if (!this->configuration.Channel.IsNull()) {
            this->configuration.Channel->Close();
        }
    } catch (Exception e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "Exception when shutting down "
            "CommServer: %s. This is usually no problem.", e.GetMsgA());
    }
    return true;
}


/*
 * vislib::net::CommServer::fireNewConnection
 */
bool vislib::net::CommServer::fireNewConnection(
        SmartRef<AbstractCommClientChannel>& channel) {
    THE_STACK_TRACE;
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

    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "CommServer informed "
        "listeners about new connection. Was accepted: %s\n", 
        retval ? "yes" : "no");
    return retval;
}


/*
 * vislib::net::CommServer::fireServerError
 */
bool vislib::net::CommServer::fireServerError(
        const vislib::Exception& exception) {
    THE_STACK_TRACE;
    bool retval = true;

    this->listeners.Lock();
    ListenerList::Iterator it = this->listeners.GetIterator();
    while (it.HasNext()) {
        retval = it.Next()->OnServerError(*this, exception) && retval;
    }
    this->listeners.Unlock();

    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "CommServer "
        "received exit request from registered error listener: %s\n", 
        !retval ? "yes" : "no");
    return retval;
}


/*
 * vislib::net::CommServer::fireServerExited
 */
void vislib::net::CommServer::fireServerExited(void) {
    THE_STACK_TRACE;
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
void vislib::net::CommServer::fireServerStarted(void) {
    THE_STACK_TRACE;
    this->listeners.Lock();
    ListenerList::Iterator it = this->listeners.GetIterator();
    while (it.HasNext()) {
        it.Next()->OnServerStarted(*this);
    }
    this->listeners.Unlock();
}
