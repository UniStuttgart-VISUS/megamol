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
#include "vislib/SocketException.h"
#include "vislib/SystemException.h"
#include "vislib/TcpCommChannel.h"
#include "vislib/Trace.h"


/*
 * vislib::net::CommServer::CommServer
 */
vislib::net::CommServer::CommServer(void) {
    VLSTACKTRACE("CommServer::CommServer", __FILE__, __LINE__);
}


/*
 * vislib::net::CommServer::~CommServer
 */
vislib::net::CommServer::~CommServer(void) {
    VLSTACKTRACE("CommServer::~CommServer", __FILE__, __LINE__);
}


/*
 * vislib::net::CommServer::AddListener
 */
void vislib::net::CommServer::AddListener(CommServerListener *listener) {
    VLSTACKTRACE("CommServer::AddListener", __FILE__, __LINE__);
    ASSERT(listener != NULL);

    this->listeners.Lock();
    if ((listener != NULL) && !this->listeners.Contains(listener)) {
        this->listeners.Append(listener);
    }
    this->listeners.Unlock();
}


/*
 * vislib::net::CommServer::Configure
 */
void vislib::net::CommServer::Configure(
        AbstractServerEndPoint *serverEndPoint, 
        const wchar_t *bindAddress) {
    VLSTACKTRACE("CommServer::Configure", __FILE__, __LINE__);
    this->bindAddress = bindAddress;
    this->serverEndPoint = serverEndPoint;
}


/*
 * vislib::net::CommServer::Configure
 */
void vislib::net::CommServer::Configure(
        SmartRef<AbstractServerEndPoint>& serverEndPoint,
        const wchar_t *bindAddress) {
    VLSTACKTRACE("CommServer::Configure", __FILE__, __LINE__);
    this->bindAddress = bindAddress;
    this->serverEndPoint = serverEndPoint;
}


/*
 * vislib::net::CommServer::RemoveListener
 */
void vislib::net::CommServer::RemoveListener(CommServerListener *listener) {
    VLSTACKTRACE("CommServer::AddListener", __FILE__, __LINE__);
    ASSERT(listener != NULL);
    this->listeners.RemoveAll(listener);
}


/*
 * vislib::net::CommServer::Run
 */
DWORD vislib::net::CommServer::Run(void *reserved) {
    VLSTACKTRACE("CommServer::Run", __FILE__, __LINE__);
    DWORD retval = 0;
    bool doServe = true;

    /* Sanity checks. */
    if (this->serverEndPoint.IsNull()) {
        throw IllegalStateException("CommServer must be configured before "
            "being started.", __FILE__, __LINE__);
    }
    if (this->bindAddress.IsEmpty()) {
        throw IllegalStateException("CommServer must be configured before "
            "being started.", __FILE__, __LINE__);
    }

    /* Prepare the socket subsystem. */
    try {
        Socket::Startup();
        VLTRACE(Trace::LEVEL_VL_VERBOSE, "Socket::Startup succeeded in "
            "CommServer::Run\n.");
    } catch (SocketException e) {
        VLTRACE(VISLIB_TRCELVL_ERROR, "Socket::Startup in CommServer::Run "
            "failed: %s\n", e.GetMsgA());
        retval = e.GetErrorCode();
        this->fireServerError(e);
        return retval;
    }

    /* Bind the end point. */
    try {
        this->serverEndPoint->Bind(this->bindAddress);
        VLTRACE(Trace::LEVEL_VL_VERBOSE, "CommServer bound to %ls.\n",
            this->bindAddress.PeekBuffer());
    } catch (sys::SystemException se) {
        VLTRACE(VISLIB_TRCELVL_ERROR, "Binding server end point to specified "
            "address failed: %s\n", se.GetMsgA());
        retval = se.GetErrorCode();
        this->fireServerError(se);
    } catch (Exception e) {
        VLTRACE(VISLIB_TRCELVL_ERROR, "Binding server end point to specified "
            "address failed: %s\n", e.GetMsgA());
        retval = -1;
        this->fireServerError(e);
    }

    /* Put the connection in listening state. */
    try {
        this->serverEndPoint->Listen(SOMAXCONN);
        VLTRACE(Trace::LEVEL_VL_VERBOSE, "CommServer is now in listen "
            "state.\n");
    } catch (sys::SystemException se) {
        VLTRACE(VISLIB_TRCELVL_ERROR, "Putting server end point in listen "
            "state failed: %s\n", se.GetMsgA());
        retval = se.GetErrorCode();
        this->fireServerError(se);
    } catch (Exception e) {
        VLTRACE(VISLIB_TRCELVL_ERROR, "Putting server end point in listen "
            "state failed: %s\n", e.GetMsgA());
        retval = -1;
        this->fireServerError(e);
    }

    /* Enter server loop if no error so far. */
    if (retval == 0) {
        VLTRACE(Trace::LEVEL_VL_VERBOSE, "CommServer is entering the server "
            "loop ...\n");
        this->fireServerStarted();

        try {
            while (doServe) {
                SmartRef<AbstractCommChannel> channel 
                    = this->serverEndPoint->Accept();
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
                        VLTRACE(Trace::LEVEL_VL_WARN, "Closing unused peer "
                            "connection caused an error: %s\n", e.GetMsgA());
                    }
                }
            }
        } catch (sys::SystemException se) {
            VLTRACE(VISLIB_TRCELVL_WARN, "Communication error in CommServer: "
                "%s\n", se.GetMsgA());
            retval = se.GetErrorCode();
            doServe = this->fireServerError(se);
        } catch (Exception e) {
            VLTRACE(VISLIB_TRCELVL_WARN, "Communication error in CommServer: "
                "%s\n", e.GetMsgA());
            retval = -1;
            doServe = this->fireServerError(e);
        }
    }
    VLTRACE(Trace::LEVEL_VL_VERBOSE, "CommServer has left the server "
        "loop.\n");

    /* Clean up connection and socket library. */
    this->Terminate();

    try {
        Socket::Cleanup();
    } catch (SocketException e) {
        VLTRACE(VISLIB_TRCELVL_WARN, "Socket::Cleanup in CommServer::Run "
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
    VLSTACKTRACE("CommServer::Terminate", __FILE__, __LINE__);
    try {
        if (!this->serverEndPoint.IsNull()) {
            this->serverEndPoint.DynamicCast<AbstractCommChannel>()->Close();
        }
    } catch (Exception e) {
        VLTRACE(Trace::LEVEL_VL_WARN, "Exception when shutting down "
            "CommServer: %s. This is usually no problem.", e.GetMsgA());
    }
    return true;
}


/*
 * vislib::net::CommServer::createNewTcpServerEndPoint
 */
void vislib::net::CommServer::createNewTcpServerEndPoint(void) {
    VLSTACKTRACE("CommServer::createNewTcpServerEndPoint", __FILE__, __LINE__);
    if (this->serverEndPoint.IsNull()) {
        this->serverEndPoint = SmartRef<AbstractServerEndPoint>(
            new TcpCommChannel(), false);

        //if (this->IsSharingAddress()) {
        //    // TODO: Problems with admin rights
        //    this->socket.SetExclusiveAddrUse(false);
        //}
        //if (this->IsReuseAddress()) {
        //    this->socket.SetReuseAddr(true);
        //}
    } else {
        throw IllegalStateException("The server cannot be configured while it "
            "is running.", __FILE__, __LINE__);
    }
}


/*
 * vislib::net::CommServer::fireNewConnection
 */
bool vislib::net::CommServer::fireNewConnection(
        SmartRef<AbstractCommChannel>& channel) {
    VLSTACKTRACE("CommServer::fireNewConnection", __FILE__, __LINE__);
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

    VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "CommServer informed "
        "listeners about new connection. Was accepted: %s\n", 
        retval ? "yes" : "no");
    return retval;
}


/*
 * vislib::net::CommServer::fireServerError
 */
bool vislib::net::CommServer::fireServerError(
        const vislib::Exception& exception) {
    VLSTACKTRACE("CommServer::fireServerError", __FILE__, __LINE__);
    bool retval = true;

    this->listeners.Lock();
    ListenerList::Iterator it = this->listeners.GetIterator();
    while (it.HasNext()) {
        retval = it.Next()->OnServerError(*this, exception) && retval;
    }
    this->listeners.Unlock();

    VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "CommServer "
        "received exit request from registered error listener: %s\n", 
        retval ? "yes" : "no");
    return retval;
}


/*
 * vislib::net::CommServer::fireServerExited
 */
void vislib::net::CommServer::fireServerExited(void) {
    VLSTACKTRACE("CommServer::fireServerExited", __FILE__, __LINE__);
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
    VLSTACKTRACE("CommServer::fireServerStarted", __FILE__, __LINE__);
    this->listeners.Lock();
    ListenerList::Iterator it = this->listeners.GetIterator();
    while (it.HasNext()) {
        it.Next()->OnServerStarted(*this);
    }
    this->listeners.Unlock();
}
