/*
 * AbstractClientNode.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractClientNode.h"

#include "vislib/assert.h"
#include "vislib/IllegalStateException.h"
#include "vislib/memutils.h"
#include "vislib/SocketException.h"
#include "vislib/SystemException.h"
#include "vislib/Trace.h"
#include "vislib/unreferenced.h"

#include "messagereceiver.h"


/*
 * vislib::net::cluster::AbstractClientNode::~AbstractClientNode
 */
vislib::net::cluster::AbstractClientNode::~AbstractClientNode(void) {
    this->disconnect(true, true);
    try {
        Socket::Cleanup();
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_WARN, "Exception while releasing "
            "AbstractClientNode: %s\n", e.GetMsgA());
    }
}


/*
 * vislib::net::cluster::AbstractClientNode::Initialise
 */
void vislib::net::cluster::AbstractClientNode::Initialise(
        sys::CmdLineProviderA& inOutCmdLine) {
    return this->initialise(inOutCmdLine);
}


/*
 * vislib::net::cluster::AbstractClientNode::Initialise
 */
void vislib::net::cluster::AbstractClientNode::Initialise(
        sys::CmdLineProviderW& inOutCmdLine) {
    return this->initialise(inOutCmdLine);
}


/*
 * vislib::net::cluster::AbstractClientNode::Run
 */
DWORD vislib::net::cluster::AbstractClientNode::Run(void) {
    if (this->socket.IsValid() || (this->receiver.IsRunning())) {
        throw IllegalStateException("AbstractClientNode::Run can only be "
            "called once for connecting to the server node.", __FILE__, 
            __LINE__);
    }

    this->reconnectAttempts++;  // First connect is for free.
    this->connect(NULL);

    return 0;
}


/*
 * vislib::net::cluster::AbstractClientNode::AbstractClientNode
 */
vislib::net::cluster::AbstractClientNode::AbstractClientNode(void)
        : Super(), reconnectAttempts(0), receiver(ReceiveMessages),
        // TODO: IPv6
        serverAddress(IPAddress::ANY, DEFAULT_PORT) {
    try {
        Socket::Startup();
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_ERROR, "Socket::Startup failed in "
            "AbstractClientNode::ctor. The instance will probably not work. "
            "Details: %s\n", e.GetMsgA());
    }
}


/*
 * vislib::net::cluster::AbstractClientNode::AbstractClientNode
 */
vislib::net::cluster::AbstractClientNode::AbstractClientNode(
        const AbstractClientNode& rhs) 
        : Super(rhs), reconnectAttempts(rhs.reconnectAttempts), 
        receiver(ReceiveMessages), serverAddress(rhs.serverAddress) {
    try {
        Socket::Startup();
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_ERROR, "Socket::Startup failed in "
            "AbstractClientNode::ctor. The instance will probably not work. "
            "Details: %s\n", e.GetMsgA());
    }
}


/*
 * vislib::net::cluster::AbstractClientNode::connect
 */
void vislib::net::cluster::AbstractClientNode::connect(
        PReceiveMessagesCtx rmc) {
    /* Clean up if in some case of illegal state. */
    this->disconnect(false, false);

    /* Remember connect attempt. */
    if (this->reconnectAttempts > 0) {
        this->reconnectAttempts--;
    }

    /* Connect to the server. */
    VLTRACE(Trace::LEVEL_VL_INFO, "Connecting to server node %s ...\n", 
        this->serverAddress.ToStringA().PeekBuffer());
    this->socket.Create(Socket::FAMILY_INET, Socket::TYPE_STREAM,
        Socket::PROTOCOL_TCP);
    this->socket.SetNoDelay(true);
    this->socket.Connect(this->serverAddress);

    /* Start the message receiver. */
    if (rmc == NULL) {
        rmc = AllocateRecvMsgCtx(this, &this->socket);
    } else {
        rmc->Receiver = this;
        rmc->Socket = &this->socket;
    }
    try {
        VERIFY(this->receiver.Start(static_cast<void *>(rmc)));
    } catch (Exception e) {
        FreeRecvMsgCtx(rmc);
        throw e;
    }

    this->onPeerConnected(this->serverAddress);
}


/*
 * vislib::net::cluster::AbstractClientNode::disconnect
 */
void vislib::net::cluster::AbstractClientNode::disconnect(const bool isSilent,
        const bool noReconnect) {
    if (noReconnect) {
        this->reconnectAttempts = 0;
    }

    try {
        this->socket.Close();
    } catch (...) {
        if (!isSilent) {
            throw;
        }
    }

    try {
        this->receiver.Join();
    } catch (...) {
        if (!isSilent) {
            throw;
        }
    }
}


/*
 * vislib::net::cluster::AbstractClientNode::countPeers
 */
SIZE_T vislib::net::cluster::AbstractClientNode::countPeers(void) const {
    return (this->socket.IsValid() ? 1 : 0);
}


/*
 * vislib::net::cluster::AbstractClientNode::forEachPeer
 */
SIZE_T vislib::net::cluster::AbstractClientNode::forEachPeer(
        ForeachPeerFunc func, void *context) {
    SIZE_T retval = 0;

    try {
        func(this, this->serverAddress, this->socket, context);
        retval = 1;
    } catch (Exception& e) {
        VL_DBGONLY_REFERENCED_LOCAL_VARIABLE(e);
        VLTRACE(Trace::LEVEL_VL_WARN, "ForeachPeerFunc failed "
            "with an exception: %s\n", e.GetMsgA());
    } catch (...) {
        VLTRACE(Trace::LEVEL_VL_WARN, "ForeachPeerFunc failed "
            "with a non-VISlib exception.\n");
    }

    return retval;
}

/*
 * vislib::net::cluster::AbstractClientNode::forPeer
 */
bool vislib::net::cluster::AbstractClientNode::forPeer(
        const PeerIdentifier& peerId, ForeachPeerFunc func, void *context) {
    bool retval = false;

    if (this->serverAddress == peerId) {
        try {
            func(this, this->serverAddress, this->socket, context);
            retval = true;
        } catch (Exception& e) {
            VL_DBGONLY_REFERENCED_LOCAL_VARIABLE(e);
            VLTRACE(Trace::LEVEL_VL_WARN, "ForeachPeerFunc failed "
                "with an exception: %s\n", e.GetMsgA());
        } catch (...) {
            VLTRACE(Trace::LEVEL_VL_WARN, "ForeachPeerFunc failed "
                "with a non-VISlib exception.\n");
        }
    }

    return retval;
}



/*
 * vislib::net::cluster::AbstractClientNode::onMessageReceiverExiting
 */
void vislib::net::cluster::AbstractClientNode::onMessageReceiverExiting(
        Socket& socket, PReceiveMessagesCtx rmc) {

    // TODO: Das funktioniert so nicht, weil der Handler im receiver, der
    // abgebrochen werden muss, aufgerufen wird.

    //while (this->reconnectAttempts > 0) {
    //    VLTRACE(Trace::LEVEL_VL_INFO, "Message receiver exited, trying to "
    //        "reconnect ...\n");
    //    try {
    //        this->connect(rmc);
    //    } catch (Exception& e) {
    //        VLTRACE(Trace::LEVEL_VL_WARN, "Reconnection attempt failed: %s\n",
    //            e.GetMsgA());
    //    } catch (...) {
    //        VLTRACE(Trace::LEVEL_VL_WARN, "Reconnection attempt failed.\n");
    //    }
    //}
    /* Specified number of reconnect attempts failed, so clean up. */

    VLTRACE(Trace::LEVEL_VL_INFO, "Should not try to reconnect any more, "
        "releasing resources ...\n");
    Super::onMessageReceiverExiting(socket, rmc);
    //this->disconnect(true, true);
}


/*
 * vislib::net::cluster::AbstractClientNode::operator =
 */
vislib::net::cluster::AbstractClientNode& 
vislib::net::cluster::AbstractClientNode::operator =(
        const AbstractClientNode& rhs) {
    if (this != &rhs) {
        Super::operator =(rhs);
        this->reconnectAttempts = rhs.reconnectAttempts;
        this->serverAddress = rhs.serverAddress;
        this->socket = Socket();
    }
    return *this;
}
