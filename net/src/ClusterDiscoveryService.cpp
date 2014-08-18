/*
 * ClusterDiscoveryService.cpp
 *
 * Copyright (C) 2006 -2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/ClusterDiscoveryService.h"

#include "vislib/assert.h"
#include "vislib/ClusterDiscoveryListener.h"
#include "vislib/IllegalParamException.h"
#include "vislib/SocketException.h"
#include "vislib/Trace.h"


/*
 * vislib::net::ClusterDiscoveryService::DEFAULT_PORT 
 */
const USHORT vislib::net::ClusterDiscoveryService::DEFAULT_PORT = 28181;


/*
 * vislib::net::ClusterDiscoveryService::DEFAULT_REQUEST_INTERVAL
 */
const UINT vislib::net::ClusterDiscoveryService::DEFAULT_REQUEST_INTERVAL
    = 10 * 1000;


/*
 * vislib::net::ClusterDiscoveryService::DEFAULT_RESPONSE_CHANCES
 */
const UINT vislib::net::ClusterDiscoveryService::DEFAULT_RESPONSE_CHANCES = 1;


/*
 * vislib::net::ClusterDiscoveryService::MSG_TYPE_USER
 */
const UINT32 vislib::net::ClusterDiscoveryService::MSG_TYPE_USER = 16;


/*
 * vislib::net::ClusterDiscoveryService::ClusterDiscoveryService
 */
vislib::net::ClusterDiscoveryService::ClusterDiscoveryService(
        const StringA& name, const IPEndPoint& responseAddr, 
        const IPAddress& bcastAddr, const USHORT bindPort, 
        const bool isObserver, const UINT requestInterval, 
        const UINT cntResponseChances)
        : bcastAddr(bcastAddr, bindPort), 
        bindAddr(IPEndPoint::FAMILY_INET, bindPort), 
        responseAddr(responseAddr), 
        cntResponseChances(cntResponseChances),
        requestInterval(requestInterval),
        isObserver(isObserver),
        name(name) {
    this->name.Truncate(MAX_NAME_LEN);

    this->peerNodes.Resize(0);  // TODO: Remove alloc crowbar!
}


/*
 * vislib::net::ClusterDiscoveryService::ClusterDiscoveryService
 */
vislib::net::ClusterDiscoveryService::ClusterDiscoveryService(
        const StringA& name, const IPEndPoint& responseAddr, 
        const IPAddress6& bcastAddr, const USHORT bindPort, 
        const bool isObserver, const UINT requestInterval, 
        const UINT cntResponseChances)
        : bcastAddr(bcastAddr, bindPort), 
        bindAddr(IPEndPoint::FAMILY_INET, bindPort), 
        responseAddr(responseAddr), 
        cntResponseChances(cntResponseChances),
        requestInterval(requestInterval),
        isObserver(isObserver),
        name(name) {
    this->name.Truncate(MAX_NAME_LEN);

    this->peerNodes.Resize(0);  // TODO: Remove alloc crowbar!
}


/*
 * vislib::net::ClusterDiscoveryService::~ClusterDiscoveryService
 */
vislib::net::ClusterDiscoveryService::~ClusterDiscoveryService(void) {
    try {
        this->senderThread.Terminate(false);
    } catch (...) {
        VLTRACE(Trace::LEVEL_VL_WARN, "The discovery sender thread could "
            "not be successfully terminated.\n");
    }
    try {
        this->receiverThread.Terminate(false);
    } catch (...) {
        VLTRACE(Trace::LEVEL_VL_WARN, "The discovery receiver thread could "
            "not be successfully terminated.\n");
    }
}


/*
 * vislib::net::ClusterDiscoveryService::AddListener
 */
void vislib::net::ClusterDiscoveryService::AddListener(
        ClusterDiscoveryListener *listener) {
    ASSERT(listener != NULL);

    this->critSect.Lock();
    if ((listener != NULL) && !this->listeners.Contains(listener)) {
        this->listeners.Append(listener);
    }
    this->critSect.Unlock();
}


/*
 * vislib::net::ClusterDiscoveryService::GetDiscoveryAddress4
 */
vislib::net::IPAddress 
vislib::net::ClusterDiscoveryService::GetDiscoveryAddress4(
        const PeerHandle& hPeer) const {
    IPAddress retval = IPAddress::NONE;
    
    this->critSect.Lock();

    if (!this->isValidPeerHandle(hPeer)) {
        this->critSect.Unlock();
        throw IllegalParamException("hPeer", __FILE__, __LINE__);
    } else {
        retval = hPeer->discoveryAddr.GetIPAddress4();
        this->critSect.Unlock();
    }

    return retval;
}


/*
 * vislib::net::ClusterDiscoveryService::GetDiscoveryAddress6
 */
vislib::net::IPAddress6 
vislib::net::ClusterDiscoveryService::GetDiscoveryAddress6(
        const PeerHandle& hPeer) const {
    IPAddress6 retval;
    
    this->critSect.Lock();

    if (!this->isValidPeerHandle(hPeer)) {
        this->critSect.Unlock();
        throw IllegalParamException("hPeer", __FILE__, __LINE__);
    } else {
        retval = hPeer->discoveryAddr.GetIPAddress6();
        this->critSect.Unlock();
    }

    return retval;
}


/*
 * vislib::net::ClusterDiscoveryService::IsRunning
 */
bool vislib::net::ClusterDiscoveryService::IsRunning(void) const {
    return (this->receiverThread.IsRunning() && this->senderThread.IsRunning());
}


/*
 * vislib::net::ClusterDiscoveryService::IsStopped
 */
bool vislib::net::ClusterDiscoveryService::IsStopped(void) const {
    return !this->IsRunning();
}


/*
 * vislib::net::ClusterDiscoveryService::RemoveListener
 */
void vislib::net::ClusterDiscoveryService::RemoveListener(
        ClusterDiscoveryListener *listener) {
    ASSERT(listener != NULL);

    this->critSect.Lock();
    this->listeners.RemoveAll(listener);
    this->critSect.Unlock();
}


/*
 * vislib::net::ClusterDiscoveryService::SendUserMessage
 */
UINT vislib::net::ClusterDiscoveryService::SendUserMessage(
        const UINT32 msgType, const void *msgBody, const SIZE_T msgSize) {
    Message msg;                // The datagram we are going to send.
    UINT retval = 0;            // Number of failed communication trials.

    this->prepareUserMessage(msg, msgType, msgBody, msgSize);
    ASSERT(this->userMsgSocket.IsValid());

    /* Send the message to all registered clients. */
    this->critSect.Lock();
    for (SIZE_T i = 0; i < this->peerNodes.Count(); i++) {
        try {
            this->userMsgSocket.Send(this->peerNodes[i]->discoveryAddr,
                &msg, sizeof(Message));
        } catch (SocketException e) {
            VLTRACE(Trace::LEVEL_VL_WARN, "ClusterDiscoveryService could not send "
                "user message (\"%s\").\n", e.GetMsgA());
            retval++;
        }
    }
    this->critSect.Unlock();

    return retval;
}

/*
 * vislib::net::ClusterDiscoveryService::SendUserMessage
 */
UINT vislib::net::ClusterDiscoveryService::SendUserMessage(
        const PeerHandle& hPeer, const UINT32 msgType, const void *msgBody, 
        const SIZE_T msgSize) {
    Message msg;                // The datagram we are going to send.
    UINT retval = 0;            // Retval, 0 in case of success.

    this->prepareUserMessage(msg, msgType, msgBody, msgSize);
    ASSERT(this->userMsgSocket.IsValid());
    
    this->critSect.Lock();

    if (!this->isValidPeerHandle(hPeer)) {
        this->critSect.Unlock();
        throw IllegalParamException("hPeer", __FILE__, __LINE__);
    }

    try {
        this->userMsgSocket.Send(hPeer->discoveryAddr, &msg, sizeof(Message));
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_WARN, "ClusterDiscoveryService could not send "
            "user message (\"%s\").\n", e.GetMsgA());
        retval++;
    }
    this->critSect.Unlock();

    return retval;
}


/*
 * vislib::net::ClusterDiscoveryService::Start
 */
void vislib::net::ClusterDiscoveryService::Start(void) {
    this->senderThread.Start(this);
    this->receiverThread.Start(this);
}


/*
 * vislib::net::ClusterDiscoveryService::Stop
 */
bool vislib::net::ClusterDiscoveryService::Stop(const bool noWait) {
    try {
        // Note: Receiver must be stopped first, in order to prevent shutdown
        // deadlock when waiting for the last message.
        this->receiverThread.TryTerminate(false);
        this->senderThread.TryTerminate(!noWait);

        return true;
    } catch (sys::SystemException e) {
        VLTRACE(Trace::LEVEL_VL_ERROR, "Stopping discovery threads failed. The "
            "error code is %d (\"%s\").\n", e.GetErrorCode(), e.GetMsgA());
        return false;
    }
}


/*
 * vislib::net::ClusterDiscoveryService::operator []
 */
vislib::net::IPEndPoint vislib::net::ClusterDiscoveryService::operator [](
        const PeerHandle& hPeer) const {
    IPEndPoint retval;
    
    this->critSect.Lock();

    if (!this->isValidPeerHandle(hPeer)) {
        this->critSect.Unlock();
        throw IllegalParamException("hPeer", __FILE__, __LINE__);
    } else {
        retval = hPeer->address;
        this->critSect.Unlock();
    }

    return retval;
}


////////////////////////////////////////////////////////////////////////////////
// Begin of nested class Sender

/*
 * vislib::net::ClusterDiscoveryService::Sender::Sender
 */
vislib::net::ClusterDiscoveryService::Sender::Sender(void) : isRunning(true) {
}


/*
 * vislib::net::ClusterDiscoveryService::Sender::~Sender
 */
vislib::net::ClusterDiscoveryService::Sender::~Sender(void) {
}


/*
 * vislib::net::ClusterDiscoveryService::Sender::Run
 */
DWORD vislib::net::ClusterDiscoveryService::Sender::Run(void *discSvc) {
    Message request;                // The UDP datagram we send.
    ClusterDiscoveryService *cds    // The discovery service we work for.
        = static_cast<ClusterDiscoveryService *>(discSvc);
                /** The socket used for the broadcast. */
    Socket socket;                  // Socket used for broadcast.
    
    ASSERT(cds != NULL);

    // Assert expected memory layout of messages.
    ASSERT(sizeof(request) == MAX_USER_DATA + 2 * sizeof(UINT32));
    ASSERT(reinterpret_cast<BYTE *>(&(request.senderBody)) 
       == reinterpret_cast<BYTE *>(&request) + 2 * sizeof(UINT32));

    /* Prepare the socket. */
    try {
        Socket::Startup();
        socket.Create(cds->bindAddr, Socket::TYPE_DGRAM, Socket::PROTOCOL_UDP);
        socket.SetBroadcast(true);
        //socket.SetLinger(true, 0);  // Force hard close.
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_ERROR, "Discovery sender thread could not "
            "create its. The error code is %d (\"%s\").\n", e.GetErrorCode(),
            e.GetMsgA());
        return e.GetErrorCode();
    }

    /* Prepare our request for initial broadcasting. */
    request.magicNumber = MAGIC_NUMBER;
    request.msgType = MSG_TYPE_IAMHERE;
    request.senderBody.sockAddr = cds->responseAddr;
#if (_MSC_VER >= 1400)
    ::strncpy_s(request.senderBody.name, MAX_NAME_LEN, cds->name, 
        MAX_NAME_LEN);
#else /* (_MSC_VER >= 1400) */
    ::strncpy(request.senderBody.name, cds->name.PeekBuffer(), MAX_NAME_LEN);
#endif /* (_MSC_VER >= 1400) */

    VLTRACE(Trace::LEVEL_VL_INFO, "The discovery sender thread is starting ...\n");

    /* Send the initial "immediate alive request" message. */
    try {
        socket.Send(cds->bcastAddr, &request, sizeof(Message));
        VLTRACE(Trace::LEVEL_VL_INFO, "Discovery service sent MSG_TYPE_IAMHERE "
            "to %s.\n", cds->bcastAddr.ToStringA().PeekBuffer());

        /*
         * If the discovery service is configured not to be member of the 
         * cluster, but to only search other members, the sender thread can 
         * leave after the first request sent above.
         * Otherwise, the thread should wait for the normal request interval
         * time before really starting.
         */
        if (cds->isObserver) {
            VLTRACE(Trace::LEVEL_VL_INFO, "Discovery service is leaving request "
                "thread as it is only discovering other nodes.\n");
            return 0;
        } else {
            sys::Thread::Sleep(cds->requestInterval);
        }

    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_WARN, "A socket error occurred in the "
            "discovery sender thread. The error code is %d (\"%s\").\n",
            e.GetErrorCode(), e.GetMsgA());
    } catch (...) {
        VLTRACE(Trace::LEVEL_VL_ERROR, "The discovery sender caught an "
            "unexpected exception.\n");
        return -1;
    }

    /* Change our request for alive broadcasting. */
    request.msgType = MSG_TYPE_IAMALIVE;

    while (this->isRunning) {
        try {
            /* Broadcast request. */
            cds->prepareRequest();
            socket.Send(cds->bcastAddr, &request, sizeof(Message));
            VLTRACE(Trace::LEVEL_VL_INFO, "Discovery service sent "
                "MSG_TYPE_IAMALIVE to %s.\n", 
                cds->bcastAddr.ToStringA().PeekBuffer());

            sys::Thread::Sleep(cds->requestInterval);
        } catch (SocketException e) {
            VLTRACE(Trace::LEVEL_VL_WARN, "A socket error occurred in the "
                "discovery sender thread. The error code is %d (\"%s\").\n",
                e.GetErrorCode(), e.GetMsgA());
        } catch (...) {
            VLTRACE(Trace::LEVEL_VL_ERROR, "The discovery sender caught an "
                "unexpected exception.\n");
            return -1;
        }
    } /* end while (this->isRunning) */

    /* Clean up. */
    try {
        /* Now inform all other nodes, that we are out. */
        request.msgType = MSG_TYPE_SAYONARA;
        socket.Send(cds->bcastAddr, &request, sizeof(Message));
        VLTRACE(Trace::LEVEL_VL_INFO, "Discovery service sent MSG_TYPE_SAYONARA to "
            "%s.\n", cds->bcastAddr.ToStringA().PeekBuffer());
        
        Socket::Cleanup();
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_ERROR, "Socket cleanup failed in the discovery "
            "request thread. The error code is %d (\"%s\").\n", 
            e.GetErrorCode(), e.GetMsgA());
        return e.GetErrorCode();
    }

    return 0;
}


/*
 * vislib::net::ClusterDiscoveryService::Sender::Terminate
 */
bool vislib::net::ClusterDiscoveryService::Sender::Terminate(void) {
    // TODO: Should perhaps be protected by crit sect.
    this->isRunning = false;
    return true;
}

// End of nested class Sender
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Begin of nested class Receiver

/*
 * vislib::net::ClusterDiscoveryService::Receiver::Receiver
 */
vislib::net::ClusterDiscoveryService::Receiver::Receiver(void) 
        : isRunning(true) {
}


/*
 * vislib::net::ClusterDiscoveryService::Receiver::~Receiver
 */
vislib::net::ClusterDiscoveryService::Receiver::~Receiver(void) {
}


/*
 * vislib::net::ClusterDiscoveryService::Receiver::Run
 */
DWORD vislib::net::ClusterDiscoveryService::Receiver::Run(void *discSvc) {
    IPEndPoint peerAddr;            // Receives address of communication peer.
    PeerNode peerNode;              // The peer node to register in our list.
    Message msg;                    // Receives the request messages.
    ClusterDiscoveryService *cds    // The discovery service we work for.
        = static_cast<ClusterDiscoveryService *>(discSvc);

    ASSERT(cds != NULL);

    // Assert expected message memory layout.
    ASSERT(sizeof(msg) == MAX_USER_DATA + 2 * sizeof(UINT32));
    ASSERT(reinterpret_cast<BYTE *>(&(msg.senderBody)) 
       == reinterpret_cast<BYTE *>(&msg) + 2 * sizeof(UINT32));

    /* 
     * Prepare a datagram socket listening for requests on the specified 
     * adapter and port. 
     */
    try {
        Socket::Startup();
        this->socket.Create(cds->bindAddr, Socket::TYPE_DGRAM, 
            Socket::PROTOCOL_UDP);
        this->socket.SetBroadcast(true);
        // TODO: Make shared socket use configurable (security issue!).
        this->socket.SetExclusiveAddrUse(false);
        this->socket.SetReuseAddr(true);
        this->socket.Bind(cds->bindAddr);
        //socket.SetLinger(false, 0);     // Force hard close.
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_ERROR, "Discovery receiver thread could not "
            "create its socket and bind it to the requested address. The "
            "error code is %d (\"%s\").\n", e.GetErrorCode(), e.GetMsgA());
        return e.GetErrorCode();
    }

    /* Register myself as known node first (observer does not know itself). */
    if (!cds->isObserver) {
        IPEndPoint discoveryAddr(cds->responseAddr);
        discoveryAddr.SetPort(cds->bindAddr.GetPort());
        cds->addPeerNode(discoveryAddr, cds->responseAddr);
        // TODO: Using the response address as discovery address is hugly.
    }

    VLTRACE(Trace::LEVEL_VL_INFO, "The discovery receiver thread is starting ...\n");

    while (this->isRunning) {
        try {

            /* Wait for next message. */
            this->socket.Receive(peerAddr, &msg, sizeof(Message));

            if (msg.magicNumber == MAGIC_NUMBER) {
                /* Message OK, look for its content. */

                if ((msg.msgType == MSG_TYPE_IAMALIVE) 
                        && (cds->name.Equals(msg.senderBody.name))) {
                    /* Got a discovery request for own cluster. */
                    VLTRACE(Trace::LEVEL_VL_INFO, "Discovery service received "
                        "MSG_TYPE_IAMALIVE from %s.\n", 
                        peerAddr.ToStringA().PeekBuffer());
                    
                    /* Add peer to local list, if not yet known. */
                    cds->addPeerNode(peerAddr, IPEndPoint(
                        msg.senderBody.sockAddr));

                } else if ((msg.msgType == MSG_TYPE_IAMHERE) 
                        && (cds->name.Equals(msg.senderBody.name))) {
                    /* 
                     * Get an initial discovery request. This triggers an 
                     * immediate alive message, but without adding the sender to
                     * the cluster.
                     *
                     * Note: Nodes sending the MSG_TYPE_IAMHERE message must
                     * be added to the list of known nodes, because nodes in 
                     * observer mode also send MSG_TYPE_IAMHERE to request an
                     * immediate response of all running nodes.
                     */
                    VLTRACE(Trace::LEVEL_VL_INFO, "Discovery service received "
                        "MSG_TYPE_IAMHERE from %s.\n", 
                        peerAddr.ToStringA().PeekBuffer());

                    if (!cds->isObserver) {
                        /* Observers must not send alive messages. */
                        peerAddr.SetPort(cds->bindAddr.GetPort());
                        ASSERT(msg.magicNumber == MAGIC_NUMBER);
                        msg.msgType = MSG_TYPE_IAMALIVE;
                        msg.senderBody.sockAddr = cds->responseAddr;
                        this->socket.Send(peerAddr, &msg, sizeof(Message));
                        VLTRACE(Trace::LEVEL_VL_INFO, "Discovery service sent "
                            "immediate MSG_TYPE_IAMALIVE answer to %s.\n",
                            peerAddr.ToStringA().PeekBuffer());
                    }

                } else if ((msg.msgType == MSG_TYPE_SAYONARA)
                        && (cds->name.Equals(msg.senderBody.name))) {
                    /* Got an explicit disconnect. */
                    VLTRACE(Trace::LEVEL_VL_INFO, "Discovery service received "
                        "MSG_TYPE_SAYONARA from %s.\n",
                        peerAddr.ToStringA().PeekBuffer());

                    cds->removePeerNode(IPEndPoint(msg.senderBody.sockAddr));

                } else if (msg.msgType >= MSG_TYPE_USER) {
                    /* Received user message. */
                    cds->fireUserMessage(peerAddr, msg.msgType, msg.userData);
                }
            } /* end if (response.magicNumber == MAGIC_NUMBER) */

        } catch (SocketException e) {
            VLTRACE(Trace::LEVEL_VL_WARN, "A socket error occurred in the "
                "discovery receiver thread. The error code is %d (\"%s\").\n",
                e.GetErrorCode(), e.GetMsgA());
        } catch (...) {
            VLTRACE(Trace::LEVEL_VL_ERROR, "The discovery receiver caught an "
                "unexpected exception.\n");
            return -1;
        }

    } /* end while (this->isRunning) */

    try {
        Socket::Cleanup();
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_ERROR, "Socket cleanup failed in the discovery "
            "receiver thread. The error code is %d (\"%s\").\n", 
            e.GetErrorCode(), e.GetMsgA());
        return e.GetErrorCode();
    }

    return 0;
}


/*
 * vislib::net::ClusterDiscoveryService::Receiver::Terminate
 */
bool vislib::net::ClusterDiscoveryService::Receiver::Terminate(void) {
    // TODO: Should perhaps be protected by crit sect.
    this->isRunning = false;
    this->socket.Close();       // Unlock blocking socket operations.
    return true;
}

// End of nested class Receiver
////////////////////////////////////////////////////////////////////////////////


/*
 * vislib::net::ClusterDiscoveryService::addPeerNode
 */
void vislib::net::ClusterDiscoveryService::addPeerNode(
        const IPEndPoint& discoveryAddr, const IPEndPoint& address) {
    INT_PTR idx = 0;            // Index of possible duplicate.
    PeerHandle hPeer = NULL;    // Old or new peer node handle.

    this->critSect.Lock();
    
    if ((idx = this->peerFromAddress(address)) >= 0) {
        /* Already known, reset disconnect chance. */
        VLTRACE(Trace::LEVEL_VL_INFO, "Peer node %s is already known.\n", 
            address.ToStringA().PeekBuffer());
        hPeer = this->peerNodes[static_cast<INT>(idx)];

    } else {
        /* Not known, so add it and fire event. */
        VLTRACE(Trace::LEVEL_VL_INFO, "I learned about node %s.\n",
            address.ToStringA().PeekBuffer());
        hPeer = new PeerNode;
        hPeer->address = address;
        hPeer->discoveryAddr = IPEndPoint(discoveryAddr, 
            this->bindAddr.GetPort());

        this->peerNodes.Append(hPeer);

        ListenerList::Iterator it = const_cast<ClusterDiscoveryService *>(
            this)->listeners.GetIterator();
        while (it.HasNext()) {
            it.Next()->OnNodeFound(*this, hPeer);
        }
    }

    hPeer->cntResponseChances = this->cntResponseChances + 1;
    this->critSect.Unlock();
}


/*
 * vislib::net::ClusterDiscoveryService::fireUserMessage
 */
void vislib::net::ClusterDiscoveryService::fireUserMessage(
        const IPEndPoint& sender, const UINT32 msgType, 
        const BYTE *msgBody) const {
    INT_PTR idx = 0;        // Index of sender PeerNode.

    this->critSect.Lock();
    
    if ((idx = this->peerFromDiscoveryAddr(sender)) >= 0) {
   
        // TODO: We need a const iterator.
        ListenerList::Iterator it = const_cast<ClusterDiscoveryService *>(
            this)->listeners.GetIterator();
        while (it.HasNext()) {
            // TODO: Should avoid cast.
            it.Next()->OnUserMessage(*this, 
                this->peerNodes[static_cast<INT>(idx)], msgType, msgBody);
        }
    }

    this->critSect.Unlock();
}


/*
 * vislib::net::ClusterDiscoveryService::isValidPeerHandle
 */
bool vislib::net::ClusterDiscoveryService::isValidPeerHandle(
        const PeerHandle& hPeer) const {
    //struct sockaddr address = static_cast<sockaddr>(hPeer->address);
    //// TODO: This is not IPv6 compatible!
    //return (reinterpret_cast<int&>(address) != 0);

    // TODO: I am not sure whether this is the intended behaviour.
    return !hPeer->address.GetIPAddress6().IsUnspecified();
}


/*
 * vislib::net::ClusterDiscoveryService::peerFromAddress
 */
INT_PTR vislib::net::ClusterDiscoveryService::peerFromAddress(
        const IPEndPoint& addr) const {
    // TODO: Think of faster solution.  
    INT_PTR retval = -1;

    //this->critSect.Lock();
    for (SIZE_T i = 0; i < this->peerNodes.Count(); i++) {
        if (this->peerNodes[i]->address == addr) {
            retval = i;
            break;
        }
    }
    //this->critSect.Unlock();

    return retval;
}


/*
 * vislib::net::ClusterDiscoveryService::peerFromDiscoveryAddr
 */
INT_PTR vislib::net::ClusterDiscoveryService::peerFromDiscoveryAddr(
        const IPEndPoint& addr) const {
    // TODO: Think of faster solution.  
    INT_PTR retval = -1;
    IPAddress6 peerIP = addr.GetIPAddress6();

    //this->critSect.Lock();
    for (SIZE_T i = 0; i < this->peerNodes.Count(); i++) {
        if (this->peerNodes[i]->discoveryAddr.GetIPAddress6() == peerIP) {
            retval = i;
            break;
        }
    }
    //this->critSect.Unlock();

    return retval;
}


/*
 * vislib::net::ClusterDiscoveryService::prepareRequest
 */
void vislib::net::ClusterDiscoveryService::prepareRequest(void) {
    this->critSect.Lock();
    struct sockaddr invalidAddr;
    reinterpret_cast<int&>(invalidAddr) = 0;

    for (SIZE_T i = 0; i < this->peerNodes.Count(); i++) {
        if (this->peerNodes[i]->cntResponseChances == 0) {
            
            /* Fire event. */
            // TODO: We need a const iterator.
            ListenerList::Iterator it = const_cast<ClusterDiscoveryService *>(
                this)->listeners.GetIterator();
            while (it.HasNext()) {
                it.Next()->OnNodeLost(*this, this->peerNodes[i],
                    ClusterDiscoveryListener::LOST_IMLICITLY);
            }

            /* Make the handle invalid. */
            this->peerNodes[i]->address = invalidAddr;

            /* Remove peer from list. */
            this->peerNodes.Erase(i);
            i--;

        } else {
            /* Decrement response chance for upcoming request. */
            this->peerNodes[i]->cntResponseChances--;
        }
    }

    this->critSect.Unlock();
}

/*
 * vislib::net::ClusterDiscoveryService::prepareUserMessage
 */
void vislib::net::ClusterDiscoveryService::prepareUserMessage(
        Message& outMsg, const UINT32 msgType, const void *msgBody, 
        const SIZE_T msgSize) {

    /* Check parameters. */
    if (msgType < MSG_TYPE_USER) {
        throw IllegalParamException("msgType", __FILE__, __LINE__);
    }
    if (msgBody == NULL) {
        throw IllegalParamException("msgBody", __FILE__, __LINE__);
    }
    if (msgSize > MAX_USER_DATA) {
        throw IllegalParamException("msgSize", __FILE__, __LINE__);
    }

    // Assert some stuff.
    ASSERT(sizeof(outMsg) == MAX_USER_DATA + 2 * sizeof(UINT32));
    ASSERT(reinterpret_cast<BYTE *>(&(outMsg.userData)) 
       == reinterpret_cast<BYTE *>(&outMsg) + 2 * sizeof(UINT32));
    ASSERT(msgType >= MSG_TYPE_USER);
    ASSERT(msgBody != NULL);
    ASSERT(msgSize <= MAX_USER_DATA);

    /* Prepare the message. */
    outMsg.magicNumber = MAGIC_NUMBER;
    outMsg.msgType = msgType;
    ::ZeroMemory(outMsg.userData, MAX_USER_DATA);
    ::memcpy(outMsg.userData, msgBody, msgSize);

    /* Lazy creation of our socket. */
    if (!this->userMsgSocket.IsValid()) {
        this->userMsgSocket.Create(Socket::FAMILY_INET, Socket::TYPE_DGRAM,
            Socket::PROTOCOL_UDP);
        //this->userMsgSocket.SetLinger(false, 0);    // Force hard close.
    }
}


/*
 * vislib::net::ClusterDiscoveryService::removePeerNode
 */
void vislib::net::ClusterDiscoveryService::removePeerNode(
        const IPEndPoint& address) {
    INT_PTR idx = 0;
    struct sockaddr invalidAddr;
    reinterpret_cast<int&>(invalidAddr) = 0;
    
    this->critSect.Lock();

    if ((idx = this->peerFromAddress(address)) >= 0) {
        PeerHandle hPeer = this->peerNodes[static_cast<INT>(idx)];

        /* Fire event. */
        // TODO: We need a const iterator.
        ListenerList::Iterator it = const_cast<ClusterDiscoveryService *>(
            this)->listeners.GetIterator();
        while (it.HasNext()) {
            it.Next()->OnNodeLost(*this, hPeer, 
                ClusterDiscoveryListener::LOST_EXPLICITLY);
        }

        /* Invalidate handle. */
        hPeer->address = invalidAddr;

        /* Remove peer from list. */
        this->peerNodes.Erase(idx);
    }

    this->critSect.Unlock();
}


/*
 * vislib::net::ClusterDiscoveryService::MAGIC_NUMBER
 */
const UINT32 vislib::net::ClusterDiscoveryService::MAGIC_NUMBER 
    = (static_cast<UINT32>('v') << 0) | (static_cast<UINT32>('c') << 8)
    | (static_cast<UINT32>('d') << 16) | (static_cast<UINT32>('s') << 24);


/*
 * vislib::net::ClusterDiscoveryService::MSG_TYPE_IAMALIVE
 */
const UINT32 vislib::net::ClusterDiscoveryService::MSG_TYPE_IAMALIVE = 2;


/*
 * vislib::net::ClusterDiscoveryService::MSG_TYPE_IAMHERE
 */
const UINT32 vislib::net::ClusterDiscoveryService::MSG_TYPE_IAMHERE = 1;


/*
 * vislib::net::ClusterDiscoveryService::MSG_TYPE_SAYONARA
 */
const UINT32 vislib::net::ClusterDiscoveryService::MSG_TYPE_SAYONARA = 3;
