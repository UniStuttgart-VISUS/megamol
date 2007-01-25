/*
 * ClusterDiscoveryService.cpp
 *
 * Copyright (C) 2006 -2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/ClusterDiscoveryService.h"

#include "vislib/assert.h"
#include "vislib/SocketException.h"
#include "vislib/Trace.h"


/*
 * vislib::net::ClusterDiscoveryService::MSG_TYPE_USER
 */
const UINT16 vislib::net::ClusterDiscoveryService::MSG_TYPE_USER = 16;


/*
 * vislib::net::ClusterDiscoveryService::ClusterDiscoveryService
 */
vislib::net::ClusterDiscoveryService::ClusterDiscoveryService(
        const StringA& name, const SocketAddress& bindAddr, 
        const IPAddress& bcastAddr, const SocketAddress& responseAddr)
        : bcastAddr(SocketAddress::FAMILY_INET, bcastAddr, bindAddr.GetPort()), 
        bindAddr(bindAddr), 
        expectedResponseCnt(2),      // TODO
        responseAddr(responseAddr), 
        requestInterval(3 * 1000),  // TODO
        name(name), 
        requester(NULL), 
        responder(NULL), 
        requestThread(NULL), 
        responseThread(NULL),
        timeoutReceive(2 * 1000),   // TODO
        timeoutSend(5 * 1000) {     // TODO
    this->name.Truncate(MAX_USER_DATA);

    this->peerNodes.Resize(0);  // TODO: Remove alloc crowbar!
}


/*
 * vislib::net::ClusterDiscoveryService::~ClusterDiscoveryService
 */
vislib::net::ClusterDiscoveryService::~ClusterDiscoveryService(void) {
    if (this->requestThread != NULL) {
        try {
            this->requestThread->Terminate(false);
            SAFE_DELETE(this->requester);
            SAFE_DELETE(this->requestThread);
        } catch (...) {
            TRACE(Trace::LEVEL_WARN, "The discovery requester thread could "
                "not be successfully terminated.\n");
        }
    }

    if (this->responseThread != NULL) {
        try {
            this->responseThread->Terminate(false);
            SAFE_DELETE(this->responder);
            SAFE_DELETE(this->responseThread);
        } catch (...) {
            TRACE(Trace::LEVEL_WARN, "The discovery responder thread could "
                "not be successfully terminated.\n");
        }
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
 * vislib::net::ClusterDiscoveryService::RemoveListener
 */
void vislib::net::ClusterDiscoveryService::RemoveListener(
        ClusterDiscoveryListener *listener) {
    ASSERT(listener != NULL);

    this->critSect.Lock();
    this->listeners.Remove(listener);
    this->critSect.Unlock();
}

/*
 * vislib::net::ClusterDiscoveryService::Start
 */
bool vislib::net::ClusterDiscoveryService::Start(void) {
    /* Prepare and start the thread generating the discovery requests. */
    this->requester = new Requester(*this);
    this->requestThread = new vislib::sys::Thread(this->requester);
    this->requestThread->Start();

    /* Prepare the response thread. */
    this->responder = new Responder(*this);
    this->responseThread = new vislib::sys::Thread(this->responder);
    this->responseThread->Start();

    // TODO: Remove debug crowbaring
    //this->requestThread->Join();
    //sys::Thread::Sleep(5 * 1000);
    //this->requestThread->Terminate(false);
    //sys::Thread::Sleep(5 * 1000);
    //this->requestThread->Join();
    return true;
}


/*
 * vislib::net::ClusterDiscoveryService::Stop
 */
bool vislib::net::ClusterDiscoveryService::Stop(void) {
    try {
        this->requestThread->Terminate(false);
        this->responseThread->Terminate(false);
        return true;
    } catch (sys::SystemException e) {
        TRACE(Trace::LEVEL_ERROR, "Stopping discovery threads failed. The "
            "error code is %d (\"%s\").\n", e.GetErrorCode(), e.GetMsgA());
        return false;
    }
}


////////////////////////////////////////////////////////////////////////////////
// Begin of nested class Requester

/*
 * vislib::net::ClusterDiscoveryService::Requester::Requester
 */
vislib::net::ClusterDiscoveryService::Requester::Requester(
        ClusterDiscoveryService& cds) : cds(cds), isRunning(true) {
}


/*
 * vislib::net::ClusterDiscoveryService::Requester::~Requester
 */
vislib::net::ClusterDiscoveryService::Requester::~Requester(void) {
}


/*
 * vislib::net::ClusterDiscoveryService::Requester::Run
 */
DWORD vislib::net::ClusterDiscoveryService::Requester::Run(
        const void *reserved) {
    SocketAddress peerAddr;
    PeerNode peerNode;
    Message request;
    Message response;
    Socket socket;

    // Assert expected memory layout of messages.
    ASSERT(sizeof(request) == MAX_USER_DATA + 4);
    ASSERT(reinterpret_cast<BYTE *>(&(request.name)) 
       == reinterpret_cast<BYTE *>(&request) + 4);

    /* Prepare the socket. */
    try {
        Socket::Startup();
        socket.Create(Socket::FAMILY_INET, Socket::TYPE_DGRAM, 
            Socket::PROTOCOL_UDP);
        socket.SetBroadcast(true);
    } catch (SocketException e) {
        TRACE(Trace::LEVEL_ERROR, "Discovery requester thread could not "
            "create its. The error code is %d (\"%s\").\n", e.GetErrorCode(),
            e.GetMsgA());
        return 1;
    }

    /* Prepare our request for multiple broadcasting. */
    request.magicNumber = MAGIC_NUMBER;
    request.msgType = MSG_TYPE_DISCOVERY_REQUEST;
#if (_MSC_VER >= 1400)
    ::strncpy_s(request.name, MAX_USER_DATA, this->cds.name, MAX_USER_DATA);
#else /* (_MSC_VER >= 1400) */
    ::strncpy(request.name, this->cds.name.PeekBuffer(), MAX_USER_DATA);
#endif /* (_MSC_VER >= 1400) */

    while (this->isRunning) {
        try {

            /* Broadcast request. */
            socket.Send(&request, sizeof(Message), this->cds.bcastAddr);
            TRACE(Trace::LEVEL_INFO, "Request thread sent "
                "MSG_TYPE_DISCOVERY_REQUEST to %s.\n",
                this->cds.bcastAddr.ToStringA().PeekBuffer());

            for (UINT i = 0; i < this->cds.expectedResponseCnt; i++) {
                socket.Receive(&response, sizeof(Message), 
                    this->cds.timeoutReceive, peerAddr);
                if (response.magicNumber == MAGIC_NUMBER) {
                    /* Message OK, look for its content. */

                    if (response.msgType == MSG_TYPE_DISCOVERY_RESPONSE) {
                        TRACE(Trace::LEVEL_INFO, "Request thread received "
                            "MSG_TYPE_DISCOVERY_RESPONSE from from %s.\n",
                            peerAddr.ToStringA().PeekBuffer());

                        /* Add peer to local list, if not yet known. */
                        peerNode.address = response.responseAddr;
                        this->cds.addPeerNode(peerNode);
                    }

                } /* end if (response.magicNumber == MAGIC_NUMBER) */
            } /* end for (UINT i = 0; i < ... */

            sys::Thread::Sleep(this->cds.requestInterval);

        } catch (SocketException e) {
            TRACE(Trace::LEVEL_WARN, "A socket error occurred in the "
                "discovery request thread. The error code is %d (\"%s\").\n",
                e.GetErrorCode(), e.GetMsgA());
        } catch (Exception e) {
            TRACE(Trace::LEVEL_WARN, "A discovery request could not be "
                "dispatched due to an unexpected exception (\"%s\").\n", 
                e.GetMsgA());
            return 2;
        } catch (...) {
            TRACE(Trace::LEVEL_WARN, "A discovery request could not be "
                "dispatched due to an unexpected exception.\n");
            return 3;
        }
    } /* end while (this->isRunning) */

    /* Clean up. */
    try {
        socket.Shutdown(Socket::BOTH);
        socket.Close();
        Socket::Cleanup();
    } catch (SocketException e) {
        TRACE(Trace::LEVEL_ERROR, "Socket cleanup failed in the discovery "
            "request thread. The error code is %d (\"%s\").\n", 
            e.GetErrorCode(), e.GetMsgA());
    }

    return 0;
}


/*
 * vislib::net::ClusterDiscoveryService::Requester::Terminate
 */
bool vislib::net::ClusterDiscoveryService::Requester::Terminate(void) {
    // TODO: Should perhaps be protected by crit sect.
    this->isRunning = false;
    return true;
}

// End of nested class Requester
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Begin of nested class Responder

/*
 * vislib::net::ClusterDiscoveryService::Responder::Responder
 */
vislib::net::ClusterDiscoveryService::Responder::Responder(
        ClusterDiscoveryService& cds) : cds(cds), isRunning(true) {
}


/*
 * vislib::net::ClusterDiscoveryService::Responder::~Responder
 */
vislib::net::ClusterDiscoveryService::Responder::~Responder(void) {
}


/*
 * vislib::net::ClusterDiscoveryService::Responder::Run
 */
DWORD vislib::net::ClusterDiscoveryService::Responder::Run(
        const void *reserved) {
    SocketAddress peerAddr;     // Receives address of UDP communication peer.
    PeerNode peerNode;          // The peer node to register in our list.
    Message request;            // Receives the request messages.
    Message discoveryResponse;  // The discovery response we send.
    Socket socket;              // The datagram socket that is listening.

    // Assert expected message memory layout.
    ASSERT(sizeof(request) == MAX_USER_DATA + 4);
    ASSERT(reinterpret_cast<BYTE *>(&(request.name)) 
       == reinterpret_cast<BYTE *>(&request) + 4);

    /* 
     * Prepare a datagram socket listening for requests on the specified 
     * adapter and port. 
     */
    try {
        Socket::Startup();
        socket.Create(Socket::FAMILY_INET, Socket::TYPE_DGRAM, 
            Socket::PROTOCOL_UDP);
        socket.Bind(this->cds.bindAddr);
    } catch (SocketException e) {
        TRACE(Trace::LEVEL_ERROR, "Discovery responder thread could not "
            "create its socket and bind it to the requested address. The "
            "error code is %d (\"%s\").\n", e.GetErrorCode(), e.GetMsgA());
        return 1;
    }

    /* Prepare a discovery response for multiple use. */
    discoveryResponse.magicNumber = MAGIC_NUMBER;
    discoveryResponse.msgType = MSG_TYPE_DISCOVERY_RESPONSE;
    discoveryResponse.responseAddr = this->cds.responseAddr;

    while (this->isRunning) {
        try {

            /* Wait for next message. */
            // Note: This receive operation must be timeouted in order to allow
            // the thread to be terminated in a controlled manner.
            socket.Receive(&request, sizeof(Message)/*, this->cds.timeoutReceive*/, 
                peerAddr);

            if (request.magicNumber == MAGIC_NUMBER) {
                /* Message OK, look for its content. */

                if ((request.msgType == MSG_TYPE_DISCOVERY_REQUEST) 
                        && (this->cds.name.Compare(request.name))) {
                    /* Got a discovery request for own cluster. */
                    TRACE(Trace::LEVEL_INFO, "Response thread received "
                        "MSG_TYPE_DISCOVERY_REQUEST from %s.\n", 
                        peerAddr.ToStringA().PeekBuffer());
                    
                    socket.Send(&discoveryResponse, sizeof(Message), 
                        this->cds.timeoutSend, peerAddr);
                    TRACE(Trace::LEVEL_INFO, "Response thread sent "
                        "MSG_TYPE_DISCOVERY_RESPONSE to %s.\n", 
                        peerAddr.ToStringA().PeekBuffer());

                    /* Add peer to local list, if not yet known. */
                    // TODO: This is nonsense. The message currently does not
                    // contain a response address. Consider inserting one.
                    //peerNode.address = request.responseAddr;
                    //this->cds.addPeerNode(peerNode);
                }
            } /* end if (response.magicNumber == MAGIC_NUMBER) */

        } catch (SocketException e) {
            TRACE(Trace::LEVEL_WARN, "A socket error occurred in the "
                "discovery response thread. The error code is %d (\"%s\").\n",
                e.GetErrorCode(), e.GetMsgA());
        } catch (Exception e) {
            TRACE(Trace::LEVEL_WARN, "A discovery request could not be "
                "answered due to an unexpected exception (\"%s\").\n", 
                e.GetMsgA());
            return 2;
        } catch (...) {
            TRACE(Trace::LEVEL_WARN, "A discovery request could not be "
                "answered due to an unexpected exception.\n");
            return 3;
        }

    } /* end while (this->isRunning) */

    try {
        socket.Shutdown(Socket::BOTH);
        socket.Close();
        Socket::Cleanup();
    } catch (SocketException e) {
        TRACE(Trace::LEVEL_ERROR, "Socket cleanup failed in the discovery "
            "response thread. The error code is %d (\"%s\").\n", 
            e.GetErrorCode(), e.GetMsgA());
    }

    return 0;
}


/*
 * vislib::net::ClusterDiscoveryService::Responder::Terminate
 */
bool vislib::net::ClusterDiscoveryService::Responder::Terminate(void) {
    // TODO: Should perhaps be protected by crit sect.
    this->isRunning = false;
    return true;
}

// End of nested class Responder
////////////////////////////////////////////////////////////////////////////////


/*
 * vislib::net::ClusterDiscoveryService::addPeerNode
 */
void vislib::net::ClusterDiscoveryService::addPeerNode(const PeerNode& node) {
    this->critSect.Lock();
    
    if (!this->peerNodes.Contains(node)) {
        this->peerNodes.Append(node);

        /* Fire event. */
        // TODO: We need a const iterator.
        ListenerList::Iterator it = const_cast<ClusterDiscoveryService *>(
            this)->listeners.GetIterator();
        while (it.HasNext()) {
            it.Next()->OnNodeFound(*this, node.address);
        }
    }

    this->critSect.Unlock();
}


/*
 * vislib::net::ClusterDiscoveryService::fireNodeFound
 */
//void vislib::net::ClusterDiscoveryService::fireNodeFound(
//        const PeerNode& node) const {
//    this->critSect.Lock();
//    
//    // TODO: We need a const iterator.
//    ListenerList::Iterator it = const_cast<ClusterDiscoveryService *>(
//        this)->listeners.GetIterator();
//    while (it.HasNext()) {
//        it.Next()->OnNodeFound(*this, node.address);
//    }
//
//    this->critSect.Unlock();
//}


/*
 * vislib::net::ClusterDiscoveryService::MAGIC_NUMBER
 */
const UINT16 vislib::net::ClusterDiscoveryService::MAGIC_NUMBER 
    = static_cast<UINT16>('v') << 8 | static_cast<UINT16>('l');


/*
 * vislib::net::ClusterDiscoveryService::MSG_TYPE_DISCOVERY_REQUEST
 */
const UINT16 vislib::net::ClusterDiscoveryService::MSG_TYPE_DISCOVERY_REQUEST
    = 1;


/*
 * vislib::net::ClusterDiscoveryService::MSG_TYPE_DISCOVERY_RESPONSE
 */
const UINT16 vislib::net::ClusterDiscoveryService::MSG_TYPE_DISCOVERY_RESPONSE 
    = 2;
