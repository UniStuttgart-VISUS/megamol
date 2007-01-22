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
 * vislib::net::ClusterDiscoveryService::ClusterDiscoveryService
 */
vislib::net::ClusterDiscoveryService::ClusterDiscoveryService(
        const StringA& name, const SocketAddress& bindAddr, 
        const IPAddress& bcastAddr)
        : bcastAddr(SocketAddress::FAMILY_INET, bcastAddr, bindAddr.GetPort()), name(name),
        requestThread(&requestFunc), responseThread(&responseFunc) {
}


/*
 * vislib::net::ClusterDiscoveryService::~ClusterDiscoveryService
 */
vislib::net::ClusterDiscoveryService::~ClusterDiscoveryService(void) {
    this->requestThread.Terminate(true);
    this->responseThread.Terminate(true);
}


/*
 * vislib::net::ClusterDiscoveryService::Start
 */
bool vislib::net::ClusterDiscoveryService::Start(void) {
    return this->requestThread.Start(this) && this->responseThread.Start(this);
}


/*
 * vislib::net::ClusterDiscoveryService::requestFunc
 */
DWORD vislib::net::ClusterDiscoveryService::requestFunc(const void *userData) {
    // TODO: 'userData' should not be const in runnable and RunnableFunc
    ClusterDiscoveryService *cds = static_cast<ClusterDiscoveryService *>(
        const_cast<void *>(userData));

    SocketAddress peerAddr;
    PeerNode peerNode;
    Message request;
    Message response;
    Socket socket;

    ASSERT(sizeof(request) == MAX_USER_DATA + 4);
    ASSERT(static_cast<void *>(&(request.name)) 
        == static_cast<void *>(&request + 4));

    try {
        socket.Create(Socket::FAMILY_INET, Socket::TYPE_DGRAM, 
            Socket::PROTOCOL_UDP);
        //socket.SetSndTimeo(3000);
    } catch (SocketException e) {
        TRACE(Trace::LEVEL_ERROR, "%d: %s", e.GetErrorCode(), e.GetMsgA());
        return -1;
    }

    request.magicNumber = MAGIC_NUMBER;
    request.msgType = MSG_TYPE_DISCOVERY_REQUEST;
#if (_MSC_VER >= 1400)
    ::strncpy_s(request.name, MAX_USER_DATA, cds->name, MAX_USER_DATA);
#else /* (_MSC_VER >= 1400) */
    ::strncpy(request.name, cds->name.PeekBuffer(), MAX_USER_DATA);
#endif /* (_MSC_VER >= 1400) */

    while (true) {
        try {
            socket.Send(&request, sizeof(Message), cds->bcastAddr);

            socket.Receive(&response, sizeof(Message), peerAddr);
            if ((response.magicNumber == MAGIC_NUMBER) 
                    && (response.msgType == MSG_TYPE_DISCVOERY_RESPONSE)) {
        cds->peerNodesCritSect.Lock();
        if (!cds->peerNodes.Contains(peerNode)) {
            cds->peerNodes.Append(peerNode);
        }
        cds->peerNodesCritSect.Unlock();                    
            }
        } catch (SocketException e) {
            TRACE(Trace::LEVEL_ERROR, "%d: %s", e.GetErrorCode(), e.GetMsgA());
        }

    }

    //msgHdr.msgType = MSG_TYPE_DISCOVERY_REQUEST;
    //msgHdr.msgSize = cds->name.Length() + 1;
    //try {
    //    socket.Send(&msgHdr, sizeof(MessageHeader), cds->bcastAddr);
    //    socket.Send(cds->name.PeekBuffer(), msgHdr.msgSize, cds->bcastAddr);
    //} catch (SocketException e) {
    //    TRACE(Trace::LEVEL_ERROR, "%d: %s", e.GetErrorCode(), e.GetMsgA());
    //}

    //try {
    //    socket.Receive(&msgHdr, sizeof(MessageHeader), peerAddr);
    //    peerNode.ipAddress = peerAddr.GetIPAddress();
    //
    //    cds->peerNodesCritSect.Lock();
    //    if (!cds->peerNodes.Contains(peerNode)) {
    //        cds->peerNodes.Append(peerNode);
    //    }
    //    cds->peerNodesCritSect.Unlock();    
    //} catch (SocketException e) {
    //    TRACE(Trace::LEVEL_ERROR, "%d: %s", e.GetErrorCode(), e.GetMsgA());
    //}

    return 0;
}


/*
 * vislib::net::ClusterDiscoveryService::responseFunc
 */
DWORD vislib::net::ClusterDiscoveryService::responseFunc(const void *userData) {
    // TODO: 'userData' should not be const in runnable and RunnableFunc
    ClusterDiscoveryService *cds = static_cast<ClusterDiscoveryService *>(
        const_cast<void *>(userData));

    //SocketAddress peerAddr;     // Peer node to respond to.
    //MessageHeader msgHdr;
    //Socket socket;              // The socket used for messages.

    //try {
    //    socket.Create(Socket::FAMILY_INET, Socket::TYPE_DGRAM, 
    //        Socket::PROTOCOL_UDP);
    //    //socket.Bind(SocketAddress(SocketAddress::FAMILY_INET, IPAddress("localhost"), 28181));
    //} catch (SocketException e) {
    //    TRACE(Trace::LEVEL_ERROR, "%d: %s", e.GetErrorCode(), e.GetMsgA());
    //    return -1;
    //}

    //try {
    //    while (true) {
    //        socket.Receive(&msgHdr, sizeof(MessageHeader), peerAddr);

    //        msgHdr.msgType = MSG_TYPE_DISCVOERY_RESPONSE;
    //        msgHdr.msgSize = 0;
    //        socket.Send(&response, sizeof(MessageHeader), peerAddr);
    //    }
    //} catch (SocketException e) {
    //    TRACE(Trace::LEVEL_ERROR, "Respond %d: %s", e.GetErrorCode(), e.GetMsgA());
    //}



    return 0;
}


/*
 * vislib::net::ClusterDiscoveryService::MAGIC_NUMBER
 */
const UINT16 vislib::net::ClusterDiscoveryService::MAGIC_NUMBER 
    = static_cast<UINT16>('v') << 16 | static_cast<UINT16>('l');


/*
 * vislib::net::ClusterDiscoveryService::MSG_TYPE_DISCOVERY_REQUEST
 */
const UINT16 vislib::net::ClusterDiscoveryService::MSG_TYPE_DISCOVERY_REQUEST
    = 1;


/*
 * vislib::net::ClusterDiscoveryService::MSG_TYPE_DISCVOERY_RESPONSE
 */
const UINT16 vislib::net::ClusterDiscoveryService::MSG_TYPE_DISCVOERY_RESPONSE 
    = 2;
