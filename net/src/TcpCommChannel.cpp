/*
 * TcpCommChannel.cpp
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/TcpCommChannel.h"

#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IPCommEndPoint.h"
#include "vislib/PeerDisconnectedException.h"
#include "vislib/SocketException.h"
#include "vislib/Trace.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::net::TcpCommChannel::FLAG_NODELAY
 */
const UINT64 vislib::net::TcpCommChannel::FLAG_NODELAY = 0x00000001;


/*
 * vislib::net::TcpCommChannel::FLAG_REUSE_ADDRESS
 */
const UINT64 vislib::net::TcpCommChannel::FLAG_REUSE_ADDRESS = 0x00000002;


/*
 * vislib::net::TcpCommChannel::Accept
 */
vislib::SmartRef<vislib::net::AbstractCommClientChannel> 
vislib::net::TcpCommChannel::Accept(void) {
    VLSTACKTRACE("TcpCommChannel::Accept", __FILE__, __LINE__);
    Socket socket = this->socket.Accept();
    // Ctor of TcpCommChannel will assign flags to actual socket.

    return SmartRef<AbstractCommClientChannel>(
        new TcpCommChannel(socket, this->flags), false);
}


/*
 * vislib::net::TcpCommChannel::Bind
 */
void vislib::net::TcpCommChannel::Bind(
        SmartRef<AbstractCommEndPoint> endPoint) {
    VLSTACKTRACE("TcpCommChannel::Bind", __FILE__, __LINE__);
    SmartRef<IPCommEndPoint> ep = endPoint.DynamicCast<IPCommEndPoint>();

    if (ep.IsNull()) {
        throw IllegalParamException("endPoint", __FILE__, __LINE__);
    }

    this->createSocket(static_cast<IPEndPoint>(*ep));   // Create lazily.
    this->socket.Bind(static_cast<IPEndPoint>(*ep));
}


/*
 * vislib::net::TcpCommChannel::Close
 */
void vislib::net::TcpCommChannel::Close(void) {
    // TODO: Find out how shutdown can be safely used.
    //try {
    //    this->socket.Shutdown();
    //} catch (SocketException e) {
    //    VLTRACE(Trace::LEVEL_VL_WARN, "SocketException when shutting down "
    //        "socket in TcpCommChannel::Close: %s\n", e.GetMsgA());
    //}
    try {
        this->socket.Close();
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_WARN, "SocketException when closing socket "
            "in TcpCommChannel::Close: %s\n", e.GetMsgA());
        throw e;
    }
}


/*
 * vislib::net::TcpCommChannel::Connect
 */
void vislib::net::TcpCommChannel::Connect(
        SmartRef<AbstractCommEndPoint> endPoint) {
    VLSTACKTRACE("TcpCommChannel::Connect", __FILE__, __LINE__);
    SmartRef<IPCommEndPoint> ep = endPoint.DynamicCast<IPCommEndPoint>();

    if (ep.IsNull()) {
        throw IllegalParamException("endPoint", __FILE__, __LINE__);
    }

    this->createSocket(static_cast<IPEndPoint>(*ep));   // Create lazily.
    this->socket.Connect(static_cast<IPEndPoint>(*ep));
}


/*
 * vislib::net::TcpCommChannel::GetLocalEndPoint
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> 
vislib::net::TcpCommChannel::GetLocalEndPoint(void) const {
    VLSTACKTRACE("TcpCommChannel::GetLocalEndPoint", __FILE__, __LINE__);
    return IPCommEndPoint::Create(this->socket.GetLocalEndPoint());
}


/*
 * vislib::net::TcpCommChannel::GetRemoteEndPoint
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint>
vislib::net::TcpCommChannel::GetRemoteEndPoint(void) const {
    VLSTACKTRACE("TcpCommChannel::GetRemoteEndPoint", __FILE__, __LINE__);
    return IPCommEndPoint::Create(this->socket.GetPeerEndPoint());
}


/*
 * vislib::net::TcpCommChannel::Listen
 */
void vislib::net::TcpCommChannel::Listen(const int backlog) {
    VLSTACKTRACE("TcpCommChannel::Listen", __FILE__, __LINE__);
    this->socket.Listen(backlog);
}


/*
 * vislib::net::TcpCommChannel::Receive
 */
SIZE_T vislib::net::TcpCommChannel::Receive(void *outData, 
        const SIZE_T cntBytes, const UINT timeout, const bool forceReceive) {
    VLSTACKTRACE("TcpCommChannel::Receive", __FILE__, __LINE__);
    SIZE_T retval = 0;
    
    if (cntBytes > 0) {
        retval = this->socket.Receive(outData, cntBytes, timeout, 0, 
            forceReceive);

        if ((retval == 0) || (forceReceive && (retval < cntBytes))) {
            throw PeerDisconnectedException(
                PeerDisconnectedException::FormatMessageForLocalEndpoint(
                this->socket.GetLocalEndPoint().ToStringW().PeekBuffer()), 
                __FILE__, __LINE__);
        }
    }

    return retval;
}


/*
 * vislib::net::TcpCommChannel::Send
 */
SIZE_T vislib::net::TcpCommChannel::Send(const void *data, 
        const SIZE_T cntBytes, const UINT timeout, const bool forceSend) {
    VLSTACKTRACE("TcpCommChannel::Send", __FILE__, __LINE__);
    return this->socket.Send(data, cntBytes, timeout, 0, forceSend);
}


/*
 * vislib::net::TcpCommChannel::TcpCommChannel
 */
vislib::net::TcpCommChannel::TcpCommChannel(const UINT64 flags) 
        : Super(), flags(flags) {
     VLSTACKTRACE("TcpCommChannel::TcpCommChannel", __FILE__, __LINE__);
}


/*
 * vislib::net::TcpCommChannel::TcpCommChannel
 */
vislib::net::TcpCommChannel::TcpCommChannel(Socket& socket, const UINT64 flags) 
        : Super(), socket(socket), flags(flags) {
    VLSTACKTRACE("TcpCommChannel::TcpCommChannel", __FILE__, __LINE__);
    socket.SetNoDelay(this->IsSetNoDelay());
    socket.SetReuseAddr(this->IsSetReuseAddress());
}


/*
 * vislib::net::TcpCommChannel::TcpCommChannel
 */
vislib::net::TcpCommChannel::TcpCommChannel(const TcpCommChannel& rhs) {
    throw UnsupportedOperationException("TcpCommChannel::TcpCommChannel", 
        __FILE__, __LINE__);
}


/*
 * vislib::net::TcpCommChannel::~TcpCommChannel
 */
vislib::net::TcpCommChannel::~TcpCommChannel(void) {
    VLSTACKTRACE("TcpCommChannel::~TcpCommChannel", __FILE__, __LINE__);

    /* Ensure that the socket is closed. */
    try {
        // Note: Must force use of correct implementation in dtor.
        TcpCommChannel::Close();
    } catch (...) {
        // Can be ignored. We expect the operation to fail as the user should
        // have closed the connection before.
    }
}


/*
 * vislib::net::TcpCommChannel::createSocket
 */
void vislib::net::TcpCommChannel::createSocket(const IPEndPoint& endPoint) {
    VLSTACKTRACE("TcpCommChannel::createSocket", __FILE__, __LINE__);

    /* Destroy old instance. */
    if (this->socket.IsValid()) {
        this->socket.Close();
    }
    this->socket.Create(endPoint, Socket::TYPE_STREAM, Socket::PROTOCOL_TCP);
    this->socket.SetNoDelay(this->IsSetNoDelay());
    this->socket.SetReuseAddr(this->IsSetReuseAddress());
}
