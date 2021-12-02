/*
 * UdpCommChannel.cpp
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/net/UdpCommChannel.h"

#include "vislib/IllegalParamException.h"
#include "vislib/Trace.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/assert.h"
#include "vislib/net/IPCommEndPoint.h"
#include "vislib/net/PeerDisconnectedException.h"
#include "vislib/net/SocketException.h"


/*
 * vislib::net::UdpCommChannel::FLAG_BROADCAST
 */
const UINT64 vislib::net::UdpCommChannel::FLAG_BROADCAST = 0x00000004;


/*
 * vislib::net::UdpCommChannel::FLAG_REUSE_ADDRESS
 */
const UINT64 vislib::net::UdpCommChannel::FLAG_REUSE_ADDRESS = 0x00000002;
// Implementation note: Trying to assign flags of different channels with the
// same name the same value (see TcpCommChannel::FLAG_REUSE_ADDRESS).


/*
 * vislib::net::UdpCommChannel::Accept
 */
vislib::SmartRef<vislib::net::AbstractCommClientChannel> vislib::net::UdpCommChannel::Accept(void) {
    Socket socket = this->socket.Accept();
    // Ctor of UdpCommChannel will assign flags to actual socket.

    return SmartRef<AbstractCommClientChannel>(new UdpCommChannel(socket, this->flags), false);
}


/*
 * vislib::net::UdpCommChannel::Bind
 */
void vislib::net::UdpCommChannel::Bind(SmartRef<AbstractCommEndPoint> endPoint) {
    SmartRef<IPCommEndPoint> ep = endPoint.DynamicCast<IPCommEndPoint>();

    if (ep.IsNull()) {
        throw IllegalParamException("endPoint", __FILE__, __LINE__);
    }

    this->createSocket(static_cast<IPEndPoint>(*ep)); // Create lazily.
    this->socket.Bind(static_cast<IPEndPoint>(*ep));
}


/*
 * vislib::net::UdpCommChannel::Close
 */
void vislib::net::UdpCommChannel::Close(void) {
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
        VLTRACE(Trace::LEVEL_VL_WARN,
            "SocketException when closing socket "
            "in UdpCommChannel::Close: %s\n",
            e.GetMsgA());
        throw e;
    }
}


/*
 * vislib::net::UdpCommChannel::Connect
 */
void vislib::net::UdpCommChannel::Connect(SmartRef<AbstractCommEndPoint> endPoint) {
    SmartRef<IPCommEndPoint> ep = endPoint.DynamicCast<IPCommEndPoint>();

    if (ep.IsNull()) {
        throw IllegalParamException("endPoint", __FILE__, __LINE__);
    }

    this->createSocket(static_cast<IPEndPoint>(*ep)); // Create lazily.
    this->socket.Connect(static_cast<IPEndPoint>(*ep));
}


/*
 * vislib::net::UdpCommChannel::GetLocalEndPoint
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> vislib::net::UdpCommChannel::GetLocalEndPoint(void) const {
    return IPCommEndPoint::Create(this->socket.GetLocalEndPoint());
}


/*
 * vislib::net::UdpCommChannel::GetRemoteEndPoint
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> vislib::net::UdpCommChannel::GetRemoteEndPoint(void) const {
    return IPCommEndPoint::Create(this->socket.GetPeerEndPoint());
}


/*
 * vislib::net::UdpCommChannel::Listen
 */
void vislib::net::UdpCommChannel::Listen(const int backlog) {
    this->socket.Listen(backlog);
}


/*
 * vislib::net::UdpCommChannel::Receive
 */
SIZE_T vislib::net::UdpCommChannel::Receive(
    void* outData, const SIZE_T cntBytes, const UINT timeout, const bool forceReceive) {
    SIZE_T retval = this->socket.Receive(outData, cntBytes, timeout, 0, forceReceive);

    if (retval == 0) {
        throw PeerDisconnectedException(PeerDisconnectedException::FormatMessageForLocalEndpoint(
                                            this->socket.GetLocalEndPoint().ToStringW().PeekBuffer()),
            __FILE__, __LINE__);
    }

    return retval;
}


/*
 * vislib::net::UdpCommChannel::Send
 */
SIZE_T vislib::net::UdpCommChannel::Send(
    const void* data, const SIZE_T cntBytes, const UINT timeout, const bool forceSend) {
    return this->socket.Send(data, cntBytes, timeout, 0, forceSend);
}


/*
 * vislib::net::UdpCommChannel::UdpCommChannel
 */
vislib::net::UdpCommChannel::UdpCommChannel(const UINT64 flags) : flags(flags) {}


/*
 * vislib::net::UdpCommChannel::UdpCommChannel
 */
vislib::net::UdpCommChannel::UdpCommChannel(Socket& socket, const UINT64 flags)
        : Super()
        , socket(socket)
        , flags(flags) {
    socket.SetReuseAddr(this->IsSetReuseAddress());
    socket.SetBroadcast(this->IsSetBroadcast());
}


/*
 * vislib::net::UdpCommChannel::~UdpCommChannel
 */
vislib::net::UdpCommChannel::~UdpCommChannel(void) {

    /* Ensure that the socket is closed. */
    try {
        // Note: Must force use of correct implementation in dtor.
        UdpCommChannel::Close();
    } catch (...) {
        // Can be ignored. We expect the operation to fail as the user should
        // have closed the connection before.
    }
}


/*
 * vislib::net::UdpCommChannel::createSocket
 */
void vislib::net::UdpCommChannel::createSocket(const IPEndPoint& endPoint) {

    /* Destroy old instance. */
    if (this->socket.IsValid()) {
        this->socket.Close();
    }
    this->socket.Create(endPoint, Socket::TYPE_DGRAM, Socket::PROTOCOL_UDP);
    this->socket.SetReuseAddr(this->IsSetReuseAddress());
    this->socket.SetBroadcast(this->IsSetBroadcast());
}
