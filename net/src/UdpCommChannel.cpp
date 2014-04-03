/*
 * UdpCommChannel.cpp
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/UdpCommChannel.h"

#include "the/assert.h"
#include "the/argument_exception.h"
#include "vislib/IPCommEndPoint.h"
#include "vislib/PeerDisconnectedException.h"
#include "vislib/SocketException.h"
#include "the/trace.h"
#include "the/not_supported_exception.h"


/*
 * vislib::net::UdpCommChannel::FLAG_BROADCAST
 */
const uint64_t vislib::net::UdpCommChannel::FLAG_BROADCAST = 0x00000004;


/*
 * vislib::net::UdpCommChannel::FLAG_REUSE_ADDRESS
 */
const uint64_t vislib::net::UdpCommChannel::FLAG_REUSE_ADDRESS = 0x00000002;
// Implementation note: Trying to assign flags of different channels with the 
// same name the same value (see TcpCommChannel::FLAG_REUSE_ADDRESS).


/*
 * vislib::net::UdpCommChannel::Accept
 */
vislib::SmartRef<vislib::net::AbstractCommClientChannel> 
vislib::net::UdpCommChannel::Accept(void) {
    THE_STACK_TRACE;
    Socket socket = this->socket.Accept();
    // Ctor of UdpCommChannel will assign flags to actual socket.

    return SmartRef<AbstractCommClientChannel>(
        new UdpCommChannel(socket, this->flags), false);
}


/*
 * vislib::net::UdpCommChannel::Bind
 */
void vislib::net::UdpCommChannel::Bind(
        SmartRef<AbstractCommEndPoint> endPoint) {
    THE_STACK_TRACE;
    SmartRef<IPCommEndPoint> ep = endPoint.DynamicCast<IPCommEndPoint>();

    if (ep.IsNull()) {
        throw the::argument_exception("endPoint", __FILE__, __LINE__);
    }

    this->createSocket(static_cast<IPEndPoint>(*ep));   // Create lazily.
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
    //    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "SocketException when shutting down "
    //        "socket in TcpCommChannel::Close: %s\n", e.what());
    //}
    try {
        this->socket.Close();
    } catch (SocketException e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "SocketException when closing socket "
            "in UdpCommChannel::Close: %s\n", e.what());
        throw e;
    }
}


/*
 * vislib::net::UdpCommChannel::Connect
 */
void vislib::net::UdpCommChannel::Connect(
        SmartRef<AbstractCommEndPoint> endPoint) {
    THE_STACK_TRACE;
    SmartRef<IPCommEndPoint> ep = endPoint.DynamicCast<IPCommEndPoint>();

    if (ep.IsNull()) {
        throw the::argument_exception("endPoint", __FILE__, __LINE__);
    }

    this->createSocket(static_cast<IPEndPoint>(*ep));   // Create lazily.
    this->socket.Connect(static_cast<IPEndPoint>(*ep));
}


/*
 * vislib::net::UdpCommChannel::GetLocalEndPoint
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> 
vislib::net::UdpCommChannel::GetLocalEndPoint(void) const {
    THE_STACK_TRACE;
    return IPCommEndPoint::Create(this->socket.GetLocalEndPoint());
}


/*
 * vislib::net::UdpCommChannel::GetRemoteEndPoint
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint>
vislib::net::UdpCommChannel::GetRemoteEndPoint(void) const {
    THE_STACK_TRACE;
    return IPCommEndPoint::Create(this->socket.GetPeerEndPoint());
}


/*
 * vislib::net::UdpCommChannel::Listen
 */
void vislib::net::UdpCommChannel::Listen(const int backlog) {
    THE_STACK_TRACE;
    this->socket.Listen(backlog);
}


/*
 * vislib::net::UdpCommChannel::Receive
 */
size_t vislib::net::UdpCommChannel::Receive(void *outData, 
        const size_t cntBytes, const unsigned int timeout, const bool forceReceive) {
    THE_STACK_TRACE;
    size_t retval = this->socket.Receive(outData, cntBytes, timeout, 0, 
        forceReceive);

    if (retval == 0) {
        throw PeerDisconnectedException(
            PeerDisconnectedException::FormatMessageForLocalEndpoint(
            this->socket.GetLocalEndPoint().ToStringW().c_str()).c_str(), 
            __FILE__, __LINE__);
    }

    return retval;
}


/*
 * vislib::net::UdpCommChannel::Send
 */
size_t vislib::net::UdpCommChannel::Send(const void *data, 
        const size_t cntBytes, const unsigned int timeout, const bool forceSend) {
    THE_STACK_TRACE;
    return this->socket.Send(data, cntBytes, timeout, 0, forceSend);
}


/*
 * vislib::net::UdpCommChannel::UdpCommChannel
 */
vislib::net::UdpCommChannel::UdpCommChannel(const uint64_t flags) : flags(flags) {
    THE_STACK_TRACE;
}


/*
 * vislib::net::UdpCommChannel::UdpCommChannel
 */
vislib::net::UdpCommChannel::UdpCommChannel(Socket& socket, const uint64_t flags) 
        : Super(), socket(socket), flags(flags) {
    THE_STACK_TRACE;
    socket.SetReuseAddr(this->IsSetReuseAddress());
    socket.SetBroadcast(this->IsSetBroadcast());
}


/*
 * vislib::net::UdpCommChannel::~UdpCommChannel
 */
vislib::net::UdpCommChannel::~UdpCommChannel(void) {
    THE_STACK_TRACE;

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
    THE_STACK_TRACE;

    /* Destroy old instance. */
    if (this->socket.IsValid()) {
        this->socket.Close();
    }
    this->socket.Create(endPoint, Socket::TYPE_DGRAM, Socket::PROTOCOL_UDP);
    this->socket.SetReuseAddr(this->IsSetReuseAddress());
    this->socket.SetBroadcast(this->IsSetBroadcast());
}
