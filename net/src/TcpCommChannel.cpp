/*
 * TcpCommChannel.cpp
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/TcpCommChannel.h"

#include "the/assert.h"
#include "the/argument_exception.h"
#include "vislib/IPCommEndPoint.h"
#include "vislib/PeerDisconnectedException.h"
#include "vislib/SocketException.h"
#include "the/trace.h"
#include "the/not_supported_exception.h"


/*
 * vislib::net::TcpCommChannel::FLAG_NODELAY
 */
const uint64_t vislib::net::TcpCommChannel::FLAG_NODELAY = 0x00000001;


/*
 * vislib::net::TcpCommChannel::FLAG_NOSENDBUFFER
 */
const uint64_t vislib::net::TcpCommChannel::FLAG_NOSENDBUFFER = 0x00000004;


/*
 * vislib::net::TcpCommChannel::FLAG_REUSE_ADDRESS
 */
const uint64_t vislib::net::TcpCommChannel::FLAG_REUSE_ADDRESS = 0x00000002;


/*
 * vislib::net::TcpCommChannel::Accept
 */
vislib::SmartRef<vislib::net::AbstractCommClientChannel> 
vislib::net::TcpCommChannel::Accept(void) {
    THE_STACK_TRACE;
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
    THE_STACK_TRACE;
    SmartRef<IPCommEndPoint> ep = endPoint.DynamicCast<IPCommEndPoint>();

    if (ep.IsNull()) {
        throw the::argument_exception("endPoint", __FILE__, __LINE__);
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
    //    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "SocketException when shutting down "
    //        "socket in TcpCommChannel::Close: %s\n", e.what());
    //}
    try {
        this->socket.Close();
    } catch (SocketException e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "SocketException when closing socket "
            "in TcpCommChannel::Close: %s\n", e.what());
        throw e;
    }
}


/*
 * vislib::net::TcpCommChannel::Connect
 */
void vislib::net::TcpCommChannel::Connect(
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
 * vislib::net::TcpCommChannel::GetLocalEndPoint
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> 
vislib::net::TcpCommChannel::GetLocalEndPoint(void) const {
    THE_STACK_TRACE;
    return IPCommEndPoint::Create(this->socket.GetLocalEndPoint());
}


/*
 * vislib::net::TcpCommChannel::GetRemoteEndPoint
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint>
vislib::net::TcpCommChannel::GetRemoteEndPoint(void) const {
    THE_STACK_TRACE;
    return IPCommEndPoint::Create(this->socket.GetPeerEndPoint());
}


/*
 * vislib::net::TcpCommChannel::Listen
 */
void vislib::net::TcpCommChannel::Listen(const int backlog) {
    THE_STACK_TRACE;
    this->socket.Listen(backlog);
}


/*
 * vislib::net::TcpCommChannel::Receive
 */
size_t vislib::net::TcpCommChannel::Receive(void *outData, 
        const size_t cntBytes, const unsigned int timeout, const bool forceReceive) {
    THE_STACK_TRACE;
    size_t retval = 0;
    
    if (cntBytes > 0) {
        retval = this->socket.Receive(outData, cntBytes, timeout, 0, 
            forceReceive);

        if ((retval == 0) || (forceReceive && (retval < cntBytes))) {
            throw PeerDisconnectedException(
                PeerDisconnectedException::FormatMessageForLocalEndpoint(
                this->socket.GetLocalEndPoint().ToStringW().c_str()).c_str(), 
                __FILE__, __LINE__);
        }
    }

    return retval;
}


/*
 * vislib::net::TcpCommChannel::Send
 */
size_t vislib::net::TcpCommChannel::Send(const void *data, 
        const size_t cntBytes, const unsigned int timeout, const bool forceSend) {
    THE_STACK_TRACE;
    return this->socket.Send(data, cntBytes, timeout, 0, forceSend);
}


/*
 * vislib::net::TcpCommChannel::TcpCommChannel
 */
vislib::net::TcpCommChannel::TcpCommChannel(const uint64_t flags) 
        : Super(), flags(flags) {
     THE_STACK_TRACE;
}


/*
 * vislib::net::TcpCommChannel::TcpCommChannel
 */
vislib::net::TcpCommChannel::TcpCommChannel(Socket& socket, const uint64_t flags) 
        : Super(), socket(socket), flags(flags) {
    THE_STACK_TRACE;
    socket.SetNoDelay(this->IsSetNoDelay());
    socket.SetReuseAddr(this->IsSetReuseAddress());
    if (this->IsSetNoSendBuffer()) {
        socket.SetSndBuf(0);
    }
}


/*
 * vislib::net::TcpCommChannel::TcpCommChannel
 */
vislib::net::TcpCommChannel::TcpCommChannel(const TcpCommChannel& rhs) {
    throw the::not_supported_exception("TcpCommChannel::TcpCommChannel", 
        __FILE__, __LINE__);
}


/*
 * vislib::net::TcpCommChannel::~TcpCommChannel
 */
vislib::net::TcpCommChannel::~TcpCommChannel(void) {
    THE_STACK_TRACE;

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
    THE_STACK_TRACE;

    /* Destroy old instance. */
    if (this->socket.IsValid()) {
        this->socket.Close();
    }
    this->socket.Create(endPoint, Socket::TYPE_STREAM, Socket::PROTOCOL_TCP);
    this->socket.SetNoDelay(this->IsSetNoDelay());
    this->socket.SetReuseAddr(this->IsSetReuseAddress());
    if (this->IsSetNoSendBuffer()) {
        socket.SetSndBuf(0);
    }
}
