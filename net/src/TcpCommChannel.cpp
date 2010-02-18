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
#include "vislib/NetworkInformation.h"
#include "vislib/PeerDisconnectedException.h"
#include "vislib/SocketException.h"
#include "vislib/StackTrace.h"
#include "vislib/Trace.h"
#include "vislib/UnsupportedOperationException.h"



/*
 * vislib::net::TcpCommChannel::FLAG_NODELAY
 */
const UINT64 vislib::net::TcpCommChannel::FLAG_NODELAY = 0x00000001;


/*
 * vislib::net::TcpCommChannel::TcpCommChannel
 */
vislib::net::TcpCommChannel::TcpCommChannel(const UINT64 flags) 
        : AbstractBidiCommChannel(),AbstractClientEndpoint() , 
        AbstractServerEndpoint(), flags(flags) {
     VLSTACKTRACE("TcpCommChannel::TcpCommChannel", __FILE__, __LINE__);
}


/*
 * vislib::net::TcpCommChannel::TcpCommChannel
 */
vislib::net::TcpCommChannel::TcpCommChannel(Socket& socket) 
        : AbstractBidiCommChannel(), AbstractClientEndpoint() , 
        AbstractServerEndpoint(), socket(socket) {
    VLSTACKTRACE("TcpCommChannel::TcpCommChannel", __FILE__, __LINE__);
}


/*
 * vislib::net::TcpCommChannel::Accept
 */
vislib::SmartRef<vislib::net::AbstractCommChannel> 
vislib::net::TcpCommChannel::Accept(void) {
    VLSTACKTRACE("TcpCommChannel::Accept", __FILE__, __LINE__);
    Socket socket = this->socket.Accept();
    socket.SetNoDelay(this->IsSetNoDelay());

    return SmartRef<AbstractCommChannel>(new TcpCommChannel(socket), false);
}


///*
// * vislib::net::TcpCommChannel::AddRef
// */
//UINT32 vislib::net::TcpCommChannel::AddRef(void) {
//    VLSTACKTRACE("TcpCommChannel::AddRef", __FILE__, __LINE__);
//    return ReferenceCounted::AddRef();
//}


/*
 * vislib::net::TcpCommChannel::Bind
 */
void vislib::net::TcpCommChannel::Bind(const char *address) {
    VLSTACKTRACE("TcpCommChannel::Bind", __FILE__, __LINE__);
    IPEndPoint endPoint;

    if (NetworkInformation::GuessLocalEndPoint(endPoint, address) > 0.0f) {
        // If there is no address match, bind to ANY.
        endPoint.SetIPAddress(
            (endPoint.GetAddressFamily() == IPEndPoint::FAMILY_INET)
            ? IPAddress::ANY : IPAddress6::ANY);
    }

    this->Bind(endPoint);
}


/*
 * vislib::net::TcpCommChannel::Bind
 */
void vislib::net::TcpCommChannel::Bind(const wchar_t *address) {
    VLSTACKTRACE("TcpCommChannel::Bind", __FILE__, __LINE__);
    IPEndPoint endPoint;

    if (NetworkInformation::GuessLocalEndPoint(endPoint, address) > 0.0f) {
        // If there is no address match, bind to ANY.
        if (endPoint.GetAddressFamily() == IPEndPoint::FAMILY_INET) {
            endPoint.SetIPAddress(IPAddress::ANY);
        } else {
            endPoint.SetIPAddress(IPAddress6::ANY);
        }
    }

    this->Bind(endPoint);
}


/*
 * vislib::net::TcpCommChannel::Bind
 */
void vislib::net::TcpCommChannel::Bind(const IPEndPoint address) {
    VLSTACKTRACE("TcpCommChannel::Bind", __FILE__, __LINE__);

    /* Lazy creation of resources. */
    if (this->socket.IsValid()) {
        this->socket.Close();
    }

    this->socket.Create(address, Socket::TYPE_STREAM, Socket::PROTOCOL_TCP);
    this->socket.SetNoDelay(this->IsSetNoDelay());

    this->socket.Bind(address);
}


/*
 * vislib::net::TcpCommChannel::Close
 */
void vislib::net::TcpCommChannel::Close(void) {
    // TODO: Find out how shutdown can be safely used.
    try {
        this->socket.Shutdown();
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_WARN, "SocketException when shutting down "
            "socket in TcpCommChannel::Close: %s\n", e.GetMsgA());
    }
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
void vislib::net::TcpCommChannel::Connect(const char *address) {
    VLSTACKTRACE("TcpCommChannel::Connect", __FILE__, __LINE__);
    IPEndPoint endPoint;

    if (NetworkInformation::GuessRemoteEndPoint(endPoint, address) > 0.0f) {
        // We only access exact matches, no wild guesses.
        throw IllegalParamException("address", __FILE__, __LINE__);
    } else {
        this->Connect(endPoint);
    }
}


/*
 * vislib::net::TcpCommChannel::Connect
 */
void vislib::net::TcpCommChannel::Connect(const wchar_t *address) {
    VLSTACKTRACE("TcpCommChannel::Connect", __FILE__, __LINE__);
    IPEndPoint endPoint;

    if (NetworkInformation::GuessRemoteEndPoint(endPoint, address) > 0.0f) {
        // We only access exact matches, no wild guesses.
        throw IllegalParamException("address", __FILE__, __LINE__);
    } else {
        this->Connect(endPoint);
    }
}


/*
 * vislib::net::TcpCommChannel::Connect
 */
void vislib::net::TcpCommChannel::Connect(const IPEndPoint& address) {
    VLSTACKTRACE("TcpCommChannel::Connect", __FILE__, __LINE__);

    /* Lazy creation of resources. */
    if (this->socket.IsValid()) {
        this->socket.Close();
    }

    this->socket.Create(address, Socket::TYPE_STREAM, Socket::PROTOCOL_TCP);
    this->socket.SetNoDelay(this->IsSetNoDelay());

    this->socket.Connect(address);
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
        const SIZE_T cntBytes, const INT timeout, const bool forceReceive) {
    VLSTACKTRACE("TcpCommChannel::Receive", __FILE__, __LINE__);
    SIZE_T retval = this->socket.Receive(outData, cntBytes, timeout, 0, 
        forceReceive);

    if (retval == 0) {
        throw PeerDisconnectedException(
            PeerDisconnectedException::FormatMessageForLocalEndpoint(
            this->socket.GetLocalEndPoint().ToStringW().PeekBuffer()), 
            __FILE__, __LINE__);
    }

    return retval;
}


///*
// * vislib::net::TcpCommChannel::Release
// */
//UINT32 vislib::net::TcpCommChannel::Release(void) {
//     VLSTACKTRACE("TcpCommChannel::Release", __FILE__, __LINE__);
//     return ReferenceCounted::Release();
//}


/*
 * vislib::net::TcpCommChannel::Send
 */
SIZE_T vislib::net::TcpCommChannel::Send(const void *data, 
        const SIZE_T cntBytes, const INT timeout, const bool forceSend) {
    VLSTACKTRACE("TcpCommChannel::Send", __FILE__, __LINE__);
    return this->socket.Send(data, cntBytes, timeout, 0, forceSend);
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
 * vislib::net::TcpCommChannel::TcpCommChannel
 */
vislib::net::TcpCommChannel::TcpCommChannel(const TcpCommChannel& rhs) {
    throw UnsupportedOperationException("TcpCommChannel::TcpCommChannel", 
        __FILE__, __LINE__);
}
