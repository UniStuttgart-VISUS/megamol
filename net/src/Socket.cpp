/*
 * Socket.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). All rights reserved.
 */

#include <cstdlib>

#ifdef _WIN32
#define TEMP_FAILURE_RETRY(e) e

#else /* _WIN32 */
#include <poll.h>
#include <sys/socket.h> // panagias: needed for shutdown()
#include <unistd.h>
#include <net/if.h>

#define SOCKET_ERROR (-1)
#endif /* _WIN32 */

#include "vislib/Socket.h"

#include "vislib/assert.h"
#include "vislib/error.h"
#include "vislib/IllegalParamException.h"
#include "vislib/SocketException.h"
#include "vislib/Trace.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::net::Socket::Cleanup
 */
void vislib::net::Socket::Cleanup(void) {
#ifdef _WIN32
    if (::WSACleanup() != 0) {
        throw SocketException(__FILE__, __LINE__);
    }
#endif /* _WIN32 */
}


/*
 * vislib::net::Socket::Startup
 */
void vislib::net::Socket::Startup(void) {
#ifdef _WIN32
    WSAData wsaData;

    // Note: Need Winsock 2 for timeouts.
    if (::WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        throw SocketException(__FILE__, __LINE__);
    }
#endif /* _WIN32 */
}


/*
 * vislib::net::Socket::TIMEOUT_INFINITE 
 */
const UINT vislib::net::Socket::TIMEOUT_INFINITE = 0;


/*
 * vislib::net::Socket::~Socket
 */
vislib::net::Socket::~Socket(void) {
}


/*
 * vislib::net::Socket::Accept
 */
vislib::net::Socket vislib::net::Socket::Accept(IPEndPoint *outConnAddr) {
    struct sockaddr_storage connAddr;
    SOCKET newSocket;

#ifdef _WIN32
    INT addrLen = static_cast<int>(sizeof(connAddr));

    if ((newSocket = ::WSAAccept(this->handle, 
            reinterpret_cast<sockaddr *>(&connAddr), &addrLen, NULL,
            0)) == INVALID_SOCKET) {
        throw SocketException(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    unsigned int addrLen = static_cast<unsigned int>(sizeof(connAddr));

    TEMP_FAILURE_RETRY(newSocket = ::accept(this->handle, 
        reinterpret_cast<sockaddr *>(&connAddr), &addrLen));

    if (newSocket == SOCKET_ERROR) {
        throw SocketException(__FILE__, __LINE__);
    }

#endif /* _WIN32 */

    if (outConnAddr != NULL) {
        *outConnAddr = IPEndPoint(connAddr);
    }

    return Socket(newSocket);
}


/*
 * vislib::net::Socket::Accept
 */
vislib::net::Socket vislib::net::Socket::Accept(SocketAddress *outConnAddr) {
    IPEndPoint endPoint;
    Socket retval = this->Accept(&endPoint);

    if (outConnAddr != NULL) {
        *outConnAddr = static_cast<SocketAddress>(endPoint);
    }

    return retval;
}


/*
 * vislib::net::Socket::Bind
 */
void vislib::net::Socket::Bind(const IPEndPoint& address) {
    if (TEMP_FAILURE_RETRY(::bind(this->handle, 
            static_cast<const struct sockaddr *>(address),
            sizeof(struct sockaddr_storage))) == SOCKET_ERROR) {
        throw SocketException(__FILE__, __LINE__);
    }
}


/*
 * vislib::net::Socket::Bind
 */
void vislib::net::Socket::Bind(const SocketAddress& address) {
    if (TEMP_FAILURE_RETRY(::bind(this->handle, 
            static_cast<const struct sockaddr *>(address),
            sizeof(struct sockaddr_in))) == SOCKET_ERROR) {
        throw SocketException(__FILE__, __LINE__);
    }
}


/*
 * vislib::net::Socket::BindToDevice
 */
void vislib::net::Socket::BindToDevice(const StringA& name) {
#ifndef _WIN32
    struct ifreq interface;

    ::strncpy(interface.ifr_ifrn.ifrn_name, name.PeekBuffer(), 
        name.Length() + 1);

    if (::setsockopt(this->handle, SOL_SOCKET, SO_BINDTODEVICE, &interface,
            sizeof(interface)) == -1) {
        throw SocketException(__FILE__, __LINE__);
    }
#endif /* !_WIN32 */
}


/*
 * vislib::net::Socket::Close
 */
void vislib::net::Socket::Close(void) {

    if (this->IsValid()) {

#ifdef _WIN32
        if (::closesocket(this->handle) == SOCKET_ERROR) {
#else /* _WIN32 */
// panagias: Use shutdown instead of close, see http://stackoverflow.com/questions/2486335/wake-up-thread-blocked-on-accept-call/2489066#2489066
        if (TEMP_FAILURE_RETRY(::shutdown(this->handle, SHUT_RDWR)) == SOCKET_ERROR) {
#endif /* _WIN32 */
            throw SocketException(__FILE__, __LINE__);
        }

        this->handle = INVALID_SOCKET;

    } /* end if (this->IsValid()) */
}


/*
 * vislib::net::Socket::Connect
 */
void vislib::net::Socket::Connect(const IPEndPoint& address) {
#ifdef _WIN32
    if (::WSAConnect(this->handle, static_cast<const struct sockaddr *>(
            address), sizeof(struct sockaddr_storage), NULL, NULL, NULL, NULL)
            == SOCKET_ERROR) {
        throw SocketException(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    // mueller: Linux of kaukerdl showed behaviour as described in 
    // http://www.madore.org/~david/computers/connect-intr.html. I took the
    // fix from there, including the inspiration for the variable naming...
    const struct sockaddr *sa = static_cast<const struct sockaddr *>(address);
    const size_t sal = sizeof(struct sockaddr_storage);
    if (::connect(this->handle, sa, sal) == SOCKET_ERROR) {
        struct pollfd linuxReallySucks;
        int someMoreJunk = 0;
        SIZE_T yetMoreUselessJunk = sizeof(someMoreJunk);

        if (errno != EINTR /* && errno != EINPROGRESS */) {
            throw SocketException(__FILE__, __LINE__);
        }
    
        linuxReallySucks.fd = this->handle;
        linuxReallySucks.events = POLLOUT;
        while (::poll(&linuxReallySucks, 1, -1) == -1) {
            if (errno != EINTR) {
                throw SocketException(__FILE__, __LINE__);
            }
        }

        this->GetOption(SOL_SOCKET, SO_ERROR, &someMoreJunk, 
            yetMoreUselessJunk);
        if (someMoreJunk != 0) {
            throw SocketException(someMoreJunk, __FILE__, __LINE__);
        }
    } /* end if (::connect(this->handle, sa, sal) == SOCKET_ERROR) */
#endif /* _Win32 */
}


/*
 * vislib::net::Socket::Connect
 */
void vislib::net::Socket::Connect(const SocketAddress& address) {
    this->Connect(IPEndPoint(address));
}


/*
 * vislib::net::Socket::Create
 */
void vislib::net::Socket::Create(const ProtocolFamily protocolFamily, 
                                 const Type type, 
                                 const Protocol protocol) {
#ifdef _WIN32
    this->handle = ::WSASocket(static_cast<const int>(protocolFamily), 
        static_cast<const int>(type), static_cast<const int>(protocol),
        NULL, 0, WSA_FLAG_OVERLAPPED);
#else /* _WIN32 */
    this->handle = ::socket(static_cast<const int>(protocolFamily), 
        static_cast<const int>(type), static_cast<const int>(protocol));
#endif /* _WIN32 */

    if (this->handle == INVALID_SOCKET) {
        throw SocketException(__FILE__, __LINE__);
    }
}


/*
 * vislib::net::Socket::Create
 */
void vislib::net::Socket::Create(const IPEndPoint& familySpecAddr, 
            const Type type, const Protocol protocol) {
    switch (familySpecAddr.GetAddressFamily()) {
        case IPEndPoint::FAMILY_INET:
            this->Create(FAMILY_INET, type, protocol);
            break;

        case IPEndPoint::FAMILY_INET6:
            this->Create(FAMILY_INET6, type, protocol);
            break;

        default:
            throw IllegalParamException("familySpecAddr", __FILE__, __LINE__);
    }
}


/*
 * vislib::net::Socket::GetLocalEndPoint
 */
vislib::net::IPEndPoint vislib::net::Socket::GetLocalEndPoint(void) const {
    IPEndPoint retval;
#ifdef _WIN32
    int len = static_cast<int>(sizeof(struct sockaddr_storage));
#else /* _WIN32 */
    socklen_t len = static_cast<socklen_t>(sizeof(struct sockaddr_storage));
#endif /* _WIN32 */

    if (::getsockname(this->handle, reinterpret_cast<sockaddr *>(
            static_cast<struct sockaddr_storage *>(retval)), &len)
            == SOCKET_ERROR) {
        throw SocketException(__FILE__, __LINE__);
    }

    return retval;
}


/*
 * vislib::net::Socket::GetMulticastInterface
 */
void vislib::net::Socket::GetMulticastInterface(IPAddress& outAddr) const {
    struct in_addr retval;
    SIZE_T size = sizeof(retval);
    this->GetOption(IPPROTO_IP, IP_MULTICAST_IF, &retval, size);
    ASSERT(size == sizeof(retval));
    outAddr = retval;
}


/*
 * vislib::net::Socket::GetMulticastLoop
 */
bool vislib::net::Socket::GetMulticastLoop(const ProtocolFamily pf) const {
    switch (pf) {
        case FAMILY_INET:
            return this->getOption(IPPROTO_IP, IP_MULTICAST_LOOP);
            break;

        case FAMILY_INET6:
            return this->getOption(IPPROTO_IPV6, IPV6_MULTICAST_LOOP);
            break;

        default:
            throw IllegalParamException("pf", __FILE__, __LINE__);
            break;
    }
}


/*
 * vislib::net::Socket::GetMulticastTimeToLive
 */
BYTE vislib::net::Socket::GetMulticastTimeToLive(
        const ProtocolFamily pf) const {
    BYTE retval = 0;
    SIZE_T size = sizeof(retval);

    switch (pf) {
        case FAMILY_INET:
            this->GetOption(IPPROTO_IP, IP_MULTICAST_TTL, &retval, size);
            break;

        default:
            throw IllegalParamException("pf", __FILE__, __LINE__);
            break;
    }

    ASSERT(size == sizeof(retval));
    return retval;
}


/*
 * vislib::net::Socket::GetOption
 */
void vislib::net::Socket::GetOption(const INT level, const INT optName, 
        void *outValue, SIZE_T& inOutValueLength) const {
#ifdef _WIN32
    int len = static_cast<int>(inOutValueLength);
#else /* _WIN32 */
    socklen_t len = static_cast<socklen_t>(inOutValueLength);
#endif /* _WIN32 */

    if (::getsockopt(this->handle, level, optName, 
            static_cast<char *>(outValue), &len) != 0) {
        throw SocketException(__FILE__, __LINE__);
    }

    inOutValueLength = static_cast<SIZE_T>(len);
}


/*
 * vislib::net::Socket::GetPeerEndPoint
 */
vislib::net::IPEndPoint vislib::net::Socket::GetPeerEndPoint(void) const {
    IPEndPoint retval;
#ifdef _WIN32
    int len = static_cast<int>(sizeof(struct sockaddr_storage));
#else /* _WIN32 */
    socklen_t len = static_cast<socklen_t>(sizeof(struct sockaddr_storage));
#endif /* _WIN32 */

    if (::getpeername(this->handle, reinterpret_cast<sockaddr *>(
            static_cast<struct sockaddr_storage *>(retval)), &len)
            == SOCKET_ERROR) {
        throw SocketException(__FILE__, __LINE__);
    }

    return retval;
}


/*
 * vislib::net::Socket::GracefulDisconnect
 */
void vislib::net::Socket::GracefulDisconnect(const bool isClose) {
    BYTE buffer[4];             // Buffer for receiving remaining data.
    
    try {
        /* Signal to server that we will not send anything else. */
        this->Shutdown(SEND);

        /* Receive all pending data from server. */
        while (this->Receive(&buffer, sizeof(buffer)) > 0);

    } catch (...) {
        /* Ensure that Close() is called in any case if requested. */
        if (isClose) {
            this->Close();
        }
        throw;
    }

    /* Close socket if requested. */
    if (isClose) {
        this->Close();
    }
}


/*
 * vislib::net::Socket::IOControl
 */
void vislib::net::Socket::IOControl(const DWORD ioControlCode, void *inBuffer,
        const DWORD cntInBuffer, void *outBuffer, const DWORD cntOutBuffer,
        DWORD& outBytesReturned) {
#ifdef _WIN32
    if (::WSAIoctl(this->handle, ioControlCode, inBuffer, cntInBuffer, 
            outBuffer, cntOutBuffer, &outBytesReturned, NULL, NULL) 
            == SOCKET_ERROR) {
        throw SocketException(__FILE__, __LINE__);
    }
#else /* _WIN32 */
    // TODO
#endif /* _WIN32 */
}


/*
 * vislib::net::Socket::LeaveMulticastGroup
 */
void vislib::net::Socket::LeaveMulticastGroup(const IPAddress& group,
                                              const IPAddress& adapter) {
    struct ip_mreq req;
    req.imr_multiaddr = *static_cast<const struct in_addr *>(group);
    req.imr_interface = *static_cast<const struct in_addr *>(adapter);
    this->SetOption(IPPROTO_IP, IP_DROP_MEMBERSHIP, &req, sizeof(req));
}


/*
 * vislib::net::Socket::LeaveMulticastGroup
 */
void vislib::net::Socket::LeaveMulticastGroup(const IPAddress6& group,
                                              const unsigned int adapter) {
    struct ipv6_mreq req;
    req.ipv6mr_multiaddr = *static_cast<const struct in6_addr *>(group);
    req.ipv6mr_interface = adapter;
    this->SetOption(IPPROTO_IPV6, IPV6_DROP_MEMBERSHIP, &req, sizeof(req));
}


/*
 * vislib::net::Socket::Listen
 */
void vislib::net::Socket::Listen(const INT backlog) {
    if (TEMP_FAILURE_RETRY(::listen(this->handle, backlog)) == SOCKET_ERROR) {
        throw SocketException(__FILE__, __LINE__);
    }
}


/*
 * vislib::net::Socket::JoinMulticastGroup
 */
void vislib::net::Socket::JoinMulticastGroup(const IPAddress& group,
                                             const IPAddress& adapter) {
    struct ip_mreq req;
    req.imr_multiaddr = *static_cast<const struct in_addr *>(group);
    req.imr_interface = *static_cast<const struct in_addr *>(adapter);
    this->SetOption(IPPROTO_IP, IP_ADD_MEMBERSHIP, &req, sizeof(req));
}


/*
 * vislib::net::Socket::JoinMulticastGroup
 */
void vislib::net::Socket::JoinMulticastGroup(const IPAddress6& group,
                                             const unsigned int adapter) {
    struct ipv6_mreq req;
    req.ipv6mr_multiaddr = *static_cast<const struct in6_addr *>(group);
    req.ipv6mr_interface = adapter;
    this->SetOption(IPPROTO_IPV6, IPV6_ADD_MEMBERSHIP, &req, sizeof(req));
}


/*
 * vislib::net::Socket::Receive
 */
SIZE_T vislib::net::Socket::Receive(void *outData, const SIZE_T cntBytes, 
        const INT timeout, const INT flags, const bool forceReceive) {
    int n = 0;                  // Highest descriptor in 'readSet' + 1.
    fd_set readSet;             // Set of socket to check for readability.
    struct timeval timeOut;     // Timeout for readability check.

    ///* Check parameter constraints. */
    //if ((timeout >= 1) && forceReceive) {
    //    throw IllegalParamException("forceReceive", __FILE__, __LINE__);
    //}

    /* Handle infinite timeout first by calling normal receive operation. */
    if (timeout < 1) {
        return this->receive(outData, cntBytes, flags, forceReceive);
    }

    /* Initialise socket set and timeout structure. */
    //ASSERT(forceReceive == false);
    FD_ZERO(&readSet);
    FD_SET(this->handle, &readSet);

    timeOut.tv_sec = timeout / 1000;
    timeOut.tv_usec = (timeout % 1000) * 1000;

    /* Wait for the socket to become readable. */
#ifndef _WIN32
    n = this->handle + 1;   // Windows does not need 'n' and will ignore it.
#endif /* !_WIN32 */
    if (TEMP_FAILURE_RETRY(::select(n, &readSet, NULL, NULL, &timeOut)) 
            == SOCKET_ERROR) {
        throw SocketException(__FILE__, __LINE__);
    }

    if (FD_ISSET(this->handle, &readSet)) {
        /* Delegate to normal receive operation. */
        return this->receive(outData, cntBytes, flags, forceReceive);

    } else {
        /* Signal timeout. */
#ifdef _WIN32
        throw SocketException(WSAETIMEDOUT, __FILE__, __LINE__);
#else /* _WIN32 */
        throw SocketException(ETIME, __FILE__, __LINE__);
#endif /* _WIN32 */
    } 
}


/*
 * vislib::net::Socket::Receive
 */
SIZE_T vislib::net::Socket::Receive(IPEndPoint& outFromAddr, void *outData,
        const SIZE_T cntBytes, const INT timeout, const INT flags,
        const bool forceReceive) {
    int n = 0;                  // Highest descriptor in 'readSet' + 1.
    fd_set readSet;             // Set of socket to check for readability.
    struct timeval timeOut;     // Timeout for readability check.

    ///* Check parameter constraints. */
    //if ((timeout >= 1) && forceReceive) {
    //    throw IllegalParamException("forceReceive", __FILE__, __LINE__);
    //}

    /* Handle infinite timeout first by calling normal receive operation. */
    if (timeout < 1) {
        return this->receiveFrom(outFromAddr, outData, cntBytes,flags, 
            forceReceive);
    }

    /* Initialise socket set and timeout structure. */
    //ASSERT(forceReceive == false);
    FD_ZERO(&readSet);
    FD_SET(this->handle, &readSet);

    timeOut.tv_sec = timeout / 1000;
    timeOut.tv_usec = (timeout % 1000) * 1000;

    /* Wait for the socket to become readable. */
#ifndef _WIN32
    n = this->handle + 1;   // Windows does not need 'n' and will ignore it.
#endif /* !_WIN32 */
    if (TEMP_FAILURE_RETRY(::select(n, &readSet, NULL, NULL, &timeOut)) 
            == SOCKET_ERROR) {
        throw SocketException(__FILE__, __LINE__);
    }

    if (FD_ISSET(this->handle, &readSet)) {
        /* Delegate to normal receive operation. */
        return this->receiveFrom(outFromAddr, outData, cntBytes, flags, 
            forceReceive);

    } else {
        /* Signal timeout. */
#ifdef _WIN32
        throw SocketException(WSAETIMEDOUT, __FILE__, __LINE__);
#else /* _WIN32 */
        throw SocketException(ETIME, __FILE__, __LINE__);
#endif /* _WIN32 */
    }
}


/*
 * vislib::net::Socket::Receive
 */
SIZE_T vislib::net::Socket::Receive(SocketAddress& outFromAddr, void *outData,
        const SIZE_T cntBytes, const INT timeout, const INT flags,
        const bool forceReceive) {
    IPEndPoint endPoint;
    SIZE_T retval = this->Receive(endPoint, outData, cntBytes, timeout, flags,
        forceReceive);
    outFromAddr = static_cast<SocketAddress>(endPoint);
    return retval;
}


/*
 * vislib::net::Socket::Send
 */
SIZE_T vislib::net::Socket::Send(const void *data, const SIZE_T cntBytes, 
        const INT timeout, const INT flags, const bool forceSend) {
    int n = 0;                  // Highest descriptor in 'writeSet' + 1.
    fd_set writeSet;            // Set of socket to check for writability.
    struct timeval timeOut;     // Timeout for writability check.

    ///* Check parameter constraints. */
    //if ((timeout >= 1) && forceSend) {
    //    throw IllegalParamException("forceSend", __FILE__, __LINE__);
    //}

    /* Handle infinite timeout first by calling normal send operation. */
    if (timeout < 1) {
        return this->send(data, cntBytes, flags, forceSend);
    }

    /* Initialise socket set and timeout structure. */
    //ASSERT(forceSend == false);
    FD_ZERO(&writeSet);
    FD_SET(this->handle, &writeSet);

    timeOut.tv_sec = timeout / 1000;
    timeOut.tv_usec = (timeout % 1000) * 1000;

    /* Wait for the socket to become readable. */
#ifndef _WIN32
    n = this->handle + 1;   // Windows does not need 'n' and will ignore it.
#endif /* !_WIN32 */
    if (TEMP_FAILURE_RETRY(::select(n, NULL, &writeSet, NULL, &timeOut)) 
            == SOCKET_ERROR) {
        throw SocketException(__FILE__, __LINE__);
    }

    if (FD_ISSET(this->handle, &writeSet)) {
        /* Delegate to normal send operation. */
        return this->send(data, cntBytes, flags, forceSend);

    } else {
        /* Signal timeout. */
#ifdef _WIN32
        throw SocketException(WSAETIMEDOUT, __FILE__, __LINE__);
#else /* _WIN32 */
        throw SocketException(ETIME, __FILE__, __LINE__);
#endif /* _WIN32 */
    }
}


/*
 * vislib::net::Socket::Send
 */
SIZE_T vislib::net::Socket::Send(const IPEndPoint& toAddr, const void *data,
        const SIZE_T cntBytes, const INT timeout, const INT flags,
        const bool forceSend) {
    int n = 0;                  // Highest descriptor in 'writeSet' + 1.
    fd_set writeSet;            // Set of socket to check for writability.
    struct timeval timeOut;     // Timeout for writability check.

    ///* Check parameter constraints. */
    //if ((timeout >= 1) && forceSend) {
    //    throw IllegalParamException("forceSend", __FILE__, __LINE__);
    //}

    /* Handle infinite timeout first by calling normal send operation. */
    if (timeout < 1) {
        return this->sendTo(toAddr, data, cntBytes, flags, forceSend);
    }

    /* Initialise socket set and timeout structure. */
    //ASSERT(forceSend == false);
    FD_ZERO(&writeSet);
    FD_SET(this->handle, &writeSet);

    timeOut.tv_sec = timeout / 1000;
    timeOut.tv_usec = (timeout % 1000) * 1000;

    /* Wait for the socket to become readable. */
#ifndef _WIN32
    n = this->handle + 1;   // Windows does not need 'n' and will ignore it.
#endif /* !_WIN32 */
    if (TEMP_FAILURE_RETRY(::select(n, NULL, &writeSet, NULL, &timeOut)) 
            == SOCKET_ERROR) {
        throw SocketException(__FILE__, __LINE__);
    }

    if (FD_ISSET(this->handle, &writeSet)) {
        /* Delegate to normal send operation. */
        return this->sendTo(toAddr, data, cntBytes, flags, forceSend);

    } else {
        /* Signal timeout. */
#ifdef _WIN32
        throw SocketException(WSAETIMEDOUT, __FILE__, __LINE__);
#else /* _WIN32 */
        throw SocketException(ETIME, __FILE__, __LINE__);
#endif /* _WIN32 */
    }
}


/*
 * vislib::net::Socket::SetMulticastLoop
 */
void vislib::net::Socket::SetMulticastLoop(const ProtocolFamily pf, 
                                           const bool enable) {
    switch (pf) {
        case FAMILY_INET:
            this->setOption(IPPROTO_IP, IP_MULTICAST_LOOP, enable);
            break;

        case FAMILY_INET6:
            this->setOption(IPPROTO_IPV6, IPV6_MULTICAST_LOOP, enable);
            break;

        default:
            throw IllegalParamException("pf", __FILE__, __LINE__);
            break;
    }
}


/*
 * vislib::net::Socket::SetMulticastTimeToLive
 */
void vislib::net::Socket::SetMulticastTimeToLive(const ProtocolFamily pf, 
                                                 const BYTE ttl) {
    switch (pf) {
        case FAMILY_INET:
            this->SetOption(IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl));
            break;

        default:
            throw IllegalParamException("pf", __FILE__, __LINE__);
            break;
    }
}


/*
 * vislib::net::Socket::SetOption
 */
void vislib::net::Socket::SetOption(const INT level, const INT optName, 
            const void *value, const SIZE_T valueLength) {
    if (::setsockopt(this->handle, level, optName, 
            static_cast<const char *>(value), static_cast<int>(valueLength)) 
            != 0) {
        throw SocketException(__FILE__, __LINE__);
    }
}


/*
 * vislib::net::Socket::SetRcvAll
 */
void vislib::net::Socket::SetRcvAll(const bool enable) {
#ifdef _WIN32
    DWORD inBuffer = enable ? 1 : 0;
    DWORD bytesReturned;
    // SIO_RCVALL
    this->IOControl(_WSAIOW(IOC_VENDOR, 1), &inBuffer, sizeof(inBuffer), NULL, 
        0, bytesReturned);
#endif /* _WIN32 */
}


/*
 * vislib::net::Socket::Shutdown
 */
void vislib::net::Socket::Shutdown(const ShutdownManifest how) {
    if (TEMP_FAILURE_RETRY(::shutdown(this->handle, how)) == SOCKET_ERROR) {
        throw SocketException(__FILE__, __LINE__);
    }
}


/*
 * vislib::net::Socket::operator =
 */
vislib::net::Socket& vislib::net::Socket::operator =(const Socket& rhs) {
    if (this != &rhs) {
        this->handle = rhs.handle;
    }

    return *this;
}


/*
 * vislib::net::Socket::operator ==
 */
bool vislib::net::Socket::operator ==(const Socket& rhs) const {
    return (this->handle == rhs.handle);
}


/*
 * vislib::net::Socket::receive
 */
SIZE_T vislib::net::Socket::receive(void *outData, const SIZE_T cntBytes, 
        const INT flags, const bool forceReceive) {
    SIZE_T totalReceived = 0;   // # of bytes totally received.
    INT lastReceived = 0;       // # of bytes received during last recv() call.
#ifdef _WIN32
    WSAOVERLAPPED overlapped;   // Overlap structure for asynchronous recv().
    WSABUF wsaBuf;              // Buffer for WSA output.
    DWORD errorCode = 0;        // WSA error during last operation.
    DWORD inOutFlags = flags;   // Flags for WSA.

    if ((overlapped.hEvent = ::WSACreateEvent()) == WSA_INVALID_EVENT) {
        throw SocketException(__FILE__, __LINE__);
    }

    wsaBuf.buf = static_cast<char *>(outData);
    wsaBuf.len = static_cast<u_long>(cntBytes);
#endif /* _WIN32 */

    do {
#ifdef _WIN32
        if (::WSARecv(this->handle, &wsaBuf, 1, reinterpret_cast<DWORD *>(
                &lastReceived), &inOutFlags, &overlapped, NULL) != 0) {
            if ((errorCode = ::WSAGetLastError()) != WSA_IO_PENDING) {
                ::WSACloseEvent(overlapped.hEvent);
                throw SocketException(errorCode, __FILE__, __LINE__);
            }
            VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "Overlapped socket I/O "
                "pending.\n");
            if (!::WSAGetOverlappedResult(this->handle, &overlapped, 
                    reinterpret_cast<DWORD *>(&lastReceived), TRUE, 
                    &inOutFlags)) {
                ::WSACloseEvent(overlapped.hEvent);
                throw SocketException(__FILE__, __LINE__);
            }
        }
        totalReceived += lastReceived;
        wsaBuf.buf += lastReceived;
        wsaBuf.len -= lastReceived;

#else /* _WIN32 */
        TEMP_FAILURE_RETRY(lastReceived = ::recv(this->handle, 
            static_cast<char *>(outData) + totalReceived, 
            static_cast<int>(cntBytes - totalReceived), flags));

        if ((lastReceived >= 0) && (lastReceived != SOCKET_ERROR)) {
            /* Successfully received new package. */
            totalReceived += static_cast<SIZE_T>(lastReceived);

        } else {
            /* Communication failed. */
            throw SocketException(__FILE__, __LINE__);
        }

#endif /* _WIN32 */
    } while (forceReceive && (totalReceived < cntBytes) && (lastReceived > 0));
    // Note: Test (lastReceived > 0) is for graceful disconnect detection.

#ifdef _WIN32
    if (!::WSACloseEvent(overlapped.hEvent)) {
        throw SocketException(__FILE__, __LINE__);
    }
#endif /*_WIN32 */

    return totalReceived;
}


/*
 * vislib::net::Socket::receiveFrom
 */
SIZE_T vislib::net::Socket::receiveFrom(IPEndPoint& outFromAddr, 
        void *outData, const SIZE_T cntBytes, const INT flags, 
        const bool forceReceive) {
    SIZE_T totalReceived = 0;   // # of bytes totally received.
    INT lastReceived = 0;       // # of bytes received during last recv() call.
#ifdef _WIN32
    WSAOVERLAPPED overlapped;   // Overlap structure for asynchronous recv().
    WSABUF wsaBuf;              // Buffer for WSA output.
    DWORD errorCode = 0;        // WSA error during last operation.
    DWORD inOutFlags = flags;   // Flags for WSA.
    INT fromLen = sizeof(struct sockaddr_storage);

    if ((overlapped.hEvent = ::WSACreateEvent()) == WSA_INVALID_EVENT) {
        throw SocketException(__FILE__, __LINE__);
    }

    wsaBuf.buf = static_cast<char *>(outData);
    wsaBuf.len = static_cast<u_long>(cntBytes);
#else /* _WIN32 */
    socklen_t fromLen = sizeof(struct sockaddr_storage);
#endif /* _WIN32 */

    do {
#ifdef _WIN32
        if (::WSARecvFrom(this->handle, &wsaBuf, 1, 
                reinterpret_cast<DWORD *>(&lastReceived), &inOutFlags,
                static_cast<sockaddr *>(outFromAddr), &fromLen, &overlapped,
                NULL) != 0) {
            if ((errorCode = ::WSAGetLastError()) != WSA_IO_PENDING) {
                ::WSACloseEvent(overlapped.hEvent);
                throw SocketException(errorCode, __FILE__, __LINE__);
            }
            VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "Overlapped socket I/O "
                "pending.\n");
            if (!::WSAGetOverlappedResult(this->handle, &overlapped, 
                    reinterpret_cast<DWORD *>(&lastReceived), TRUE, 
                    &inOutFlags)) {
                ::WSACloseEvent(overlapped.hEvent);
                throw SocketException(__FILE__, __LINE__);
            }
        }
        totalReceived += lastReceived;
        wsaBuf.buf += lastReceived;
        wsaBuf.len -= lastReceived;

#else /* _WIN32 */
        TEMP_FAILURE_RETRY(lastReceived = ::recvfrom(this->handle, 
            static_cast<char *>(outData) + totalReceived, 
            static_cast<int>(cntBytes - totalReceived), 
            flags, static_cast<sockaddr *>(outFromAddr), &fromLen));

        if ((lastReceived >= 0) && (lastReceived != SOCKET_ERROR)) {
            /* Successfully received new package. */
            totalReceived += static_cast<SIZE_T>(lastReceived);

        } else {
            /* Communication failed. */
            throw SocketException(__FILE__, __LINE__);
        }

#endif /* _WIN32 */

    } while (forceReceive && (totalReceived < cntBytes) && (lastReceived > 0));
    // Note: Test (lastReceived > 0) is for graceful disconnect detection.

#ifdef _WIN32
    if (!::WSACloseEvent(overlapped.hEvent)) {
        throw SocketException(__FILE__, __LINE__);
    }
#endif /*_WIN32 */

    return totalReceived;
}


/*
 * vislib::net::Socket::send
 */
SIZE_T vislib::net::Socket::send(const void *data, const SIZE_T cntBytes, 
        const INT flags, const bool forceSend) {
    SIZE_T totalSent = 0;       // # of bytes totally sent.      
    INT lastSent = 0;           // # of bytes sent during last send() call.
#ifdef _WIN32
    WSAOVERLAPPED overlapped;   // Overlap structure for asynchronous recv().
    WSABUF wsaBuf;              // Buffer for WSA output.
    DWORD errorCode = 0;        // WSA error during last operation.
    DWORD inOutFlags = flags;   // Flags for WSA.

    if ((overlapped.hEvent = ::WSACreateEvent()) == WSA_INVALID_EVENT) {
        throw SocketException(__FILE__, __LINE__);
    }

    wsaBuf.buf = const_cast<char *>(static_cast<const char *>(data));
    wsaBuf.len = static_cast<u_long>(cntBytes);
#endif /* _WIN32 */

    do {
#ifdef _WIN32
        if (::WSASend(this->handle, &wsaBuf, 1, reinterpret_cast<DWORD *>(
                &lastSent), flags, &overlapped, NULL) != 0) {
            if ((errorCode = ::WSAGetLastError()) != WSA_IO_PENDING) {
                ::WSACloseEvent(overlapped.hEvent);
                throw SocketException(errorCode, __FILE__, __LINE__);
            }
            VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "Overlapped socket I/O "
                "pending.\n");
            if (!::WSAGetOverlappedResult(this->handle, &overlapped, 
                    reinterpret_cast<DWORD *>(&lastSent), TRUE, &inOutFlags)) {
                ::WSACloseEvent(overlapped.hEvent);
                throw SocketException(__FILE__, __LINE__);
            }
        }
        totalSent += lastSent;
        wsaBuf.buf += lastSent;
        wsaBuf.len -= lastSent;

#else /* _WIN32 */
        TEMP_FAILURE_RETRY(lastSent = ::send(this->handle, 
            static_cast<const char *>(data), 
            static_cast<int>(cntBytes - totalSent), flags));

        if ((lastSent >= 0) && (lastSent != SOCKET_ERROR)) {
            totalSent += static_cast<SIZE_T>(lastSent);

        } else {
            throw SocketException(__FILE__, __LINE__);
        }
#endif /* _WIN32 */

    } while (forceSend && (totalSent < cntBytes));

#ifdef _WIN32
    if (!::WSACloseEvent(overlapped.hEvent)) {
        throw SocketException(__FILE__, __LINE__);
    }
#endif /*_WIN32 */

    return totalSent;
}


/*
 * vislib::net::Socket::sendTo
 */
SIZE_T vislib::net::Socket::sendTo(const IPEndPoint& toAddr, 
        const void *data, const SIZE_T cntBytes, const INT flags, 
        const bool forceSend) {
    SIZE_T totalSent = 0;       // # of bytes totally sent.      
    INT lastSent = 0;           // # of bytes sent during last send() call.
    const sockaddr *to = static_cast<const sockaddr *>(toAddr);
    INT toLen = sizeof(sockaddr_storage);
#ifdef _WIN32
    WSAOVERLAPPED overlapped;   // Overlap structure for asynchronous recv().
    WSABUF wsaBuf;              // Buffer for WSA output.
    DWORD errorCode = 0;        // WSA error during last operation.
    DWORD inOutFlags = flags;   // Flags for WSA.

    if ((overlapped.hEvent = ::WSACreateEvent()) == WSA_INVALID_EVENT) {
        throw SocketException(__FILE__, __LINE__);
    }

    wsaBuf.buf = const_cast<char *>(static_cast<const char *>(data));
    wsaBuf.len = static_cast<u_long>(cntBytes);
#endif /* _WIN32 */

    do {
#ifdef _WIN32
        if (::WSASendTo(this->handle, &wsaBuf, 1, reinterpret_cast<DWORD *>(
                &lastSent), flags, to, toLen, &overlapped, NULL) != 0) {
            if ((errorCode = ::WSAGetLastError()) != WSA_IO_PENDING) {
                ::WSACloseEvent(overlapped.hEvent);
                throw SocketException(errorCode, __FILE__, __LINE__);
            }
            VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "Overlapped socket I/O "
                "pending.\n");
            if (!::WSAGetOverlappedResult(this->handle, &overlapped, 
                    reinterpret_cast<DWORD *>(&lastSent), TRUE, &inOutFlags)) {
                ::WSACloseEvent(overlapped.hEvent);
                throw SocketException(__FILE__, __LINE__);
            }
        }
        totalSent += lastSent;
        wsaBuf.buf += lastSent;
        wsaBuf.len -= lastSent;

#else /* _WIN32 */
        TEMP_FAILURE_RETRY(::sendto(this->handle, 
            static_cast<const char *>(data),
            static_cast<int>(cntBytes - totalSent), flags, to, toLen));

        if ((lastSent >= 0) && (lastSent != SOCKET_ERROR)) {
            totalSent += static_cast<SIZE_T>(lastSent);

        } else {
            throw SocketException(__FILE__, __LINE__);
        }
#endif /* _WIN32 */

    } while (forceSend && (totalSent < cntBytes));

#ifdef _WIN32
    if (!::WSACloseEvent(overlapped.hEvent)) {
        throw SocketException(__FILE__, __LINE__);
    }
#endif /*_WIN32 */

    return totalSent;
}
