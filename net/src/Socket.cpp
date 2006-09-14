/*
 * SocketAddress.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). All rights reserved.
 */

#include <cstdlib>

#ifndef _WIN32
#include <unistd.h>

#define SOCKET_ERROR (-1)
#endif /* _WIN32 */

#include "vislib/Socket.h"

#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
#include "vislib/SocketException.h"
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

    if (::WSAStartup(MAKEWORD(1, 1), &wsaData) != 0) {
        throw SocketException(__FILE__, __LINE__);
    }
#endif /* _WIN32 */
}


/*
 * vislib::net::Socket::~Socket
 */
vislib::net::Socket::~Socket(void) {
}


/*
 * vislib::net::Socket::Accept
 */
vislib::net::Socket vislib::net::Socket::Accept(SocketAddress *outConnAddr) {
    struct sockaddr connAddr;
    SOCKET newSocket;

#ifdef _WIN32
    INT addrLen = static_cast<int>(sizeof(connAddr));

    if ((newSocket = ::WSAAccept(this->handle, &connAddr, &addrLen, NULL, 
            0)) == INVALID_SOCKET) {
        throw SocketException(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    unsigned int addrLen = static_cast<unsigned int>(sizeof(connAddr));

    if ((newSocket = ::accept(this->handle, &connAddr, &addrLen))
            == SOCKET_ERROR) {
        throw SocketException(__FILE__, __LINE__);
    }

#endif /* _WIN32 */

    if (outConnAddr != NULL) {
        *outConnAddr = SocketAddress(connAddr);
    }

    return Socket(newSocket);
}


/*
 * vislib::net::Socket::Bind
 */
void vislib::net::Socket::Bind(const SocketAddress& address) {
    if (::bind(this->handle, &static_cast<const struct sockaddr>(address),
            sizeof(struct sockaddr_in)) == SOCKET_ERROR) {
        throw SocketException(__FILE__, __LINE__);
    }
}


/*
 * vislib::net::Socket::Close
 */
void vislib::net::Socket::Close(void) {

    if (this->IsValid()) {

#ifdef _WIN32
        if (::closesocket(this->handle) == SOCKET_ERROR) {
#else /* _WIN32 */
        if (::close(this->handle) == SOCKET_ERROR) {
#endif /* _WIN32 */
            throw SocketException(__FILE__, __LINE__);
        }

    } /* end if (this->IsValid()) */
}


/*
 * vislib::net::Socket::Connect
 */
void vislib::net::Socket::Connect(const SocketAddress& address) {
#ifdef _WIN32
    if (::WSAConnect(this->handle, &static_cast<const struct sockaddr>(address),
            sizeof(struct sockaddr), NULL, NULL, NULL, NULL) == SOCKET_ERROR) {
#else /* _WIN32 */
    if (::connect(this->handle, &static_cast<const struct sockaddr>(address),
            sizeof(struct sockaddr)) == SOCKET_ERROR) {
#endif /* _Win32 */
        throw SocketException(__FILE__, __LINE__);
    }
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
        NULL, 0, 0);
#else /* _WIN32 */
    this->handle = ::socket(static_cast<const int>(protocolFamily), 
        static_cast<const int>(type), static_cast<const int>(protocol));
#endif /* _WIN32 */

    if (this->handle == INVALID_SOCKET) {
        throw SocketException(__FILE__, __LINE__);
    }
}


/*
 * vislib::net::Socket::GetOption
 */
void vislib::net::Socket::GetOption(const INT level, const INT optName, 
        void *outValue, SIZE_T& inOutValueLength) const {
#ifdef _WIN32
	int len = static_cast<INT>(inOutValueLength);
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
 * vislib::net::Socket::Listen
 */
void vislib::net::Socket::Listen(const INT backlog) {
    if (::listen(this->handle, backlog) == SOCKET_ERROR) {
        throw SocketException(__FILE__, __LINE__);
    }
}


/*
 * vislib::net::Socket::Receive
 */
SIZE_T vislib::net::Socket::Receive(void *outData, const SIZE_T cntBytes, 
        const INT flags, const bool forceReceive) {
    SIZE_T totalReceived = 0;   // # of bytes totally received.
    INT lastReceived = 0;       // # of bytes received during last recv() call.

    do {
        lastReceived = ::recv(this->handle, static_cast<char *>(outData) 
            + totalReceived, static_cast<int>(cntBytes - totalReceived), flags);

        if ((lastReceived >= 0) && (lastReceived != SOCKET_ERROR)) {
            /* Successfully received new package. */
            totalReceived += static_cast<SIZE_T>(lastReceived);

        } else {
            /* Communication failed. */
            throw SocketException(__FILE__, __LINE__);
        }

    } while (forceReceive && (totalReceived < cntBytes));

    return totalReceived;
}


/*
 * vislib::net::Socket::Send
 */
SIZE_T vislib::net::Socket::Send(const void *data, const SIZE_T cntBytes, 
        const INT flags, const bool forceSend) {
    SIZE_T totalSent = 0;       // # of bytes totally sent.      
    INT lastSent = 0;           // # of bytes sent during last send() call.

    do {
        lastSent = ::send(this->handle, static_cast<const char *>(data), 
            static_cast<int>(cntBytes - totalSent), flags);

        if ((lastSent >= 0) && (lastSent != SOCKET_ERROR)) {
            totalSent += static_cast<SIZE_T>(lastSent);

        } else {
            throw SocketException(__FILE__, __LINE__);
        }

    } while (forceSend && (totalSent < cntBytes));
    
    return totalSent;
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
 * vislib::net::Socket::Shutdown
 */
void vislib::net::Socket::Shutdown(const ShutdownManifest how) {
    if (::shutdown(this->handle, how) == SOCKET_ERROR) {
        throw SocketException(__FILE__, __LINE__);
    }
}


/*
 * vislib::net::Socket::Socket
 */
vislib::net::Socket::Socket(const Socket& rhs) {
    throw UnsupportedOperationException("vislib::net::Socket::Socket",
        __FILE__, __LINE__);
}


/*
 * vislib::net::Socket::operator =
 */
vislib::net::Socket& vislib::net::Socket::operator =(const Socket& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
