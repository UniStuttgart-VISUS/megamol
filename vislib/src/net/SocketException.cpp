/*
 * SocketException.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

// Workaround global WIN32_LEAN_AND_MEAN definition
#ifdef _WIN32
#include <winsock.h>
#endif

#include "vislib/net/SocketException.h"
#include "vislib/sys/error.h"


/*
 * vislib::net::SocketException::SocketException
 */
vislib::net::SocketException::SocketException(const DWORD errorCode, const char* file, const int line)
        : SystemException(errorCode, file, line) {}


/*
 * vislib::net::SocketException::SocketException
 */
vislib::net::SocketException::SocketException(const DWORD errorCode, const char* msg, const char* file, const int line)
        : SystemException(errorCode, file, line) {
    if (msg != NULL) {
        this->setMsg(msg);
    }
}


/*
 * vislib::net::SocketException::SocketException
 */
vislib::net::SocketException::SocketException(const char* file, const int line)
#ifdef _WIN32
        : SystemException(::WSAGetLastError(), file, line) {
#else  /* _WIN32 */
        : SystemException(::GetLastError(), file, line) {
#endif /* _WIN32 */
}


/*
 * vislib::net::SocketException::SocketException
 */
vislib::net::SocketException::SocketException(const SocketException& rhs) : SystemException(rhs) {}


/*
 * vislib::net::SocketException::~SocketException
 */
vislib::net::SocketException::~SocketException() {}


/*
 * vislib::net::SocketException::operator =
 */
vislib::net::SocketException& vislib::net::SocketException::operator=(const SocketException& rhs) {
    SystemException::operator=(rhs);
    return *this;
}


/*
 * vislib::net::SocketException::IsTimeout
 */
bool vislib::net::SocketException::IsTimeout() const {
#ifdef _WIN32
    return (this->GetErrorCode() == WSAETIMEDOUT);
#else  /* _WIN32 */
    return (this->GetErrorCode() == ETIMEDOUT);
#endif /* _WIN32 */
}
