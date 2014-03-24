/*
 * SocketException.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/error.h"
#include "vislib/SocketException.h"


/*
 * vislib::net::SocketException::SocketException
 */
vislib::net::SocketException::SocketException(const unsigned int errorCode, 
        const char *file, const int line)
        : the::system::system_exception(errorCode, file, line) {
}


/*
 * vislib::net::SocketException::SocketException
 */
vislib::net::SocketException::SocketException(const unsigned int errorCode,
        const char *msg, const char *file, const int line)
        : the::system::system_exception(errorCode, file, line) {
    if (msg != NULL) {
        this->set_msg(msg);
    }
}


/*
 * vislib::net::SocketException::SocketException
 */
vislib::net::SocketException::SocketException(const char *file, const int line)
#ifdef _WIN32
        : the::system::system_exception(::WSAGetLastError(), file, line) {
#else /* _WIN32 */
        : the::system::system_exception(::GetLastError(), file, line) {
#endif /* _WIN32 */
}


/*
 * vislib::net::SocketException::SocketException
 */
vislib::net::SocketException::SocketException(const SocketException& rhs) 
        : the::system::system_exception(rhs) {
}


/*
 * vislib::net::SocketException::~SocketException
 */
vislib::net::SocketException::~SocketException(void) {
}


/*
 * vislib::net::SocketException::operator =
 */
vislib::net::SocketException& vislib::net::SocketException::operator =(
        const SocketException& rhs) {
    the::system::system_exception::operator =(rhs);
    return *this;
}


/*
 * vislib::net::SocketException::IsTimeout
 */
bool vislib::net::SocketException::IsTimeout(void) const {
#ifdef _WIN32
    return (this->get_error().native_error() == WSAETIMEDOUT);
#else /* _WIN32 */
    return (this->get_error().native_error() == ETIMEDOUT);
#endif /* _WIN32 */
}
