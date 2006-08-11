/*
 * SocketException.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/error.h"
#include "vislib/SocketException.h"


/*
 * vislib::sys::SocketException::SocketException
 */
vislib::sys::SocketException::SocketException(const DWORD errorCode, 
        const char *file, const int line) 
		: SystemException(errorCode, file, line) {
}


/*
 * vislib::sys::SocketException::SocketException
 */
vislib::sys::SocketException::SocketException(const char *file, const int line)
#ifdef _WIN32
        : SystemException(::WSAGetLastError(), file, line) {
#else /* _WIN32 */
        : SystemException(::GetLastError(), file, line) {
#endif /* _WIN32 */
}


/*
 * vislib::sys::SocketException::SocketException
 */
vislib::sys::SocketException::SocketException(const SocketException& rhs) 
        : SystemException(rhs) {
}


/*
 * vislib::sys::SocketException::~SocketException
 */
vislib::sys::SocketException::~SocketException(void) {
}


/*
 * vislib::sys::SocketException::operator =
 */
vislib::sys::SocketException& vislib::sys::SocketException::operator =(
		const SocketException& rhs) {
    SystemException::operator =(rhs);
    return *this;
}

