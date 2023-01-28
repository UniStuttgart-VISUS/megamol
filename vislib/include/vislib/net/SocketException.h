/*
 * SocketException.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/sys/SystemException.h"


namespace vislib::net {

/**
 * This exception indicates a socket error.
 */
class SocketException : public sys::SystemException {

public:
    /**
     * Create a new exception using the system message for the given
     * error code as exception message.
     *
     * @param errorCode A socket error code.
     * @param file      The file the exception was thrown in.
     * @param line      The line the exception was thrown in.
     */
    SocketException(const DWORD errorCode, const char* file, const int line);

    /**
     * Create a new exception using the specified error code, but
     * overwrite the message with the specified value. If 'msg' is
     * NULL, this ctor behaves exactly as SocketException(
     * const DWORD errorCode, const char *file, const int line).
     *
     * @param errorCode A socket error code.
     * @param msg       A user-defined message to be used instead of
     *                  the system message associated with the given
     *                  error code.
     * @param file      The file the exception was thrown in.
     * @param line      The line the exception was thrown in.
     */
    SocketException(const DWORD errorCode, const char* msg, const char* file, const int line);

    /**
     * Create a new exception using the last socket error code, i. e.
     * ::WSAGetLastError() on Windows or ::errno on Linux systems.
     *
     * @param file The file the exception was thrown in.
     * @param line The line the exception was thrown in.
     */
    SocketException(const char* file, const int line);

    /**
     * Create a clone of 'rhs'.
     *
     * @param rhs The object to be cloned.
     */
    SocketException(const SocketException& rhs);

    /** Dtor. */
    ~SocketException() override;

    /**
     * Assignment operator.
     *
     * @param rhs The right hand side operand.
     *
     * @return *this.
     */
    virtual SocketException& operator=(const SocketException& rhs);

    /**
     * Answer whether the exception represents a timeout.
     *
     * @return true, if the exception represents a timeout, false otherwise.
     */
    bool IsTimeout() const;
};

} // namespace vislib::net

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
