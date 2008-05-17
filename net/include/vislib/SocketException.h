/*
 * SocketException.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef SOCKETEXCEPTION_H_INCLUDED
#define SOCKETEXCEPTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/SystemException.h"


namespace vislib {
namespace net {

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
        SocketException(const DWORD errorCode, const char *file,
            const int line);

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
        SocketException(const DWORD errorCode, const char *msg, 
            const char *file, const int line);

        /**
         * Create a new exception using the last socket error code, i. e.
         * ::WSAGetLastError() on Windows or ::errno on Linux systems.
         *
         * @param file The file the exception was thrown in.
         * @param line The line the exception was thrown in.
         */
        SocketException(const char *file, const int line);

        /**
         * Create a clone of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        SocketException(const SocketException& rhs);

        /** Dtor. */
        virtual ~SocketException(void);

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        virtual SocketException& operator =(const SocketException& rhs);

        /**
         * Answer whether the exception represents a timeout.
         *
         * @return true, if the exception represents a timeout, false otherwise.
         */
        bool IsTimeout(void) const;

    };

} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* SOCKETEXCEPTION_H_INCLUDED */
