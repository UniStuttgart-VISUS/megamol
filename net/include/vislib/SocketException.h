/*
 * SocketException.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef SOCKETEXCEPTION_H_INCLUDED
#define SOCKETEXCEPTION_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/SystemException.h"


namespace vislib {
namespace net {

    /**
     * This exception indicates a socket error.
     */
    class SocketException : public sys::SystemException {

    public:

        /**
         * Ctor.
         *
		 * @param errorCode A socket error code.
         * @param file      The file the exception was thrown in.
         * @param line      The line the exception was thrown in.
         */
        SocketException(const DWORD errorCode, const char *file,
            const int line);

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

    };

} /* end namespace net */
} /* end namespace vislib */

#endif /* SOCKETEXCEPTION_H_INCLUDED */
