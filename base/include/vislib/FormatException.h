/*
 * FormatException.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_FORMATEXCEPTION_H_INCLUDED
#define VISLIB_FORMATEXCEPTION_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/Exception.h"


namespace vislib {


    /**
     * This exception indicates an format error.
     * E.g. if a request for an integer value is answered with a non-convertable string.
     */
    class FormatException : public Exception {

    public:

        /**
         * Ctor.
         *
         * @param msg  The exception detail message.
         * @param file The file the exception was thrown in.
         * @param line The line the exception was thrown in.
         */
        FormatException(const char *msg, const char *file, 
            const int line);

        /**
         * Ctor.
         *
         * @param msg  The exception detail message.
         * @param file The file the exception was thrown in.
         * @param line The line the exception was thrown in.
         */
        FormatException(const wchar_t *msg, const char *file, 
            const int line);

        /**
         * Create a clone of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        FormatException(const FormatException& rhs);

        /** Dtor. */
        virtual ~FormatException(void);

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        virtual FormatException& operator =(const FormatException& rhs);

    };
    
} /* end namespace vislib */

#endif /* VISLIB_FORMATEXCEPTION_H_INCLUDED */

