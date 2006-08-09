/*
 * IOException.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef IOEXCEPTION_H_INCLUDED
#define IOEXCEPTION_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/SystemException.h"


namespace vislib {
namespace sys {

    /**
     * This exception indicates an I/O error.
     *
     * @author Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de)
     */
    class IOException : public SystemException {

    public:

        /**
         * Ctor.
         *
		 * @param errorCode A system dependent error code.
         * @param file      The file the exception was thrown in.
         * @param line      The line the exception was thrown in.
         */
        IOException(const DWORD errorCode, const char *file,
            const int line);

        /**
         * Ctor.
         *
		 * @param errorCode A system dependent error code.
		 * @param msg       A detailed error message.
         * @param file      The file the exception was thrown in.
         * @param line      The line the exception was thrown in.
         */
        IOException(const DWORD errorCode, const char *msg, const char *file,
            const int line);

        /**
         * Create a clone of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        IOException(const IOException& rhs);

        /** Dtor. */
        virtual ~IOException(void);

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        virtual IOException& operator =(const IOException& rhs);

    };

} /* end namespace sys */
} /* end namespace vislib */

#endif /* IOEXCEPTION_H_INCLUDED */
