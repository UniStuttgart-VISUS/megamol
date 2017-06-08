/*
 * UnsupportedOperationException.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_UNSUPPORTEDOPERATIONEXCEPTION_H_INCLUDED
#define VISLIB_UNSUPPORTEDOPERATIONEXCEPTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "Exception.h"


namespace vislib {

    /**
     * This exception indicates an illegal parameter value.
     *
     * @author Christoph Mueller
     */
    class UnsupportedOperationException : public Exception {

    public:

        /**
         * Ctor.
         *
         * @param funcName Name of the unsupported method or function.
         * @param file     The file the exception was thrown in.
         * @param line     The line the exception was thrown in.
         */
        UnsupportedOperationException(const char *funcName, const char *file, 
            const int line);

        /**
         * Ctor.
         *
         * @param funcName Name of the unsupported method or function.
         * @param file     The file the exception was thrown in.
         * @param line     The line the exception was thrown in.
         */
        UnsupportedOperationException(const wchar_t *funcName, const char *file, 
            const int line);

        /**
         * Create a clone of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        UnsupportedOperationException(const UnsupportedOperationException& rhs);

        /** Dtor. */
        virtual ~UnsupportedOperationException(void);

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        virtual UnsupportedOperationException& operator =(
			const UnsupportedOperationException& rhs);
    };
}

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_UNSUPPORTEDOPERATIONEXCEPTION_H_INCLUDED */
