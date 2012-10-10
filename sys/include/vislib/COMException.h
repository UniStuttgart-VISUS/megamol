/*
 * COMException.h
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_COMEXCEPTION_H_INCLUDED
#define VISLIB_COMEXCEPTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifdef _WIN32
#include <comdef.h>
#endif /* _WIN32 */

#include "vislib/Exception.h"


namespace vislib {
namespace sys {


    /**
     * This exception represents a COM error (HRESULT).
     */
    class COMException : public Exception {

    public:

#ifdef _WIN32
        /**
         * Ctor.
         *
         * @param hr   A COM error code.
         * @param file The file the exception was thrown in.
         * @param line The line the exception was thrown in.
         */
        COMException(const HRESULT hr, const char *file, const int line);
# else /* _WIN32 */
        COMException(const char *file, const int line);
#endif /* _WIN32 */

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        COMException(const COMException& rhs);


        /** Dtor. */
        virtual ~COMException(void);

#ifdef _WIN32
        /**
         * Gets the COM error code associated with the exception.
         *
         * @return The error code.
         */
        inline HRESULT GetErrorCode(void) const {
            return this->hr;
        }
#endif /* _WIN32 */

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        COMException& operator =(const COMException& rhs);

    private:

        /** Superclass alias. */
        typedef Exception Super;

#ifdef _WIN32
        /** The COM error code. */
        HRESULT hr;
#endif /* _WIN32 */

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_COMEXCEPTION_H_INCLUDED */

