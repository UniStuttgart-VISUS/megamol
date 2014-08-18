/*
 * PAMException.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_PAMEXCEPTION_H_INCLUDED
#define VISLIB_PAMEXCEPTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifndef _WIN32
#include <security/pam_appl.h>
#endif /* !_WIN32 */

#include "vislib/Exception.h"


namespace vislib {
namespace sys {


    /**
     * This exception represents a Linux PAM error. It has no meaning on Windows
     * systems and should never been thrown there.
     */
    class PAMException : public Exception {

    public:

#ifndef _WIN32
        /** 
         * Create a new instance.
         *
         * @param hPAM      The handle of the PAM context.
         * @param errorCode The PAM error code represented by the exception. 
         *                  Note, that the exception is in an invalid state, if
         *                  this is not a valid PAM error code.
         * @param file      The file the exception was thrown in.
         * @param line      The line the exception was thrown in.
         */
        PAMException(pam_handle_t *hPAM, const int errorCode, const char *file, 
            const int line);
#endif /* !_WIN32 */

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        PAMException(const PAMException& rhs);

        /** Dtor. */
        virtual ~PAMException(void);

        /**
         * Answer the PAM error code associated with this exception.
         *
         * @return the error code.
         */
        inline int GetErrorCode(void) const {
            return this->errorCode;
        }

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        PAMException& operator =(const PAMException& rhs);

    private:

#ifdef _WIN32
        /** Disallow instances on windows. */
        PAMException(void);
#endif /* _WIN32 */

        /** The PAM error code. */
        int errorCode;

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_PAMEXCEPTION_H_INCLUDED */
