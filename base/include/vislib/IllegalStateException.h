/*
 * IllegalStateException.h  22.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ILLEGALSTATEEXCEPTION_H_INCLUDED
#define VISLIB_ILLEGALSTATEEXCEPTION_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "Exception.h"


namespace vislib {

    /**
     * This exception indicates that an object would enter an illegal state when
     * calling a method or is already in an illegal state.
     *
     * @author Christoph Mueller
     */
    class IllegalStateException : public Exception {

    public:

        /**
         * Ctor.
         *
         * @param msg  The exception detail message.
         * @param file The file the exception was thrown in.
         * @param line The line the exception was thrown in.
         */
        IllegalStateException(const TCHAR *msg, const char *file, 
            const int line);

        /**
         * Create a clone of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        IllegalStateException(const IllegalStateException& rhs);

        /** Dtor. */
        virtual ~IllegalStateException(void);

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        virtual IllegalStateException& operator =(
			const IllegalStateException& rhs);
    };
}

#endif /* VISLIB_ILLEGALSTATEEXCEPTION_H_INCLUDED */
