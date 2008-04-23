/*
 * AlreadyExistsException.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ALREADYEXISTSEXCEPTION_H_INCLUDED
#define VISLIB_ALREADYEXISTSEXCEPTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/Exception.h"


namespace vislib {


    /**
     * This exception should be used to indicated that a creation of an 
	 * element failed because it must be unique and there already is another
	 * equal element.
     */
	class AlreadyExistsException : public Exception {
    public:

        /**
         * Ctor.
         *
         * @param msg  A human readable message.
         * @param file The file the exception was thrown in.
         * @param line The line the exception was thrown in.
         */
        AlreadyExistsException(const char *msg, const char *file, 
            const int line);

        /**
         * Create a clone of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        AlreadyExistsException(const AlreadyExistsException& rhs);

        /** Dtor. */
        ~AlreadyExistsException(void);

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AlreadyExistsException& operator =(const AlreadyExistsException& rhs);

	};
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ALREADYEXISTSEXCEPTION_H_INCLUDED */

