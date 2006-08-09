/*
 * SystemException.h  09.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SYSTEMEXCEPTION_H_INCLUDED
#define VISLIB_SYSTEMEXCEPTION_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/Exception.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {

    /**
     * This exception class represents a system error. It is instantiated with 
     * a system error code and retrieves the error message from the system.
     *
     * @author Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de)
     */
    class SystemException : public Exception {

    public:

        /**
         * Ctor.
         *
		 * @param errorCode A system dependent error code.
         * @param file      The file the exception was thrown in.
         * @param line      The line the exception was thrown in.
         */
        SystemException(const DWORD errorCode, const char *file,
            const int line);

        /**
         * Create a clone of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        SystemException(const SystemException& rhs);

        /** Dtor. */
        virtual ~SystemException(void);

		/**
		 * Answer the system dependent error code associated with this 
		 * exception.
		 *
		 * @return The system error code.
		 */
		inline DWORD GetErrorCode(void) const {
			return this->errorCode;
		}

        /**
         * Answer the file the exception description text. The onwnership of the
         * memory remains at the object.
         *
         * @return The exception message.
         */
        virtual const TCHAR *GetMsg(void) const;

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        virtual SystemException& operator =(const SystemException& rhs);

	private:

		/** A system dependent error code. */
		DWORD errorCode;
    };

} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_SYSTEMEXCEPTION_H_INCLUDED */
