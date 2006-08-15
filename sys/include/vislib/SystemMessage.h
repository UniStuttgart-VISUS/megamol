/*
 * SystemMessage.h  15.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SYSTEMMESSAGE_H_INCLUDED
#define VISLIB_SYSTEMMESSAGE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/Exception.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {

    /**
     * This class wraps the human readable string of a system error code. It 
     * does the lookup, allocates appropriate memory and releases the memory
     * on destruction.
     */
    class SystemMessage {

    public:

        /**
         * Ctor.
         *
		 * @param errorCode A system dependent error code.
         */
        SystemMessage(const DWORD errorCode);

        /**
         * Create a clone of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        SystemMessage(const SystemMessage& rhs);

        /** Dtor. */
        ~SystemMessage(void);

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        SystemMessage& operator =(const SystemMessage& rhs);

        /**
         * Cast to TCHAR string. 
         *
         * This operator provides access to the human readable error message.
         * The object remains owner of the memory allocated for the string.
         *
         * The error message string is created lazily.
         *
         * @return The human readable error message.
         */
        operator const TCHAR *(void) const;

	private:

		/** A system dependent error code. */
		DWORD errorCode;

        /** The formatted message string. */
        mutable TCHAR *msg;
    };

} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_SYSTEMMESSAGE_H_INCLUDED */
