/*
 * SystemMessage.h  15.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SYSTEMMESSAGE_H_INCLUDED
#define VISLIB_SYSTEMMESSAGE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


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
         * Cast to char string. 
         *
         * This operator provides access to the human readable error message.
         * The returned pointer is valid until the object is destroyed or 
         * another cast operator is called.
         * The object remains owner of the memory allocated for the string.
         *
         * The error message string is created lazily.
         *
         * @return The human readable error message.
         */
        operator const char *(void) const;

        /**
         * Cast to wchar_t string. 
         *
         * This operator provides access to the human readable error message.
         * The returned pointer is valid until the object is destroyed or 
         * another cast operator is called.
         * The object remains owner of the memory allocated for the string.
         *
         * The error message string is created lazily.
         *
         * @return The human readable error message.
         */
        operator const wchar_t *(void) const;

        /**
         * Answer the system dependent error code associated with this 
         * message.
         *
         * @return The system error code.
         */
        inline DWORD GetErrorCode(void) const {
            return this->errorCode;
        }

    private:

        /** A system dependent error code. */
        DWORD errorCode;

        /** Remember whether 'msg' points to a Unicode or ANSI string. */
        mutable bool isMsgUnicode;

        /** The formatted message string. */
        mutable void *msg;
    };

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SYSTEMMESSAGE_H_INCLUDED */
