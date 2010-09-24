/*
 * UnexpectedMessageException.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_UNEXPECTEDMESSAGEEXCEPTION_H_INCLUDED
#define VISLIB_UNEXPECTEDMESSAGEEXCEPTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Exception.h"
#include "vislib/SimpleMessageHeaderData.h"


namespace vislib {
namespace net {


    /**
     * This exception should be thrown in case a SimpleMessage was received
     * which was not expected.
     */
    class UnexpectedMessageException : public Exception {

    public:

        /**
         * Ctor.
         *
         * @param actualID   The ID of the message which was actually received.
         * @param expectedID The ID of the message which was expected.
         * @param file       The file the exception was thrown in.
         * @param line       The line the exception was thrown in.
         */
        UnexpectedMessageException(const SimpleMessageID actualID,
            const SimpleMessageID expectedID, const char *file, const int line);

        /**
         * Create a clone of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        UnexpectedMessageException(const UnexpectedMessageException& rhs);

        /** Dtor. */
        virtual ~UnexpectedMessageException(void);

        /**
         * Answer the actually received message ID.
         *
         * @return The actually received message ID.
         */
        inline SimpleMessageID GetActualID(void) const {
            return this->actualID;
        }

        /**
         * Answer the expected message ID.
         *
         * @return The expected message ID.
         */
        inline SimpleMessageID GetExpectedID(void) const {
            return this->expectedID;
        }

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        virtual UnexpectedMessageException& operator =(
			const UnexpectedMessageException& rhs);

    private:

        /** The ID of the message which was actually received. */
        SimpleMessageID actualID;

        /** The ID of the message which was expected. */
        SimpleMessageID expectedID;
    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_UNEXPECTEDMESSAGEEXCEPTION_H_INCLUDED */

