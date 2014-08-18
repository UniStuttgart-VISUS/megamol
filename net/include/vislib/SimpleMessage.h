/*
 * SimpleMessage.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SIMPLEMESSAGE_H_INCLUDED
#define VISLIB_SIMPLEMESSAGE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractSimpleMessage.h"
#include "vislib/RawStorage.h"


namespace vislib {
namespace net {


    /**
     * This class implements a message that manages the storage for the
     * header and the message body. It is normally recommended using this
     * message class.
     */
    class SimpleMessage : public AbstractSimpleMessage {

    public:

        /** 
         * Create a new message with the specified number of bytes for the 
         * message body. The storage for the message header is added 
         * automatically. The body size is automatically set in the message
         * header.
         *
         * @param bodySize The size of the message body to be allocated in
         *                 bytes.
         */
        SimpleMessage(const SIZE_T bodySize = 0);

        /**
         * Create a new message using the size specified in 'header'. The 
         * complete header will be used to initialise the message. If a
         * message body is specified, it is copied, too. 'body' must be a
         * memory block of at least header->GetBodySize() bytes.
         *
         * @param header The message header containing the description of the
         *               message to be created.
         * @param body   The initial data for the message body. If this is a
         *               NULL pointer, the message is not initialised.
         */
        SimpleMessage(const AbstractSimpleMessageHeader& header, 
            const void *body = NULL);

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        SimpleMessage(const SimpleMessage& rhs);
        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        SimpleMessage(const AbstractSimpleMessage& rhs);

        /** Dtor. */
        virtual ~SimpleMessage(void);

        /**
         * Trim the storage of the message to hold the actual size of the
         * header and the current content.
         */
        void Trim(void);

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline SimpleMessage& operator =(const AbstractSimpleMessage& rhs) {
            VLSTACKTRACE("SimpleMessage::operator =", __FILE__, __LINE__);
            Super::operator =(rhs);
            return *this;
        }

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline SimpleMessage& operator =(const SimpleMessage& rhs) {
            VLSTACKTRACE("SimpleMessage::operator =", __FILE__, __LINE__);
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /**
         * Ensure that the underlying storage has enough memory to store a message 
         * (including header) with the specified size. The caller
         * must add the size of the header.
         *
         * @param outStorage This variable receives the pointer to the begin of
         *                   the storage.
         * @param size       The size of the memory to be allocated in bytes.
         *
         * @return true if the storage has been reallocated, false if it remains
         *         the same (i. e. the pointer has not been changed).
         *
         * @throws Exception or derived in case of an error.
         */
        virtual bool assertStorage(void *& outStorage, const SIZE_T size);

    private:

        /** Superclass typedef. */
        typedef AbstractSimpleMessage Super;

        /** Storage for the message and its header. */
        RawStorage storage;

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SIMPLEMESSAGE_H_INCLUDED */
