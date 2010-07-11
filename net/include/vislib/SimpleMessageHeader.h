/*
 * SimpleMessageHeader.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SIMPLEMESSAGEHEADER_H_INCLUDED
#define VISLIB_SIMPLEMESSAGEHEADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractSimpleMessageHeader.h"


namespace vislib {
namespace net {


    /**
     * This class represents a message header consisting of 
     * SimpleMessageHeaderData.
     */
    class SimpleMessageHeader : public AbstractSimpleMessageHeader {

    public:

        /** Ctor. */
        SimpleMessageHeader(void);

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned. 
         */
        SimpleMessageHeader(const SimpleMessageHeader& rhs);

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned. 
         */
        SimpleMessageHeader(const AbstractSimpleMessageHeader& rhs);

        /**
         * Assign the given header data to this message header.
         *
         * @param data The message header data.
         */
        SimpleMessageHeader(const SimpleMessageHeaderData& data);

        /**
         * Assign the given header data to this message header.
         *
         * @param data Pointer to the message header data. This must not be 
         *             NULL.
         */
        explicit SimpleMessageHeader(const SimpleMessageHeaderData *data);

        /** Dtor. */
        virtual ~SimpleMessageHeader(void);

        /**
         * Provides direct access to the underlying SimpleMessageHeaderData.
         *
         * @return A pointer to the message header data.
         */
        virtual SimpleMessageHeaderData *PeekData(void);

        /**
         * Provides direct access to the underlying SimpleMessageHeaderData.
         *
         * @return A pointer to the message header data.
         */
        virtual const SimpleMessageHeaderData *PeekData(void) const;

        /**
         * Assignment operator.
         *
         * @param The right hand side operand.
         *
         * @return *this
         */
        inline SimpleMessageHeader& operator =(const SimpleMessageHeader& rhs) {
            VLSTACKTRACE("SimpleMessageHeader::operator =", __FILE__, __LINE__);
            Super::operator =(rhs);
            return *this;
        }

        /**
         * Assignment operator.
         *
         * @param The right hand side operand.
         *
         * @return *this
         */
        inline SimpleMessageHeader& operator =(
                const AbstractSimpleMessageHeader& rhs) {
            VLSTACKTRACE("SimpleMessageHeader::operator =", __FILE__, __LINE__);
            Super::operator =(rhs);
            return *this;
        }

        /**
         * Assignment operator.
         *
         * @param The right hand side operand.
         *
         * @return *this
         */
        inline SimpleMessageHeader& operator =(
                const SimpleMessageHeaderData& rhs) {
            VLSTACKTRACE("SimpleMessageHeader::operator =", __FILE__, __LINE__);
            Super::operator =(rhs);
            return *this;
        }

        /**
         * Assignment operator.
         *
         * @param The right hand side operand. This must not be NULL.
         *
         * @return *this
         */
        inline SimpleMessageHeader& operator =(
                const SimpleMessageHeaderData *rhs) {
            VLSTACKTRACE("SimpleMessageHeader::operator =", __FILE__, __LINE__);
            Super::operator =(rhs);
            return *this;
        }

    private:

        /** Superclass typedef. */
        typedef AbstractSimpleMessageHeader Super;

        /** The actual header data. */
        SimpleMessageHeaderData data;

    };


} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SIMPLEMESSAGEHEADER_H_INCLUDED */
