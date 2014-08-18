/*
 * AbstractSimpleMessageHeader.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTSIMPLEMESSAGEHEADER_H_INCLUDED
#define VISLIB_ABSTRACTSIMPLEMESSAGEHEADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/SimpleMessageHeaderData.h"
#include "vislib/StackTrace.h"


namespace vislib {
namespace net {


    /**
     * This class provides the interface for a message header in the simple 
	 * network protocol implementation of VISlib. There are two subclasses of
	 * this abstract class: SimpleMessageHeader is an implementation which
	 * provides storage for the header data. ShallowSimpleMessageHeader does
	 * not provide any storage, but takes a pointer to memory containing the
	 * header data.
     */
    class AbstractSimpleMessageHeader {

    public:

        /** Dtor. */
        virtual ~AbstractSimpleMessageHeader(void);

		/**
		 * Answer the body size stored in the message header.
		 *
		 * @return The body size.
		 */
		inline SimpleMessageSize GetBodySize(void) const {
			VLSTACKTRACE("SimpleMessageHeader::GetBodySize", __FILE__, 
				__LINE__);
			return this->PeekData()->BodySize;
		}

		/**
		 * Answer the size of the header packet. This is the size of the data
		 * returned by PeekData().
		 *
		 * @return The size of the header data in bytes.
		 */
		inline SimpleMessageSize GetHeaderSize(void) const {
			VLSTACKTRACE("SimpleMessageHeader::GetHeaderSize", __FILE__, 
				__LINE__);
			return sizeof(SimpleMessageHeaderData);
		}

		/**
		 * Answer the message ID.
		 *
		 * @return The message ID.
		 */
		inline SimpleMessageID GetMessageID(void) const {
			VLSTACKTRACE("SimpleMessageHeader::GetMessageID", __FILE__, 
				__LINE__);
			return this->PeekData()->MessageID;
		}

		/**
		 * Answer whether the body size is not zero.
		 *
		 * @return true if the body size is larger than zero, false otherwise.
		 */
		inline bool HasBody(void) const {
			VLSTACKTRACE("SimpleMessageHeader::HasBody", __FILE__, __LINE__);
			return (this->PeekData()->BodySize > 0);
		}

		/**
		 * Provides direct access to the underlying SimpleMessageHeaderData.
		 *
		 * @return A pointer to the message header data.
		 */
		virtual SimpleMessageHeaderData *PeekData(void) = 0;

		/**
		 * Provides direct access to the underlying SimpleMessageHeaderData.
		 *
		 * @return A pointer to the message header data.
		 */
		virtual const SimpleMessageHeaderData *PeekData(void) const = 0;

		/**
		 * Set the body size.
		 *
		 * @param bodySize The body size.
		 */
		inline void SetBodySize(const SimpleMessageSize bodySize) {
			VLSTACKTRACE("SimpleMessageHeader::SetBodySize", __FILE__, 
				__LINE__);
			this->PeekData()->BodySize = bodySize;
		}

		/**
		 * Set a new message ID.
		 *
		 * @param messageID  The new message ID.
		 * @param isSystemID Disables the system ID check. Must be false.
		 *
		 * @throw IllegalParamException If the message ID is a system ID.
		 */
		void SetMessageID(const SimpleMessageID messageID, 
            bool isSystemID = false);

		/**
		 * Assignment operator.
		 *
		 * @param The right hand side operand.
		 *
		 * @return *this
		 */
		AbstractSimpleMessageHeader& operator =(
			const AbstractSimpleMessageHeader& rhs);

		/**
		 * Assignment operator.
		 *
		 * @param The right hand side operand.
		 *
		 * @return *this
		 */
		AbstractSimpleMessageHeader& operator =(
			const SimpleMessageHeaderData& rhs);

		/**
		 * Assignment operator.
		 *
		 * @param The right hand side operand. This must not be NULL.
		 *
		 * @return *this
		 */
		AbstractSimpleMessageHeader& operator =(
			const SimpleMessageHeaderData *rhs);

		/**
		 * Test for equality.
		 *
		 * @param The right hand side operand.
		 * 
		 * @return true in case this object and 'rhs' are equal, 
		 *         false otherwise.
		 */
		bool operator ==(const AbstractSimpleMessageHeader& rhs) const;

		/**
		 * Test for inequality.
		 *
		 * @param The right hand side operand.
		 * 
		 * @return true in case this object and 'rhs' are not equal, 
		 *         false otherwise.
		 */
		inline bool operator !=(const AbstractSimpleMessageHeader& rhs) const {
			VLSTACKTRACE("AbstractSimpleMessageHeader::operator !=", __FILE__,
				__LINE__);
			return !(*this == rhs);
		}

    protected:

        /** Ctor. */
        AbstractSimpleMessageHeader(void);

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTSIMPLEMESSAGEHEADER_H_INCLUDED */
