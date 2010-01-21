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


#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
#include "vislib/SimpleMessageHeaderData.h"
#include "vislib/StackTrace.h"


namespace vislib {
namespace net {


    /**
     * TODO: comment class
	 *
	 * The template parameter S must be either SimpleMessageHeaderData[1] or
	 * SimpleMessageHeaderData *
     */
    template<class S> class AbstractSimpleMessageHeader {

    public:

        /** Dtor. */
        virtual ~AbstractSimpleMessageHeader(void);

		/**
		 * Answer the body size stored in the message header.
		 *
		 * @return The body size.
		 */
		inline UINT32 GetBodySize(void) const {
			VLSTACKTRACE("SimpleMessageHeader::GetBodySize", __FILE__, 
				__LINE__);
			return this->data->BodySize;
		}

		/**
		 * Answer the size of the header packet. This is the size of the data
		 * returned by PeekData().
		 *
		 * @return The size of the header data in bytes.
		 */
		inline UINT32 GetHeaderSize(void) const {
			VLSTACKTRACE("SimpleMessageHeader::GetHeaderSize", __FILE__, 
				__LINE__);
			// Note: As S is a pointer, we cannot do anything else here than
			// hard-coding the header data type.
			return sizeof(SimpleMessageHeaderData);
		}

		/**
		 * Answer the message ID.
		 *
		 * @return The message ID.
		 */
		inline UINT32 GetMessageID(void) const {
			VLSTACKTRACE("SimpleMessageHeader::GetMessageID", __FILE__, 
				__LINE__);
			return this->data->MessageID;
		}

		/**
		 * Answer whether the body size is not zero.
		 *
		 * @return true if the body size is larger than zero, false otherwise.
		 */
		inline bool HasBody(void) const {
			VLSTACKTRACE("SimpleMessageHeader::HasBody", __FILE__, __LINE__);
			return (this->data->BodySize > 0);
		}

		/**
		 * Provides direct access to the underlying SimpleMessageHeaderData.
		 *
		 * @return A pointer to the message header data.
		 */
		inline SimpleMessageHeaderData *PeekData(void) {
			VLSTACKTRACE("SimpleMessageHeader::PeekData", __FILE__, __LINE__);
			return this->data;
		}

		/**
		 * Provides direct access to the underlying SimpleMessageHeaderData.
		 *
		 * @return A pointer to the message header data.
		 */
		inline const SimpleMessageHeaderData *PeekData(void) const {
			VLSTACKTRACE("SimpleMessageHeader::PeekData", __FILE__, __LINE__);
			return this->data;
		}

		/**
		 * Set the body size.
		 *
		 * @param bodySize The body size.
		 */
		inline void SetBodySize(const UINT32 bodySize) {
			VLSTACKTRACE("SimpleMessageHeader::SetBodySize", __FILE__, 
				__LINE__);
			this->data->BodySize = bodySize;
		}

		/**
		 * Set a new message ID.
		 *
		 * @param messageID  The new message ID.
		 * @param isSystemID Disables the system ID check. Must be false.
		 *
		 * @throw IllegalParamException If the message ID is a system ID.
		 */
		void SetMessageID(const UINT32 messageID, bool isSystemID = false);

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
		AbstractSimpleMessageHeader& operator =(const SimpleMessageHeaderData& rhs);

		/**
		 * Assignment operator.
		 *
		 * @param The right hand side operand. This must not be NULL.
		 *
		 * @return *this
		 */
		AbstractSimpleMessageHeader& operator =(const SimpleMessageHeaderData *rhs);

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

		/** 
		 * The message header data. This can be either a pointer to an external
		 * piece of memory or an array of SimpleMessageHeaderData with exactly
		 * one element.
		 */
		S data;

    };


	/*
	 * vislib::net::AbstractSimpleMessageHeader<S>::~AbstractSimpleMessageHeader
	 */
	template<class S>
	AbstractSimpleMessageHeader<S>::~AbstractSimpleMessageHeader(void) {
		VLSTACKTRACE("AbstractSimpleMessageHeader::"
			"~AbstractSimpleMessageHeader", __FILE__, __LINE__);
	}


	/*
	 * vislib::net::AbstractSimpleMessageHeader<S>::SetMessageID
	 */
	template<class S>
	void vislib::net::AbstractSimpleMessageHeader<S>::SetMessageID(
			const UINT32 messageID, bool isSystemID) {
		VLSTACKTRACE("AbstractSimpleMessageHeader::SetMessageID", __FILE__, 
			__LINE__);
		if (isSystemID && (messageID < VLSNP1_FIRST_RESERVED_MESSAGE_ID)) {
			throw IllegalParamException("messageID", __FILE__, __LINE__);
		} else if (!isSystemID && (messageID >= VLSNP1_FIRST_RESERVED_MESSAGE_ID)) {
			throw IllegalParamException("messageID", __FILE__, __LINE__);
		}

		this->data->MessageID = messageID;
	}


	/*
	 * vislib::net::AbstractSimpleMessageHeader<S>::operator =
	 */
	template<class S>
	AbstractSimpleMessageHeader<S>& AbstractSimpleMessageHeader<S>::operator =(
			const AbstractSimpleMessageHeader& rhs) {
		VLSTACKTRACE("AbstractSimpleMessageHeader::operator =", __FILE__, 
			__LINE__);

		if (this != &rhs) {
			::memcpy(this->data, rhs.data, this->GetHeaderSize());
		}

		return *this;
	}


	/*
	 * vislib::net::AbstractSimpleMessageHeader<S>::operator =
	 */
	template<class S>
	AbstractSimpleMessageHeader<S>& AbstractSimpleMessageHeader<S>::operator =(
			const SimpleMessageHeaderData& rhs) {
		VLSTACKTRACE("AbstractSimpleMessageHeader::operator =", __FILE__, 
			__LINE__);

		if (this->data != &rhs) {
			::memcpy(this->data, &rhs, this->GetHeaderSize());
		}

		return *this;
	}


	/*
	 * vislib::net::AbstractSimpleMessageHeader<S>::operator =
	 */
	template<class S>
	AbstractSimpleMessageHeader<S>& AbstractSimpleMessageHeader<S>::operator =(
			const SimpleMessageHeaderData *rhs) {
		VLSTACKTRACE("AbstractSimpleMessageHeader::operator =", __FILE__, 
			__LINE__);
		ASSERT(rhs != NULL);

		if (this->data != rhs) {
			::memcpy(this->data, rhs, this->GetHeaderSize());
		}

		return *this;
	}


	/*
	 * vislib::net::AbstractSimpleMessageHeader<S>::operator ==
	 */
	template<class S>
	bool AbstractSimpleMessageHeader<S>::operator ==(
			const AbstractSimpleMessageHeader& rhs) const {
		VLSTACKTRACE("AbstractSimpleMessageHeader::operator ==", __FILE__,
			__LINE__);
		return (::memcmp(this->data, rhs.data, this->GetHeaderSize()) 
			== 0);
		
	}


	/*
	 * vislib::net::AbstractSimpleMessageHeader<S>::AbstractSimpleMessageHeader
	 */
	template<class S>
	AbstractSimpleMessageHeader<S>::AbstractSimpleMessageHeader(void) {
		VLSTACKTRACE("AbstractSimpleMessageHeader::AbstractSimpleMessageHeader",
			__FILE__, __LINE__);
	}
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTSIMPLEMESSAGEHEADER_H_INCLUDED */

