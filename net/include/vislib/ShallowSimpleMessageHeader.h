/*
 * ShallowSimpleMessageHeader.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SHALLOWSIMPLEMESSAGEHEADER_H_INCLUDED
#define VISLIB_SHALLOWSIMPLEMESSAGEHEADER_H_INCLUDED
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
     * TODO: comment class
     */
    class ShallowSimpleMessageHeader : public AbstractSimpleMessageHeader {

    public:

		explicit ShallowSimpleMessageHeader(SimpleMessageHeaderData *data);

        /** Dtor. */
        virtual ~ShallowSimpleMessageHeader(void);

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
		inline ShallowSimpleMessageHeader& operator =(
				const ShallowSimpleMessageHeader& rhs) {
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
		inline ShallowSimpleMessageHeader& operator =(
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
		inline ShallowSimpleMessageHeader& operator =(
				const SimpleMessageHeaderData *rhs) {
			VLSTACKTRACE("SimpleMessageHeader::operator =", __FILE__, __LINE__);
			Super::operator =(rhs);
			return *this;
		}

	private:

		/** Superclass typedef. */
		typedef AbstractSimpleMessageHeader Super;

		/** 
		 * Pointer to the actual data. The object is not the owner of the memory
		 * designated by this pointer.
		 */
		SimpleMessageHeaderData *data;

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SHALLOWSIMPLEMESSAGEHEADER_H_INCLUDED */

