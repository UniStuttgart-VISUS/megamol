/*
 * AbstractSimpleMessageHeader.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractSimpleMessageHeader.h"

#include "vislib/IllegalParamException.h"
#include "vislib/assert.h"

/*
 * vislib::net::AbstractSimpleMessageHeader::~AbstractSimpleMessageHeader
 */
vislib::net::AbstractSimpleMessageHeader::~AbstractSimpleMessageHeader(void) {
	THE_STACK_TRACE;
}


/*
 * vislib::net::AbstractSimpleMessageHeader::SetMessageID
 */
void vislib::net::AbstractSimpleMessageHeader::SetMessageID(
		const SimpleMessageID messageID, bool isSystemID) {
	THE_STACK_TRACE;

	if (isSystemID && (messageID < VLSNP1_FIRST_RESERVED_MESSAGE_ID)) {
		throw IllegalParamException("messageID", __FILE__, __LINE__);
	} else if (!isSystemID && (messageID >= VLSNP1_FIRST_RESERVED_MESSAGE_ID)) {
		throw IllegalParamException("messageID", __FILE__, __LINE__);
	}

	this->PeekData()->MessageID = messageID;
}


/*
 * vislib::net::AbstractSimpleMessageHeader::operator =
 */
vislib::net::AbstractSimpleMessageHeader& 
vislib::net::AbstractSimpleMessageHeader::operator =(
		const AbstractSimpleMessageHeader& rhs) {
	THE_STACK_TRACE;

	if (this != &rhs) {
		::memcpy(this->PeekData(), rhs.PeekData(), this->GetHeaderSize());
	}

	return *this;
}


/*
 * vislib::net::AbstractSimpleMessageHeader::operator =
 */
vislib::net::AbstractSimpleMessageHeader& 
vislib::net::AbstractSimpleMessageHeader::operator =(
		const SimpleMessageHeaderData& rhs) {
	THE_STACK_TRACE;

	if (this->PeekData() != &rhs) {
		::memcpy(this->PeekData(), &rhs, this->GetHeaderSize());
	}

	return *this;
}


/*
 * vislib::net::AbstractSimpleMessageHeader::operator =
 */
vislib::net::AbstractSimpleMessageHeader& 
vislib::net::AbstractSimpleMessageHeader::operator =(
		const SimpleMessageHeaderData *rhs) {
	THE_STACK_TRACE;
	ASSERT(rhs != NULL);

	if (this->PeekData() != rhs) {
		::memcpy(this->PeekData(), rhs, this->GetHeaderSize());
	}

	return *this;
}


/*
 * vislib::net::AbstractSimpleMessageHeader::operator ==
 */
bool vislib::net::AbstractSimpleMessageHeader::operator ==(
		const AbstractSimpleMessageHeader& rhs) const {
	THE_STACK_TRACE;

	return (::memcmp(this->PeekData(), rhs.PeekData(),
		this->GetHeaderSize()) == 0);
}


/*
 * vislib::net::AbstractSimpleMessageHeader::AbstractSimpleMessageHeader
 */
vislib::net::AbstractSimpleMessageHeader::AbstractSimpleMessageHeader(void) {
	THE_STACK_TRACE;
}
