/*
 * AbstractSimpleMessageHeader.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractSimpleMessageHeader.h"

#include "vislib/IllegalParamException.h"


/*
 * vislib::net::AbstractSimpleMessageHeader::~AbstractSimpleMessageHeader
 */
vislib::net::AbstractSimpleMessageHeader::~AbstractSimpleMessageHeader(void) {
	VLSTACKTRACE("AbstractSimpleMessageHeader::~AbstractSimpleMessageHeader",
		__FILE__, __LINE__);
}


/*
 * vislib::net::AbstractSimpleMessageHeader::SetMessageID
 */
void vislib::net::AbstractSimpleMessageHeader::SetMessageID(
		const SimpleMessageID messageID, bool isSystemID) {
	VLSTACKTRACE("AbstractSimpleMessageHeader::SetMessageID", 
		__FILE__, __LINE__);

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
	VLSTACKTRACE("AbstractSimpleMessageHeader::operator =", 
		__FILE__, __LINE__);

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
	VLSTACKTRACE("AbstractSimpleMessageHeader::operator =", 
		__FILE__, __LINE__);

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
	VLSTACKTRACE("AbstractSimpleMessageHeader::operator =", 
		__FILE__, __LINE__);
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
	VLSTACKTRACE("AbstractSimpleMessageHeader::operator ==", 
		__FILE__, __LINE__);

	return (::memcmp(this->PeekData(), rhs.PeekData(),
		this->GetHeaderSize()) == 0);
}


/*
 * vislib::net::AbstractSimpleMessageHeader::AbstractSimpleMessageHeader
 */
vislib::net::AbstractSimpleMessageHeader::AbstractSimpleMessageHeader(void) {
	VLSTACKTRACE("AbstractSimpleMessageHeader::AbstractSimpleMessageHeader",
		__FILE__, __LINE__);
}
