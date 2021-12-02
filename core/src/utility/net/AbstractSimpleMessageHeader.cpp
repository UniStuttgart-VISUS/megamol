/*
 * AbstractSimpleMessageHeader.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "mmcore/utility/net/AbstractSimpleMessageHeader.h"

#include "vislib/IllegalParamException.h"
#include "vislib/assert.h"

#include <cstring>


/*
 * vislib::net::AbstractSimpleMessageHeader::~AbstractSimpleMessageHeader
 */
vislib::net::AbstractSimpleMessageHeader::~AbstractSimpleMessageHeader(void) {}


/*
 * vislib::net::AbstractSimpleMessageHeader::SetMessageID
 */
void vislib::net::AbstractSimpleMessageHeader::SetMessageID(const SimpleMessageID messageID, bool isSystemID) {

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
vislib::net::AbstractSimpleMessageHeader& vislib::net::AbstractSimpleMessageHeader::operator=(
    const AbstractSimpleMessageHeader& rhs) {

    if (this != &rhs) {
        std::memcpy(this->PeekData(), rhs.PeekData(), this->GetHeaderSize());
    }

    return *this;
}


/*
 * vislib::net::AbstractSimpleMessageHeader::operator =
 */
vislib::net::AbstractSimpleMessageHeader& vislib::net::AbstractSimpleMessageHeader::operator=(
    const SimpleMessageHeaderData& rhs) {

    if (this->PeekData() != &rhs) {
        std::memcpy(this->PeekData(), &rhs, this->GetHeaderSize());
    }

    return *this;
}


/*
 * vislib::net::AbstractSimpleMessageHeader::operator =
 */
vislib::net::AbstractSimpleMessageHeader& vislib::net::AbstractSimpleMessageHeader::operator=(
    const SimpleMessageHeaderData* rhs) {
    ASSERT(rhs != NULL);

    if (this->PeekData() != rhs) {
        std::memcpy(this->PeekData(), rhs, this->GetHeaderSize());
    }

    return *this;
}


/*
 * vislib::net::AbstractSimpleMessageHeader::operator ==
 */
bool vislib::net::AbstractSimpleMessageHeader::operator==(const AbstractSimpleMessageHeader& rhs) const {

    return (std::memcmp(this->PeekData(), rhs.PeekData(), this->GetHeaderSize()) == 0);
}


/*
 * vislib::net::AbstractSimpleMessageHeader::AbstractSimpleMessageHeader
 */
vislib::net::AbstractSimpleMessageHeader::AbstractSimpleMessageHeader(void) {}
