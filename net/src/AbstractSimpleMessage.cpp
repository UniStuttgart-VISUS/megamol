/*
 * AbstractSimpleMessage.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractSimpleMessage.h"

#include <climits>


/*
 * vislib::net::AbstractSimpleMessage::~AbstractSimpleMessage
 */
vislib::net::AbstractSimpleMessage::~AbstractSimpleMessage(void) {
    VLSTACKTRACE("AbstractSimpleMessage::~AbstractSimpleMessage", 
        __FILE__, __LINE__);
}


/*
 * vislib::net::AbstractSimpleMessage::AbstractSimpleMessage
 */
vislib::net::AbstractSimpleMessage::AbstractSimpleMessage(void) {
    VLSTACKTRACE("AbstractSimpleMessage::AbstractSimpleMessage", 
        __FILE__, __LINE__);
}


/*
 * vislib::net::AbstractSimpleMessage::GetBody
 */
const void *vislib::net::AbstractSimpleMessage::GetBody(void) const {
    VLSTACKTRACE("AbstractSimpleMessage::GetBody", __FILE__, __LINE__);
    // If that asserts, the child class probably does not initialise correctly.
    ASSERT(this->header.PeekData() != NULL);
    return (this->header.PeekData() + 1);
}


/*
 * vislib::net::AbstractSimpleMessage::GetBody
 */
void *vislib::net::AbstractSimpleMessage::GetBody(void) {
    VLSTACKTRACE("AbstractSimpleMessage::GetBody", __FILE__, __LINE__);
    // If that asserts, the child class probably does not initialise correctly.
    ASSERT(this->header.PeekData() != NULL);
    return const_cast<SimpleMessageHeaderData *>(this->header.PeekData() + 1);
}



/*
 * vislib::net::AbstractSimpleMessage::SetBody
 */
void vislib::net::AbstractSimpleMessage::SetBody(const void *body, 
        const SIZE_T bodySize) {
    VLSTACKTRACE("AbstractSimpleMessage::SetBody", __FILE__, __LINE__);
    // If that asserts, the child class probably does not initialise correctly.
    ASSERT(this->header.PeekData() != NULL);

    void *b = NULL;
    SIZE_T bs = bodySize;
    
    if (body != NULL) {
        if (bs == 0) {
            /* No body size was passed, assume old size. */
            bs = this->header.GetBodySize();
        } 

        this->header.SetBodySize(
            static_cast<vislib::net::SimpleMessageSize>(bs));
        this->AssertBodySize();
        b = this->GetBody();
        ASSERT(b != NULL);

        ::memcpy(b, body, bs);

    } else {
        /* No body was passed, force size to zero bytes. */
        this->header.SetBodySize(0);
    }
}


/*
 * vislib::net::AbstractSimpleMessage::SetHeader
 */
void vislib::net::AbstractSimpleMessage::SetHeader(
        const AbstractSimpleMessageHeader& header, const bool reallocateBody) {
    VLSTACKTRACE("AbstractSimpleMessage::SetHeader", __FILE__, __LINE__);
    // If that asserts, the child class probably does not initialise correctly.
    ASSERT(this->header.PeekData() != NULL);
    this->header = header;
    if (reallocateBody) {
        this->assertStorage(this->GetHeader().GetBodySize());
    }
}


/*
 * vislib::net::AbstractSimpleMessage::operator =
 */
vislib::net::AbstractSimpleMessage& 
vislib::net::AbstractSimpleMessage::operator =(
        const AbstractSimpleMessage& rhs) {
    VLSTACKTRACE("AbstractSimpleMessage::operator =", __FILE__, __LINE__);

    if ((this != &rhs) && (static_cast<const void *>(*this) 
            != static_cast<const void *>(rhs))) {
        void *data = this->assertStorage(rhs.GetHeader().GetBodySize());
        ::memcpy(data, static_cast<const void *>(rhs), rhs.GetMessageSize());
    }

    return *this;
}


/*
 * vislib::net::AbstractSimpleMessage::operator ==
 */
bool vislib::net::AbstractSimpleMessage::operator ==(
        const AbstractSimpleMessage& rhs) const {
    VLSTACKTRACE("AbstractSimpleMessage::operator ==", __FILE__, __LINE__);

    // Note: Order of tests ensures correct test range and performance
    if (this->GetHeader() == rhs.GetHeader()) {
        return (::memcmp(this->GetBody(), rhs.GetBody(), 
            this->GetHeader().GetBodySize()) == 0);
    } else {
        return false;
    }
}


/*
 * vislib::net::AbstractSimpleMessage::operator const void *
 */
vislib::net::AbstractSimpleMessage::operator const void *(void) const {
    VLSTACKTRACE("AbstractSimpleMessage::operator const void *",
        __FILE__, __LINE__);
    return static_cast<const void *>(this->header.PeekData());
}


/*
 * vislib::net::AbstractSimpleMessage::operator void *
 */
vislib::net::AbstractSimpleMessage::operator void *(void) {
    VLSTACKTRACE("AbstractSimpleMessage::operator void *",
        __FILE__, __LINE__);
    return const_cast<void *>(static_cast<const void *>(
        this->header.PeekData()));
}


/*
 * vislib::net::AbstractSimpleMessage::assertStorage
 */
void *vislib::net::AbstractSimpleMessage::assertStorage(
        const SIZE_T bodySize) {
    VLSTACKTRACE("AbstractSimpleMessage::assertStorage", __FILE__, __LINE__);
    ASSERT(bodySize <= UINT_MAX);
    
    SimpleMessageHeader oldHeader;
    void *retval = NULL;

    if (this->header.PeekData() != NULL) {
        oldHeader = this->header;
    }
    
    if (this->assertStorage(retval, sizeof(SimpleMessageHeaderData)
            +  bodySize)) {
        this->header.SetData(static_cast<SimpleMessageHeaderData *>(retval));
        this->header = oldHeader;
    }

    ASSERT(retval != NULL);
    return retval;
}
