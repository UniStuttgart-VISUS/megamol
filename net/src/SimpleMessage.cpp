/*
 * SimpleMessage.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/SimpleMessage.h"

#include <climits>

#include "vislib/OutOfRangeException.h"


/*
 * vislib::net::SimpleMessage::SimpleMessage
 */
vislib::net::SimpleMessage::SimpleMessage(const size_t bodySize) : Super() {
    THE_STACK_TRACE;
    // This will force the superclass to (i) allocate memory for the message
    // header and the body itself and (ii) to update the message header pointer.
    THE_ASSERT(bodySize < UINT_MAX);
    Super::assertStorage(bodySize);
    this->GetHeader().SetBodySize(static_cast<UINT32>(bodySize));
}


/*
 * vislib::net::SimpleMessage::SimpleMessage
 */
vislib::net::SimpleMessage::SimpleMessage(
    const AbstractSimpleMessageHeader& header, const void *body) : Super() {
    THE_STACK_TRACE;

    Super::assertStorage(header.GetBodySize());
    this->SetHeader(header);

    if (body != NULL) {
        this->SetBody(body);
    }
}


/*
 * vislib::net::SimpleMessage::SimpleMessage
 */
vislib::net::SimpleMessage::SimpleMessage(const SimpleMessage& rhs) : Super() {
    THE_STACK_TRACE;
    *this = rhs;
}


/*
 * vislib::net::SimpleMessage::SimpleMessage
 */
vislib::net::SimpleMessage::SimpleMessage(const AbstractSimpleMessage& rhs) 
        : Super() {
    THE_STACK_TRACE;
    *this = rhs;
}


/*
 * vislib::net::SimpleMessage::~SimpleMessage
 */
vislib::net::SimpleMessage::~SimpleMessage(void) {
    THE_STACK_TRACE;
}


/*
 * vislib::net::SimpleMessage::Trim
 */
void vislib::net::SimpleMessage::Trim(void) {
    THE_STACK_TRACE;
    this->storage.EnforceSize(this->GetMessageSize(), true);
}


/*
 * vislib::net::SimpleMessage::assertStorage
 */
bool vislib::net::SimpleMessage::assertStorage(void *& outStorage, 
        const size_t size) {
    THE_STACK_TRACE;
    bool retval = this->storage.AssertSize(size, false);
    outStorage = static_cast<void *>(this->storage);
    return retval;
}
