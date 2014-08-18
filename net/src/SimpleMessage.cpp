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
vislib::net::SimpleMessage::SimpleMessage(const SIZE_T bodySize) : Super() {
    VLSTACKTRACE("SimpleMessage::SimpleMessage", __FILE__, __LINE__);
    // This will force the superclass to (i) allocate memory for the message
    // header and the body itself and (ii) to update the message header pointer.
    ASSERT(bodySize < UINT_MAX);
    Super::assertStorage(bodySize);
    this->GetHeader().SetBodySize(static_cast<UINT32>(bodySize));
}


/*
 * vislib::net::SimpleMessage::SimpleMessage
 */
vislib::net::SimpleMessage::SimpleMessage(
    const AbstractSimpleMessageHeader& header, const void *body) : Super() {
    VLSTACKTRACE("SimpleMessage::SimpleMessage", __FILE__, __LINE__);

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
    VLSTACKTRACE("SimpleMessage::SimpleMessage", __FILE__, __LINE__);
    *this = rhs;
}


/*
 * vislib::net::SimpleMessage::SimpleMessage
 */
vislib::net::SimpleMessage::SimpleMessage(const AbstractSimpleMessage& rhs) 
        : Super() {
    VLSTACKTRACE("SimpleMessage::SimpleMessage", __FILE__, __LINE__);
    *this = rhs;
}


/*
 * vislib::net::SimpleMessage::~SimpleMessage
 */
vislib::net::SimpleMessage::~SimpleMessage(void) {
    VLSTACKTRACE("SimpleMessage::~SimpleMessage", __FILE__, __LINE__);
}


/*
 * vislib::net::SimpleMessage::Trim
 */
void vislib::net::SimpleMessage::Trim(void) {
    VLSTACKTRACE("SimpleMessage::Trim", __FILE__, __LINE__);
    this->storage.EnforceSize(this->GetMessageSize(), true);
}


/*
 * vislib::net::SimpleMessage::assertStorage
 */
bool vislib::net::SimpleMessage::assertStorage(void *& outStorage, 
        const SIZE_T size) {
    VLSTACKTRACE("SimpleMessage::assertStorage", __FILE__, __LINE__);
    bool retval = this->storage.AssertSize(size, false);
    outStorage = static_cast<void *>(this->storage);
    return retval;
}
