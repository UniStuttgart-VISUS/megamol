/*
 * UnexpectedMessageException.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/UnexpectedMessageException.h"
#include "the/text/string_builder.h"


/*
 * vislib::net::UnexpectedMessageException::UnexpectedMessageException
 */
vislib::net::UnexpectedMessageException::UnexpectedMessageException(
        const SimpleMessageID actualID, const SimpleMessageID expectedID, 
        const char *file, const int line) 
        : the::exception(__FILE__, __LINE__), actualID(actualID), 
        expectedID(expectedID) {
    this->set_msg(the::text::astring_builder::format(
        "Received SimpleMessageID %u, but %u was expected.",
        this->actualID, this->expectedID).c_str());
}


/*
 * vislib::net::UnexpectedMessageException::UnexpectedMessageException
 */
vislib::net::UnexpectedMessageException::UnexpectedMessageException(
        const UnexpectedMessageException& rhs) 
        : the::exception(rhs), actualID(rhs.actualID), expectedID(rhs.expectedID) {
    // Nothing to do.
}


/*
 * vislib::net::UnexpectedMessageException::~UnexpectedMessageException
 */
vislib::net::UnexpectedMessageException::~UnexpectedMessageException(void) throw() {
}

vislib::net::UnexpectedMessageException& 
vislib::net::UnexpectedMessageException::operator =(
        const UnexpectedMessageException& rhs) {
    if (this != &rhs) {
        the::exception::operator =(rhs);
        this->actualID = rhs.actualID;
        this->expectedID = rhs.expectedID;
    }

    return *this;
}
