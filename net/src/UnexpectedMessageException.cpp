/*
 * UnexpectedMessageException.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/UnexpectedMessageException.h"


/*
 * vislib::net::UnexpectedMessageException::UnexpectedMessageException
 */
vislib::net::UnexpectedMessageException::UnexpectedMessageException(
        const SimpleMessageID actualID, const SimpleMessageID expectedID, 
        const char *file, const int line) 
        : Exception(__FILE__, __LINE__), actualID(actualID), 
        expectedID(expectedID) {
    this->formatMsg("Received SimpleMessageID %u, but %u was expected.",
        this->actualID, this->expectedID);
}


/*
 * vislib::net::UnexpectedMessageException::UnexpectedMessageException
 */
vislib::net::UnexpectedMessageException::UnexpectedMessageException(
        const UnexpectedMessageException& rhs) 
        : Exception(rhs), actualID(rhs.actualID), expectedID(rhs.expectedID) {
    // Nothing to do.
}


/*
 * vislib::net::UnexpectedMessageException::~UnexpectedMessageException
 */
vislib::net::UnexpectedMessageException::~UnexpectedMessageException(void) {
}

vislib::net::UnexpectedMessageException& 
vislib::net::UnexpectedMessageException::operator =(
        const UnexpectedMessageException& rhs) {
    if (this != &rhs) {
        Exception::operator =(rhs);
        this->actualID = rhs.actualID;
        this->expectedID = rhs.expectedID;
    }

    return *this;
}