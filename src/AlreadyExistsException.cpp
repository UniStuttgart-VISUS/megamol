/*
 * AlreadyExistsException.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/AlreadyExistsException.h"


/*
 * vislib::AlreadyExistsException::AlreadyExistsException
 */
vislib::AlreadyExistsException::AlreadyExistsException(const char *msg, 
        const char *file, const int line) : Exception(msg, file, line) {
}


/*
 * vislib::AlreadyExistsException::~AlreadyExistsException
 */
vislib::AlreadyExistsException::AlreadyExistsException(
        const AlreadyExistsException& rhs) : Exception(rhs) {
}


/*
 * vislib::AlreadyExistsException::~AlreadyExistsException
 */
vislib::AlreadyExistsException::~AlreadyExistsException(void) {
}


/*
 * vislib::NoSuchElementException::operator =
 */
vislib::AlreadyExistsException& 
vislib::AlreadyExistsException::operator =(const AlreadyExistsException& rhs) {
    Exception::operator =(rhs);
    return *this;
}
