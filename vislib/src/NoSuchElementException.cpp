/*
 * NoSuchElementException.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */


#include "vislib/NoSuchElementException.h"


/*
 * vislib::NoSuchElementException::NoSuchElementException
 */
vislib::NoSuchElementException::NoSuchElementException(const char *msg, 
        const char *file, const int line) : Exception(msg, file, line) {
}


/*
 * vislib::NoSuchElementException::~NoSuchElementException
 */
vislib::NoSuchElementException::NoSuchElementException(
        const NoSuchElementException& rhs) : Exception(rhs) {
}


/*
 * vislib::NoSuchElementException::~NoSuchElementException
 */
vislib::NoSuchElementException::~NoSuchElementException(void) {
}


/*
 * vislib::NoSuchElementException::operator =
 */
vislib::NoSuchElementException& 
vislib::NoSuchElementException::operator =(const NoSuchElementException& rhs) {
    Exception::operator =(rhs);
    return *this;
}
