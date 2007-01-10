/*
 * FormatException.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/FormatException.h"


/*
 * vislib::FormatException::FormatException
 */
vislib::FormatException::FormatException(const char *msg,
        const char *file, const int line) 
        : Exception(msg, file, line) {
}


/*
 * vislib::FormatException::FormatException
 */
vislib::FormatException::FormatException(const wchar_t *msg,
        const char *file, const int line) 
        : Exception(msg, file, line) {
}


/*
 * vislib::FormatException::FormatException
 */
vislib::FormatException::FormatException(const FormatException& rhs) 
		: Exception(rhs) {
}


/*
 * vislib::FormatException::~FormatException
 */
vislib::FormatException::~FormatException(void) {
}


/*
 * vislib::FormatException::operator =
 */
vislib::FormatException& vislib::FormatException::operator =(
		const FormatException& rhs) {
	Exception::operator =(rhs);
	return *this;
}
