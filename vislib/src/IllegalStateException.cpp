/*
 * IllegalStateException.cpp  22.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/IllegalStateException.h"


/*
 * vislib::IllegalStateException::IllegalStateException
 */
vislib::IllegalStateException::IllegalStateException(const char *msg,
        const char *file, const int line) 
        : Exception(msg, file, line) {
}


/*
 * vislib::IllegalStateException::IllegalStateException
 */
vislib::IllegalStateException::IllegalStateException(const wchar_t *msg,
        const char *file, const int line) 
        : Exception(msg, file, line) {
}


/*
 * vislib::IllegalStateException::IllegalStateException
 */
vislib::IllegalStateException::IllegalStateException(
		const IllegalStateException& rhs) 
		: Exception(rhs) {
}


/*
 * vislib::IllegalStateException::~IllegalStateException
 */
vislib::IllegalStateException::~IllegalStateException(void) {
}


/*
 * vislib::IllegalStateException::operator =
 */
vislib::IllegalStateException& vislib::IllegalStateException::operator =(
		const IllegalStateException& rhs) {
	Exception::operator =(rhs);
	return *this;
}
