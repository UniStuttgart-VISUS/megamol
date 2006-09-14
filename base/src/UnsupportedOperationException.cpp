/*
 * UnsupportedOperationException.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */


#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::UnsupportedOperationException::UnsupportedOperationException
 */
vislib::UnsupportedOperationException::UnsupportedOperationException(
		const char *funcName, const char *file, const int line) 
        : Exception(file, line) {
    Exception::formatMsg("'%s' is an unsupported operation.", funcName);
}


/*
 * vislib::UnsupportedOperationException::UnsupportedOperationException
 */
vislib::UnsupportedOperationException::UnsupportedOperationException(
		const wchar_t *funcName, const char *file, const int line) 
        : Exception(file, line) {
    Exception::formatMsg(L"'%s' is an unsupported operation.", funcName);
}


/*
 * vislib::UnsupportedOperationException::UnsupportedOperationException
 */
vislib::UnsupportedOperationException::UnsupportedOperationException(
		const UnsupportedOperationException& rhs) 
		: Exception(rhs) {
}


/*
 * vislib::UnsupportedOperationException::~UnsupportedOperationException
 */
vislib::UnsupportedOperationException::~UnsupportedOperationException(void) {
}


/*
 * vislib::UnsupportedOperationException::operator =
 */
vislib::UnsupportedOperationException& 
		vislib::UnsupportedOperationException::operator =(
		const UnsupportedOperationException& rhs) {
	Exception::operator =(rhs);
	return *this;
}

