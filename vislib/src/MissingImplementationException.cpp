/*
 * MissingImplementationException.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */


#include "vislib/MissingImplementationException.h"


/*
 * vislib::MissingImplementationException::MissingImplementationException
 */
vislib::MissingImplementationException::MissingImplementationException(
        const char *method, const char *file, const int line) 
        : Exception(file, line) {
	Exception::formatMsg("Implementation of '%s' is missing.", method);
}


/*
 * vislib::MissingImplementationException::MissingImplementationException
 */
vislib::MissingImplementationException::MissingImplementationException(
        const wchar_t *method, const char *file, const int line) 
        : Exception(file, line) {
	Exception::formatMsg(L"Implementation of '%s' is missing.", method);
}


/*
 * vislib::MissingImplementationException::MissingImplementationException
 */
vislib::MissingImplementationException::MissingImplementationException(
        const MissingImplementationException& rhs)
        : Exception(rhs) {
}


/*
 * vislib::MissingImplementationException::MissingImplementationException
 */
vislib::MissingImplementationException::~MissingImplementationException(void) {
}


/*
 * vislib::MissingImplementationException::MissingImplementationException
 */
vislib::MissingImplementationException& 
vislib::MissingImplementationException::operator =(
        const MissingImplementationException& rhs) {
	Exception::operator =(rhs);
	return *this;
}
