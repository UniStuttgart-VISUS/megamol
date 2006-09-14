/*
 * IOException.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/IOException.h"


/*
 * vislib::sys::IOException::IOException
 */
vislib::sys::IOException::IOException(const DWORD errorCode, const char *file,
		const int line) 
		: SystemException(errorCode, file, line) {
}


/*
 * vislib::sys::IOException::IOException
 */
vislib::sys::IOException::IOException(const DWORD errorCode, const char *msg, 
		const char *file, const int line) 
		: SystemException(errorCode, file, line) {
    Exception::setMsg(msg);
}


/*
 * vislib::sys::IOException::IOException
 */
vislib::sys::IOException::IOException(const DWORD errorCode, const wchar_t *msg, 
		const char *file, const int line) 
		: SystemException(errorCode, file, line) {
    Exception::setMsg(msg);
}


/*
 * vislib::sys::IOException::IOException
 */
vislib::sys::IOException::IOException(const IOException& rhs) 
        : SystemException(rhs) {
}


/*
 * vislib::sys::IOException::~IOException
 */
vislib::sys::IOException::~IOException(void) {
}


/*
 * vislib::sys::IOException::operator =
 */
vislib::sys::IOException& vislib::sys::IOException::operator =(
		const IOException& rhs) {
    SystemException::operator =(rhs);
    return *this;
}

