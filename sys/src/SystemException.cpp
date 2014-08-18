/*
 * SystemException.cpp  09.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/SystemException.h"

#include <cstring>

#include "vislib/error.h"


/*
 * vislib::sys::SystemException::SystemException
 */
vislib::sys::SystemException::SystemException(const DWORD errorCode, 
        const char *file, const int line) 
		: Exception(static_cast<const char *>(NULL), file, line), sysMsg(errorCode) {
}


/*
 * vislib::sys::SystemException::SystemException
 */
vislib::sys::SystemException::SystemException(const char *file, const int line)
        : Exception(static_cast<const char *>(NULL), file, line), 
        sysMsg(::GetLastError()) {
}


/*
 * vislib::sys::SystemException::SystemException
 */
vislib::sys::SystemException::SystemException(const SystemException& rhs) 
		: Exception(rhs), sysMsg(rhs.sysMsg) {
}


/*
 * vislib::sys::SystemException::~SystemException
 */
vislib::sys::SystemException::~SystemException(void) {
}


/*
 * vislib::sys::SystemException::GetMsgA
 */
const char *vislib::sys::SystemException::GetMsgA(void) const {
    return static_cast<const char *>(this->sysMsg);
}


/*
 * vislib::sys::SystemException::GetMsgW
 */
const wchar_t *vislib::sys::SystemException::GetMsgW(void) const {
    return static_cast<const wchar_t *>(this->sysMsg);
}


/*
 * vislib::sys::SystemException::operator =
 */
vislib::sys::SystemException& vislib::sys::SystemException::operator =(
		const SystemException& rhs) {
	if (this != &rhs) {
		Exception::operator =(rhs);
		this->sysMsg = rhs.sysMsg;
	}
	return *this;
}

