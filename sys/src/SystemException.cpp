/*
 * SystemException.cpp  09.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include <cstring>

#include "vislib/SystemException.h"

#include "vislib/error.h"
#include "vislib/SystemMessage.h"



/*
 * vislib::sys::SystemException::SystemException
 */
vislib::sys::SystemException::SystemException(const DWORD errorCode, 
        const char *file, const int line) 
		: Exception(NULL, file, line), errorCode(errorCode) {
}


/*
 * vislib::sys::SystemException::SystemException
 */
vislib::sys::SystemException::SystemException(const char *file, const int line)
: Exception(NULL, file, line), errorCode(::GetLastError()) {
}


/*
 * vislib::sys::SystemException::SystemException
 */
vislib::sys::SystemException::SystemException(const SystemException& rhs) 
		: Exception(rhs), errorCode(rhs.errorCode) {
}


/*
 * vislib::sys::SystemException::~SystemException
 */
vislib::sys::SystemException::~SystemException(void) {
}


/*
 * vislib::sys::SystemException::GetMsg
 */
const TCHAR *vislib::sys::SystemException::GetMsg(void) const {

    if (Exception::GetMsg() == NULL) {
        this->setMsg(static_cast<const TCHAR *>(
            SystemMessage(this->GetErrorCode())));
    }

    return Exception::GetMsg();
}


/*
 * vislib::sys::SystemException::operator =
 */
vislib::sys::SystemException& vislib::sys::SystemException::operator =(
		const SystemException& rhs) {
	if (this != &rhs) {
		Exception::operator =(rhs);
		this->errorCode = rhs.errorCode;
	}
	return *this;
}

