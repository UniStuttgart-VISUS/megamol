/*
 * SystemException.cpp  09.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include <cstring>

#include "vislib/error.h"
#include "vislib/SystemException.h"


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
#ifdef _WIN32
        LPVOID msgBuf = NULL;

        ::FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER 
            | FORMAT_MESSAGE_FROM_SYSTEM, NULL, this->GetErrorCode(),
		    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
		    reinterpret_cast<TCHAR *>(&msgBuf), 0, NULL);
        this->setMsg(reinterpret_cast<TCHAR *>(msgBuf));
	    ::LocalFree(msgBuf);
#else /* _WIN32 */
#if defined(UNICODE) || defined (_UNICODE)
#error TODO Implementation missing
#endif
        this->setMsg(strerror(this->GetErrorCode()));
#endif /* _WIN32 */
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

