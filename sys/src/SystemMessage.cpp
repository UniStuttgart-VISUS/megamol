/*
 * SystemMessage.cpp  15.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef _WIN32
#include <cstring>

#include "vislib/StringConverter.h"
#endif /* _WIN32 */

#include "vislib/error.h"
#include "vislib/SystemMessage.h"


/*
 * vislib::sys::SystemMessage::SystemMessage
 */
vislib::sys::SystemMessage::SystemMessage(const DWORD errorCode)
		: errorCode(errorCode), msg(NULL) {
}


/*
 * vislib::sys::SystemMessage::SystemMessage
 */
vislib::sys::SystemMessage::SystemMessage(const SystemMessage& rhs) 
		: errorCode(rhs.errorCode), msg(NULL) {
}


/*
 * vislib::sys::SystemMessage::~SystemMessage
 */
vislib::sys::SystemMessage::~SystemMessage(void) {
#ifdef _WIN32
    if (this->msg != NULL) {
        ::LocalFree(this->msg);
    }
#else /* _WIN32 */
    ARY_SAFE_DELETE(this->msg);
#endif /* _WIN32 */
}


/*
 * vislib::sys::SystemMessage::operator =
 */
vislib::sys::SystemMessage& vislib::sys::SystemMessage::operator =(
		const SystemMessage& rhs) {
	if (this != &rhs) {
		this->errorCode = rhs.errorCode;
	}
	return *this;
}


/*
 * vislib::sys::SystemMessage::operator const TCHAR *
 */
vislib::sys::SystemMessage::operator const TCHAR *(void) const {

    if (this->msg == NULL) {
#ifdef _WIN32
        // TODO: Possible hazard: FormatMessage could fail.
        ::FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER 
            | FORMAT_MESSAGE_FROM_SYSTEM, NULL, this->errorCode,
		    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            reinterpret_cast<TCHAR *>(&this->msg), 0, NULL);

#else /* _WIN32 */
        EXTENT bufLen = 128;
        CHAR *buf = new CHAR[bufLen];
        CHAR *msg = NULL;

#ifdef _GNU_SOURCE
        msg = ::strerror_r(this->errorCode, buf, bufLen);

        // Ensure end of string in case 'buf' was used and too short.
        buf[bufLen - 1] = 0;                

#else /* _GNU_SOURCE */
        while (::strerror_r(this->errorCode, buf, bufLen) == ERANGE) {
            delete[] buf;
            bufLen += bufLen / 2;
            buf = new CHAR[bufLen];
        }     
        msg = buf;
#endif /* _GNU_SOURCE */

        bufLen = ::strlen(msg) + 1;
        this->msg = new TCHAR[bufLen];
        ::_tcscpy(this->msg, A2T(msg));
        ARY_SAFE_DELETE(buf);
#endif /* _WIN32 */
    }

    return this->msg;
}
