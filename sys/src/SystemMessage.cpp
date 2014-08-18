/*
 * SystemMessage.cpp  15.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/SystemMessage.h"

#ifndef _WIN32
#include <wchar.h>
#endif /* _WIN32 */

#include "vislib/error.h"
#include "vislib/memutils.h"


/*
 * vislib::sys::SystemMessage::SystemMessage
 */
vislib::sys::SystemMessage::SystemMessage(const DWORD errorCode)
        : errorCode(errorCode), isMsgUnicode(false), msg(NULL) {
}


/*
 * vislib::sys::SystemMessage::SystemMessage
 */
vislib::sys::SystemMessage::SystemMessage(const SystemMessage& rhs) 
        : errorCode(rhs.errorCode), isMsgUnicode(false), msg(NULL) {
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
    SAFE_OPERATOR_DELETE(this->msg);
#endif /* _WIN32 */
}


/*
 * vislib::sys::SystemMessage::operator =
 */
vislib::sys::SystemMessage& vislib::sys::SystemMessage::operator =(
        const SystemMessage& rhs) {
    if (this != &rhs) {
#ifdef _WIN32
        if (this->msg != NULL) {
            ::LocalFree(this->msg);
            this->msg = NULL;
        }
#else /* _WIN32 */
        SAFE_OPERATOR_DELETE(this->msg);
#endif /* _WIN32 */
		this->errorCode = rhs.errorCode;
	}
	return *this;
}


/*
 * vislib::sys::SystemMessage::operator const char *
 */
vislib::sys::SystemMessage::operator const char *(void) const {

    if ((this->msg != NULL) && this->isMsgUnicode) {
#ifdef _WIN32
        ::LocalFree(this->msg);
        this->msg = NULL;
#else /* _WIN32 */
        SAFE_OPERATOR_DELETE(this->msg);
#endif /* _WIN32 */
    }

    if (this->msg == NULL) {
#ifdef _WIN32
        // TODO: Possible hazard: FormatMessage could fail.
        ::FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER 
            | FORMAT_MESSAGE_FROM_SYSTEM, NULL, this->errorCode,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            reinterpret_cast<char *>(&this->msg), 0, NULL);

#else /* _WIN32 */
        SIZE_T bufLen = 128;
        char *buf = new char[bufLen];
        char *msg = NULL;

#ifdef _GNU_SOURCE
        msg = ::strerror_r(this->errorCode, buf, bufLen);

        // Ensure end of string in case 'buf' was used and too short.
        buf[bufLen - 1] = 0;                

#else /* _GNU_SOURCE */
        while (::strerror_r(this->errorCode, buf, bufLen) == -1) {
            if (::GetLastError() != ERANGE) {
                buf[0] = 0;
                break;
            }
            delete[] buf;
            bufLen += bufLen / 2;
            buf = new char[bufLen];
        }     
        msg = buf;
#endif /* _GNU_SOURCE */

        bufLen = ::strlen(msg) + 1;
        this->msg = ::operator new(bufLen * sizeof(char));
        ::memcpy(this->msg, msg, bufLen * sizeof(char));

        if (msg == buf) {
            // Assume that we only have to free memory that we have
            // allocated ourselves, but only our own buffer.
            // Manpages do not contain sufficient information for doing
            // something more intelligent.
            ARY_SAFE_DELETE(buf);
        }
#endif /* _WIN32 */

        this->isMsgUnicode = false;
    }

    return static_cast<const char *>(this->msg);
}


/*
 * vislib::sys::SystemMessage::operator const wchar_t *
 */
vislib::sys::SystemMessage::operator const wchar_t *(void) const {

    if ((this->msg != NULL) && !this->isMsgUnicode) {
#ifdef _WIN32
        ::LocalFree(this->msg);
        this->msg = NULL;
#else /* _WIN32 */
        SAFE_OPERATOR_DELETE(this->msg);
#endif /* _WIN32 */
    }

    if (this->msg == NULL) {
#ifdef _WIN32
        // TODO: Possible hazard: FormatMessage could fail.
        ::FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER 
            | FORMAT_MESSAGE_FROM_SYSTEM, NULL, this->errorCode,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            reinterpret_cast<wchar_t *>(&this->msg), 0, NULL);

#else /* _WIN32 */
        SIZE_T bufLen = 128;
        char *buf = new char[bufLen];
        char *msg = NULL;

#ifdef _GNU_SOURCE
        msg = ::strerror_r(this->errorCode, buf, bufLen);

        // Ensure end of string in case 'buf' was used and too short.
        buf[bufLen - 1] = 0;                

#else /* _GNU_SOURCE */
        while (::strerror_r(this->errorCode, buf, bufLen) == -1) {
            if (::GetLastError() != ERANGE) {
                buf[0] = 0;
                break;
            }
            delete[] buf;
            bufLen += bufLen / 2;
            buf = new char[bufLen];
        }     
        msg = buf;
#endif /* _GNU_SOURCE */

        bufLen = ::strlen(msg) + 1;
        this->msg = ::operator new(bufLen * sizeof(wchar_t));
        ::swprintf(static_cast<wchar_t *>(this->msg), bufLen - 1, L"%hs", msg);

        if (msg == buf) {
            // Assume that we only have to free memory that we have
            // allocated ourselves, but only our own buffer.
            // Manpages do not contain sufficient information for doing
            // something more intelligent.
            ARY_SAFE_DELETE(buf);
        }
#endif /* _WIN32 */

        this->isMsgUnicode = true;
    }

    return static_cast<const wchar_t *>(this->msg);
}
