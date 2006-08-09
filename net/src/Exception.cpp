/*
 * Exception.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#include <cstring>
#include <cstdio>
#include <cstdarg>

#include "vislib/Exception.h"
#include "vislib/memutils.h"


/*
 * vislib::Exception::Exception
 */
vislib::Exception::Exception(const char *msg, const char *file, const int line)
        : file(NULL), line(line), msg(NULL) {
    this->setFile(file);
    this->setMsg(msg);
}


/*
 * vislib::Exception::Exception
 */
vislib::Exception::Exception(const char *file, const int line) 
        : file(NULL), line(line), msg(NULL) {
    this->setFile(file);
    this->setMsg("Exception");
}


/*
 * vislib::Exception::Exception
 */
vislib::Exception::Exception(const Exception& rhs) 
        : file(NULL), line(rhs.line), msg(NULL) {
    this->setFile(rhs.file);
    this->setMsg(rhs.msg);
}


/* 
 * vislib::Exception::~Exception
 */
vislib::Exception::~Exception(void) {
    ARY_SAFE_DELETE(this->file);
    ARY_SAFE_DELETE(this->msg);
}


/*
 * vislib::Exception::operator =
 */
vislib::Exception& vislib::Exception::operator =(const Exception& rhs) {

    if (this != &rhs) {
        this->setFile(rhs.file);
        this->line = rhs.line;
        this->setMsg(rhs.msg);
    }

    return *this;
}


/*
 * vislib::Exception::formatMsg
 */
void vislib::Exception::formatMsg(const char *fmt, ...) {
	const float bufGrowFactor = 1.5f;

	va_list arglist;
	va_start(arglist, fmt);

	if (fmt != NULL) {
		int bufLen = static_cast<int>(::strlen(fmt) + 1);

		do {
			ARY_SAFE_DELETE(this->msg);
			bufLen = static_cast<int>(bufGrowFactor * bufLen);
			this->msg = new char[bufLen];
#if (_MSC_VER >= 1400)
		} while (::vsnprintf_s(this->msg, bufLen, _TRUNCATE, fmt, arglist) < 0);
#elif _WIN32 /* (_MSC_VER >= 1400) */
		} while (::_vsnprintf(this->msg, bufLen, fmt, arglist) < 0);
#else /* (_MSC_VER >= 1400) */
		} while (::vsnprintf(this->msg, bufLen, fmt, arglist) < 0);
#endif /* (_MSC_VER >= 1400) */

	} else {
		/* Delete old message and set pointer NULL. */
		ARY_SAFE_DELETE(this->msg);
	}
}


/*
 * vislib::Exception::setFile
 */
void vislib::Exception::setFile(const char *file) {
    ARY_SAFE_DELETE(this->file);

    if (file != NULL) {
		size_t bufLen = ::strlen(file) + 1;
        this->file = new char[bufLen];
#if (_MSC_VER >= 1400)
		::_snprintf_s(this->file, bufLen, bufLen, file);
#else /* (_MSC_VER >= 1400) */
		::strcpy(this->file, file);
#endif /* (_MSC_VER >= 1400) */
    }
}


/*
 * vislib::Exception::setMsg
 */
void vislib::Exception::setMsg(const char *msg) {
    ARY_SAFE_DELETE(this->msg);

    if (msg != NULL) {
		size_t bufLen = ::strlen(msg) + 1;
        this->msg = new char[bufLen];
#if (_MSC_VER >= 1400)
		::_snprintf_s(this->msg, bufLen, bufLen, msg);
#else /* (_MSC_VER >= 1400) */
		::strcpy(this->msg, msg);
#endif /* (_MSC_VER >= 1400) */
    }
}
