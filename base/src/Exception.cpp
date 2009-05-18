/*
 * Exception.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#include "vislib/Exception.h"

#include <cstring>
#include <cstdio>
#include <cstdarg>

#include "vislib/memutils.h"
#include "vislib/StackTrace.h"
#include "vislib/StringConverter.h"


/*
 * vislib::Exception::Exception
 */
vislib::Exception::Exception(const char *msg, const char *file, 
        const int line)
        : file(NULL), line(line), msg(NULL), stack(NULL) {
    this->setFile(file);
    this->setMsg(msg);
    this->fetchStack();
}


/*
 * vislib::Exception::Exception
 */
vislib::Exception::Exception(const wchar_t *msg, const char *file, 
        const int line)
        : file(NULL), line(line), msg(NULL), stack(NULL) {
    this->setFile(file);
    this->setMsg(msg);
    this->fetchStack();
}


/*
 * vislib::Exception::Exception
 */
vislib::Exception::Exception(const char *file, const int line) 
        : file(NULL), line(line), msg(NULL), stack(NULL) {
    this->setFile(file);
    this->fetchStack();
}


/*
 * vislib::Exception::Exception
 */
vislib::Exception::Exception(const Exception& rhs) 
        : file(NULL), line(rhs.line), msg(NULL), stack(NULL) {
    this->setFile(rhs.file);

    if (rhs.isMsgUnicode) {
        this->setMsg(static_cast<const wchar_t *>(rhs.GetMsgW()));
    } else {
        this->setMsg(static_cast<const char *>(rhs.GetMsgA()));
    }

    if (rhs.stack != NULL) {
        SIZE_T len = ::strlen(rhs.stack) + 1;
        this->stack = new char[len];
        ::memcpy(this->stack, rhs.stack, len);
    }
}


/* 
 * vislib::Exception::~Exception
 */
vislib::Exception::~Exception(void) {
    ARY_SAFE_DELETE(this->file);
    SAFE_OPERATOR_DELETE(this->msg);
    ARY_SAFE_DELETE(this->stack);
}


/*
 * vislib::Exception::GetMsgA
 */
const char *vislib::Exception::GetMsgA(void) const {
    if (this->msg == NULL) {
        return "Exception";
    }

    if (this->isMsgUnicode) {
        this->setMsg(W2A(static_cast<wchar_t *>(this->msg)));
    }

    return static_cast<const char *>(this->msg);
}


/*
 * vislib::Exception::GetMsgW
 */
const wchar_t *vislib::Exception::GetMsgW(void) const {
    if (this->msg == NULL) {
        return L"Exception";
    }

    if (!this->isMsgUnicode) {
        this->setMsg(A2W(static_cast<char *>(this->msg)));
    }

    return static_cast<const wchar_t *>(this->msg);
}


/*
 * vislib::Exception::operator =
 */
vislib::Exception& vislib::Exception::operator =(const Exception& rhs) {

    if (this != &rhs) {
        this->setFile(rhs.file);
        this->line = rhs.line;

        // using GetMsgX() because derived classes may use lazy initialisation
        if (rhs.isMsgUnicode) {
            this->setMsg(rhs.GetMsgW());
        } else {
            this->setMsg(rhs.GetMsgA());
        }

        ARY_SAFE_DELETE(this->stack);
        if (rhs.stack != NULL) {
            SIZE_T len = ::strlen(rhs.stack) + 1;
            this->stack = new char[len];
            ::memcpy(this->stack, rhs.stack, len);
        }
    }

    return *this;
}


/*
 * vislib::Exception::fetchStack
 */
void vislib::Exception::fetchStack(void) {
    unsigned int size;
    vislib::StackTrace::GetStackString((char*)NULL, size);
    if (size <= 1) {
        //ARY_SAFE_DELETE(this->stack);
        this->stack = NULL;
        return;
    }

    this->stack = new char[size];
    vislib::StackTrace::GetStackString(this->stack, size);
}


/*
 * vislib::Exception::formatMsg
 */
void vislib::Exception::formatMsg(const char *fmt, ...) {
    const float bufGrowFactor = 1.5f;

    va_list arglist;
    va_start(arglist, fmt);

    if (fmt != NULL) {
        this->isMsgUnicode = false;
        int bufLen = static_cast<int>(::strlen(fmt) + 1);

        do {
            SAFE_OPERATOR_DELETE(this->msg);
            bufLen = static_cast<int>(bufGrowFactor * bufLen);
            this->msg = ::operator new(bufLen * sizeof(char));
#if defined(_MSC_VER) && (_MSC_VER >= 1400)
        } while (::_vsnprintf_s(static_cast<char *>(this->msg), bufLen, 
                _TRUNCATE, fmt, arglist) < 0);
#elif _WIN32 
        } while (::_vsnprintf(static_cast<char *>(this->msg), bufLen,
                fmt, arglist) < 0);
#else /* defined(_MSC_VER) && (_MSC_VER >= 1400) */
        } while (::vsnprintf(static_cast<char *>(this->msg), bufLen, 
                fmt, arglist) < 0);
#endif /* defined(_MSC_VER) && (_MSC_VER >= 1400) */

    } else {
        /* Delete old message and set pointer NULL. */
        SAFE_OPERATOR_DELETE(this->msg);
    }
}


/*
 * vislib::Exception::formatMsg
 */
void vislib::Exception::formatMsg(const wchar_t *fmt, ...) {
    const float bufGrowFactor = 1.5f;

    va_list arglist;
    va_start(arglist, fmt);

    if (fmt != NULL) {
        this->isMsgUnicode = true;
        int bufLen = static_cast<int>(::wcslen(fmt) + 1);

        do {
            SAFE_OPERATOR_DELETE(this->msg);
            bufLen = static_cast<int>(bufGrowFactor * bufLen);
            this->msg = ::operator new(bufLen * sizeof(wchar_t));
#if defined(_MSC_VER) && (_MSC_VER >= 1400)
        } while (::_vsnwprintf_s(static_cast<wchar_t *>(this->msg), bufLen, 
                _TRUNCATE, fmt, arglist) < 0);
#elif _WIN32 
        } while (::_vsnwprintf(static_cast<wchar_t *>(this->msg), bufLen,
                fmt, arglist) < 0);
#else /* defined(_MSC_VER) && (_MSC_VER >= 1400) */
        } while (::vswprintf(static_cast<wchar_t *>(this->msg), bufLen, 
                fmt, arglist) < 0);
#endif /* defined(_MSC_VER) && (_MSC_VER >= 1400) */

    } else {
        /* Delete old message and set pointer NULL. */
        SAFE_OPERATOR_DELETE(this->msg);
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
        ::memcpy(this->file, file, bufLen * sizeof(char));
    }
}


/*
 * vislib::Exception::setMsg
 */
void vislib::Exception::setMsg(const char *msg) const {
    SAFE_OPERATOR_DELETE(this->msg);

    if (msg != NULL) {
        this->isMsgUnicode = false;
        size_t bufLen = ::strlen(msg) + 1;
        this->msg = ::operator new(bufLen * sizeof(char));
        ::memcpy(this->msg, msg, bufLen * sizeof(char));
    }
}


/*
 * vislib::Exception::setMsg
 */
void vislib::Exception::setMsg(const wchar_t *msg) const {
    SAFE_OPERATOR_DELETE(this->msg);

    if (msg != NULL) {
        this->isMsgUnicode = true;
        size_t bufLen = ::wcslen(msg) + 1;
        this->msg = ::operator new(bufLen * sizeof(wchar_t));
        ::memcpy(this->msg, msg, bufLen * sizeof(wchar_t));
    }
}
