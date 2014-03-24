/*
 * PAMException.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/PAMException.h"


#ifndef _WIN32
/*
 * vislib::sys::PAMException::PAMException
 */
vislib::sys::PAMException::PAMException(pam_handle_t *hPAM, const int errorCode,
        const char *file, const int line) 
        : the::exception(::pam_strerror(hPAM, errorCode), file, line), 
        errorCode(errorCode) {
}
#endif /* !_WIN32 */


/*
 * vislib::sys::PAMException::PAMException
 */
vislib::sys::PAMException::PAMException(const PAMException& rhs) 
        : the::exception(rhs), errorCode(errorCode) {
}


/*
 * vislib::sys::PAMException::~PAMException
 */
vislib::sys::PAMException::~PAMException(void) {
}


/*
 * vislib::sys::PAMException::operator =
 */
vislib::sys::PAMException& vislib::sys::PAMException::operator =(
        const PAMException& rhs) {
    if (this != &rhs) {
        the::exception::operator =(rhs);
        this->errorCode = rhs.errorCode;

    }

    return *this;
}


#ifdef _WIN32
/*
 * vislib::sys::PAMException::PAMException
 */
vislib::sys::PAMException::PAMException(void) : the::exception(__FILE__, __LINE__) {
}
#endif /* _WIN32 */
