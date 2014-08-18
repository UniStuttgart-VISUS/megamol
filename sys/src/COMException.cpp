/*
 * COMException.cpp
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/COMException.h"


/*
 * vislib::sys::COMException::COMException
 */
#ifdef _WIN32
vislib::sys::COMException::COMException(const HRESULT hr, const char *file, 
        const int line) : Super(file, line) {
    _com_error ce(hr);
    this->setMsg(ce.ErrorMessage());
}
#else /* _WIN32 */
vislib::sys::COMException::COMException(const char *file, const int line) 
        : Super("COMException", file, line) {
    // Nothing to do.
}
#endif /* _WIN32 */


/*
 * vislib::sys::COMException::COMException
 */
vislib::sys::COMException::COMException(const COMException& rhs) : Super(rhs) {
    // Nothing to do.
}

/*
 * vislib::sys::COMException::~COMException
 */
vislib::sys::COMException::~COMException(void) {
    // Nothing to do.
}


/*
 * vislib::sys::COMException::operator =
 */
vislib::sys::COMException& vislib::sys::COMException::operator =(
        const COMException& rhs) {
    if (this != &rhs) {
        Super::operator=(rhs);
#ifdef _WIN32
        this->hr = rhs.hr;
#endif /* _WIN32 */
    }
    return *this;
}
