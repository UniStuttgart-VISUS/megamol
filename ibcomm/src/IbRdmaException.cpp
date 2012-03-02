/*
 * IbRdmaException.cpp
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/IbRdmaException.h"


/*
 * vislib::net::ib::IbRdmaException::IbRdmaException
 */
vislib::net::ib::IbRdmaException::IbRdmaException(const int errorCode, 
        const char *file, const int line) 
        : Super(file, line), errorCode(errorCode) {
}


/*
 * vislib::net::ib::IbRdmaException::IbRdmaException
 */
vislib::net::ib::IbRdmaException::IbRdmaException(const char *call, 
        const int errorCode, const char *file, const int line) 
        : Super(file, line), errorCode(errorCode) {
    this->formatMsg("%s failed with error code %d.", call, this->errorCode);
}


/*
 * vislib::net::ib::IbRdmaException
 */
vislib::net::ib::IbRdmaException::IbRdmaException(const IbRdmaException& rhs) 
        : Super(rhs), errorCode(rhs.errorCode) {
    // Nothing to do.
}

/*
 * vislib::net::ib::IbRdmaException::~IbRdmaException
 */
vislib::net::ib::IbRdmaException::~IbRdmaException(void) {
    // Nothing to do.
}


/*
 * vislib::net::ib::IbRdmaException::operator =
 */
vislib::net::ib::IbRdmaException& 
vislib::net::ib::IbRdmaException::operator =(const IbRdmaException& rhs) {
    if (this != &rhs) {
        Super::operator =(rhs);
    }
    return *this;
}