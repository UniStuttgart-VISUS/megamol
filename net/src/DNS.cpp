/*
 * DNS.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/DNS.h"

#include "vislib/IllegalParamException.h"
#include "vislib/UnsupportedOperationException.h"

#include "vislib/MissingImplementationException.h"



/* 
 * vislib::net::DNS::GetHostEntry
 */
vislib::net::IPHostEntry vislib::net::DNS::GetHostEntry(const IPAddress& hostAddress) {
    throw MissingImplementationException("GetHostEntry", __FILE__, __LINE__);

}


/* 
 * vislib::net::DNS::GetHostEntry
 */
vislib::net::IPHostEntry vislib::net::DNS::GetHostEntry(const IPAddress6& hostAddress) {
    throw MissingImplementationException("GetHostEntry", __FILE__, __LINE__);
}


/* 
 * vislib::net::DNS::GetHostEntry
 */
vislib::net::IPHostEntry vislib::net::DNS::GetHostEntry(const char *hostName) {
    throw MissingImplementationException("GetHostEntry", __FILE__, __LINE__);
}


/*
 * vislib::net::DNS::~DNS
 */
vislib::net::DNS::~DNS(void) {
    // Nothing to do.
}


/*
 * vislib::net::DNS::DNS
 */
vislib::net::DNS::DNS(void) {
    // Nothing to do.
}


/*
 * vislib::net::DNS::DNS
 */
vislib::net::DNS::DNS(const DNS& rhs) {
    throw UnsupportedOperationException("DNS::DNS", __FILE__, __LINE__);
}


/*
 * vislib::net::DNS::DNS
 */
vislib::net::DNS& vislib::net::DNS::operator =(const DNS& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
