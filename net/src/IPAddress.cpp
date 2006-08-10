/*
 * IPAddress.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). All rights reserved.
 */

#ifndef _WIN32
#include <cstring>
#include <netdb.h>
#endif /* !_WIN32 */

#include "vislib/assert.h"
#include "vislib/IPAddress.h"


/*
 * vislib::net::IPAddress::IPAddress
 */
vislib::net::IPAddress::IPAddress(const TCHAR *address) {
    VERIFY(this->Lookup(address));
}


/*
 * vislib::net::IPAddress::~IPAddress
 */
vislib::net::IPAddress::~IPAddress(void) {
}


/*
 * vislib::net::IPAddress::Lookup
 */
bool vislib::net::IPAddress::Lookup(const TCHAR *hostname) {
    ASSERT(false);  // TCHAR implementation!
    return false;

    ///* Try to find the host by its name first. */
    //hostent *host = ::gethostbyname(hostname);

    //if (host != NULL) {
    //    /* Host found. */
    //    ::memcpy(&this->address.s_addr, host->h_addr_list[0], host->h_length);

    //} else {
    //    /* Host not found, assume IP address. */

    //    if ((this->address.s_addr = ::inet_addr(hostname)) == INADDR_NONE) {
    //        /* IP address is invalid, return error. */
    //        return false;
    //    }
    //}

    //return true;
}


/*
 * vislib::net::IPAddress::operator =
 */
vislib::net::IPAddress& vislib::net::IPAddress::operator =(const IPAddress& rhs) {
    if (this != &rhs) {
        ::memcpy(&this->address, &rhs.address, sizeof(in_addr));
    }

    return *this;
}


/*
 * vislib::net::IPAddress::operator ==
 */
bool vislib::net::IPAddress::operator ==(const IPAddress& rhs) const {
    return (::memcmp(&this->address, &rhs.address, sizeof(in_addr)) == 0);
}
