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

#include "vislib/IPAddress.h"

#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
#include "vislib/StringConverter.h"


/*
 * vislib::net::IPAddress::ANY
 */
const vislib::net::IPAddress vislib::net::IPAddress::ANY(
    static_cast<unsigned long>(INADDR_ANY));


/*
 * vislib::net::IPAddress::LOCALHOST
 */
const vislib::net::IPAddress vislib::net::IPAddress::LOCALHOST("127.0.0.1");


/*
 * vislib::net::IPAddress::NONE
 */
const vislib::net::IPAddress vislib::net::IPAddress::NONE(
    static_cast<unsigned long>(INADDR_NONE));


/*
 * vislib::net::IPAddress::Create
 */
vislib::net::IPAddress vislib::net::IPAddress::Create(const char *address) {
    IPAddress retval;

    if (!retval.Lookup(address)) {
        throw IllegalParamException("address", __FILE__, __LINE__);
    }

    return retval;
}


/*
 * vislib::net::IPAddress::IPAddress
 */
vislib::net::IPAddress::IPAddress(const char *address) {
    VERIFY(this->Lookup(address));
}


/*
 * vislib::net::IPAddress::IPAddress
 */
vislib::net::IPAddress::IPAddress(unsigned char i1, unsigned char i2, 
        unsigned char i3, unsigned char i4) {
#ifdef _WIN32
    this->address.S_un.S_un_b.s_b1 = i1;
    this->address.S_un.S_un_b.s_b2 = i2;
    this->address.S_un.S_un_b.s_b3 = i3;
    this->address.S_un.S_un_b.s_b4 = i4;
#else /* _WIN32 */
    this->address.s_addr
        = (static_cast<unsigned int>(i1))
        + (static_cast<unsigned int>(i2) << 8)
        + (static_cast<unsigned int>(i3) << 16)
        + (static_cast<unsigned int>(i4) << 24);
#endif /* _WIN32 */
}


/*
 * vislib::net::IPAddress::~IPAddress
 */
vislib::net::IPAddress::~IPAddress(void) {
}


/*
 * vislib::net::IPAddress::Lookup
 */
bool vislib::net::IPAddress::Lookup(const char *hostname) {

    /* Try to find the host by its name first. */
    hostent *host = ::gethostbyname(hostname);

    if (host != NULL) {
        /* Host found. */
        ::memcpy(&this->address.s_addr, host->h_addr_list[0], host->h_length);

    } else {
        /* Host not found, assume IP address. */

        if ((this->address.s_addr = ::inet_addr(hostname)) == INADDR_NONE) {
            /* IP address is invalid, return error. */
            return false;
        }
    }

    return true;
}


/*
 * vislib::net::IPAddress::ToStringA
 */
vislib::StringA vislib::net::IPAddress::ToStringA(void) const {
    StringA retval(::inet_ntoa(this->address));
    return retval;
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


/*
 * vislib::net::IPAddress::operator &=
 */
vislib::net::IPAddress& vislib::net::IPAddress::operator &=(
        const IPAddress& mask) {
#ifdef _WIN32
    this->address.S_un.S_addr &= mask.address.S_un.S_addr;
#else /* _WIN32 */
    this->address.s_addr &= mask.address.s_addr;
#endif /* _WIN32 */
    return *this;
}

/*
 * vislib::net::IPAddress::IPAddress
 */
vislib::net::IPAddress::IPAddress(const unsigned long address) {
    this->address.s_addr = address;
}
