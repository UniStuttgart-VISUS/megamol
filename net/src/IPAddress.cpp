/*
 * IPAddress.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). All rights reserved.
 */

#ifndef _WIN32
#include <arpa/inet.h>
#include <cstring>
#include <netdb.h>
#endif /* !_WIN32 */

#include "vislib/IPAddress.h"

#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
#include "vislib/NetworkInformation.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/StringConverter.h"


/*
 * vislib::net::IPAddress::ALL_NODES_ON_LINK
 */
const vislib::net::IPAddress vislib::net::IPAddress::ALL_NODES_ON_LINK
#ifdef _WIN32
    (::in4addr_allnodesonlink);
#else /* _WIN32 */
    = vislib::net::IPAddress::Create("224.0.0.1");
#endif /* _WIN32 */


/*
 * vislib::net::IPAddress::ALL_ROUTERS_ON_LINK
 */
const vislib::net::IPAddress vislib::net::IPAddress::ALL_ROUTERS_ON_LINK
#ifdef _WIN32
    (::in4addr_allroutersonlink);
#else /* _WIN32 */
    = vislib::net::IPAddress::Create("224.0.0.2");
#endif /* _WIN32 */


/*
 * vislib::net::IPAddress::ANY
 */
const vislib::net::IPAddress vislib::net::IPAddress::ANY(
    static_cast<unsigned long>(INADDR_ANY), true);


/*
 * vislib::net::IPAddress::BROADCAST
 */
const  vislib::net::IPAddress vislib::net::IPAddress::BROADCAST(
    static_cast<unsigned long>(INADDR_BROADCAST), true);


/*
 * vislib::net::IPAddress::LOCALHOST
 */
const vislib::net::IPAddress vislib::net::IPAddress::LOCALHOST(
    static_cast<unsigned long>(INADDR_LOOPBACK), true);


/*
 * vislib::net::IPAddress::NONE
 */
const vislib::net::IPAddress vislib::net::IPAddress::NONE(
    static_cast<unsigned long>(INADDR_NONE), true);


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
 * vislib::net::IPAddress::IPAddress
 */
vislib::net::IPAddress::IPAddress(const unsigned long address, 
                                  const bool isHostByteOrder) {
    this->address.s_addr = isHostByteOrder ? htonl(address) : address;
}


/*
 * vislib::net::IPAddress::~IPAddress
 */
vislib::net::IPAddress::~IPAddress(void) {
}


/*
 * vislib::net::IPAddress::GetPrefix
 */
vislib::net::IPAddress vislib::net::IPAddress::GetPrefix(
        const ULONG prefixLength) const {
    IPAddress netmask = NetworkInformation::PrefixToNetmask4(prefixLength);
    return (netmask & *this);
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
 * vislib::net::IPAddress::operator []
 */
BYTE vislib::net::IPAddress::operator [](const int i) const {
    if ((i > 0) && (i < static_cast<int>(sizeof(this->address)))) {
        return reinterpret_cast<const BYTE *>(&this->address)[i];
    } else {
        throw OutOfRangeException(i, 0, sizeof(this->address), __FILE__,
            __LINE__);
    }
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
