/*
 * IPAddress6.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/IPAddress6.h"

#include <cstdlib>

#include "vislib/assert.h"
#include "vislib/DNS.h"
#include "vislib/IllegalStateException.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/SocketException.h"
#include "vislib/Trace.h"


/*
 * vislib::net::IPAddress6::Create
 */
vislib::net::IPAddress6 vislib::net::IPAddress6::Create(
        const char *hostNameOrAddress) {
    IPAddress6 retval;
    DNS::GetHostAddress(retval, hostNameOrAddress);
    return retval;
}


/*
 * vislib::net::IPAddress6::Create
 */
vislib::net::IPAddress6 vislib::net::IPAddress6::Create(
        const wchar_t *hostNameOrAddress) {
    IPAddress6 retval;
    DNS::GetHostAddress(retval, hostNameOrAddress);
    return retval;
}


/*
 * vislib::net::IPAddress6::ALL_NODES_ON_LINK
 */
const vislib::net::IPAddress6 vislib::net::IPAddress6::ALL_NODES_ON_LINK
#ifdef _WIN32
    (::in6addr_allnodesonlink);
#else /* _WIN32 */
    = vislib::net::IPAddress6::Create("ff02::1");
#endif /* _WIN32 */


/*
 * vislib::net::IPAddress6::ALL_ROUTERS_ON_LINK
 */
const vislib::net::IPAddress6 vislib::net::IPAddress6::ALL_ROUTERS_ON_LINK
#ifdef _WIN32
    (::in6addr_allroutersonlink);
#else /* _WIN32 */
    = vislib::net::IPAddress6::Create("ff02::2");
#endif /* _WIN32 */


/*
 * vislib::net::IPAddress6::ALL_NODES_ON_NODE
 */
const vislib::net::IPAddress6 vislib::net::IPAddress6::ALL_NODES_ON_NODE
#ifdef _WIN32
    (::in6addr_allnodesonnode);
#else /* _WIN32 */
    = vislib::net::IPAddress6::Create("ff01::1");
#endif /* _WIN32 */


/*
 * vislib::net::IPAddress6::ANY
 */
const vislib::net::IPAddress6 vislib::net::IPAddress6::ANY(::in6addr_any);


/*
 * vislib::net::IPAddress6::LOCALHOST
 */
const vislib::net::IPAddress6& vislib::net::IPAddress6::LOCALHOST
    = vislib::net::IPAddress6::LOOPBACK;


/*
 * vislib::net::IPAddress6::LOOPBACK
 */
const vislib::net::IPAddress6 vislib::net::IPAddress6::LOOPBACK(
    ::in6addr_loopback);


/*
 * vislib::net::IPAddress6::UNSPECIFIED
 */
const vislib::net::IPAddress6& vislib::net::IPAddress6::UNSPECIFIED
    = vislib::net::IPAddress6::ANY;


/*
 * vislib::net::IPAddress6::IPAddress6
 */
vislib::net::IPAddress6::IPAddress6(void) {
    *this = ::in6addr_loopback;
}


/*
 * vislib::net::IPAddress6::IPAddress6
 */
vislib::net::IPAddress6::IPAddress6(const struct in6_addr& address) {
    *this = address;
}


/*
 * vislib::net::IPAddress6::IPAddress6
 */
vislib::net::IPAddress6::IPAddress6(
        const BYTE b1, const BYTE b2, const BYTE b3, const BYTE b4,
        const BYTE b5, const BYTE b6, const BYTE b7, const BYTE b8,
        const BYTE b9, const BYTE b10, const BYTE b11, const BYTE b12,
        const BYTE b13, const BYTE b14, const BYTE b15, const BYTE b16) {
#define VLIPADDR6_INITBYTE(n) this->address.s6_addr[n - 1] = b##n
    VLIPADDR6_INITBYTE(1);
    VLIPADDR6_INITBYTE(2);
    VLIPADDR6_INITBYTE(3);
    VLIPADDR6_INITBYTE(4);
    VLIPADDR6_INITBYTE(5);
    VLIPADDR6_INITBYTE(6);
    VLIPADDR6_INITBYTE(7);
    VLIPADDR6_INITBYTE(8);
    VLIPADDR6_INITBYTE(9);
    VLIPADDR6_INITBYTE(10);
    VLIPADDR6_INITBYTE(11);
    VLIPADDR6_INITBYTE(12);
    VLIPADDR6_INITBYTE(13);
    VLIPADDR6_INITBYTE(14);
    VLIPADDR6_INITBYTE(15);
    VLIPADDR6_INITBYTE(16);
#undef VLIPADDR6_INITBYTE
}


/*
 * vislib::net::IPAddress6::IPAddress6
 */
vislib::net::IPAddress6::IPAddress6(const IPAddress& address) {
    this->MapV4Address(address);
}


/*
 * vislib::net::IPAddress6::IPAddress6
 */
vislib::net::IPAddress6::IPAddress6(const struct in_addr& address) {
    this->MapV4Address(address);
}


/*
 * vislib::net::IPAddress6::IPAddress6
 */
vislib::net::IPAddress6::IPAddress6(const IPAddress6& rhs) {
    *this = rhs;
}


/*
 * vislib::net::IPAddress6::~IPAddress6
 */
vislib::net::IPAddress6::~IPAddress6(void) {
}


/*
 * vislib::net::IPAddress6::GetPrefix
 */ 
vislib::net::IPAddress6 
vislib::net::IPAddress6::GetPrefix(const ULONG prefixLength) const {
    IPAddress6 retval;
    int cntBytes = sizeof(retval.address.s6_addr); 
    int cntPrefix = prefixLength > static_cast<ULONG>(8 * cntBytes)
        ? cntBytes : static_cast<int>(prefixLength);
    div_t cntCopy = ::div(cntPrefix, 8);

    /* Zero out everything. */
    ::ZeroMemory(retval.address.s6_addr, cntBytes);

    /* Copy complete bytes. */
    ::memcpy(retval.address.s6_addr, this->address.s6_addr, cntCopy.quot);

    /* Copy fraction of incomplete byte if necessary. */
    if (cntCopy.rem > 0) {
        retval.address.s6_addr[cntCopy.quot] 
            = this->address.s6_addr[cntCopy.quot] << (8 - cntCopy.rem);
    }

    return retval;
}


/*
 * vislib::net::IPAddress6::MapV4Address
 */
void vislib::net::IPAddress6::MapV4Address(const struct in_addr& address) {
    /* Zero out first 80 bits. */
    for (int i = 0; i < 10; i++) {
        this->address.s6_addr[i] = 0;
    }

    /* Bits 80 to 95 must be all 1. */
    for (int i = 10; i < 12; i++) {
        this->address.s6_addr[i] = 0xff;
    }

    /* Embed IPv4 in the last 32 bits. */
    ASSERT(sizeof(in_addr) == 4);
    ::memcpy(this->address.s6_addr + 12, &address, sizeof(in_addr));

    ASSERT(this->IsV4Mapped());
}


/*
 * vislib::net::IPAddress6::ToStringA
 */
vislib::StringA vislib::net::IPAddress6::ToStringA(void) const {
    struct sockaddr_in6 addr;   // Dummy socket address used for lookup.
    char buffer[NI_MAXHOST];    // Receives the stringised address.
    int err = 0;                // OS operation return value.

    ::ZeroMemory(&addr, sizeof(addr));
    addr.sin6_family = AF_INET6;
    ::memcpy(&addr.sin6_addr, &this->address, sizeof(struct in6_addr));

    if ((err = ::getnameinfo(reinterpret_cast<struct sockaddr *>(&addr),
            sizeof(struct sockaddr_in6), buffer, sizeof(buffer), NULL, 0,
            NI_NUMERICHOST)) != 0) {
        VLTRACE(Trace::LEVEL_VL_ERROR, "::getnameinfo failed in "
            "IPAddress6::ToStringA(): %s\n",
#ifdef _WIN32
            ::gai_strerrorA(err)
#else /* _WIN32 */
            ::gai_strerror(err)
#endif /* _WIN32 */
        );
        buffer[0] = 0;
    }

    return StringA(buffer);
}


/*
 * vislib::net::IPAddress6::UnmapV4Address
 */
vislib::net::IPAddress vislib::net::IPAddress6::UnmapV4Address(void) const {
    if (this->IsV4Mapped()) {
        return IPAddress(*reinterpret_cast<const in_addr *>(
            this->address.s6_addr + 12));
    } else {
        throw IllegalStateException("The IPv6 address is not a mapped IPv4 "
            "address.", __FILE__, __LINE__);
    }
}


/*
 * vislib::net::IPAddress6::operator []
 */
BYTE vislib::net::IPAddress6::operator [](const int i) const {
    if ((i > 0) && (i < static_cast<int>(sizeof(this->address)))) {
        return reinterpret_cast<const BYTE *>(&this->address)[i];
    } else {
        throw OutOfRangeException(i, 0, sizeof(this->address), __FILE__,
            __LINE__);
    }
}


/*
 * vislib::net::IPAddress6::operator =
 */
vislib::net::IPAddress6& vislib::net::IPAddress6::operator =(
        const struct in6_addr& rhs) {
    if (&this->address != &rhs) {
        ::memcpy(&this->address, &rhs, sizeof(struct in6_addr));
    }

    return *this;
}


/*
 * vislib::net::IPAddress6::operator =
 */
vislib::net::IPAddress6& vislib::net::IPAddress6::operator =(
        const IPAddress& rhs) {
    this->MapV4Address(rhs);
    return *this;
}


/*
 * vislib::net::IPAddress6::operator ==
 */
bool vislib::net::IPAddress6::operator ==(const struct in6_addr& rhs) const {
#ifndef _WIN32
#define IN6_ADDR_EQUAL IN6_ARE_ADDR_EQUAL
#endif /* !_WIN32 */
    return (IN6_ADDR_EQUAL(&this->address, &rhs) != 0);
}


/*
 * vislib::net::IPAddress6::IPAddress
 */
vislib::net::IPAddress6::operator vislib::net::IPAddress(void) const {
    if (this->IsV4Compatible()) {
        return IPAddress(*reinterpret_cast<const in_addr *>(
            this->address.s6_addr + 12));
    } else {
        return this->UnmapV4Address();
    }
}
