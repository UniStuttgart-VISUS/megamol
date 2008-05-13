/*
 * IPAddress6.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/IPAddress6.h"

#include "vislib/assert.h"


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
#define VLIPADDR5_INITBYTE(n) this->address.s6_addr[n - 1] = b##n
    VLIPADDR5_INITBYTE(1);
    VLIPADDR5_INITBYTE(2);
    VLIPADDR5_INITBYTE(3);
    VLIPADDR5_INITBYTE(4);
    VLIPADDR5_INITBYTE(5);
    VLIPADDR5_INITBYTE(6);
    VLIPADDR5_INITBYTE(7);
    VLIPADDR5_INITBYTE(8);
    VLIPADDR5_INITBYTE(9);
    VLIPADDR5_INITBYTE(10);
    VLIPADDR5_INITBYTE(11);
    VLIPADDR5_INITBYTE(12);
    VLIPADDR5_INITBYTE(13);
    VLIPADDR5_INITBYTE(14);
    VLIPADDR5_INITBYTE(15);
    VLIPADDR5_INITBYTE(16);
#undef VLIPADDR5_INITBYTE
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
