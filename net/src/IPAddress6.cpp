/*
 * IPAddress6.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/IPAddress6.h"


/*
 * vislib::net::IPAddress6::IPAddress6
 */
vislib::net::IPAddress6::IPAddress6(void) {
    // TODO: Implement
}


/*
 * vislib::net::IPAddress6::~IPAddress6
 */
vislib::net::IPAddress6::~IPAddress6(void) {
    // TODO: Implement
}


/*
 * vislib::net::IPAddress6::operator =
 */
vislib::net::IPAddress6& vislib::net::IPAddress6::operator =(
        const IPAddress6& rhs) {
    if (this != &rhs) {
        ::memcpy(&this->address, &rhs.address, sizeof(struct in6_addr));
    }

    return *this;
}


/*
 * vislib::net::IPAddress6::operator ==
 */
bool vislib::net::IPAddress6::operator ==(const IPAddress6& rhs) const {
    return (::memcmp(&this->address, &rhs.address, sizeof(struct in6_addr)) 
        == 0);
}
