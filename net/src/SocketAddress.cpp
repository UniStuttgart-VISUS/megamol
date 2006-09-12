/*
 * SocketAddress.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). All rights reserved.
 */

#include "vislib/SocketAddress.h"

#include "vislib/assert.h"
#include "vislib/memutils.h"


/*
 * vislib::net::SocketAddress::SocketAddress
 */
vislib::net::SocketAddress::SocketAddress(const AddressFamily addressFamily,
										  const IPAddress& ipAddress, 
										  const unsigned short port) {
    ASSERT(sizeof(this->genericAddress) == sizeof(this->inetAddress));
    ::memset(&this->genericAddress, 0, sizeof(this->genericAddress));
    this->inetAddress.sin_family = static_cast<unsigned short>(addressFamily);
    this->inetAddress.sin_port = htons(port);
    this->inetAddress.sin_addr = static_cast<struct in_addr>(ipAddress);
}


/*
 * vislib::net::SocketAddress::SocketAddress
 */
vislib::net::SocketAddress::SocketAddress(struct sockaddr address) {
    ASSERT(sizeof(this->genericAddress) == sizeof(this->inetAddress));
    ::memcpy(&this->genericAddress, &address, sizeof(struct sockaddr));
}


/*
 * vislib::net::SocketAddress::SocketAddress
 */
vislib::net::SocketAddress::SocketAddress(void) {
    ASSERT(sizeof(this->genericAddress) == sizeof(this->inetAddress));
    ::memset(&this->genericAddress, 0, sizeof(this->genericAddress));
}


/*
 * vislib::net::SocketAddress::SocketAddress
*/
vislib::net::SocketAddress::SocketAddress(const SocketAddress& rhs) {
    ASSERT(sizeof(this->genericAddress) == sizeof(this->inetAddress));
    ::memcpy(&this->genericAddress, &rhs.genericAddress, 
        sizeof(this->genericAddress));
}


/*
 * vislib::net::SocketAddress::~SocketAddress
 */
vislib::net::SocketAddress::~SocketAddress(void) {
}


/*
 * vislib::net::SocketAddress::SetIPAdress
 */
void vislib::net::SocketAddress::SetIPAdress(const IPAddress& ipAddress) {
    this->inetAddress.sin_addr = static_cast<struct in_addr>(ipAddress);
}


/*
 * vislib::net::SocketAddress::operator =
 */ 
vislib::net::SocketAddress& vislib::net::SocketAddress::operator =(
		const SocketAddress& rhs) {
    if (this != &rhs) {
        ::memcpy(&this->genericAddress, &rhs.genericAddress, 
            sizeof(this->genericAddress));
    }

    return *this;
}


/*
 * vislib::net::SocketAddress::operator ==
 */
bool vislib::net::SocketAddress::operator ==(const SocketAddress& rhs) const {
    return (::memcmp(&this->genericAddress, &rhs.genericAddress, 
        sizeof(this->genericAddress)) == 0);
}
