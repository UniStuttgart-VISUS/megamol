/*
 * IPEndPoint.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/IPEndPoint.h"

#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/memutils.h"


/*
 * vislib::net::IPEndPoint::IPEndPoint
 */
vislib::net::IPEndPoint::IPEndPoint(const IPAddress& ipAddress,
                                    const unsigned short port) {
    ::ZeroMemory(&this->address, sizeof(this->address));
    this->SetIPAddress(ipAddress);
    this->SetPort(port);            // IP address must have been set before!
}


/*
 * vislib::net::IPEndPoint::IPEndPoint
 */
vislib::net::IPEndPoint::IPEndPoint(const IPAddress6& ipAddress,
                                    const unsigned short port) {
    ::ZeroMemory(&this->address, sizeof(this->address));
    this->SetIPAddress(ipAddress);
    this->SetPort(port);            // IP address must have been set before!
    // TODO flow stuff etc!
}


/*
 * vislib::net::IPEndPoint::IPEndPoint
 */
vislib::net::IPEndPoint::IPEndPoint(const AddressFamily addressFamily,
                                    const unsigned short port) {
    switch (addressFamily) {

        case FAMILY_INET:
            this->SetIPAddress(IPAddress::ANY);
            break;

        case FAMILY_INET6:
            this->SetIPAddress(IPAddress6::ANY);
            break;

        default:
            // This should be legal as we have not allocated heap memory or any
            // other OS resource until now.
            throw IllegalParamException("addressFamily", __FILE__, __LINE__);
    }

    this->SetPort(port);            // IP address must have been set before!
}


/*
 * vislib::net::IPEndPoint::IPEndPoint
 */
vislib::net::IPEndPoint::IPEndPoint(const IPEndPoint& rhs) {
    *this = rhs;
}


/*
 * vislib::net::IPEndPoint::~IPEndPoint
 */
vislib::net::IPEndPoint::~IPEndPoint(void) {
    // Nothing to do.
}


/*
 * vislib::net::IPEndPoint::GetIPAddress4
 */
vislib::net::IPAddress vislib::net::IPEndPoint::GetIPAddress4(void) const {
    switch (this->address.ss_family) {
        case AF_INET:
            return IPAddress(this->asV4().sin_addr);

        case AF_INET6:
            // TODO
            throw 1;

        default:
            throw IllegalStateException("The address family of the IPEndPoint "
                "is not in the expected range.", __FILE__, __LINE__);
    }
}


/*
 * vislib::net::IPEndPoint::GetIPAddress6
 */
vislib::net::IPAddress6 vislib::net::IPEndPoint::GetIPAddress6(void) const {
    switch (this->address.ss_family) {
        case AF_INET:
            return IPAddress6(this->asV4().sin_addr);

        case AF_INET6:
            return IPAddress6(this->asV6().sin6_addr);

        default:
            throw IllegalStateException("The address family of the IPEndPoint "
                "is not in the expected range.", __FILE__, __LINE__);
    }
}


/*
 * vislib::net::IPEndPoint::GetPort
 */
unsigned short vislib::net::IPEndPoint::GetPort(void) const {
    switch (this->address.ss_family) {
        case AF_INET:
            return ntohs(this->asV4().sin_port);

        case AF_INET6:
            return ntohs(this->asV6().sin6_port);

        default:
            throw IllegalStateException("The address family of the IPEndPoint "
                "is not in the expected range.", __FILE__, __LINE__);
    }
}


/*
 * vislib::net::IPEndPoint::SetIPAddress
 */
void vislib::net::IPEndPoint::SetIPAddress(const IPAddress& ipAddress) {
    this->address.ss_family = AF_INET;
    ::memcpy(&(this->asV4().sin_addr),
        static_cast<const struct in_addr *>(ipAddress),
        sizeof(struct in_addr));
}


/*
 * vislib::net::IPEndPoint::SetIPAddress
 */
void vislib::net::IPEndPoint::SetIPAddress(const IPAddress6& ipAddress) {
    this->address.ss_family = AF_INET6;
    ::memcpy(&(this->asV6().sin6_addr),
        static_cast<const struct in6_addr *>(ipAddress),
        sizeof(struct in6_addr));
}


/*
 * vislib::net::IPEndPoint::SetPort
 */
void vislib::net::IPEndPoint::SetPort(const unsigned int port) {
    switch (this->address.ss_family) {
        case AF_INET:
            this->asV4().sin_port = htons(port);
            break;

        case AF_INET6:
            this->asV6().sin6_port = htons(port);
            break;

        default:
            ASSERT(false);          // This should be unreachable.
    }
}


/*
 * vislib::net::IPEndPoint::operator =
 */
vislib::net::IPEndPoint& vislib::net::IPEndPoint::operator =(
        const IPEndPoint& rhs) {
    if (this != &rhs) {
        ::memcpy(&this->address, &rhs.address, sizeof(sockaddr_storage));
    }

    return *this;
}


/*
 * vislib::net::IPEndPoint::operator =
 */
vislib::net::IPEndPoint& vislib::net::IPEndPoint::operator =(
        const SocketAddress& rhs) {
    this->SetIPAddress(rhs.GetIPAddress());
    this->SetPort(rhs.GetPort());
    return *this;
}


/*
 * vislib::net::IPEndPoint::operator ==
 */
bool vislib::net::IPEndPoint::operator ==(const IPEndPoint& rhs) {
    return (::memcpy(&this->address, &rhs.address, sizeof(sockaddr_storage))
        == 0);
}
