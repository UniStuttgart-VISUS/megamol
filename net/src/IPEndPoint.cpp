/*
 * IPEndPoint.cpp
 *
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/IPEndPoint.h"

#include "vislib/assert.h"
#include "vislib/DNS.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/memutils.h"
#include "vislib/SocketException.h"
#include "vislib/SystemMessage.h"
#include "vislib/Trace.h"


#ifdef _WIN32
    // Crowbar to work around windows naming conflict
#ifdef _MSC_VER
#pragma push_macro("SetPort")
#undef SetPort
#else /* _MSC_VER */
#ifdef SetPort
#error SetPort Macro defined!
#endif /* SetPort */
#endif /* _MSC_VER */
#endif /* _WIN32 */


/*
 * vislib::net::IPEndPoint::CreateIPv4
 */
vislib::net::IPEndPoint vislib::net::IPEndPoint::CreateIPv4(
        const char *hostNameOrAddress, const unsigned short port) {
    IPAddress addr;
    try {
        DNS::GetHostAddress(addr, hostNameOrAddress);
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_ERROR, "Could not lookup \"%s\": %s\n", 
            hostNameOrAddress, e.GetMsgA());
        throw IllegalParamException("hostNameOrAddress", __FILE__, __LINE__);
    }

    return IPEndPoint(addr, port);
}


/*
 * vislib::net::IPEndPoint::CreateIPv6
 */
vislib::net::IPEndPoint vislib::net::IPEndPoint::CreateIPv6(
        const char *hostNameOrAddress, const unsigned short port) {
    IPAddress6 addr;
    try {
        DNS::GetHostAddress(addr, hostNameOrAddress);
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_ERROR, "Could not lookup \"%s\": %s\n", 
            hostNameOrAddress, e.GetMsgA());
        throw IllegalParamException("hostNameOrAddress", __FILE__, __LINE__);
    }

    return IPEndPoint(addr, port);
}


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
vislib::net::IPEndPoint::IPEndPoint(const IPAgnosticAddress& ipAddress, 
                                    const unsigned short port) {
    ::ZeroMemory(&this->address, sizeof(this->address));
    this->SetIPAddress(ipAddress);
    this->SetPort(port);            // IP address must have been set before!
}


/*
 * vislib::net::IPEndPoint::IPEndPoint
 */
vislib::net::IPEndPoint::IPEndPoint(const AddressFamily addressFamily,
                                    const unsigned short port) {
    ::ZeroMemory(&this->address, sizeof(this->address));

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
vislib::net::IPEndPoint::IPEndPoint(const SocketAddress& address) {
    ::ZeroMemory(&this->address, sizeof(this->address));
    *this = address;
}


/*
 * vislib::net::IPEndPoint::IPEndPoint
 */
vislib::net::IPEndPoint::IPEndPoint(const struct sockaddr_storage& address) {
    //::ZeroMemory(&this->address, sizeof(this->address));
    ::memcpy(&this->address, &address, sizeof(struct sockaddr_storage));
}


/*
 * vislib::net::IPEndPoint::IPEndPoint
 */
vislib::net::IPEndPoint::IPEndPoint(const struct sockaddr_in& address) {
    ::ZeroMemory(&this->address, sizeof(this->address));
    *this = address;
}


/*
 * vislib::net::IPEndPoint::IPEndPoint
 */
vislib::net::IPEndPoint::IPEndPoint(const struct sockaddr_in6& address) {
    ::ZeroMemory(&this->address, sizeof(this->address));
    *this = address;
}


/*
 * vislib::net::IPEndPoint::IPEndPoint
 */
vislib::net::IPEndPoint::IPEndPoint(const IPEndPoint& rhs) {
    ::ZeroMemory(&this->address, sizeof(this->address));
    *this = rhs;
}


/*
 * vislib::net::IPEndPoint::IPEndPoint
 */
vislib::net::IPEndPoint::IPEndPoint(const IPEndPoint& address, 
                                    const unsigned short newPort) {
    ::ZeroMemory(&this->address, sizeof(this->address));
    *this = address;
    this->SetPort(newPort);
}


/*
 * vislib::net::IPEndPoint::~IPEndPoint
 */
vislib::net::IPEndPoint::~IPEndPoint(void) {
    // Nothing to do.
}


/*
 * vislib::net::IPAgnosticAddress
 */
vislib::net::IPAgnosticAddress vislib::net::IPEndPoint::GetIPAddress(
        void) const {
    switch (this->address.ss_family) {
        case AF_INET:
            return IPAgnosticAddress(this->asV4().sin_addr);

        case AF_INET6:
            return IPAgnosticAddress(this->asV6().sin6_addr);

        default:
            throw IllegalStateException("The address family of the IPEndPoint "
                "is not in the expected range.", __FILE__, __LINE__);
    }
}


/*
 * vislib::net::IPEndPoint::GetIPAddress4
 */
vislib::net::IPAddress vislib::net::IPEndPoint::GetIPAddress4(void) const {
    switch (this->address.ss_family) {
        case AF_INET:
            return IPAddress(this->asV4().sin_addr);

        case AF_INET6: {
            const struct in6_addr *addr = &this->asV6().sin6_addr;
            if (IN6_IS_ADDR_V4MAPPED(addr) || IN6_IS_ADDR_V4COMPAT(addr)) {
                const BYTE *bytes = reinterpret_cast<const BYTE *>(addr);
                return IPAddress(bytes[12], bytes[13], bytes[14], bytes[15]);
            } else {
                throw IllegalStateException("The IPEndPoint is an IPv6 end "
                    "point which has no IPv4-mapped or -compatible address.",
                    __FILE__, __LINE__);
            }
            } break;

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
 * vislib::net::IPEndPoint::SetIPAddress
 */
void vislib::net::IPEndPoint::SetIPAddress(const IPAgnosticAddress& ipAddress) {
    switch (ipAddress.GetAddressFamily()) {
        case IPAgnosticAddress::FAMILY_INET:
            this->SetIPAddress(static_cast<const IPAddress&>(ipAddress));
            break;

        case IPAgnosticAddress::FAMILY_INET6:
            // SetIPAddress produces deep copy. Cast to object does not work
            // on gcc, because this program is utter sh**.
            this->SetIPAddress(static_cast<const IPAddress6&>(ipAddress));
            break;

        default:
            throw IllegalParamException("ipAddress", __FILE__, __LINE__);
    }
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
 * vislib::net::IPEndPoint::ToStringA
 */
vislib::StringA vislib::net::IPEndPoint::ToStringA(void) const {
    StringA retval;

    switch (this->address.ss_family) {
        case AF_INET:
            retval.Format("%s:%u", 
                this->GetIPAddress4().ToStringA().PeekBuffer(),
                this->GetPort());
            break;

        case AF_INET6:
            retval.Format("[%s]:%u", 
                this->GetIPAddress6().ToStringA().PeekBuffer(),
                this->GetPort());
            break;

        default:
            ASSERT(false);          // This should be unreachable.
    }

    return retval;
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
 * vislib::net::IPEndPoint::operator =
 */
vislib::net::IPEndPoint& vislib::net::IPEndPoint::operator =(
        const struct sockaddr_in& rhs) {
    if (reinterpret_cast<struct sockaddr_in *>(&this->address) != &rhs) {
        ::ZeroMemory(&this->address, sizeof(struct sockaddr_storage));
        ::memcpy(&this->address, &rhs, sizeof(struct sockaddr_in));
    }
    return *this;
}


/*
 * vislib::net::IPEndPoint::operator =
 */
vislib::net::IPEndPoint& vislib::net::IPEndPoint::operator =(
        const struct sockaddr_in6& rhs) {
    if (reinterpret_cast<struct sockaddr_in6 *>(&this->address) != &rhs) {
        ::ZeroMemory(&this->address, sizeof(struct sockaddr_storage));
        ::memcpy(&this->address, &rhs, sizeof(struct sockaddr_in6));
    }
    return *this;
}


/*
 * vislib::net::IPEndPoint::operator ==
 */
bool vislib::net::IPEndPoint::operator ==(const IPEndPoint& rhs) const {
    size_t size  = 0;

    switch (this->address.ss_family) {
        case AF_INET: 
            size = sizeof(struct sockaddr_in);
            break;

        case AF_INET6: 
            size = sizeof(struct sockaddr_in6);
            break;

        default:
            size = sizeof(struct sockaddr_storage);
            break;
    }

    return (::memcmp(&this->address, &rhs.address, size) == 0);
}


/*
 * vislib::net::IPEndPoint::operator vislib::net::SocketAddress
 */
vislib::net::IPEndPoint::operator vislib::net::SocketAddress(void) const {
    switch (this->address.ss_family) {
        case AF_INET:
            return SocketAddress(this->asV4());

        case AF_INET6:
            return SocketAddress(SocketAddress::FAMILY_INET, this->GetIPAddress4(),
                this->GetPort());

        default:
            throw IllegalStateException("The address family of the IPEndPoint "
                "is not in the expected range.", __FILE__, __LINE__);
    }
}


#ifdef _WIN32
    // Crowbar to work around windows naming conflict
#ifdef _MSC_VER
#pragma pop_macro("SetPort")
#endif /* _MSC_VER */
#endif /* _WIN32 */
