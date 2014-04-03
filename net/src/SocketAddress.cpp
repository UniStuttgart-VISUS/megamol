/*
 * SocketAddress.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). All rights reserved.
 */

#include "vislib/SocketAddress.h"

#include "the/assert.h"
#include "the/memory.h"
#include "the/string.h"
#include "the/text/string_builder.h"


/*
 * vislib::net::SocketAddress::Create
 */
vislib::net::SocketAddress vislib::net::SocketAddress::Create(
        const AddressFamily addressFamily, const char *host, 
        const unsigned short port) {
    return SocketAddress(addressFamily, IPAddress::Create(host), port);
}


/*
 * vislib::net::SocketAddress::CreateInet
 */
vislib::net::SocketAddress vislib::net::SocketAddress::CreateInet(
        const char *host, const unsigned short port) {
    return SocketAddress(FAMILY_INET, IPAddress::Create(host), port);
}


/*
 * vislib::net::SocketAddress::CreateInet
 */
vislib::net::SocketAddress vislib::net::SocketAddress::CreateInet(
        const unsigned short port) {
    return SocketAddress(FAMILY_INET, port);
}


/*
 * vislib::net::SocketAddress::SocketAddress
 */
vislib::net::SocketAddress::SocketAddress(const AddressFamily addressFamily,
                                          const IPAddress& ipAddress, 
                                          const unsigned short port) {
    THE_ASSERT(sizeof(this->genericAddress) == sizeof(this->inetAddress));
    ::memset(&this->genericAddress, 0, sizeof(this->genericAddress));
    this->inetAddress.sin_family = static_cast<unsigned short>(addressFamily);
    this->inetAddress.sin_port = htons(port);
    this->inetAddress.sin_addr = ipAddress.operator in_addr();
}


/*
 * vislib::net::SocketAddress::SocketAddress
 */
vislib::net::SocketAddress::SocketAddress(const AddressFamily addressFamily,
                                          const unsigned short port) {
    THE_ASSERT(sizeof(this->genericAddress) == sizeof(this->inetAddress));
    ::memset(&this->genericAddress, 0, sizeof(this->genericAddress));
    this->inetAddress.sin_family = static_cast<unsigned short>(addressFamily);
    this->inetAddress.sin_port = htons(port);
    this->inetAddress.sin_addr.s_addr = htonl(INADDR_ANY);
}


/*
 * vislib::net::SocketAddress::SocketAddress
 */
vislib::net::SocketAddress::SocketAddress(const struct sockaddr& address) {
    THE_ASSERT(sizeof(this->genericAddress) == sizeof(this->inetAddress));
    ::memcpy(&this->genericAddress, &address, sizeof(struct sockaddr));
}


/*
 * vislib::net::SocketAddress::SocketAddress
 */
vislib::net::SocketAddress::SocketAddress(const struct sockaddr_in& address) {
    THE_ASSERT(sizeof(this->genericAddress) == sizeof(this->inetAddress));
    ::memcpy(&this->inetAddress, &address, sizeof(struct sockaddr_in));
}



/*
 * vislib::net::SocketAddress::SocketAddress
 */
vislib::net::SocketAddress::SocketAddress(void) {
    THE_ASSERT(sizeof(this->genericAddress) == sizeof(this->inetAddress));
    ::memset(&this->genericAddress, 0, sizeof(this->genericAddress));
}


/*
 * vislib::net::SocketAddress::SocketAddress
*/
vislib::net::SocketAddress::SocketAddress(const SocketAddress& rhs) {
    THE_ASSERT(sizeof(this->genericAddress) == sizeof(this->inetAddress));
    ::memcpy(&this->genericAddress, &rhs.genericAddress, 
        sizeof(this->genericAddress));
}


/*
 * vislib::net::SocketAddress::SocketAddress
 */
vislib::net::SocketAddress::SocketAddress(const SocketAddress& address, 
        const unsigned short newPort) {
    THE_ASSERT(sizeof(this->genericAddress) == sizeof(this->inetAddress));
    this->inetAddress.sin_family = address.inetAddress.sin_family;
    this->inetAddress.sin_port = htons(newPort);
    this->inetAddress.sin_addr.s_addr = address.inetAddress.sin_addr.s_addr;
}


/*
 * vislib::net::SocketAddress::~SocketAddress
 */
vislib::net::SocketAddress::~SocketAddress(void) {
}


/*
 * vislib::net::SocketAddress::SetIPAddress
 */
void vislib::net::SocketAddress::SetIPAddress(const IPAddress& ipAddress) {
    this->inetAddress.sin_addr = ipAddress.operator in_addr();
}


/* 
 * vislib::net::SocketAddress::ToStringA
 */
the::astring vislib::net::SocketAddress::ToStringA(void) const {
    the::astring retval;
    the::text::astring_builder::format_to(retval, "%s:%u", static_cast<const char *>(
        this->GetIPAddress().ToStringA().c_str()), this->GetPort());
    return retval;
}


/* 
 * vislib::net::SocketAddress::ToStringW
 */
the::wstring vislib::net::SocketAddress::ToStringW(void) const {
    the::wstring retval;
    // TODO: might fail on linux!!
    the::text::wstring_builder::format_to(retval, L"%s:%u", static_cast<const wchar_t *>(
        this->GetIPAddress().ToStringW().c_str()), this->GetPort());
    return retval;
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
