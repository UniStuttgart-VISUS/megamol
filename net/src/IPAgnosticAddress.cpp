/*
 * IPAgnosticAddress.cpp
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/IPAgnosticAddress.h"

#include <cstdlib>

#include "vislib/DNS.h"
#include "vislib/IllegalStateException.h"
#include "vislib/OutOfRangeException.h"
#include "the/memory.h"


/*
 * vislib::net::IPAgnosticAddress::Create
 */
vislib::net::IPAgnosticAddress vislib::net::IPAgnosticAddress::Create(
        const char *hostNameOrAddress, const AddressFamily inCaseOfDoubt) {
    THE_STACK_TRACE;
    IPAgnosticAddress retval;
    DNS::GetHostAddress(retval, hostNameOrAddress, inCaseOfDoubt);
    return retval;
}


/*
 * vislib::net::IPAgnosticAddress::Create
 */
vislib::net::IPAgnosticAddress vislib::net::IPAgnosticAddress::Create(
        const wchar_t *hostNameOrAddress, const AddressFamily inCaseOfDoubt) {
    THE_STACK_TRACE;
    IPAgnosticAddress retval;
    DNS::GetHostAddress(retval, hostNameOrAddress, inCaseOfDoubt);
    return retval;
}


/*
 * vislib::net::IPAgnosticAddress::CreateAny
 */
vislib::net::IPAgnosticAddress vislib::net::IPAgnosticAddress::CreateAny(
        const AddressFamily addressFamily) {
    THE_STACK_TRACE;

    switch (addressFamily) {
        case FAMILY_INET:
            return IPAgnosticAddress(IPAddress::ANY);

        case FAMILY_INET6:
            return IPAgnosticAddress(IPAddress6::ANY);

        default:
            throw IllegalParamException("addressFamily", __FILE__, __LINE__);
    }
}


/*
 * vislib::net::IPAgnosticAddress::ALL_NODES_ON_LINK4
 */
const vislib::net::IPAddress& 
vislib::net::IPAgnosticAddress::ALL_NODES_ON_LINK4
    = vislib::net::IPAddress::ALL_NODES_ON_LINK;


/*
 * vislib::net::IPAgnosticAddress::ALL_NODES_ON_LINK6
 */
const vislib::net::IPAddress6& 
vislib::net::IPAgnosticAddress::ALL_NODES_ON_LINK6
    = vislib::net::IPAddress6::ALL_NODES_ON_LINK;


/*
 * vislib::net::IPAgnosticAddress::ALL_ROUTERS_ON_LINK4
 */
const vislib::net::IPAddress& 
vislib::net::IPAgnosticAddress::ALL_ROUTERS_ON_LINK4
    = vislib::net::IPAddress::ALL_ROUTERS_ON_LINK;


/*
 * vislib::net::IPAgnosticAddress::ALL_ROUTERS_ON_LINK6
 */
const vislib::net::IPAddress6& 
vislib::net::IPAgnosticAddress::ALL_ROUTERS_ON_LINK6
    = vislib::net::IPAddress6::ALL_ROUTERS_ON_LINK;


/*
 * vislib::net::IPAgnosticAddress::ANY4
 */
const vislib::net::IPAddress& vislib::net::IPAgnosticAddress::ANY4
    = vislib::net::IPAddress::ANY;


/*
 * vislib::net::IPAgnosticAddress::ANY6
 */
const vislib::net::IPAddress6& vislib::net::IPAgnosticAddress::ANY6
    = vislib::net::IPAddress6::ANY;


/*
 * vislib::net::IPAgnosticAddress::LOOPBACK4
 */
const vislib::net::IPAddress& vislib::net::IPAgnosticAddress::LOOPBACK4
    = vislib::net::IPAddress::LOCALHOST;


/*
 * vislib::net::IPAgnosticAddress::LOOPBACK6
 */
const vislib::net::IPAddress6& vislib::net::IPAgnosticAddress::LOOPBACK6
    = vislib::net::IPAddress6::LOOPBACK;


/*
 * vislib::net::IPAgnosticAddress::NONE4
 */
const vislib::net::IPAddress& vislib::net::IPAgnosticAddress::NONE4
    = vislib::net::IPAddress::NONE;


/*
 * vislib::net::IPAgnosticAddress::NONE6
 */
const vislib::net::IPAddress6& vislib::net::IPAgnosticAddress::NONE6
    = vislib::net::IPAddress6::UNSPECIFIED;


/*
 * vislib::net::IPAgnosticAddress::IPAgnosticAddress
 */
vislib::net::IPAgnosticAddress::IPAgnosticAddress(void) : v4(NULL), v6(NULL) {
    THE_STACK_TRACE;
    // Nothing else to do.
}


/*
 * vislib::net::IPAgnosticAddress::IPAgnosticAddress
 */
vislib::net::IPAgnosticAddress::IPAgnosticAddress(const IPAddress& address)
        : v4(NULL), v6(NULL) {
    THE_STACK_TRACE;
    *this = address;
}


/*
 * vislib::net::IPAgnosticAddress::IPAgnosticAddress
 */
vislib::net::IPAgnosticAddress::IPAgnosticAddress(const IPAddress6& address)
        : v4(NULL), v6(NULL) {
    THE_STACK_TRACE;
    *this = address;
}


/*
 * vislib::net::IPAgnosticAddress::IPAgnosticAddress
 */
vislib::net::IPAgnosticAddress::IPAgnosticAddress(const struct in_addr& address)
        : v4(NULL), v6(NULL) {
    THE_STACK_TRACE;
    *this = address;
}


/*
 * vislib::net::IPAgnosticAddress::IPAgnosticAddress
 */
vislib::net::IPAgnosticAddress::IPAgnosticAddress(const struct in6_addr& address)
        : v4(NULL), v6(NULL) {
    THE_STACK_TRACE;
    *this = address;
}


/*
 * vislib::net::IPAgnosticAddress::IPAgnosticAddress
 */
vislib::net::IPAgnosticAddress::IPAgnosticAddress(
        const uint8_t b1, const uint8_t b2, const uint8_t b3, const uint8_t b4)
        : v4(NULL), v6(NULL) {
    THE_STACK_TRACE;
    this->v4 = new IPAddress(b1, b2, b3, b4);
}


/*
 * vislib::net::IPAgnosticAddress::IPAgnosticAddress
 */
vislib::net::IPAgnosticAddress::IPAgnosticAddress(
        const uint8_t b1, const uint8_t b2, const uint8_t b3, const uint8_t b4,
        const uint8_t b5, const uint8_t b6, const uint8_t b7, const uint8_t b8,
        const uint8_t b9, const uint8_t b10, const uint8_t b11, const uint8_t b12,
        const uint8_t b13, const uint8_t b14, const uint8_t b15, const uint8_t b16)
        : v4(NULL), v6(NULL) {
    THE_STACK_TRACE;
    this->v6 = new IPAddress6(b1, b2, b3, b4, b5, b6, b7, b8,
        b9, b10, b11, b12, b13, b14, b15, b16);
}


/*
 * vislib::net::IPAgnosticAddress::IPAgnosticAddress
 */
vislib::net::IPAgnosticAddress::IPAgnosticAddress(const IPAgnosticAddress& rhs) 
        : v4(NULL), v6(NULL) {
    THE_STACK_TRACE;
    *this = rhs;
}


/*
 * vislib::net::IPAgnosticAddress::~IPAgnosticAddress
 */
vislib::net::IPAgnosticAddress::~IPAgnosticAddress(void) {
    THE_STACK_TRACE;
    the::safe_delete(this->v4);
    the::safe_delete(this->v6);
}


/*
 * vislib::net::IPAgnosticAddress::GetAddressFamily
 */
vislib::net::IPAgnosticAddress::AddressFamily 
vislib::net::IPAgnosticAddress::GetAddressFamily(void) const {
    THE_STACK_TRACE;

    if (this->IsV4()) {
        return FAMILY_INET;

    } else if (this->IsV6()) {
        return FAMILY_INET6;

    } else {
        return FAMILY_UNSPECIFIED;
    }
}


/*
 * vislib::net::IPAgnosticAddress::GetPrefix
 */
vislib::net::IPAgnosticAddress vislib::net::IPAgnosticAddress::GetPrefix(
        const ULONG prefixLength) const {
    THE_STACK_TRACE;

    if (this->IsV4()) {
        return IPAgnosticAddress(this->v4->GetPrefix(prefixLength));

    } else if (this->IsV6()) {
        return IPAgnosticAddress(this->v6->GetPrefix(prefixLength));

    } else {
        return IPAgnosticAddress();
    }
}


/*
 * vislib::net::IPAgnosticAddress::ToStringA
 */
vislib::StringA vislib::net::IPAgnosticAddress::ToStringA(void) const {
    THE_STACK_TRACE;

    if (this->IsV4()) {
        return this->v4->ToStringA();

    } else if (this->IsV6()) {
        return this->v6->ToStringA();

    } else {
        return StringA::EMPTY;
    }
}


/*
 * vislib::net::IPAgnosticAddress::ToStringW
 */
vislib::StringW vislib::net::IPAgnosticAddress::ToStringW(void) const {
    THE_STACK_TRACE;

    if (this->IsV4()) {
        return this->v4->ToStringW();

    } else if (this->IsV6()) {
        return this->v6->ToStringW();

    } else {
        return StringW::EMPTY;
    }
}


/*
 * vislib::net::IPAgnosticAddress::operator []
 */
uint8_t vislib::net::IPAgnosticAddress::operator [](const int i) const {
    THE_STACK_TRACE;

    if (this->IsV4()) {
        return this->v4->operator [](i);

    } else if (this->IsV6()) {
        return this->v6->operator [](i);

    } else {
        throw IllegalStateException("The IPAgnosticAddress has no data to "
            "be accessed.", __FILE__, __LINE__);
    }
}


/*
 * vislib::net::IPAgnosticAddress::operator =
 */
vislib::net::IPAgnosticAddress& vislib::net::IPAgnosticAddress::operator =(
        const IPAgnosticAddress& rhs) {
    THE_STACK_TRACE;

    if (this != &rhs) {
        the::safe_delete(this->v4);
        the::safe_delete(this->v6);

        if (rhs.IsV4()) {
            THE_ASSERT(rhs.v6 == NULL);
            this->v4 = new IPAddress(*(rhs.v4));

        } else if (rhs.IsV6()) {
            THE_ASSERT(rhs.v4 == NULL);
            this->v6 = new IPAddress6(*(rhs.v6));
        }
    }

    return *this;
}


/*
 * vislib::net::IPAgnosticAddress::operator =
 */
vislib::net::IPAgnosticAddress& vislib::net::IPAgnosticAddress::operator =(
        const IPAddress& rhs) {
    THE_STACK_TRACE;

    if (this->v4 != &rhs) {
        the::safe_delete(this->v4);
        the::safe_delete(this->v6);
        this->v4 = new IPAddress(rhs);
    }

    return *this;
}


/*
 * vislib::net::IPAgnosticAddress::operator =
 */
vislib::net::IPAgnosticAddress& vislib::net::IPAgnosticAddress::operator =(
        const IPAddress6& rhs) {
    THE_STACK_TRACE;

    if (this->v6 != &rhs) {
        the::safe_delete(this->v4);
        the::safe_delete(this->v6);
        this->v6 = new IPAddress6(rhs);
    }

    return *this;
}


/*
 * vislib::net::IPAgnosticAddress::operator =
 */
vislib::net::IPAgnosticAddress& vislib::net::IPAgnosticAddress::operator =(
        const struct in_addr& rhs) {
    THE_STACK_TRACE;

    // TODO: This might be unsafe if s.o. copies internal data of v4
    the::safe_delete(this->v4);
    the::safe_delete(this->v6);
    this->v4 = new IPAddress(rhs);

    return *this;
}


/*
 * vislib::net::IPAgnosticAddress::operator =
 */
vislib::net::IPAgnosticAddress& vislib::net::IPAgnosticAddress::operator =(
        const struct in6_addr& rhs) {
    THE_STACK_TRACE;

    // TODO: This might be unsafe if s.o. copies internal data of v6
    the::safe_delete(this->v4);
    the::safe_delete(this->v6);
    this->v6 = new IPAddress6(rhs);

    return *this;
}


/*
 * vislib::net::IPAgnosticAddress::operator ==
 */
bool vislib::net::IPAgnosticAddress::operator ==(
        const IPAgnosticAddress& rhs) const {
    THE_STACK_TRACE;

    if (this->IsV4() && rhs.IsV4()) {
        return this->v4->operator ==(*(rhs.v4));

    } else if (this->IsV6() && rhs.IsV6()) {
        return this->v6->operator ==(*(rhs.v6));

    } else if (this->IsV4() && rhs.IsV6()) {
        // TODO: Diese Implementierung ist ranzig!
        return rhs.v6->operator ==(IPAddress6(*(this->v4)));

    } else if (this->IsV6() && rhs.IsV4()) {
        // TODO: Diese Implementierung ist ranzig!
        return this->v6->operator ==(IPAddress6(*(rhs.v4)));

    } else {
        THE_ASSERT(this->v4 == NULL);
        THE_ASSERT(rhs.v4 == NULL);
        THE_ASSERT(this->v6 == NULL);
        THE_ASSERT(rhs.v6 == NULL);
        return true;
    }
}


/*
 * vislib::net::IPAgnosticAddress::operator ==
 */
bool vislib::net::IPAgnosticAddress::operator ==(const IPAddress& rhs) const {
    THE_STACK_TRACE;
    return (this->IsV4()) ? this->v4->operator ==(rhs) : false;
}


/*
 * vislib::net::IPAgnosticAddress::operator ==
 */
bool vislib::net::IPAgnosticAddress::operator ==(const IPAddress6& rhs) const {
    THE_STACK_TRACE;
    return (this->IsV6()) ? this->v6->operator ==(rhs) : false;
}


/*
 * vislib::net::IPAgnosticAddress::operator ==
 */
bool vislib::net::IPAgnosticAddress::operator ==(
        const struct in_addr& rhs) const {
    THE_STACK_TRACE;
    return (this->IsV4()) ? this->v4->operator ==(rhs) : false;
}


/*
 * vislib::net::IPAgnosticAddress::operator ==
 */
bool vislib::net::IPAgnosticAddress::operator ==(
        const struct in6_addr& rhs) const {
    THE_STACK_TRACE;
    return (this->IsV6()) ? this->v6->operator ==(rhs) : false;
}


/*
 * vislib::net::IPAgnosticAddress::operator vislib::net::IPAddress
 */
vislib::net::IPAgnosticAddress::operator vislib::net::IPAddress(void) const {
    THE_STACK_TRACE;

    if (this->IsV4()) {
        return *(this->v4);

    } else if (this->IsV6()) {
        return static_cast<IPAddress>(*(this->v6));

    } else {
        throw IllegalStateException("The IPAgnosticAddress cannot be converted "
            "to an IPv4 address.", __FILE__, __LINE__);
    }
}


/*
 * vislib::net::IPAgnosticAddress::operator const vislib::net::IPAddress *
 */
vislib::net::IPAgnosticAddress::operator const vislib::net::IPAddress *(
        void) const {
    THE_STACK_TRACE;

    if (this->IsV4()) {
        return this->v4;
    } else {
        throw IllegalStateException("The IPAgnosticAddress does not represent "
            "an IPv4 address.", __FILE__, __LINE__);
    }
}


/*
 * vislib::net::IPAgnosticAddress::operator vislib::net::IPAddress *
 */
vislib::net::IPAgnosticAddress::operator vislib::net::IPAddress *(void) {
    THE_STACK_TRACE;

    if (this->IsV4()) {
        return this->v4;
    } else {
        throw IllegalStateException("The IPAgnosticAddress does not represent "
            "an IPv4 address.", __FILE__, __LINE__);
    }
}


/*
 * vislib::net::IPAgnosticAddress::operator vislib::net::IPAddress6
 */
vislib::net::IPAgnosticAddress::operator vislib::net::IPAddress6(void) const {
    THE_STACK_TRACE;

    if (this->IsV4()) {
        return IPAddress6(*(this->v4));

    } else if (this->IsV6()) {
        return *(this->v6);

    } else {
        return IPAgnosticAddress::NONE6;
    }
}


/*
 * vislib::net::IPAgnosticAddress::operator const vislib::net::IPAddress6 *
 */
vislib::net::IPAgnosticAddress::operator const vislib::net::IPAddress6 *(
        void) const {
    THE_STACK_TRACE;

    if (this->IsV6()) {
        return this->v6;
    } else {
        throw IllegalStateException("The IPAgnosticAddress does not represent "
            "an IPv6 address.", __FILE__, __LINE__);
    }
}


/*
 * vislib::net::IPAgnosticAddress::operator vislib::net::IPAddress6 *
 */
vislib::net::IPAgnosticAddress::operator vislib::net::IPAddress6 *(void) {
    THE_STACK_TRACE;

    if (this->IsV6()) {
        return this->v6;
    } else {
        throw IllegalStateException("The IPAgnosticAddress does not represent "
            "an IPv6 address.", __FILE__, __LINE__);
    }
}
