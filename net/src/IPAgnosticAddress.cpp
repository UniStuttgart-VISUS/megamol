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
#include "vislib/memutils.h"


/*
 * vislib::net::IPAgnosticAddress::Create
 */
vislib::net::IPAgnosticAddress vislib::net::IPAgnosticAddress::Create(
        const char *hostNameOrAddress, const AddressFamily inCaseOfDoubt) {
    VLSTACKTRACE("IPAgnosticAddress::Create", __FILE__, __LINE__);
    IPAgnosticAddress retval;
    DNS::GetHostAddress(retval, hostNameOrAddress, inCaseOfDoubt);
    return retval;
}


/*
 * vislib::net::IPAgnosticAddress::Create
 */
vislib::net::IPAgnosticAddress vislib::net::IPAgnosticAddress::Create(
        const wchar_t *hostNameOrAddress, const AddressFamily inCaseOfDoubt) {
    VLSTACKTRACE("IPAgnosticAddress::Create", __FILE__, __LINE__);
    IPAgnosticAddress retval;
    DNS::GetHostAddress(retval, hostNameOrAddress, inCaseOfDoubt);
    return retval;
}


/*
 * vislib::net::IPAgnosticAddress::CreateAny
 */
vislib::net::IPAgnosticAddress vislib::net::IPAgnosticAddress::CreateAny(
        const AddressFamily addressFamily) {
    VLSTACKTRACE("IPAgnosticAddress::CreateAny", __FILE__, __LINE__);

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
    VLSTACKTRACE("IPAgnosticAddress::IPAgnosticAddress", __FILE__, __LINE__);
    // Nothing else to do.
}


/*
 * vislib::net::IPAgnosticAddress::IPAgnosticAddress
 */
vislib::net::IPAgnosticAddress::IPAgnosticAddress(const IPAddress& address)
        : v4(NULL), v6(NULL) {
    VLSTACKTRACE("IPAgnosticAddress::IPAgnosticAddress", __FILE__, __LINE__);
    *this = address;
}


/*
 * vislib::net::IPAgnosticAddress::IPAgnosticAddress
 */
vislib::net::IPAgnosticAddress::IPAgnosticAddress(const IPAddress6& address)
        : v4(NULL), v6(NULL) {
    VLSTACKTRACE("IPAgnosticAddress::IPAgnosticAddress", __FILE__, __LINE__);
    *this = address;
}


/*
 * vislib::net::IPAgnosticAddress::IPAgnosticAddress
 */
vislib::net::IPAgnosticAddress::IPAgnosticAddress(const struct in_addr& address)
        : v4(NULL), v6(NULL) {
    VLSTACKTRACE("IPAgnosticAddress::IPAgnosticAddress", __FILE__, __LINE__);
    *this = address;
}


/*
 * vislib::net::IPAgnosticAddress::IPAgnosticAddress
 */
vislib::net::IPAgnosticAddress::IPAgnosticAddress(const struct in6_addr& address)
        : v4(NULL), v6(NULL) {
    VLSTACKTRACE("IPAgnosticAddress::IPAgnosticAddress", __FILE__, __LINE__);
    *this = address;
}


/*
 * vislib::net::IPAgnosticAddress::IPAgnosticAddress
 */
vislib::net::IPAgnosticAddress::IPAgnosticAddress(
        const BYTE b1, const BYTE b2, const BYTE b3, const BYTE b4)
        : v4(NULL), v6(NULL) {
    VLSTACKTRACE("IPAgnosticAddress::IPAgnosticAddress", __FILE__, __LINE__);
    this->v4 = new IPAddress(b1, b2, b3, b4);
}


/*
 * vislib::net::IPAgnosticAddress::IPAgnosticAddress
 */
vislib::net::IPAgnosticAddress::IPAgnosticAddress(
        const BYTE b1, const BYTE b2, const BYTE b3, const BYTE b4,
        const BYTE b5, const BYTE b6, const BYTE b7, const BYTE b8,
        const BYTE b9, const BYTE b10, const BYTE b11, const BYTE b12,
        const BYTE b13, const BYTE b14, const BYTE b15, const BYTE b16)
        : v4(NULL), v6(NULL) {
    VLSTACKTRACE("IPAgnosticAddress::IPAgnosticAddress", __FILE__, __LINE__);
    this->v6 = new IPAddress6(b1, b2, b3, b4, b5, b6, b7, b8,
        b9, b10, b11, b12, b13, b14, b15, b16);
}


/*
 * vislib::net::IPAgnosticAddress::IPAgnosticAddress
 */
vislib::net::IPAgnosticAddress::IPAgnosticAddress(const IPAgnosticAddress& rhs) 
        : v4(NULL), v6(NULL) {
    VLSTACKTRACE("IPAgnosticAddress::IPAgnosticAddress", __FILE__, __LINE__);
    *this = rhs;
}


/*
 * vislib::net::IPAgnosticAddress::~IPAgnosticAddress
 */
vislib::net::IPAgnosticAddress::~IPAgnosticAddress(void) {
    VLSTACKTRACE("IPAgnosticAddress::~IPAgnosticAddress", __FILE__, __LINE__);
    SAFE_DELETE(this->v4);
    SAFE_DELETE(this->v6);
}


/*
 * vislib::net::IPAgnosticAddress::GetAddressFamily
 */
vislib::net::IPAgnosticAddress::AddressFamily 
vislib::net::IPAgnosticAddress::GetAddressFamily(void) const {
    VLSTACKTRACE("IPAgnosticAddress::GetAddressFamily", __FILE__, __LINE__);

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
    VLSTACKTRACE("IPAgnosticAddress::GetPrefix", __FILE__, __LINE__);

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
    VLSTACKTRACE("IPAgnosticAddress::ToStringA", __FILE__, __LINE__);

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
    VLSTACKTRACE("IPAgnosticAddress::ToStringW", __FILE__, __LINE__);

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
BYTE vislib::net::IPAgnosticAddress::operator [](const int i) const {
    VLSTACKTRACE("IPAgnosticAddress::operator []", __FILE__, __LINE__);

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
    VLSTACKTRACE("IPAgnosticAddress::operator =", __FILE__, __LINE__);

    if (this != &rhs) {
        SAFE_DELETE(this->v4);
        SAFE_DELETE(this->v6);

        if (rhs.IsV4()) {
            ASSERT(rhs.v6 == NULL);
            this->v4 = new IPAddress(*(rhs.v4));

        } else if (rhs.IsV6()) {
            ASSERT(rhs.v4 == NULL);
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
    VLSTACKTRACE("IPAgnosticAddress::operator =", __FILE__, __LINE__);

    if (this->v4 != &rhs) {
        SAFE_DELETE(this->v4);
        SAFE_DELETE(this->v6);
        this->v4 = new IPAddress(rhs);
    }

    return *this;
}


/*
 * vislib::net::IPAgnosticAddress::operator =
 */
vislib::net::IPAgnosticAddress& vislib::net::IPAgnosticAddress::operator =(
        const IPAddress6& rhs) {
    VLSTACKTRACE("IPAgnosticAddress::operator =", __FILE__, __LINE__);

    if (this->v6 != &rhs) {
        SAFE_DELETE(this->v4);
        SAFE_DELETE(this->v6);
        this->v6 = new IPAddress6(rhs);
    }

    return *this;
}


/*
 * vislib::net::IPAgnosticAddress::operator =
 */
vislib::net::IPAgnosticAddress& vislib::net::IPAgnosticAddress::operator =(
        const struct in_addr& rhs) {
    VLSTACKTRACE("IPAgnosticAddress::operator =", __FILE__, __LINE__);

    // TODO: This might be unsafe if s.o. copies internal data of v4
    SAFE_DELETE(this->v4);
    SAFE_DELETE(this->v6);
    this->v4 = new IPAddress(rhs);

    return *this;
}


/*
 * vislib::net::IPAgnosticAddress::operator =
 */
vislib::net::IPAgnosticAddress& vislib::net::IPAgnosticAddress::operator =(
        const struct in6_addr& rhs) {
    VLSTACKTRACE("IPAgnosticAddress::operator =", __FILE__, __LINE__);

    // TODO: This might be unsafe if s.o. copies internal data of v6
    SAFE_DELETE(this->v4);
    SAFE_DELETE(this->v6);
    this->v6 = new IPAddress6(rhs);

    return *this;
}


/*
 * vislib::net::IPAgnosticAddress::operator ==
 */
bool vislib::net::IPAgnosticAddress::operator ==(
        const IPAgnosticAddress& rhs) const {
    VLSTACKTRACE("IPAgnosticAddress::operator ==", __FILE__, __LINE__);

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
        ASSERT(this->v4 == NULL);
        ASSERT(rhs.v4 == NULL);
        ASSERT(this->v6 == NULL);
        ASSERT(rhs.v6 == NULL);
        return true;
    }
}


/*
 * vislib::net::IPAgnosticAddress::operator ==
 */
bool vislib::net::IPAgnosticAddress::operator ==(const IPAddress& rhs) const {
    VLSTACKTRACE("IPAgnosticAddress::operator ==", __FILE__, __LINE__);
    return (this->IsV4()) ? this->v4->operator ==(rhs) : false;
}


/*
 * vislib::net::IPAgnosticAddress::operator ==
 */
bool vislib::net::IPAgnosticAddress::operator ==(const IPAddress6& rhs) const {
    VLSTACKTRACE("IPAgnosticAddress::operator ==", __FILE__, __LINE__);
    return (this->IsV6()) ? this->v6->operator ==(rhs) : false;
}


/*
 * vislib::net::IPAgnosticAddress::operator ==
 */
bool vislib::net::IPAgnosticAddress::operator ==(
        const struct in_addr& rhs) const {
    VLSTACKTRACE("IPAgnosticAddress::operator ==", __FILE__, __LINE__);
    return (this->IsV4()) ? this->v4->operator ==(rhs) : false;
}


/*
 * vislib::net::IPAgnosticAddress::operator ==
 */
bool vislib::net::IPAgnosticAddress::operator ==(
        const struct in6_addr& rhs) const {
    VLSTACKTRACE("IPAgnosticAddress::operator ==", __FILE__, __LINE__);
    return (this->IsV6()) ? this->v6->operator ==(rhs) : false;
}


/*
 * vislib::net::IPAgnosticAddress::operator vislib::net::IPAddress
 */
vislib::net::IPAgnosticAddress::operator vislib::net::IPAddress(void) const {
    VLSTACKTRACE("IPAgnosticAddress::operator IPAddress", __FILE__, __LINE__);

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
    VLSTACKTRACE("IPAgnosticAddress::operator const IPAddress *", __FILE__, 
        __LINE__);

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
    VLSTACKTRACE("IPAgnosticAddress::operator IPAddress *", __FILE__, __LINE__);

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
    VLSTACKTRACE("IPAgnosticAddress::operator IPAddress6", __FILE__, __LINE__);

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
    VLSTACKTRACE("IPAgnosticAddress::operator const IPAddress6 *", __FILE__, 
        __LINE__);

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
    VLSTACKTRACE("IPAgnosticAddress::operator IPAddress6 *", __FILE__, __LINE__);

    if (this->IsV6()) {
        return this->v6;
    } else {
        throw IllegalStateException("The IPAgnosticAddress does not represent "
            "an IPv6 address.", __FILE__, __LINE__);
    }
}
