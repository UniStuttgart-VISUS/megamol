/*
 * IPCommEndPoint.cpp
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/net/IPCommEndPoint.h"

#include "vislib/IllegalParamException.h"
#include "vislib/net/NetworkInformation.h"


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> vislib::net::IPCommEndPoint::Create(const IPEndPoint& endPoint) {
    return SmartRef<AbstractCommEndPoint>(new IPCommEndPoint(endPoint), false);
}


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> vislib::net::IPCommEndPoint::Create(
    const IPAgnosticAddress& ipAddress, const unsigned short port) {
    return IPCommEndPoint::Create(IPEndPoint(ipAddress, port));
}


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> vislib::net::IPCommEndPoint::Create(
    const IPAddress& ipAddress, const unsigned short port) {
    return IPCommEndPoint::Create(IPEndPoint(ipAddress, port));
}


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> vislib::net::IPCommEndPoint::Create(
    const IPAddress6& ipAddress, const unsigned short port) {
    return IPCommEndPoint::Create(IPEndPoint(ipAddress, port));
}


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> vislib::net::IPCommEndPoint::Create(
    const ProtocolVersion protocolVersion, const unsigned short port) {
    switch (protocolVersion) {
    case IPV4:
        return IPCommEndPoint::Create(IPAgnosticAddress::ANY4, port);
        /* Unreachable. */

    case IPV6:
        return IPCommEndPoint::Create(IPAgnosticAddress::ANY6, port);
        /* Unreachable. */

    default:
        throw IllegalParamException("protocolVersion", __FILE__, __LINE__);
        /* Unreachable. */
    }
}


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> vislib::net::IPCommEndPoint::Create(
    const ProtocolVersion protocolVersion, const char* hostNameOrAddress, const unsigned short port) {
    switch (protocolVersion) {
    case IPV4:
        return IPCommEndPoint::Create(IPEndPoint::CreateIPv4(hostNameOrAddress, port));
        /* Unreachable. */

    case IPV6:
        return IPCommEndPoint::Create(IPEndPoint::CreateIPv6(hostNameOrAddress, port));
        /* Unreachable. */

    default:
        throw IllegalParamException("protocolVersion", __FILE__, __LINE__);
        /* Unreachable. */
    }
}


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> vislib::net::IPCommEndPoint::Create(
    IPAgnosticAddress::AddressFamily addressFamily, const char* str) {
    IPEndPoint ep;
    if (NetworkInformation::GuessRemoteEndPoint(ep, str, addressFamily) > 0.0f) {
        throw IllegalParamException("str", __FILE__, __LINE__);
    } else {
        return IPCommEndPoint::Create(ep);
    }
}


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> vislib::net::IPCommEndPoint::Create(const char* str) {
    SmartRef<AbstractCommEndPoint> retval = IPCommEndPoint::Create(IPV4, static_cast<unsigned short>(0));
    retval->Parse(str);
    return retval;
}


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> vislib::net::IPCommEndPoint::Create(const wchar_t* str) {
    SmartRef<AbstractCommEndPoint> retval = IPCommEndPoint::Create(IPV4, static_cast<unsigned short>(0));
    retval->Parse(str);
    return retval;
}


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> vislib::net::IPCommEndPoint::Create(
    const struct sockaddr_storage& address) {
    IPEndPoint ep(address);
    return SmartRef<AbstractCommEndPoint>(new IPCommEndPoint(ep), false);
}


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> vislib::net::IPCommEndPoint::Create(
    const struct sockaddr_in& address) {
    IPEndPoint ep(address);
    return SmartRef<AbstractCommEndPoint>(new IPCommEndPoint(ep), false);
}


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> vislib::net::IPCommEndPoint::Create(
    const struct sockaddr_in6& address) {
    IPEndPoint ep(address);
    return SmartRef<AbstractCommEndPoint>(new IPCommEndPoint(ep), false);
}


/*
 * vislib::net::IPCommEndPoint::Parse
 */
void vislib::net::IPCommEndPoint::Parse(const StringA& str) {
    IPEndPoint ep;
    if (NetworkInformation::GuessRemoteEndPoint(ep, str.PeekBuffer()) > 0.0f) {
        throw IllegalParamException("str", __FILE__, __LINE__);
    } else {
        this->endPoint = ep;
    }
}


/*
 * vislib::net::IPCommEndPoint::Parse
 */
void vislib::net::IPCommEndPoint::Parse(const StringA& str, const ProtocolVersion preferredProtocolVersion) {
    IPEndPoint ep;
    if (NetworkInformation::GuessRemoteEndPoint(
            ep, str.PeekBuffer(), static_cast<IPAgnosticAddress::AddressFamily>(preferredProtocolVersion)) > 0.0f) {
        throw IllegalParamException("str", __FILE__, __LINE__);
    } else {
        this->endPoint = ep;
    }
}


/*
 * vislib::net::IPCommEndPoint::Parse
 */
void vislib::net::IPCommEndPoint::Parse(const StringW& str) {
    IPEndPoint ep;
    if (NetworkInformation::GuessRemoteEndPoint(ep, str.PeekBuffer()) > 0.0f) {
        throw IllegalParamException("str", __FILE__, __LINE__);
    } else {
        this->endPoint = ep;
    }
}


/*
 * vislib::net::IPCommEndPoint::Parse
 */
void vislib::net::IPCommEndPoint::Parse(const StringW& str, const ProtocolVersion preferredProtocolVersion) {
    IPEndPoint ep;
    if (NetworkInformation::GuessRemoteEndPoint(
            ep, str.PeekBuffer(), static_cast<IPAgnosticAddress::AddressFamily>(preferredProtocolVersion)) > 0.0f) {
        throw IllegalParamException("str", __FILE__, __LINE__);
    } else {
        this->endPoint = ep;
    }
}

/*
 * vislib::net::IPCommEndPoint::ToStringA
 */
vislib::StringA vislib::net::IPCommEndPoint::ToStringA() const {
    return this->endPoint.ToStringA();
}


/*
 * vislib::net::IPCommEndPoint::ToStringW
 */
vislib::StringW vislib::net::IPCommEndPoint::ToStringW() const {
    return this->endPoint.ToStringW();
}


/*
 * vislib::net::IPCommEndPoint:operator ==
 */
bool vislib::net::IPCommEndPoint::operator==(const AbstractCommEndPoint& rhs) const {
    try {
        const IPCommEndPoint& ep = dynamic_cast<const IPCommEndPoint&>(rhs);
        return (this->endPoint == static_cast<IPEndPoint>(ep));
    } catch (...) {
        // If the conversion fails, 'rhs' is not an IPCommEndPoint and
        // therefore cannot be equal.
        return false;
    }
}


/*
 * vislib::net::IPCommEndPoint::convertAddressFamily
 */
vislib::net::IPCommEndPoint::ProtocolVersion vislib::net::IPCommEndPoint::convertAddressFamily(
    const IPAgnosticAddress::AddressFamily addressFamily) {
    switch (addressFamily) {
    case IPAgnosticAddress::FAMILY_INET:
        /* Falls through. */
    case IPAgnosticAddress::FAMILY_INET6:
        return static_cast<ProtocolVersion>(addressFamily);
        /* unreachable. */

    default:
        throw IllegalParamException("addressFamily", __FILE__, __LINE__);
        /* unreachable. */
    }
}


/*
 * vislib::net::IPCommEndPoint::IPCommEndPoint
 */
vislib::net::IPCommEndPoint::IPCommEndPoint(const IPEndPoint& endPoint) : Super(), endPoint(endPoint) {}


/*
 * vislib::net::IPCommEndPoint::~IPCommEndPoint
 */
vislib::net::IPCommEndPoint::~IPCommEndPoint() {}
