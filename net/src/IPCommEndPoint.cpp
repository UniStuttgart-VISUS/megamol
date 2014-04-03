/*
 * IPCommEndPoint.cpp
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/IPCommEndPoint.h"

#include "vislib/NetworkInformation.h"
#include "the/argument_exception.h"


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> 
vislib::net::IPCommEndPoint::Create(
        const IPEndPoint& endPoint) {
    THE_STACK_TRACE;
    return SmartRef<AbstractCommEndPoint>(new IPCommEndPoint(endPoint), false);
}


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> 
vislib::net::IPCommEndPoint::Create(
        const IPAgnosticAddress& ipAddress, const unsigned short port) {
    THE_STACK_TRACE;
    return IPCommEndPoint::Create(IPEndPoint(ipAddress, port));
}
 

/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> 
vislib::net::IPCommEndPoint::Create(
        const IPAddress& ipAddress, const unsigned short port) {
    THE_STACK_TRACE;
    return IPCommEndPoint::Create(IPEndPoint(ipAddress, port));
}
 

/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> 
vislib::net::IPCommEndPoint::Create(
        const IPAddress6& ipAddress, const unsigned short port) {
    THE_STACK_TRACE;
    return IPCommEndPoint::Create(IPEndPoint(ipAddress, port));
}


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> 
vislib::net::IPCommEndPoint::Create(
        const ProtocolVersion protocolVersion, const unsigned short port) {
    THE_STACK_TRACE;
    switch (protocolVersion) {
        case IPV4:
            return IPCommEndPoint::Create(IPAgnosticAddress::ANY4, port);
            /* Unreachable. */

        case IPV6:
            return IPCommEndPoint::Create(IPAgnosticAddress::ANY6, port);
            /* Unreachable. */

        default:
            throw the::argument_exception("protocolVersion", __FILE__, __LINE__);
            /* Unreachable. */
    }
}


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> 
vislib::net::IPCommEndPoint::Create(
            const ProtocolVersion protocolVersion,
            const char *hostNameOrAddress,
            const unsigned short port) {
    THE_STACK_TRACE;
    switch (protocolVersion) {
        case IPV4:
            return IPCommEndPoint::Create(IPEndPoint::CreateIPv4(
                hostNameOrAddress, port));
            /* Unreachable. */

        case IPV6:
            return IPCommEndPoint::Create(IPEndPoint::CreateIPv6(
                hostNameOrAddress, port));
            /* Unreachable. */

        default:
            throw the::argument_exception("protocolVersion", __FILE__, __LINE__);
            /* Unreachable. */
    }
}


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> 
vislib::net::IPCommEndPoint::Create(
        IPAgnosticAddress::AddressFamily addressFamily,
        const char *str) {
    THE_STACK_TRACE;
    IPEndPoint ep;
    if (NetworkInformation::GuessRemoteEndPoint(ep, str, addressFamily) 
            > 0.0f) {
        throw the::argument_exception("str", __FILE__, __LINE__);
    } else {
        return IPCommEndPoint::Create(ep);
    }
}


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> 
vislib::net::IPCommEndPoint::Create(
        const char *str) {
    THE_STACK_TRACE;
    SmartRef<AbstractCommEndPoint> retval = IPCommEndPoint::Create(IPV4, 
        static_cast<unsigned short>(0));
    retval->Parse(str);
    return retval;
}


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> 
vislib::net::IPCommEndPoint::Create(
        const wchar_t *str) {
    THE_STACK_TRACE;
    SmartRef<AbstractCommEndPoint> retval = IPCommEndPoint::Create(IPV4, 
        static_cast<unsigned short>(0));
    retval->Parse(str);
    return retval;
}


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> 
vislib::net::IPCommEndPoint::Create(const struct sockaddr_storage& address) {
    THE_STACK_TRACE;
    IPEndPoint ep(address);
    return SmartRef<AbstractCommEndPoint>(new IPCommEndPoint(ep), false);
}


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> 
vislib::net::IPCommEndPoint::Create(const struct sockaddr_in& address) {
    THE_STACK_TRACE;
    IPEndPoint ep(address);
    return SmartRef<AbstractCommEndPoint>(new IPCommEndPoint(ep), false);
}


/*
 * vislib::net::IPCommEndPoint::Create
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> 
vislib::net::IPCommEndPoint::Create(const struct sockaddr_in6& address) {
    THE_STACK_TRACE;
    IPEndPoint ep(address);
    return SmartRef<AbstractCommEndPoint>(new IPCommEndPoint(ep), false);
}


/*
 * vislib::net::IPCommEndPoint::Parse
 */
void vislib::net::IPCommEndPoint::Parse(const the::astring& str) {
    THE_STACK_TRACE;
    IPEndPoint ep;
    if (NetworkInformation::GuessRemoteEndPoint(ep, str.c_str()) > 0.0f) {
        throw the::argument_exception("str", __FILE__, __LINE__);
    } else {
        this->endPoint = ep;
    }
}


/*
 * vislib::net::IPCommEndPoint::Parse
 */
void vislib::net::IPCommEndPoint::Parse(const the::astring& str,
        const ProtocolVersion preferredProtocolVersion) {
    THE_STACK_TRACE;
    IPEndPoint ep;
    if (NetworkInformation::GuessRemoteEndPoint(ep, str.c_str(),
            static_cast<IPAgnosticAddress::AddressFamily>
            (preferredProtocolVersion)) > 0.0f) {
        throw the::argument_exception("str", __FILE__, __LINE__);
    } else {
        this->endPoint = ep;
    }
}


/*
 * vislib::net::IPCommEndPoint::Parse
 */
void vislib::net::IPCommEndPoint::Parse(const the::wstring& str)  {
    THE_STACK_TRACE;
    IPEndPoint ep;
    if (NetworkInformation::GuessRemoteEndPoint(ep, str.c_str()) > 0.0f) {
        throw the::argument_exception("str", __FILE__, __LINE__);
    } else {
        this->endPoint = ep;
    }
}


/*
 * vislib::net::IPCommEndPoint::Parse
 */
void vislib::net::IPCommEndPoint::Parse(const the::wstring& str,
        const ProtocolVersion preferredProtocolVersion) {
    THE_STACK_TRACE;
    IPEndPoint ep;
    if (NetworkInformation::GuessRemoteEndPoint(ep, str.c_str(),
            static_cast<IPAgnosticAddress::AddressFamily>
            (preferredProtocolVersion)) > 0.0f) {
        throw the::argument_exception("str", __FILE__, __LINE__);
    } else {
        this->endPoint = ep;
    }
}

/*
 * vislib::net::IPCommEndPoint::ToStringA
 */
the::astring vislib::net::IPCommEndPoint::ToStringA(void) const {
    THE_STACK_TRACE;
    return this->endPoint.ToStringA();
}


/*
 * vislib::net::IPCommEndPoint::ToStringW
 */
the::wstring vislib::net::IPCommEndPoint::ToStringW(void) const {
    THE_STACK_TRACE;
    return this->endPoint.ToStringW();
}


/*
 * vislib::net::IPCommEndPoint:operator ==
 */
bool vislib::net::IPCommEndPoint::operator ==(
        const AbstractCommEndPoint& rhs) const {
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
vislib::net::IPCommEndPoint::ProtocolVersion 
vislib::net::IPCommEndPoint::convertAddressFamily(
        const IPAgnosticAddress::AddressFamily addressFamily) {
    THE_STACK_TRACE;
    switch (addressFamily) {
        case IPAgnosticAddress::FAMILY_INET:
            /* Falls through. */
        case IPAgnosticAddress::FAMILY_INET6:
            return static_cast<ProtocolVersion>(addressFamily);
            /* unreachable. */

        default:
            throw the::argument_exception("addressFamily", __FILE__, __LINE__);
            /* unreachable. */
    }
}


/*
 * vislib::net::IPCommEndPoint::IPCommEndPoint
 */
vislib::net::IPCommEndPoint::IPCommEndPoint(
        const IPEndPoint& endPoint) : Super(), endPoint(endPoint) {
    THE_STACK_TRACE;
}


/*
 * vislib::net::IPCommEndPoint::~IPCommEndPoint
 */
vislib::net::IPCommEndPoint::~IPCommEndPoint(void) {
    THE_STACK_TRACE;
}
