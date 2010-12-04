/*
 * IPCommEndPointAddress.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/IPCommEndPointAddress.h"

#include "vislib/NetworkInformation.h"
#include "vislib/IllegalParamException.h"


/*
 * vislib::net::IPCommEndPointAddress::Create
 */
vislib::net::IPCommEndPointAddress *vislib::net::IPCommEndPointAddress::Create(
        const IPEndPoint& endPoint) {
    VLSTACKTRACE("IPCommEndPointAddress::Create", __FILE__, __LINE__);
    return new IPCommEndPointAddress(endPoint);
}


/*
 * vislib::net::IPCommEndPointAddress::Create
 */
vislib::net::IPCommEndPointAddress *vislib::net::IPCommEndPointAddress::Create(
        const IPAgnosticAddress& ipAddress, const unsigned short port) {
    VLSTACKTRACE("IPCommEndPointAddress::Create", __FILE__, __LINE__);
    return IPCommEndPointAddress::Create(IPEndPoint(ipAddress, port));
}
 

/*
 * vislib::net::IPCommEndPointAddress::Create
 */
vislib::net::IPCommEndPointAddress *vislib::net::IPCommEndPointAddress::Create(
        const IPAddress& ipAddress, const unsigned short port) {
    VLSTACKTRACE("IPCommEndPointAddress::Create", __FILE__, __LINE__);
    return IPCommEndPointAddress::Create(IPEndPoint(ipAddress, port));
}
 

/*
 * vislib::net::IPCommEndPointAddress::Create
 */
vislib::net::IPCommEndPointAddress *vislib::net::IPCommEndPointAddress::Create(
        const IPAddress6& ipAddress, const unsigned short port) {
    VLSTACKTRACE("IPCommEndPointAddress::Create", __FILE__, __LINE__);
    return IPCommEndPointAddress::Create(IPEndPoint(ipAddress, port));
}


/*
 * vislib::net::IPCommEndPointAddress::Create
 */
vislib::net::IPCommEndPointAddress *vislib::net::IPCommEndPointAddress::Create(
        const ProtocolVersion protocolVersion, const unsigned short port) {
    VLSTACKTRACE("IPCommEndPointAddress::Create", __FILE__, __LINE__);
    switch (protocolVersion) {
        case IPV4:
            return IPCommEndPointAddress::Create(IPAgnosticAddress::ANY4, port);
            /* Unreachable. */

        case IPV6:
            return IPCommEndPointAddress::Create(IPAgnosticAddress::ANY6, port);
            /* Unreachable. */

        default:
            throw IllegalParamException("protocolVersion", __FILE__, __LINE__);
            /* Unreachable. */
    }
}


/*
 * vislib::net::IPCommEndPointAddress::Create
 */
vislib::net::IPCommEndPointAddress *vislib::net::IPCommEndPointAddress::Create(
            const ProtocolVersion protocolVersion,
            const char *hostNameOrAddress,
            const unsigned short port) {
    VLSTACKTRACE("IPCommEndPointAddress::Create", __FILE__, __LINE__);
    switch (protocolVersion) {
        case IPV4:
            return IPCommEndPointAddress::Create(IPEndPoint::CreateIPv4(
                hostNameOrAddress, port));
            /* Unreachable. */

        case IPV6:
            return IPCommEndPointAddress::Create(IPEndPoint::CreateIPv6(
                hostNameOrAddress, port));
            /* Unreachable. */

        default:
            throw IllegalParamException("protocolVersion", __FILE__, __LINE__);
            /* Unreachable. */
    }
}


/*
 * vislib::net::IPCommEndPointAddress::Create
 */
vislib::net::IPCommEndPointAddress *vislib::net::IPCommEndPointAddress::Create(
        IPAgnosticAddress::AddressFamily addressFamily,
        const char *str) {
    VLSTACKTRACE("IPCommEndPointAddress::Create", __FILE__, __LINE__);
    IPEndPoint ep;
    if (NetworkInformation::GuessRemoteEndPoint(ep, str, addressFamily) 
            > 0.0f) {
        throw IllegalParamException("str", __FILE__, __LINE__);
    } else {
        return IPCommEndPointAddress::Create(ep);
    }
}


/*
 * vislib::net::IPCommEndPointAddress::Create
 */
vislib::net::IPCommEndPointAddress *vislib::net::IPCommEndPointAddress::Create(
        const char *str) {
    VLSTACKTRACE("IPCommEndPointAddress::Create", __FILE__, __LINE__);
    IPCommEndPointAddress *retval = IPCommEndPointAddress::Create(IPV4, 
        static_cast<unsigned short>(0));
    try {
        retval->Parse(str);
    } catch (...) {
        retval->Release();
        throw;
    }
    return retval;
}


/*
 * vislib::net::IPCommEndPointAddress::Create
 */
vislib::net::IPCommEndPointAddress *vislib::net::IPCommEndPointAddress::Create(
        const wchar_t *str) {
    VLSTACKTRACE("IPCommEndPointAddress::Create", __FILE__, __LINE__);
    IPCommEndPointAddress *retval = IPCommEndPointAddress::Create(IPV4, 
        static_cast<unsigned short>(0));
    try {
        retval->Parse(str);
    } catch (...) {
        retval->Release();
        throw;
    }
    return retval;
}


/*
 * vislib::net::IPCommEndPointAddress::Parse
 */
void vislib::net::IPCommEndPointAddress::Parse(const StringA& str) {
    VLSTACKTRACE("IPCommEndPointAddress::Parse", __FILE__, __LINE__);
    IPEndPoint ep;
    if (NetworkInformation::GuessRemoteEndPoint(ep, str.PeekBuffer()) > 0.0f) {
        throw IllegalParamException("str", __FILE__, __LINE__);
    } else {
        this->endPoint = ep;
    }
}


/*
 * vislib::net::IPCommEndPointAddress::Parse
 */
void vislib::net::IPCommEndPointAddress::Parse(const StringA& str,
        const ProtocolVersion preferredProtocolVersion) {
    VLSTACKTRACE("IPCommEndPointAddress::Parse", __FILE__, __LINE__);
    IPEndPoint ep;
    if (NetworkInformation::GuessRemoteEndPoint(ep, str.PeekBuffer(),
            static_cast<IPAgnosticAddress::AddressFamily>
            (preferredProtocolVersion)) > 0.0f) {
        throw IllegalParamException("str", __FILE__, __LINE__);
    } else {
        this->endPoint = ep;
    }
}


/*
 * vislib::net::IPCommEndPointAddress::Parse
 */
void vislib::net::IPCommEndPointAddress::Parse(const StringW& str)  {
    VLSTACKTRACE("IPCommEndPointAddress::Parse", __FILE__, __LINE__);
    IPEndPoint ep;
    if (NetworkInformation::GuessRemoteEndPoint(ep, str.PeekBuffer()) > 0.0f) {
        throw IllegalParamException("str", __FILE__, __LINE__);
    } else {
        this->endPoint = ep;
    }
}


/*
 * vislib::net::IPCommEndPointAddress::Parse
 */
void vislib::net::IPCommEndPointAddress::Parse(const StringW& str,
        const ProtocolVersion preferredProtocolVersion) {
    VLSTACKTRACE("IPCommEndPointAddress::Parse", __FILE__, __LINE__);
    IPEndPoint ep;
    if (NetworkInformation::GuessRemoteEndPoint(ep, str.PeekBuffer(),
            static_cast<IPAgnosticAddress::AddressFamily>
            (preferredProtocolVersion)) > 0.0f) {
        throw IllegalParamException("str", __FILE__, __LINE__);
    } else {
        this->endPoint = ep;
    }
}

/*
 * vislib::net::IPCommEndPointAddress::ToStringA
 */
vislib::StringA vislib::net::IPCommEndPointAddress::ToStringA(void) const {
    VLSTACKTRACE("IPCommEndPointAddress::ToStringA", __FILE__, __LINE__);
    return this->endPoint.ToStringA();
}


/*
 * vislib::net::IPCommEndPointAddress::ToStringW
 */
vislib::StringW vislib::net::IPCommEndPointAddress::ToStringW(void) const {
    VLSTACKTRACE("IPCommEndPointAddress::ToStringW", __FILE__, __LINE__);
    return this->endPoint.ToStringW();
}


/*
 * vislib::net::IPCommEndPointAddress::convertAddressFamily
 */
vislib::net::IPCommEndPointAddress::ProtocolVersion 
vislib::net::IPCommEndPointAddress::convertAddressFamily(
        const IPAgnosticAddress::AddressFamily addressFamily) {
    VLSTACKTRACE("IPCommEndPointAddress::convertAddressFamily", __FILE__, 
        __LINE__);
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
 * vislib::net::IPCommEndPointAddress::IPCommEndPointAddress
 */
vislib::net::IPCommEndPointAddress::IPCommEndPointAddress(
        const IPEndPoint& endPoint) : Super(), endPoint(endPoint) {
    VLSTACKTRACE("IPCommEndPointAddress::IPCommEndPointAddress", __FILE__, 
        __LINE__);
}


/*
 * vislib::net::IPCommEndPointAddress::~IPCommEndPointAddress
 */
vislib::net::IPCommEndPointAddress::~IPCommEndPointAddress(void) {
    VLSTACKTRACE("IPCommEndPointAddress::IPCommEndPointAddress", __FILE__, 
        __LINE__);
}
