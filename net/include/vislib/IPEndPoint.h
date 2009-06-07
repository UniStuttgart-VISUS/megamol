/*
 * IPEndPoint.h
 *
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_IPENDPOINT_H_INCLUDED
#define VISLIB_IPENDPOINT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/IPAgnosticAddress.h"


namespace vislib {
namespace net {

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

    /**
     * The IPEndPoint class represents a socket address that is IP-agnostic,
     * i.e. can be used with both, IPv4 and IPv6. It supersedes the
     * SocketAddress class of VISlib, which supports only IPv4. In contrast to
     * SocketAddress, IPEndPoint only supports the address families AF_INET and
     * AF_INET6.
     */
    class IPEndPoint {

    public:

        /** 
         * The available address families. Only FAMILY_INET and FAMILY_INET6 are
         * valid for IPEndPoints.
         */
        enum AddressFamily {
            FAMILY_UNSPECIFIED = IPAgnosticAddress::FAMILY_UNSPECIFIED,
            FAMILY_INET = IPAgnosticAddress::FAMILY_INET,
            FAMILY_INET6 = IPAgnosticAddress::FAMILY_INET6
        };

        /**
         * Create an IPEndPoint by looking up the specified host or IP 
         * address and using FAMILY_INET.
         *
         * @param hostNameOrAddress An IP address in the dotted string format 
         *                          or a host name.
         * @param port              The port of the address.
         * 
         * @return The IPEndPoint if all parameters are valid.
         *
         * @throws IllegalParamException If 'host' is not a valid host name or
         *                               IP address.
         */
        static IPEndPoint CreateIPv4(const char *hostNameOrAddress,
            const unsigned short port);

        /**
         * Create an IPEndPoint by looking up the specified host or IP 
         * address and using FAMILY_INET6.
         *
         * @param hostNameOrAddress An IP address in the dotted string format 
         *                          or a host name.
         * @param port              The port of the address.
         * 
         * @return The IPEndPoint if all parameters are valid.
         *
         * @throws IllegalParamException If 'host' is not a valid host name or
         *                               IP address.
         */
        static IPEndPoint CreateIPv6(const char *hostNameOrAddress,
            const unsigned short port);

        /**
         * Creates a new IPv4 end point using the specified address and port.
         *
         * @param ipAddress The IP address of the end point.
         * @param port      The port number of the end point.
         */
        IPEndPoint(const IPAddress& ipAddress = IPAddress::ANY,
            const unsigned short port = 0);

        /**
         * Creates a new IPv6 end point using the specified address and port.
         *
         * @param ipAddress The IP address of the end point.
         * @param port      The port number of the end point.
         */
        IPEndPoint(const IPAddress6& ipAddress, const unsigned short port);

        /**
         * Creates a new end point using the specified address and port.
         *
         * @param ipAddress The IP address of the end point.
         * @param port      The port number of the end point. 
         */
        IPEndPoint(const IPAgnosticAddress& ipAddress, 
            const unsigned short port);

        /**
         * Create a new unspecified end point (ANY) for the given address 
         * family, which must be one of FAMILY_INET or FAMILY_INET6, and the
         * given port.
         *
         * @param addressFamily The address family to create the end point for.
         * @param port          The port number of the end point.
         *
         * @throw IllegalParamException If 'addressFamily' is not one of the
         *                              supported families.
         */
        IPEndPoint(const AddressFamily addressFamily,
            const unsigned short port);

        /**
         * Create an IPEndPoint from a legacy SocketAddress.
         *
         * @param address The socket address to convert.
         */
        IPEndPoint(const SocketAddress& address);

        /**
         * Create a copy of 'address' but change the port to 'newPort'.
         *
         * @param address The address to be cloned.
         * @param newPort The new port.
         */
        IPEndPoint(const IPEndPoint& address, const unsigned short newPort); 

        /**
         * Create an IPEndPoint from an OS IP-agnostic address structure.
         *
         * @param address The address storage.
         */
        explicit IPEndPoint(const struct sockaddr_storage& address);

        /**
         * Create an IPEndPoint from an OS IPv4 structure.
         *
         * @param address The IPv4 address to set.
         */
        explicit IPEndPoint(const struct sockaddr_in& address);

        /**
         * Create an IPEndPoint from an OS IPv4 structure.
         *
         * @param address The IPv6 address to set.
         */
        explicit IPEndPoint(const struct sockaddr_in6& address);

        /**
         * Clone 'rhs'
         * 
         * @param rhs The object to be cloned.
         */
        IPEndPoint(const IPEndPoint& rhs);

        /** Dtor. */
        ~IPEndPoint(void);

        /**
         * Answer the address family of the IP end point.
         *
         * @retunr The address family of the IP end point.
         */
        inline AddressFamily GetAddressFamily(void) const {
            return static_cast<AddressFamily>(this->address.ss_family);
        }

        /**
         * Answer the IP address of the IP end point.
         *
         * @return The IP address of the end point.
         *
         * @throws IllegalStateException If the address familiy is illegal.
         */
        IPAgnosticAddress GetIPAddress(void) const;

        /**
         * Answer the IPv4 address of the IP end point. This might fail if the
         * end point is an IPv6 end point and the address cannot be converted.
         *
         * @return The IP address of the end point.
         *
         * @throws IllegalStateException If the address familiy is illegal.
         */
        IPAddress GetIPAddress4(void) const;

        /**
         * Answer the IPv6 address of the IP end point. If the end point is an
         * IPv4 end point, the address will be mapped.
         *
         * @return The IP address of the end point.
         *
         * @throws IllegalStateException If the address familiy is illegal.
         */
        IPAddress6 GetIPAddress6(void) const;

        /**
         * Answer the port of the IP end point.
         *
         * @return The port of the IP end point.
         *
         * @throws IllegalStateException If the address familiy is illegal.
         */
        unsigned short GetPort(void) const;

        /**
         * Set a new IPv4 address. This will also change the address family.
         *
         * @param ipAddress The new IP address.
         */
        void SetIPAddress(const IPAddress& ipAddress);

        /**
         * Set a new IPv6 address. This will also change the address family.
         *
         * @param ipAddress The new IP address.
         */
        void SetIPAddress(const IPAddress6& ipAddress);

        /**
         * Set a new IP address. This will also change the address family.
         *
         * @param ipAddress The new IP address.
         */
        void SetIPAddress(const IPAgnosticAddress& ipAddress);

        /**
         * Set a new port number.
         *
         * @param port The new port number.
         */
        void SetPort(const unsigned int port);

        /**
         * Set a new port number.
         *
         * @param port The new port number.
         */
        inline void SetPortA(const unsigned int port) {
            this->SetPort(port);
        }

        /**
         * Set a new port number.
         *
         * @param port The new port number.
         */
        inline void SetPortW(const unsigned int port) {
            this->SetPort(port);
        }

        /**
         * Convert the socket address into a human readable format.
         *
         * @return The string representation of the IP address.
         */
        StringA ToStringA(void) const;

        /**
         * Convert the socket address into a human readable format.
         *
         * @return The string representation of the IP address.
         */
        StringW ToStringW(void) const {
            return StringW(this->ToStringA());
        }

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        IPEndPoint& operator =(const IPEndPoint& rhs);

        /**
         * Create an IPEndPoint for the specified IPv4 socket address.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        IPEndPoint& operator =(const SocketAddress& rhs);

        /**
         * Create an IPEndPoint that represents the specified IPv4 address.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        IPEndPoint& operator =(const struct sockaddr_in& rhs);

        /**
         * Create an IPEndPoint that represents the specified IPv6 address.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        IPEndPoint& operator =(const struct sockaddr_in6& rhs);

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this object and rhs are equal, false otherwise.
         */
        bool operator ==(const IPEndPoint& rhs) const;

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this object and rhs are not equal, false otherwise.
         */
        inline bool operator !=(const IPEndPoint& rhs) const {
            return !(*this == rhs);
        }

        /**
         * Access the internal storage member as generic sockaddr pointer.
         *
         * @return Pointer to the sockaddr_storage used for the end point.
         */
        inline operator const struct sockaddr *const(void) const {
            return reinterpret_cast<const sockaddr *>(&this->address);
        }

        /**
         * Access the internal storage member as generic sockaddr pointer.
         *
         * @return Pointer to the sockaddr_storage used for the end point.
         */
        inline operator struct sockaddr *(void) {
            return reinterpret_cast<sockaddr *>(&this->address);
        }

        /**
         * Access the internal storage member.
         *
         * @return Reference to the sockaddr_storage used for the end point.
         */
        inline operator const struct sockaddr_storage&(void) const {
            return this->address;
        }

        /**
         * Access the internal storage member.
         *
         * @return Reference to the sockaddr_storage used for the end point.
         */
        inline operator struct sockaddr_storage&(void) {
            return this->address;
        }

        /**
         * Access the internal storage member.
         *
         * @return Pointer to the sockaddr_storage used for the end point.
         */
        inline operator const struct sockaddr_storage *const(void) const {
            return &this->address;
        }

        /**
         * Access the internal storage member.
         *
         * @return Pointer to the sockaddr_storage used for the end point.
         */
        inline operator struct sockaddr_storage *(void) {
            return &this->address;
        }

        /**
         * Convert the IPEndPoint to a legacy SocketAddress.
         *
         * Note that this operation can fail if the end point is not an IPv4
         * end point.
         *
         * @return The legacy SocketAdddress that is equivalent to this 
         *         IPEndPoint.
         *
         * @throws IllegalStateException If the end point cannot be converted
         *                               into an IPv4 SocketAddress.
         */
        operator SocketAddress(void) const;

    private:

        /**
         * Access the address storage as IPv4 socket address.
         *
         * @return Reference to the address storage reinterpreted for IPv4.
         */
        inline struct sockaddr_in& asV4(void) {
            return reinterpret_cast<struct sockaddr_in&>(this->address);
        }

        /**
         * Access the address storage as IPv4 socket address.
         *
         * @return Reference to the address storage reinterpreted for IPv4.
         */
        inline const struct sockaddr_in& asV4(void) const {
            return reinterpret_cast<const struct sockaddr_in&>(this->address);
        }

        /**
         * Access the address storage as IPv6 socket address.
         *
         * @return Reference to the address storage reinterpreted for IPv6.
         */
        inline struct sockaddr_in6& asV6(void) {
            return reinterpret_cast<struct sockaddr_in6&>(this->address);
        }

        /**
         * Access the address storage as IPv6 socket address.
         *
         * @return Reference to the address storage reinterpreted for IPv6.
         */
        inline const struct sockaddr_in6& asV6(void) const {
            return reinterpret_cast<const struct sockaddr_in6&>(this->address);
        }

        /** The storage for the generic address. */
        struct sockaddr_storage address;

    };

#ifdef _WIN32
    // Crowbar to work around windows naming conflict
#ifdef _MSC_VER
#pragma pop_macro("SetPort")
#endif /* _MSC_VER */
#endif /* _WIN32 */

} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_IPENDPOINT_H_INCLUDED */
