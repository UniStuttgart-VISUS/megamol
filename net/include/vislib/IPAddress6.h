/*
 * IPAddress6.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_IPADDRESS6_H_INCLUDED
#define VISLIB_IPADDRESS6_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else /* _WIN32 */
#include <netinet/in.h>
#endif /* _WIN32 */

#include "vislib/IPAddress.h"


namespace vislib {
namespace net {


    /**
     * This class represents an IPv6 address.
     *
     * In contrast to the IPv4 class IPAddress, IPAddress6 does not provide
     * Lookup() capabilities. Use the static methods of vislib::net::DNS
     * instead. There is, however, the same static Create convenience method
     * that uses the DNS class for performing the lookup.
     */
    class IPAddress6 {

    public:

        /**
         * This method tries to lookup the specified host name or human readable
         * IP address via DNS and creates an IPAddress6 for it in case of
         * success.
         *
         * @param hostNameOrAddress The host name or stringised IP address to 
         *                          search.
         *
         * @return One of the IP addresses that are assigned to the host.
         *
         * @throws SocketException In case the operation fails, e.g. the host 
         *                         could not be found.
         */
        static IPAddress6 Create(const char *hostNameOrAddress);

        /**
         * This method tries to lookup the specified host name or human readable
         * IP address via DNS and creates an IPAddress6 for it in case of
         * success.
         *
         * @param hostNameOrAddress The host name or stringised IP address to 
         *                          search.
         *
         * @return One of the IP addresses that are assigned to the host.
         *
         * @throws SocketException In case the operation fails, e.g. the host 
         *                         could not be found.
         */
        static IPAddress6 Create(const wchar_t *hostNameOrAddress);

        /** 
         * The all-hosts group multicast address addressing all nodes in the 
         * link local (subnet) scope.
         */
        static const IPAddress6 ALL_NODES_ON_LINK;

        /** 
         * The all-routers group multicast address addressing all nodes in the
         * link local (subnet) scope.
         */
        static const IPAddress6 ALL_ROUTERS_ON_LINK;

        /** 
         * The all-hosts group multicast address addressing all nodes in the 
         * node local (interface) scope.
         */
        static const IPAddress6 ALL_NODES_ON_NODE;

        /** 
         * Constant special IP address that allows receiving from all available
         * adapters and sending from the default (lowest-numbered adapter)
         * interface.
         */
        static const IPAddress6 ANY;

        /** 
         * Constant loopback address (alias for LOOPBACK because of old 
         * IPAddress class usage compatibility).
         */
        static const IPAddress6& LOCALHOST;

        /** Constant loopback address. */
        static const IPAddress6 LOOPBACK;

        ///** 
        // * The all-hosts group multicast address addressing all nodes in the 
        // * site local scope (packets may be routed, but not by border routers).
        // */
        //static const IPAddress6 MULTICAST_ALL_SITE_LOCAL_HOSTS;

        ///** 
        // * The all-routers group multicast address addressing all nodes in the 
        // * site local scope (packets may be routed, but not by border routers).
        // */
        //static const IPAddress6 MULTICAST_ALL_SITE_LOCAL_ROUTERS;

        /** Alias for ANY. */
        static const IPAddress6& UNSPECIFIED;

        /**
         * Create a new loopback IPAddress6.
         */
        IPAddress6(void);

        /**
         * Create an IPAddress6 that represents the specified struct in6_addr.
         *
         * @param address The address value.
         */
        IPAddress6(const struct in6_addr& address);

        /**
         * Create an IPAddress6 from individual bytes. The bytes must be 
         * specified in network byte order.
         *
         * @param b1  Byte number 1 of the IP address.
         * @param b2  Byte number 2 of the IP address.
         * @param b3  Byte number 3 of the IP address.
         * @param b4  Byte number 4 of the IP address.
         * @param b5  Byte number 5 of the IP address.
         * @param b6  Byte number 6 of the IP address.
         * @param b7  Byte number 7 of the IP address.
         * @param b8  Byte number 8 of the IP address.
         * @param b9  Byte number 9 of the IP address.
         * @param b10 Byte number 10 of the IP address.
         * @param b11 Byte number 11 of the IP address.
         * @param b12 Byte number 12 of the IP address.
         * @param b13 Byte number 13 of the IP address.
         * @param b14 Byte number 14 of the IP address.
         * @param b15 Byte number 15 of the IP address.
         * @param b16 Byte number 16 of the IP address.
         */
        IPAddress6(const BYTE b1, const BYTE b2, const BYTE b3, const BYTE b4,
            const BYTE b5, const BYTE b6, const BYTE b7, const BYTE b8,
            const BYTE b9, const BYTE b10, const BYTE b11, const BYTE b12,
            const BYTE b13, const BYTE b14, const BYTE b15, const BYTE b16);

        /**
         * Create an IPv4 mapped address.
         *
         * @param address The IPv4 address to be mapped.
         */
        explicit IPAddress6(const IPAddress& address);

        /**
         * Create a mapped IPv4 address. This ctor is equal to creating a new
         * IPAddress and calling MapV4Address(address) on it.
         *
         * @param address The IPv4 address to be mapped.
         */
        explicit IPAddress6(const struct in_addr& address);

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        IPAddress6(const IPAddress6& rhs);

        /** Dtor. */
        ~IPAddress6(void);

        /**
         * Get the prefix of length 'prefixLength' bits of the address. The rest
         * of the returned address will be filled with zeroes.
         *
         * @param prefixLength The length of the prefix. If it is out of range, 
         *                     the method will succeed and return the complete
         *                     address.
         */
        IPAddress6 GetPrefix(const ULONG prefixLength) const;

        /**
         * Determines whether the address is a link local address.
         *
         * @return true if the address is a link local address, false otherwise.
         */
        inline bool IsLinkLocal(void) const {
            return (IN6_IS_ADDR_LINKLOCAL(&this->address) != 0);
        }

        /**
         * Answer whether the address is the loopback address.
         *
         * @return true if the address is the loopback address, false otherwise.
         */
        inline bool IsLoopback(void) const {
            return (IN6_IS_ADDR_LOOPBACK(&this->address) != 0);
        }

        /**
         * Answer whether the IP address is an IPv6 multicast global address.
         *
         * @return true if the IP address is multicast global address,
         *         false otherwise.
         */
        inline bool IsMulticast(void) const {
            return (IN6_IS_ADDR_MULTICAST(&this->address) != 0);
        }

        /**
         * Answer whether the IP address is an IPv6 site local address.
         *
         * @return true if the address is a site local address, false otherwise.
         */
        inline bool IsSiteLocal(void) const {
            return (IN6_IS_ADDR_SITELOCAL(&this->address) != 0);
        }

        /**
         * Answer whether the address is unspecified, i.e. ANY.
         *
         * @return true if the address is unspecified, false otherwise.
         */
        inline bool IsUnspecified(void) const {
            return (IN6_IS_ADDR_UNSPECIFIED(&this->address) != 0);
        }

        /**
         * Answer whether the IP address is an IPv4-compatible IPv6 address.
         *
         * An IPv6 address is IPv4 compatible if it is assigned to an IPv6/IPv4 
         * node, which bears the high-order 96-bit prefix 0:0:0:0:0:0, and an
         * IPv4 address in the low-order 32-bits. IPv4-compatible addresses are 
         * used by the automatic tunneling mechanism.
         *
         * @return true if the IPv6 address is an IPv4-compatible address,
         *         false otherwise.
         */
        inline bool IsV4Compatible(void) const {
            return (IN6_IS_ADDR_V4COMPAT(&this->address) != 0);
        }

        /**
         * Answer thether the IP address is an IPv4-mapped IPv6 address.
         *
         * @return true if the IPv6 address is an IPv4-mapped address,
         *         false otherwise.
         */
        inline bool IsV4Mapped(void) const {
            return (IN6_IS_ADDR_V4MAPPED(&this->address) != 0);
        }

        /**
         * Set this IP address to be the mapped IPv4 address 'address'.
         *
         * @param address The IPv4 address to map.
         */
        void MapV4Address(const struct in_addr& address);

        /**
         * Set this IP address to be the mapped IPv4 address 'address'.
         *
         * @param address The IPv4 address to map.
         */
        inline void MapV4Address(const IPAddress& address) {
            this->MapV4Address(*static_cast<const struct in_addr *>(address));
        }

        /**
         * Convert the IP address into a human readable format.
         *
         * @return The string representation of the IP address.
         */
        StringA ToStringA(void) const;

        /**
         * Convert the IP address into a human readable format.
         *
         * @return The string representation of the IP address.
         */
        inline StringW ToStringW(void) const {
            return StringW(this->ToStringA());
        }

        /**
         * If this IPv6 address is a mapped IPv4 address, i. e. if 
         * this->IsV4Mapped() returns true, this method returns the mapped
         * IPv4 address.
         *
         * @return The mapped IPv4 address.
         *
         * @throws IllegalStateException If this address is not a mapped IPv4
         *                               address.
         */
        IPAddress UnmapV4Address(void) const;

        /**
         * Provides access to the single bytes of the IP address.
         *
         * @param i The index of the byte to access, which must be within 
         *          [0, 15].
         *
         * @return The 'i'th byte of the address.
         *
         * @throws OutOfRangeException If 'i' is not a legal byte number.
         */
        BYTE operator [](const int i) const;

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline IPAddress6& operator =(const IPAddress6& rhs) {
            return (*this = rhs.address);
        }

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        IPAddress6& operator =(const struct in6_addr& rhs);

        /**
         * Assignment operator for IPv4 addresses. 'rhs' will be mapped to an 
         * IPv6 address. This operator is equal to calling MapV4Address(rhs).
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        IPAddress6& operator =(const IPAddress& rhs);

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this IP address are equal, false otherwise.
         */
        inline bool operator ==(const IPAddress6& rhs) const {
            return (*this == rhs.address);
        }

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this IP address are not equal, 
         *        false otherwise.
         */
        inline bool operator !=(const IPAddress6& rhs) const {
            return !(*this == rhs);
        }

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this IP address are equal, false otherwise.
         */
        bool operator ==(const struct in6_addr& rhs) const;

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this IP address are not equal, 
         *        false otherwise.
         */
        inline bool operator !=(const struct in6_addr& rhs) const {
            return !(*this == rhs);
        }

        /**
         * Cast to struct in6_addr. Note, that a deep copy is returned.
         *
         * @return The in6_addr that is represented by this object.
         */
        inline operator struct in6_addr(void) const {
            return this->address;
        }

        /**
         * Cast to struct in6_addr. The operation returns a reference to the 
         * internal in6_addr structure.
         *
         * @return The in6_addr that is represented by this object.
         */
        inline operator struct in6_addr&(void) {
            return this->address;
        }

        /**
         * Cast to pointer to struct in6_addr. The operation exposes the 
         * internal in6_addr structure.
         *
         * @return Pointer to the internal in6_addr that is represented by this 
         *         object.
         */
        inline operator const struct in6_addr *(void) const {
            return &this->address;
        }

        /**
         * Cast to pointer to struct in6_addr. The operation exposes the 
         * internal in6_addr structure.
         *
         * @return Pointer to the internal in6_addr that is represented by this 
         *         object.
         */
        inline operator struct in6_addr *(void) {
            return &this->address;
        }

        /**
         * Cast to an IPv4 address.
         *
         * This operation is only valid if the IPv6 address is either a mapped
         * IPv4 address or if the IPv6 address is IPv4 compatible.
         *
         * @return The IPv4 address that is equivalent to this IPv6 address if 
         *         such a conversion is possible.
         *
         * @throws IllegalStateException If the IPv6 address is neither an IPv4
         *                               mapped address nor IPv4 compatible.
         */
        operator IPAddress(void) const;

    private:

        /** The actual IPv6 address. */
        struct in6_addr address;

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_IPADDRESS6_H_INCLUDED */
