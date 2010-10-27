/*
 * IPAgnosticAddress.h
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_IPAGNOSTICADDRESS_H_INCLUDED
#define VISLIB_IPAGNOSTICADDRESS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/IPAddress.h"
#include "vislib/IPAddress6.h"
#include "vislib/SocketAddress.h"
#include "vislib/StackTrace.h"


namespace vislib {
namespace net {


    /**
     * This class is used to store IP-agnostic addresses, which can be both, 
     * IPv4 or IPv6. 
     *
     * Please note, that the use of this class is less efficient than using
     * vislib::net::IPAddress or vislib::net::IPAddress6.
     */
    class IPAgnosticAddress {

    public:

        /** 
         * The available address families. Only FAMILY_INET and FAMILY_INET6 are
         * valid for IPAgnosticAddress.
         */
        enum AddressFamily {
            FAMILY_UNSPECIFIED = SocketAddress::FAMILY_UNSPEC,
            FAMILY_INET = SocketAddress::FAMILY_INET,
            FAMILY_INET6 = SocketAddress::FAMILY_INET6
        };

        /**
         * This method tries to lookup the specified host name or human readable
         * IP address via DNS and creates an IPAgnosticAddress for it in case of
         * success.
         *
         * The method tries to retrieve the address family from the string 
         * provided, i. e. of an IPv4 address string, a FAMILY_INET address 
         * is returned and for an IPv6 string a FAMILY_INET6. In case of doubt, 
         * e. g. if a host name is specified and both address families are valid
         * for that host, the family specified for 'inCaseOfDoubt' is used.
         *
         * @param hostNameOrAddress The host name or stringised IP address to 
         *                          search.
         * @param inCaseOfDoubt     The address family to be used if it is not
         *                          clear from the 'hostNameOrAddress' 
         *                          parameter. This parameter defaults to
         *                          FAMILY_INET6
         *
         * @return One of the IP addresses that are assigned to the host.
         *
         * @throws SocketException In case the operation fails, e.g. the host 
         *                         could not be found.
         */
        static IPAgnosticAddress Create(const char *hostNameOrAddress,
            const AddressFamily inCaseOfDoubt = FAMILY_INET6);

        /**
         * This method tries to lookup the specified host name or human readable
         * IP address via DNS and creates an IPAgnosticAddress for it in case of
         * success.
         *
         * The method tries to retrieve the address family from the string 
         * provided, i. e. of an IPv4 address string, a FAMILY_INET address 
         * is returned and for an IPv6 string a FAMILY_INET6. In case of doubt, 
         * e. g. if a host name is specified and both address families are valid
         * for that host, the family specified for 'inCaseOfDoubt' is used.
         *
         * @param hostNameOrAddress The host name or stringised IP address to 
         *                          search.
         * @param inCaseOfDoubt     The address family to be used if it is not
         *                          clear from the 'hostNameOrAddress' 
         *                          parameter. This parameter defaults to
         *                          FAMILY_INET6
         *
         * @return One of the IP addresses that are assigned to the host.
         *
         * @throws SocketException In case the operation fails, e.g. the host 
         *                         could not be found.
         */
        static IPAgnosticAddress Create(const wchar_t *hostNameOrAddress,
            const AddressFamily inCaseOfDoubt = FAMILY_INET6);

        /**
         * Convenience method for creating an "any address" for the given
         * address family.
         *
         * @param addressFamily The address family to get the "any address" for.
         *                      This must be FAMILY_INET or FAMILY_INET6.
         *
         * @return ANY4 or ANY6 depending on 'addressFamily'.
         *
         * @throws IllegalParamException If 'addressFamily' is not supported.
         */
        static IPAgnosticAddress CreateAny(const AddressFamily addressFamily);

        /** 
         * The all-hosts group multicast address addressing all nodes in the 
         * subnet.
         */
        static const IPAddress& ALL_NODES_ON_LINK4;

        /** 
         * The all-hosts group multicast address addressing all nodes in the 
         * subnet.
         */
        static const IPAddress6& ALL_NODES_ON_LINK6;

        /** 
         * The all-routers multicast group address addressing all routers in the
         * subnet.
         */
        static const IPAddress& ALL_ROUTERS_ON_LINK4;

        /** 
         * The all-routers multicast group address addressing all routers in the
         * subnet.
         */
        static const IPAddress6& ALL_ROUTERS_ON_LINK6;

        /**
         * Provides an IP address that indicates that a server must listen for 
         * client activity on all network interfaces.
         */
        static const IPAddress& ANY4;

        /**
         * Provides an IP address that indicates that a server must listen for 
         * client activity on all network interfaces.
         */
        static const IPAddress6& ANY6;

        /** Provides the IP loopback address. */
        static const IPAddress& LOOPBACK4;

        /** Provides the IP loopback address. */
        static const IPAddress6& LOOPBACK6;

        /**
         * Provides an IP address that indicates that no network interface 
         * should be used.
         */
        static const IPAddress& NONE4;

        /**
         * Provides an IP address that indicates that no network interface 
         * should be used.
         */
        static const IPAddress6& NONE6;

        /** 
         * Creates an unspecified, invalid address.
         */
        IPAgnosticAddress(void);

        /** 
         * Create an IPAgnosticAddress from an IPAddress.
         *
         * @param address The IPAddress to be represented by the new object.
         */
        explicit IPAgnosticAddress(const IPAddress& address);

        /** 
         * Create an IPAgnosticAddress from an IPAddress6.
         *
         * @param address The IPAddress6 to be represented by the new object.
         */
        explicit IPAgnosticAddress(const IPAddress6& address);

        /** 
         * Create an IPAgnosticAddress from an in_addr structure.
         *
         * @param address The in_addr to be represented by the new object.
         */
        explicit IPAgnosticAddress(const struct in_addr& address);

        /** 
         * Create an IPAgnosticAddress from an in6_addr structure.
         *
         * @param address The in6_addr to be represented by the new object.
         */
        explicit IPAgnosticAddress(const struct in6_addr& address);

        /**
         * Create an IPv4 address from individual bytes. The bytes must be 
         * specified in network byte order.
         *
         * @param b1  Byte number 1 of the IP address.
         * @param b2  Byte number 2 of the IP address.
         * @param b3  Byte number 3 of the IP address.
         * @param b4  Byte number 4 of the IP address.
         */
        IPAgnosticAddress(
            const BYTE b1, const BYTE b2, const BYTE b3, const BYTE b4);

        /**
         * Create an IPv6 address from individual bytes. The bytes must be 
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
        IPAgnosticAddress(
            const BYTE b1, const BYTE b2, const BYTE b3, const BYTE b4,
            const BYTE b5, const BYTE b6, const BYTE b7, const BYTE b8,
            const BYTE b9, const BYTE b10, const BYTE b11, const BYTE b12,
            const BYTE b13, const BYTE b14, const BYTE b15, const BYTE b16);

        /**
         * Copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        IPAgnosticAddress(const IPAgnosticAddress& rhs);

        /** Dtor. */
        virtual ~IPAgnosticAddress(void);

        /**
         * Gets the address familiy of the IP address.
         */
        AddressFamily GetAddressFamily(void) const;

        /**
         * Get the prefix of length 'prefixLength' bits of the address. The rest
         * of the returned address will be filled with zeroes.
         *
         * @param prefixLength The length of the prefix. If it is out of range, 
         *                     the method will succeed and return the complete
         *                     address.
         */
        IPAgnosticAddress GetPrefix(const ULONG prefixLength) const;

        /**
         * Answer whether the address is the ANY4 or ANY6 address.
         *
         * @return true if the address represents "any" address, false otherwise.
         */
        inline bool IsAny(void) const {
            VLSTACKTRACE("IPAgnosticAddress::IsAny", __FILE__, __LINE__);
            return ((*this == ANY4) || (*this == ANY6));
        }

        /**
         * Answer whether the address family version is 4.
         *
         * @return true if the address is v4, false otherwise.
         */
        inline bool IsV4(void) const {
            VLSTACKTRACE("IPAgnosticAddress::IsV4", __FILE__, __LINE__);
            return (this->v4 != NULL);
        }

        /**
         * Answer whether the address family version is 6.
         *
         * @return true if the address is v6, false otherwise.
         */
        inline bool IsV6(void) const {
            VLSTACKTRACE("IPAgnosticAddress::IsV6", __FILE__, __LINE__);
            return (this->v6 != NULL);
        }

        /** 
         * Answer a string representation of the IP address. 
         *
         * @return A string representation.
         */
        StringA ToStringA(void) const;

        /** 
         * Answer a string representation of the IP address.
         *
         * @return A string representation.
         */
        StringW ToStringW(void) const;

        /**
         * Provides access to the single bytes of the IP address.
         *
         * @param i The index of the byte to access, which must be within 
         *          [0, 4] for an IPv4 address and [0, 15] for an IPv6 address.
         *
         * @return The 'i'th byte of the address.
         *
         * @throws OutOfRangeException If 'i' is not a legal byte number.
         * @throws IllegalStateException If the address is invalid.
         */
        BYTE operator [](const int i) const;

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        IPAgnosticAddress& operator =(const IPAgnosticAddress& rhs);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        IPAgnosticAddress& operator =(const IPAddress& rhs);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        IPAgnosticAddress& operator =(const IPAddress6& rhs);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        IPAgnosticAddress& operator =(const struct in_addr& rhs);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        IPAgnosticAddress& operator =(const struct in6_addr& rhs);

        /** 
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this object and 'rhs' are equal, false otherwise.
         */
        bool operator ==(const IPAgnosticAddress& rhs) const;

        /** 
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this object and 'rhs' are equal, false otherwise.
         */
        bool operator ==(const IPAddress& rhs) const;

        /** 
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this object and 'rhs' are equal, false otherwise.
         */
        bool operator ==(const IPAddress6& rhs) const;

        /** 
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this object and 'rhs' are equal, false otherwise.
         */
        bool operator ==(const struct in_addr& rhs) const;

        /** 
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this object and 'rhs' are equal, false otherwise.
         */
        bool operator ==(const struct in6_addr& rhs) const;

        /** 
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this object and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const IPAgnosticAddress& rhs) const {
            VLSTACKTRACE("IPAgnosticAddress::operator !=", __FILE__, __LINE__);
            return !(*this == rhs);
        }

        /** 
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this object and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const IPAddress& rhs) const {
            VLSTACKTRACE("IPAgnosticAddress::operator !=", __FILE__, __LINE__);
            return !(*this == rhs);
        }

        /** 
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this object and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const IPAddress6& rhs) const {
            VLSTACKTRACE("IPAgnosticAddress::operator !=", __FILE__, __LINE__);
            return !(*this == rhs);
        }

        /** 
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this object and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const struct in_addr& rhs) const {
            VLSTACKTRACE("IPAgnosticAddress::operator !=", __FILE__, __LINE__);
            return !(*this == rhs);
        }

        /** 
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this object and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const struct in6_addr& rhs) const {
            VLSTACKTRACE("IPAgnosticAddress::operator !=", __FILE__, __LINE__);
            return !(*this == rhs);
        }

        /**
         * Cast to an IPv4 address.
         *
         * This operation is only valid if the IPAgnosticAddress represents an
         * IPv4 address or an IPv6 address that can be casted to a valid IPv4
         * address.
         *
         * @return The IPv4 address that is equivalent to this address.
         *
         * @throws IllegalStateException If the address is neither IPv4 nor can
         *                               be casted to an IPv4 address.
         */
        operator IPAddress(void) const;

        /**
         * Cast to a pointer to an IPv4 address.
         *
         * This operation is only valid if the IPAgnosticAddress represents an
         * IPv4 address.
         *
         * @return The IPv4 address that is equivalent to this address.
         *
         * @throws IllegalStateException If the address is not IPv4.
         */
        operator const IPAddress *(void) const;

        /**
         * Cast to a pointer to an IPv4 address. This cast exposes the internal
         * address structure, i.e. results in aliasing. The callee remains owner
         * of the memory, which is valid as long as the object is unchanged.
         *
         * This operation is only valid if the IPAgnosticAddress represents an
         * IPv4 address.
         *
         * @return The IPv4 address that is equivalent to this address.
         *
         * @throws IllegalStateException If the address is not IPv4.
         */
        operator IPAddress *(void);

        /**
         * Cast to an IPv6 address. This cast exposes the internal
         * address structure, i.e. results in aliasing. The callee remains owner
         * of the memory, which is valid as long as the object is unchanged.
         *
         * This operation is only valid if the IPAgnosticAddress represents an
         * IPv6 address or an IPv4 address.
         *
         * @return The IPv6 address that is equivalent to this address.
         *
         * @throws IllegalStateException If the address is neither IPv6 nor can
         *                               be casted to an IPv6 address.
         */
        operator IPAddress6(void) const;

        /**
         * Cast to a pointer to an IPv6 address. This cast exposes the internal
         * address structure, i.e. results in aliasing. The callee remains owner
         * of the memory, which is valid as long as the object is unchanged.
         *
         * This operation is only valid if the IPAgnosticAddress represents an
         * IPv6 address.
         *
         * @return The IPv6 address that is equivalent to this address.
         *
         * @throws IllegalStateException If the address is not IPv6.
         */
        operator const IPAddress6 *(void) const;

        /**
         * Cast to a pointer to an IPv6 address. This cast exposes the internal
         * address structure, i.e. results in aliasing. The callee remains owner
         * of the memory, which is valid as long as the object is unchanged.
         *
         * This operation is only valid if the IPAgnosticAddress represents an
         * IPv6 address.
         *
         * @return The IPv6 address that is equivalent to this address.
         *
         * @throws IllegalStateException If the address is not IPv6.
         */
        operator IPAddress6 *(void);

    private:

        /** If the address is v4, it is stored here. */
        IPAddress *v4;

        /** If the address is v6, it is stored here. */
        IPAddress6 *v6;
    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_IPAGNOSTICADDRESS_H_INCLUDED */
