/*
 * IPAddress.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). All rights reserved.
 */

#ifndef VISLIB_IPADDRESS_H_INCLUDED
#define VISLIB_IPADDRESS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifdef _WIN32
#include <winsock2.h>
#else /* _WIN32 */
#include <arpa/inet.h>
#endif /* _WIN32 */

#ifdef _MSC_VER
#pragma comment(lib, "Ws2_32")
#endif /* _MSC_VER */

#include "vislib/String.h"


namespace vislib {
namespace net {

    /**
     * Represents an IPv4 address.
     *
     * @author Christoph Mueller
     */
    class IPAddress {

    public:

        /** 
         * The all-hosts group multicast address addressing all nodes in the 
         * subnet.
         */
        static const IPAddress ALL_NODES_ON_LINK;

        /** 
         * The all-routers multicast group address addressing all routers in the
         * subnet.
         */
        static const IPAddress ALL_ROUTERS_ON_LINK;

        /** 
         * Constant special IP address that allows receiving from all available
         * adapters and sending from the default (lowest-numbered adapter)
         * interface.
         */
        static const IPAddress ANY;

        /** Constant broadcast address (255.255.255.255). */
        static const IPAddress BROADCAST;

        /** Constant loopback address (127.0.0.1). */
        static const IPAddress LOCALHOST;

        /** Constant invalid IP address. */
        static const IPAddress NONE;

        /**
         * Create an IPAddress form the dottet string format or host name.
         *
         * @param address The IP address in the dotted string format or a valid
         *                host name.
         *
         * @return The IP address matching 'address'.
         *
         * @throws IllegalParamException If 'address' is not a valid IP address 
         *                               or host name.
         */
        static IPAddress Create(const char *address = "127.0.0.1");

        /**
         * Create an IPAddress from the dotted string format.
         *
         * NOTE: The address might be invalid after construction, if 'address'
         * is not a valid host name or IP address in the dotted string format.
         * It is recommended to use the Lookup() method for finding a host or
         * creating the address by IPAddress::Create.
         *
         * @param address The IP address in the dotted string format.
         */
        explicit IPAddress(const char *address = "127.0.0.1");

        /**
         * Create an IPAddress from four characters.
         *
         * @param i1 The first number of the IP address.
         * @param i2 The first number of the IP address.
         * @param i3 The first number of the IP address.
         * @param i4 The first number of the IP address.
         */
        IPAddress(unsigned char i1, unsigned char i2, unsigned char i3, 
            unsigned char i4);

        /**
         * Cast constructor from struct in_addr.
         *
         * @param address The in_addr structure to be copied.
         */
        inline IPAddress(const struct in_addr& address) : address(address) { };

        /**
         * Create an IPAddress from its representation as a single integer.
         *
         * @param address         The address as integer. The byte order of this
         *                        integer must be according to the value of the
         *                        'isHostByteOrder' flag.
         * @param isHostByteOrder Determines whether 'address' is host byte 
         *                        order or network byte order (false, default).
         */
        explicit IPAddress(const unsigned long address, 
                const bool isHostByteOrder = false);

        /**
         * Copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        inline IPAddress(const IPAddress& rhs) : address(rhs.address) { };

        /** Dtor. */
        ~IPAddress(void);

        /**
         * Get the prefix of length 'prefixLength' bits of the address. The rest
         * of the returned address will be filled with zeroes.
         *
         * @param prefixLength The length of the prefix. If it is out of range, 
         *                     the method will succeed and return the complete
         *                     address.
         */
        IPAddress GetPrefix(const ULONG prefixLength) const;

        /**
         * Lookup and set the IP address of the specified host. If the host is
         * not found, nothing is changed and the return value is false.
         *
         * Note, that 'hostname' can either be an IP address in the dotted 
         * string format or the host name. Both will be tested.
         *
         * @param hostname Either a hostname or an IP address in the dotted 
         *                 string format.
         *
         * @return true, if the hostname could be resolved or the IP address was
         *         valid, false otherwise.
         */
        bool Lookup(const char *hostname);

        /**
         * Convert the IP address into dotted string format.
         *
         * @return The string representation of the IP address.
         */
        StringA ToStringA(void) const;

        /**
         * Convert the IP address into dotted string format.
         *
         * @return The string representation of the IP address.
         */
        inline StringW ToStringW(void) const {
            return StringW(this->ToStringA());
        }

        /**
         * Provides access to the single bytes of the IP address.
         *
         * @param i The index of the byte to access, which must be within 
         *          [0, 4].
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
        IPAddress& operator =(const IPAddress& rhs);

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this IP address are equal, false otherwise.
         */
        bool operator ==(const IPAddress& rhs) const;

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this IP address are not equal, 
         *        false otherwise.
         */
        inline bool operator !=(const IPAddress& rhs) const {
            return !(*this == rhs);
        }

        /**
         * Cast to struct in_addr. Note, that a deep copy is returned.
         *
         * @return The in_addr that is represented by this object.
         */
        inline operator struct in_addr(void) const {
            return this->address;
        }

        /**
         * Cast to struct in_addr. The operation returns a reference to the 
         * internal in_addr structure.
         *
         * @return The in_addr that is represented by this object.
         */
        inline operator struct in_addr&(void) {
            return this->address;
        }

        /**
         * Cast to pointer to struct in_addr. The operation exposes the 
         * internal in6_addr structure.
         *
         * @return Pointer to the internal in6_addr that is represented by this 
         *         object.
         */
        inline operator const struct in_addr *(void) const {
            return &this->address;
        }

        /**
         * Cast to pointer to struct in_addr. The operation exposes the 
         * internal in6_addr structure.
         *
         * @return Pointer to the internal in_addr that is represented by this 
         *         object.
         */
        inline operator struct in_addr *(void) {
            return &this->address;
        }

        /**
         * Applies a subnet mask on this IP address.
         *
         * @param mask The subnet mask.
         *
         * @return *this.
         */
        IPAddress& operator &=(const IPAddress& mask);

        /**
         * Applies a subnet mask to this IP address and returns the result.
         *
         * @param mask The subnet mask.
         *
         * @return The IP address with the subnet mask applied.
         */
        inline IPAddress operator &(const IPAddress& mask) const {
            IPAddress retval(*this);
            retval &= mask;
            return retval;
        }

    private:

        /** The IP address wrapped by this object. */
        struct in_addr address;
    };

} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_IPADDRESS_H_INCLUDED */
