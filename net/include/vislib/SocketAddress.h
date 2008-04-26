/*
 * SocketAddress.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). All rights reserved.
 */

#ifndef VISLIB_SOCKETADDRESS_H_INCLUDED
#define VISLIB_SOCKETADDRESS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/IPAddress.h"


namespace vislib {
namespace net {

    /**
     * A wrapper for a socket address.
     *
     * @author Christoph Mueller
     */

    class SocketAddress {

    public:

        /** The possible address families. */
        enum AddressFamily {
            FAMILY_UNIX = AF_UNIX,          // Local to host.
            FAMILY_UNSPEC = AF_UNSPEC,      // Unspecified address family.
            FAMILY_INET = AF_INET,          // Internet address.
            //FAMILY_IMPLINK = AF_IMPLINK,    // arpanet imp addresses.
            //FAMILIY_PUP = AF_PUP,           // pup protocols: e.g. BSP
            //FAMILY_CHAOS = AF_CHAOS,        // mit CHAOS protocols
            //FAMILY_NS = AF_NS,              // XEROX NS protocols
            //FAMILY_IPX = AF_IPX,            // IPX protocols: IPX, SPX, etc.
            //FAMILY_ISO = AF_ISO,            // ISO protocols
            //FAMILY_OSI = AF_OSI,            // OSI is ISO
            //FAMILY_ECMA = AF_ECMA,          // european computer manufacturers
            //FAMILY_DATAKIT = AF_DATAKIT,    // datakit protocols
            //FAMILY_CCITT = AF_CCITT,        // CCITT protocols, X.25 etc.
            //FAMILY_SNA = AF_SNA,            // IBM SNA
            //FAMILY_DECNET = AF_DECnet,      // DECnet
            //FAMILY_DLI = AF_DLI,            // Direct data link interface
            //FAMILY_LAT = AF_LAT,            // LAT
            //FAMILY_HYLINK = AF_HYLINK,      // NSC Hyperchannel
            //FAMILY_APPLETALK = AF_APPLETALK,// AppleTalk 
            //FAMILY_NETBIOS = AF_NETBIOS,    // NetBios-style addresses
            //FAMILY_VOICEVIEW = AF_VOICEVIEW,// VoiceView
            //FAMILY_FIREFOX = AF_FIREFOX,    // Protocols from Firefox
            //FAMILY_UNKNOWN1 = AF_UNKNOWN1,  // Somebody is using this!
            //FAMILY_BAN = AF_BAN,            // Banyan 
            //FAMILY_ATM = AF_ATM,            // Native ATM Services
            FAMILY_INET6 = AF_INET6        // Internetwork Version 6
            //FAMILY_CLUSTER = AF_CLUSTER,    // Microsoft Wolfpack 
            //FAMILY_12844 = AF_12844,        // IEEE 1284.4 WG AF
            //FAMILY_IRDA = AF_IRDA,          // IrDA
            //FAMILY_NETDES = AF_NETDES,      // Network Designers OSI & gateway
            //FAMILY_TCNPROCESS = AF_TCNPROCESS, 
            //FAMILY_TCNMESSAGE = AF_TCNMESSAGE, 
            //FAMILY_ICLFXBM = AF_ICLFXBM
        };

        /**
         * Create a socket address by looking up the specified host or IP 
         * address.
         *
         * @param addressFamily The address family to create the address for.
         * @param host          An IP address in the dotted string format or a 
         *                      host name.
         * @param port          The port of the address.
         * 
         * @return The socket address, if all parameters are valid.
         *
         * @throws IllegalParamException If 'host' is not a valid host name or
         *                               IP address.
         */
        static SocketAddress Create(const AddressFamily addressFamily, 
            const char *host, const unsigned short port);

        /**
         * Create a socket address by looking up the specified host or IP 
         * address and using FAMILY_INET.
         *
         * @param host An IP address in the dotted string format or a host name.
         * @param port The port of the address.
         * 
         * @return The socket address, if all parameters are valid.
         *
         * @throws IllegalParamException If 'host' is not a valid host name or
         *                               IP address.
         */
        static SocketAddress CreateInet(const char *host, 
            const unsigned short port);

        /**
         * Create a socket address accepting any IP address on the specified 
         * port and using FAMILY_INET.
         *
         * @param port The port of the address.
         */
        static SocketAddress CreateInet(const unsigned short port);

        /**
         * Create a socket address accepting 'ipAddress' on the specified port.
         *
         * @param addressFamily The address family to create the address for.
         * @param ipAddress     The IP address part of the address.
         * @param port          The port of the address.
         */
        SocketAddress(const AddressFamily addressFamily, 
                      const IPAddress& ipAddress, 
                      const unsigned short port);

        /**
         * Create a socket address accepting any IP address on the specified 
         * port.
         *
         * @param addressFamily The address family to create the address for.
         * @param port          The port of the address.
         */
        SocketAddress(const AddressFamily addressFamily, 
                      const unsigned short port);

        /**
         * Cast ctor for struct sockaddr.
         *
         * @param address A struct to be copied.
         */
        SocketAddress(const struct sockaddr& address);

        /**
         * Cast ctor for struct sockaddr_in.
         *
         * @param address A struct to be copied.
         */
        SocketAddress(const struct sockaddr_in& address);

        /** Default ctor. */
        SocketAddress(void);

        /**
         * Copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        SocketAddress(const SocketAddress& rhs);

        /**
         * Create a copy of 'address' but change the port to 'newPort'.
         *
         * @param address The address to be cloned.
         * @param newPort The new port.
         */
        SocketAddress(const SocketAddress& address, 
                      const unsigned short newPort); 
  
        /** Dtor. */
        ~SocketAddress(void);

        /**
         * Answer the address family of the socket address.
         *
         * @return The address family.
         */
        inline AddressFamily GetAddressFamily(void) const {
            return static_cast<AddressFamily>(this->genericAddress.sa_family);
        }

        /**
         * Answer the IP address of the socket address. This operation is only
         * supported, if the address family is FAMILY_INET.
         *
         * @return The IP address.
         */
        inline IPAddress GetIPAddress(void) const {
            return this->inetAddress.sin_addr;
        }

        /**
         * Answer the port of the socket address. The operation is only 
         * supported, if the socket address

         *
         * @return The port number.
         */
        inline unsigned short GetPort(void) const {
            return ntohs(this->inetAddress.sin_port);
        }

        /**
         * Set a new address family.
         *
         * @param addressFamily The new address family.
         */
        inline void SetAddressFamily(const AddressFamily addressFamily) {
            this->genericAddress.sa_family 
                = static_cast<unsigned short>(addressFamily);
        }

        /**
         * Set a new IP address.
         *
         * @param ipAddress The new IP address.
         */
        void SetIPAddress(const IPAddress& ipAddress);

        /**
         * Set a new port number.
         *
         * @param port The new port number.
         */
        inline void SetPort(const unsigned short port) {
            this->inetAddress.sin_port = htons(port);
        }

        /**
         * Convert the socket address into dotted string colon port format.
         *
         * @return The string representation of the socket address.
         */
        StringA ToStringA(void) const;

        /**
         * Convert the socket address into dotted string colon port format.
         *
         * @return The string representation of the socket address.
         */
        StringW ToStringW(void) const;

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        SocketAddress& operator =(const SocketAddress& rhs);

        /**
         * Test for equality. A bitwise comparison of the two socket addresses 
         * is performed.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if the objects are equal, false otherwise.
         */
        bool operator ==(const SocketAddress& rhs) const;

        /**
         * Test for inequality. A bitwise comparison of the two socket addresses
         * is performed.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if the objects are not equal, false otherwise.
         */
        inline bool operator !=(const SocketAddress& rhs) const {
            return !(*this == rhs);
        }

        inline operator const struct sockaddr_in &(void) const {
            return this->inetAddress;
        }

        inline operator const struct sockaddr_in *const(void) const {
            return &this->inetAddress;
        }

        inline operator const struct sockaddr &(void) const {
            // We know, that the data are aligned in the same way.
            return this->genericAddress;
        }

        inline operator const struct sockaddr *const(void) const {
            // We know, that the data are aligned in the same way.
            return &this->genericAddress;
        }

    private:

        /** The wrapped socket address. */
        union {
            struct sockaddr_in inetAddress;
            struct sockaddr genericAddress;
        };
    };

} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SOCKETADDRESS_H_INCLUDED */
