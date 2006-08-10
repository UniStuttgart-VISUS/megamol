/*
 * IPAddress.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). All rights reserved.
 */

#ifndef VISLIB_IPADDRESS_H_INCLUDED
#define VISLIB_IPADDRESS_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#ifdef _WIN32
#include <winsock2.h>
#else /* _WIN32 */
#include <arpa/inet.h>
#endif /* _WIN32 */

#include "vislib/tchar.h"


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
         * Create an IPAddress from the dotted string format.
		 *
		 * NOTE: The address might be invalid after construction, if 'address'
		 * is not a valid host name or IP address in the dotted string format.
		 * It is recommended to use the Lookup() method for finding a host.
         *
         * @param address The IP address in the dotted string format.
         */
        explicit IPAddress(const TCHAR *address = _T("127.0.0.1"));

        /**
         * Cast constructor from struct in_addr.
         *
         * @param address The in_addr structure to be copied.
         */
        inline IPAddress(const struct in_addr& address) : address(address) { };

        /**
         * Copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        inline IPAddress(const IPAddress& rhs) : address(rhs.address) { };

        /** Dtor. */
        virtual ~IPAddress(void);

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
        virtual bool Lookup(const TCHAR *hostname);

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
         * @param true, if 'rhs' and this vector are equal, false otherwise.
         */
        bool operator ==(const IPAddress& rhs) const;

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this vector are not equal, false otherwise.
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
        inline operator struct in_addr&(void)  {
            return this->address;
        }

    protected:

        /** The IP address wrapped by this object. */
        struct in_addr address;
    };

} /* end namespace net */
} /* end namespace vislib */

#endif /* VISLIB_IPADDRESS_H_INCLUDED */
