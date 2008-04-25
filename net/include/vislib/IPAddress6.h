/*
 * IPAddress6.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
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
     */
    class IPAddress6 {

    public:

        /** Ctor. */
        IPAddress6(void);

        /** Dtor. */
        ~IPAddress6(void);

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        IPAddress6& operator =(const IPAddress6& rhs);

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @param true, if 'rhs' and this IP address are equal, false otherwise.
         */
        bool operator ==(const IPAddress6& rhs) const;

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

    private:

        /** The acutal IPv6 address. */
        struct in6_addr address;

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_IPADDRESS6_H_INCLUDED */

