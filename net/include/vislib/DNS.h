/*
 * DNS.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_DNS_H_INCLUDED
#define VISLIB_DNS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/IPAddress.h"
#include "vislib/IPAddress6.h"
#include "vislib/IPHostEntry.h"
#include "vislib/String.h"


namespace vislib {
namespace net {


    /**
     * This class povides DNS queries via static methods.
     */
    class DNS {

    public:

        //static void GetHostEntry(IPHostEntryA& outEntry, 
        //    const IPAddress& hostAddress);

        //static void GetHostEntry(IPHostEntryA& outEntry,
        //    const IPAddress6& hostAddress);

        //static void GetHostEntry(IPHostEntryW& outEntry,
        //    const IPAddress& hostAddress);

        //static void GetHostEntry(IPHostEntryW& outEntry, 
        //    const IPAddress6& hostAddress);

        /**
         * Answer the IPHostEntry for the specified host name.
         *
         * @param outEntry Receives the host entry.
         * @param hostName The host name to search.
         *
         * @throws SocketException In case the operation fails, e.g. the host 
         *                         could not be found.
         */
        static void GetHostEntry(IPHostEntryA& outEntry, 
            const char *hostName);

        /**
         * Answer the IPHostEntry for the specified host name.
         *
         * @param outEntry Receives the host entry.
         * @param hostName The host name to search.
         *
         * @throws SocketException In case the operation fails, e.g. the host 
         *                         could not be found.
         */
        static void GetHostEntry(IPHostEntryW& outEntry,
            const wchar_t *hostName);

        /** Dtor. */
        ~DNS(void);

     private:

        /** 
         * Disallow instances.
         */
        DNS(void);

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        DNS(const DNS& rhs);

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If (this != &rhs).
         */
        DNS& operator =(const DNS& rhs);

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_DNS_H_INCLUDED */

