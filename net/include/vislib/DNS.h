/*
 * DNS.h
 *
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
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
#include "vislib/IPAgnosticAddress.h"
#include "vislib/IPHostEntry.h"
#include "vislib/String.h"


namespace vislib {
namespace net {


    /**
     * This class povides DNS queries via static methods.
     */
    class DNS {

    public:

        /**
         * Answer any of the IP addresses of the host identified by the given
         * host name or human readable IP address.
         *
         * The method first tries to retrieve the address by using the rather 
         * new getaddrinfo function. If this fails, it tries to recover using
         * the older gethostbyname function. If this fails either, the exception
         * raised by the first try is rethrown.
         *
         * @param outAddress        Receives the IP address.
         * @param hostNameOrAddress The host name or stringised IP address to 
         *                          search.
         *
         * @throws SocketException In case the operation fails, e.g. the host 
         *                         could not be found.
         */
        static void GetHostAddress(IPAddress& outAddress,
            const char *hostNameOrAddress);

        /**
         * Answer any of the IP addresses of the host identified by the given
         * host name or human readable IP address.
         *
         * The method first tries to retrieve the address by using the rather 
         * new getaddrinfo function. If this fails, it tries to recover using
         * the older gethostbyname function. If this fails either, the exception
         * raised by the first try is rethrown.
         *
         * @param outAddress        Receives the IP address.
         * @param hostNameOrAddress The host name or stringised IP address to 
         *                          search.
         *
         * @throws SocketException In case the operation fails, e.g. the host 
         *                         could not be found.
         */
        static void GetHostAddress(IPAddress& outAddress,
            const wchar_t *hostNameOrAddress);

        /**
         * Answer any of the IP addresses of the host identified by the given
         * host name or human readable IP address.
         *
         * @param outAddress        Receives the IP address.
         * @param hostNameOrAddress The host name or stringised IP address to 
         *                          search.
         *
         * @throws SocketException In case the operation fails, e.g. the host 
         *                         could not be found.
         */
        static void GetHostAddress(IPAddress6& outAddress,
            const char *hostNameOrAddress);

        /**
         * Answer any of the IP addresses of the host identified by the given
         * host name or human readable IP address.
         *
         * @param outAddress        Receives the IP address.
         * @param hostNameOrAddress The host name or stringised IP address to 
         *                          search.
         *
         * @throws SocketException In case the operation fails, e.g. the host 
         *                         could not be found.
         */
        static void GetHostAddress(IPAddress6& outAddress,
            const wchar_t *hostNameOrAddress);

        /**
         * Answer any of the IP addresses of the host identified by the given
         * host name or human readable IP address.
         *
         * @param outAddress        Receives the IP address.
         * @param hostNameOrAddress The host name or stringised IP address to 
         *                          search.
         * @param inCaseOfDoubt     The address family to be used if it is not
         *                          clear from the 'hostNameOrAddress' 
         *                          parameter. This parameter defaults to
         *                          FAMILY_INET6.
         *
         * @throws SocketException In case the operation fails, e.g. the host 
         *                         could not be found.
         */
        static void GetHostAddress(IPAgnosticAddress& outAddress,
            const char *hostNameOrAddress, 
            const IPAgnosticAddress::AddressFamily inCaseOfDoubt 
            = IPAgnosticAddress::FAMILY_INET6);

        /**
         * Answer any of the IP addresses of the host identified by the given
         * host name or human readable IP address.
         *
         * @param outAddress        Receives the IP address.
         * @param hostNameOrAddress The host name or stringised IP address to 
         *                          search.
         * @param inCaseOfDoubt     The address family to be used if it is not
         *                          clear from the 'hostNameOrAddress' 
         *                          parameter. This parameter defaults to
         *                          FAMILY_INET6.
         *
         * @throws SocketException In case the operation fails, e.g. the host 
         *                         could not be found.
         */
        static void GetHostAddress(IPAgnosticAddress& outAddress,
            const wchar_t *hostNameOrAddress,
            const IPAgnosticAddress::AddressFamily inCaseOfDoubt 
            = IPAgnosticAddress::FAMILY_INET6);


        //static void GetHostAddresses(vislib::Array<IPAddress>& outAddresses,
        //    const char *hostNameOrAddress);

        //static void GetHostAddresses(vislib::Array<IPAddress>& outAddresses,
        //    const wchar_t *hostNameOrAddress);

        //static void GetHostAddresses(vislib::Array<IPAddress6>& outAddresses,
        //    const char *hostNameOrAddress);

        //static void GetHostAddresses(vislib::Array<IPAddress6>& outAddresses,
        //    const wchar_t *hostNameOrAddress);

        //static void GetHostEntry(IPHostEntryA& outEntry, 
        //    const IPAddress& hostAddress);

        //static void GetHostEntry(IPHostEntryA& outEntry,
        //    const IPAddress6& hostAddress);

        //static void GetHostEntry(IPHostEntryW& outEntry,
        //    const IPAddress& hostAddress);

        //static void GetHostEntry(IPHostEntryW& outEntry, 
        //    const IPAddress6& hostAddress);

        /**
         * Answer the IPHostEntry for the specified host name or IP address 
         * string.
         *
         * @param outEntry          Receives the host entry.
         * @param hostNameOrAddress The host name or stringised IP address to 
         *                          search.
         *
         * @throws SocketException       In case the operation fails, e.g. the
         *                               host could not be found.
         * @throws IllegalParamException If the specified host does not use 
         *                               IPv4 or IPv6.
         */
        static void GetHostEntry(IPHostEntryA& outEntry, 
            const char *hostNameOrAddress);

        /**
         * Answer the IPHostEntry for the specified host name or IP address
         * string.
         *
         * @param outEntry          Receives the host entry.
         * @param hostNameOrAddress The host name or stringised IP address to 
         *                          search.
         *
         * @throws SocketException In case the operation fails, e.g. the host 
         *                         could not be found.
         * @throws IllegalParamException If the specified host does not use 
         *                               IPv4 or IPv6.
         */
        static void GetHostEntry(IPHostEntryW& outEntry,
            const wchar_t *hostNameOrAddress);

        /**
         * Answer the IPHostEntry for the specified IP address.
         *
         * @param outEntry Receives the host entry.
         * @param addrress The reference IP address to search.
         *
         * @throws SocketException       In case the operation fails, e.g. the
         *                               host could not be found.
         * @throws IllegalParamException If the specified host does not use 
         *                               IPv4 or IPv6.
         */
        static void GetHostEntry(IPHostEntryA& outEntry, 
            const IPAgnosticAddress& address);

        /**
         * Answer the IPHostEntry for the specified IP address.
         *
         * @param outEntry Receives the host entry.
         * @param addrress The reference IP address to search.
         *
         * @throws SocketException       In case the operation fails, e.g. the
         *                               host could not be found.
         * @throws IllegalParamException If the specified host does not use 
         *                               IPv4 or IPv6.
         */
        static void GetHostEntry(IPHostEntryW& outEntry, 
            const IPAgnosticAddress& address);

        /**
         * Answer the IPHostEntry for the specified IP address.
         *
         * @param outEntry Receives the host entry.
         * @param addrress The reference IP address to search.
         *
         * @throws SocketException       In case the operation fails, e.g. the
         *                               host could not be found.
         * @throws IllegalParamException If the specified host does not use 
         *                               IPv4 or IPv6.
         */
        static void GetHostEntry(IPHostEntryA& outEntry, 
            const IPAddress& address);

        /**
         * Answer the IPHostEntry for the specified IP address.
         *
         * @param outEntry Receives the host entry.
         * @param addrress The reference IP address to search.
         *
         * @throws SocketException       In case the operation fails, e.g. the
         *                               host could not be found.
         * @throws IllegalParamException If the specified host does not use 
         *                               IPv4 or IPv6.
         */
        static void GetHostEntry(IPHostEntryW& outEntry, 
            const IPAddress& address);

        /**
         * Answer the IPHostEntry for the specified IP address.
         *
         * @param outEntry Receives the host entry.
         * @param addrress The reference IP address to search.
         *
         * @throws SocketException       In case the operation fails, e.g. the
         *                               host could not be found.
         * @throws IllegalParamException If the specified host does not use 
         *                               IPv4 or IPv6.
         */
        static void GetHostEntry(IPHostEntryA& outEntry, 
            const IPAddress6& address);

        /**
         * Answer the IPHostEntry for the specified IP address.
         *
         * @param outEntry Receives the host entry.
         * @param addrress The reference IP address to search.
         *
         * @throws SocketException       In case the operation fails, e.g. the
         *                               host could not be found.
         * @throws IllegalParamException If the specified host does not use 
         *                               IPv4 or IPv6.
         */
        static void GetHostEntry(IPHostEntryW& outEntry, 
            const IPAddress6& address);

        /** Dtor. */
        ~DNS(void);

     private:

        /**
         * Retrieve the addrinfo list for the specified host name of IP address
         * string.
         *
         * The caller takes the ownership of the returned list and must release
         * it using the appropriate API function.
         *
         * @param hostNameOrAddress The host name or stringised IP address to 
         *                          search.
         * @param addressFamily     The address familiy to limit the search to.
         *
         * @returns The addrinfo list if the host was found. The caller must 
         *          free this list.
         *
         * @throws SocketException In case the operation fails, e.g. the host 
         *                         could not be found.
         */
        static struct addrinfo *getAddrInfo(const char *hostNameOrAddress,
            const int addressFamily);

        /**
         * Retrieve the addrinfo list for the specified host name of IP address
         * string.
         *
         * The caller takes the ownership of the returned list and must release
         * it using the appropriate API function.
         *
         * @param hostNameOrAddress The host name or stringised IP address to 
         *                          search.
         * @param addressFamily     The address familiy to limit the search to.
         *
         * @returns The addrinfo list if the host was found. The caller must 
         *          free this list.
         *
         * @throws SocketException In case the operation fails, e.g. the host 
         *                         could not be found.
         */
#if (defined(_WIN32) && defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502))
        static ADDRINFOW *getAddrInfo(const wchar_t *hostNameOrAddress,
            const int addressFamily);
#else /* (defined(_WIN32) && defined(_WIN32_WINNT) && ... */
        static struct addrinfo *getAddrInfo(const wchar_t *hostNameOrAddress,
            const int addressFamily);
#endif /* (defined(_WIN32) && defined(_WIN32_WINNT) && ... */

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

