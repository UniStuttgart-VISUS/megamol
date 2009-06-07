/*
 * IPHostEntry.h
 *
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_IPHOSTENTRY_H_INCLUDED
#define VISLIB_IPHOSTENTRY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifndef _WIN32
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#endif /* !_WIN32 */

#include "vislib/IPAgnosticAddress.h"       // Must be first include!
#include "vislib/Array.h"
#include "vislib/IllegalParamException.h"
#include "vislib/String.h"


namespace vislib {
namespace net {


    /**
     * The IPHostEntry class is used as a helper to return answers from DNS. It
     * associates DNS host names with their respective IP addresses.
     */
    template<class T> class IPHostEntry {

    public:

        /** Ctor. */
        IPHostEntry(void);

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        IPHostEntry(const IPHostEntry& rhs);

        /** Dtor. */
        ~IPHostEntry(void);

        /**
         * Get the IP end points assigned to the host.
         *
         * @return An array of IP end points, which of the object remains 
         *         owner.
         */
        inline const Array<IPAgnosticAddress>& GetAddresses(void) const {
            return this->addresses;
        }

        /**
         * Answer the canonical name of the host.
         *
         * @return The canonical name of the host.
         */
        inline const String<T>& GetCanonicalName(void) const {
            return this->canonicalName;
        }

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        IPHostEntry& operator =(const IPHostEntry& rhs);

    private:

        /**
         * Fill the IPHostEntry with the specified address info.
         *
         * @param addrInfo The structure holding the data to copy.
         *
         * @throws IllegalParamException If the adress info uses an address
         *                               family other then AF_INET4 or AF_INET6.
         */
        void set(const struct addrinfo *addrInfo);

#ifdef _WIN32
        /**
         * Fill the IPHostEntry with the specified address info.
         *
         * @param addrInfo The structure holding the data to copy.
         *
         * @throws IllegalParamException If the adress info uses an address
         *                               family other then AF_INET4 or AF_INET6.
         */
        void set(const ADDRINFOW *addrInfo);
#endif /* _WIN32 */

        /** The socket addresses of the host (IPv4 or IPv6). */
        Array<IPAgnosticAddress> addresses;

        /** The The official name of the host. */
        String<T> canonicalName;

        /* Allow DNS creating IPHostEntry objects with actual data. */
        friend class DNS;
    };


    /*
     * vislib::net::IPHostEntry<T>::IPHostEntry
     */
    template<class T> IPHostEntry<T>::IPHostEntry(void) {
        // Nothing to do.
    }


    /*
     * vislib::net::IPHostEntry:<T>:IPHostEntry
     */
    template<class T> IPHostEntry<T>::IPHostEntry(const IPHostEntry& rhs) {
        *this = rhs;
    }


    /*
     * vislib::net::IPHostEntry<T>::~IPHostEntry
     */
    template<class T> IPHostEntry<T>::~IPHostEntry(void) {
        // Nothing to do.
    }


    /*
     * vislib::net::IPHostEntry<T>::operator =
     */
    template<class T> IPHostEntry<T>& IPHostEntry<T>::operator =(
            const IPHostEntry& rhs) {
        if (this != &rhs) {
            this->addresses = rhs.addresses;
            this->canonicalName = rhs.canonicalName;
        }

        return *this;
    }


    /*
     * IPHostEntry<T>::IPHostEntry
     */
    template<class T> void IPHostEntry<T>::set(
            const struct addrinfo *addrInfo) {
        const struct addrinfo *ai = addrInfo;   // Cursor through linked list.

        // Clear old values
        this->addresses.Clear();
        this->canonicalName.Clear();

        while (ai != NULL) {
            switch (ai->ai_family) {
                case AF_INET:
                    this->addresses.Add(IPAgnosticAddress(
                        reinterpret_cast<const sockaddr_in *>(
                        ai->ai_addr)->sin_addr));
                    break;

                case AF_INET6:
                    this->addresses.Add(IPAgnosticAddress(
                        reinterpret_cast<const sockaddr_in6 *>(
                        ai->ai_addr)->sin6_addr));
                    break;

                default:
                    throw IllegalParamException("ai", __FILE__, __LINE__);
            }
        
            if (((ai->ai_flags & AI_CANONNAME) != 0) 
                    && (ai->ai_canonname != NULL)) {
                this->canonicalName = ai->ai_canonname;
            }

            ai = ai->ai_next;
        }
    }


#ifdef _WIN32
    /*
     * IPHostEntry<T>::IPHostEntry
     */
    template<class T> void IPHostEntry<T>::set(const ADDRINFOW *addrInfo) {
        const ADDRINFOW *ai = addrInfo;         // Cursor through linked list.

        this->addresses.Clear();
        this->canonicalName.Clear();

        while (ai != NULL) {
            switch (ai->ai_family) {
                case AF_INET:
                    this->addresses.Add(IPAgnosticAddress(
                        reinterpret_cast<const sockaddr_in *>(
                        ai->ai_addr)->sin_addr));
                    break;

                case AF_INET6:
                    this->addresses.Add(IPAgnosticAddress(
                        reinterpret_cast<const sockaddr_in6 *>(
                        ai->ai_addr)->sin6_addr));
                    break;

                default:
                    throw IllegalParamException("ai", __FILE__, __LINE__);
            }
        
            if (((ai->ai_flags & AI_CANONNAME) != 0) 
                    && (ai->ai_canonname != NULL)) {
                this->canonicalName = ai->ai_canonname;
            }

            ai = ai->ai_next;
        }
    }
#endif /* _WIN32 */


    /** Template instantiation for ANSI strings. */
    typedef IPHostEntry<CharTraitsA> IPHostEntryA;

    /** Template instantiation for wide strings. */
    typedef IPHostEntry<CharTraitsW> IPHostEntryW;

    /** Template instantiation for TCHARs. */
    typedef IPHostEntry<TCharTraits> TIPHostEntry;
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_IPHOSTENTRY_H_INCLUDED */
