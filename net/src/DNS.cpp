/*
 * DNS.cpp
 *
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/DNS.h"

#ifdef _WIN32
#include <ws2tcpip.h>
#endif /* _WIN32 */

#include "vislib/IllegalParamException.h"
#include "vislib/SocketException.h"
#include "vislib/StackTrace.h"
#include "vislib/StringConverter.h"
#include "vislib/UnsupportedOperationException.h"

#include "vislib/MissingImplementationException.h"


// This is a little hack for making the implementation more readable.
// The macros are #undef'ed at the end of this file.
#if !(defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502))
#define ADDRINFOW struct addrinfo
#define FreeAddrInfoW freeaddrinfo
#endif /* !(defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502)) */


/*
 * vislib::net::DNS::GetHostAddress
 */
void vislib::net::DNS::GetHostAddress(IPAddress& outAddress,
                                      const char *hostNameOrAddress) {
    VLSTACKTRACE("DNS::GetHostAddress", __FILE__, __LINE__);
    struct addrinfo *entries = NULL;

    try {
        entries = DNS::getAddrInfo(hostNameOrAddress, AF_INET);
        ASSERT(entries != NULL);
        outAddress = reinterpret_cast<sockaddr_in *>(
            entries->ai_addr)->sin_addr;
        ::freeaddrinfo(entries);
    } catch (...) {
        /* 
         * Try to use the old implementation (gethostbyname) as fallback and 
         * fail, if this does not work either.
         */
        ASSERT(entries == NULL);
        if (!outAddress.Lookup(hostNameOrAddress)) {
            throw;
        }
    }
}


/*
 * vislib::net::DNS::GetHostAddress
 */
void vislib::net::DNS::GetHostAddress(IPAddress& outAddress,
                                      const wchar_t *hostNameOrAddress) {
    VLSTACKTRACE("DNS::GetHostAddress", __FILE__, __LINE__);
    ADDRINFOW *entries = NULL;

    try {
        entries = DNS::getAddrInfo(hostNameOrAddress, AF_INET);
        ASSERT(entries != NULL);
        outAddress = reinterpret_cast<sockaddr_in *>(
            entries->ai_addr)->sin_addr;
        ::FreeAddrInfoW(entries);
    } catch (...) {
        /* 
         * Try to use the old implementation (gethostbyname) as fallback and 
         * fail, if this does not work either.
         */
        ASSERT(entries == NULL);
        if (!outAddress.Lookup(W2A(hostNameOrAddress))) {
            throw;
        }
    }
}


/*
 * vislib::net::DNS::GetHostAddress
 */
void vislib::net::DNS::GetHostAddress(IPAddress6& outAddress,
                                      const char *hostNameOrAddress) {
    VLSTACKTRACE("DNS::GetHostAddress", __FILE__, __LINE__);

    struct addrinfo *entries = DNS::getAddrInfo(hostNameOrAddress, AF_INET6);
    ASSERT(entries != NULL);
    outAddress = reinterpret_cast<sockaddr_in6 *>(entries->ai_addr)->sin6_addr;
    ::freeaddrinfo(entries);
}


/*
 * vislib::net::DNS::GetHostAddress
 */
void vislib::net::DNS::GetHostAddress(IPAddress6& outAddress,
                                      const wchar_t *hostNameOrAddress) {
    VLSTACKTRACE("DNS::GetHostAddress", __FILE__, __LINE__);

    ADDRINFOW *entries = DNS::getAddrInfo(hostNameOrAddress, AF_INET6);
    ASSERT(entries != NULL);
    outAddress = reinterpret_cast<sockaddr_in6 *>(entries->ai_addr)->sin6_addr;
    ::FreeAddrInfoW(entries);
}


/*
 * vislib::net::DNS::GetHostAddress
 */
void vislib::net::DNS::GetHostAddress(IPAgnosticAddress& outAddress,
        const char *hostNameOrAddress, 
        const IPAgnosticAddress::AddressFamily inCaseOfDoubt) {
    VLSTACKTRACE("DNS::GetHostAddress", __FILE__, __LINE__);

    int addrFam = static_cast<int>(inCaseOfDoubt);
    struct addrinfo *entries = NULL;

    /* Use the ":" heuristic as check for IPv6 address. */
    if ((hostNameOrAddress != NULL) 
            && (::strchr(hostNameOrAddress, ':') != NULL)) {
        addrFam = AF_INET6;
    }

    try {
        entries = DNS::getAddrInfo(hostNameOrAddress, addrFam);
    } catch (...) {
        /* Try again using IPv4 in case of IPv6 failure. */
        if (addrFam != AF_INET) {
            addrFam = AF_INET;
            entries = DNS::getAddrInfo(hostNameOrAddress, addrFam);
        } else {
            throw;
        }
    }

    ASSERT(entries != NULL);    // Must have succeeded or exception.
    switch (entries->ai_family) {
        case AF_INET:
            outAddress = reinterpret_cast<const sockaddr_in *>(
                entries->ai_addr)->sin_addr;
            break;

        case AF_INET6:
            outAddress = reinterpret_cast<const sockaddr_in6 *>(
                entries->ai_addr)->sin6_addr;
            break;

        default:
            ASSERT(false);
            outAddress = IPAgnosticAddress();
            break;
    }
    
    ::freeaddrinfo(entries);
}


/*
 * vislib::net::DNS::GetHostAddress
 */
void vislib::net::DNS::GetHostAddress(IPAgnosticAddress& outAddress,
        const wchar_t *hostNameOrAddress,
        const IPAgnosticAddress::AddressFamily inCaseOfDoubt) {
    VLSTACKTRACE("DNS::GetHostAddress", __FILE__, __LINE__);

    int addrFam = static_cast<int>(inCaseOfDoubt);
    ADDRINFOW *entries = NULL;

    /* Use the ":" heuristic as check for IPv6 address. */
    if ((hostNameOrAddress != NULL) 
            && (::wcsstr(hostNameOrAddress, L":") != NULL)) {
        addrFam = AF_INET6;
    }

    try {
        entries = DNS::getAddrInfo(hostNameOrAddress, addrFam);
    } catch (...) {
        /* Try again using IPv4 in case of IPv6 failure. */
        if (addrFam != AF_INET) {
            addrFam = AF_INET;
            entries = DNS::getAddrInfo(hostNameOrAddress, addrFam);
        } else {
            throw;
        }
    }

    ASSERT(entries != NULL);    // Must have succeeded or exception.
    switch (entries->ai_family) {
        case AF_INET:
            outAddress = reinterpret_cast<const sockaddr_in *>(
                entries->ai_addr)->sin_addr;
            break;

        case AF_INET6:
            outAddress = reinterpret_cast<const sockaddr_in6 *>(
                entries->ai_addr)->sin6_addr;
            break;

        default:
            ASSERT(false);
            outAddress = IPAgnosticAddress();
            break;
    }
    
    ::FreeAddrInfoW(entries);
}


///* 
// * vislib::net::DNS::GetHostEntry
// */
//void vislib::net::DNS::GetHostEntry(IPHostEntryA& outEntry,
//        const IPAddress& hostAddress) {
//    throw MissingImplementationException("GetHostEntry", __FILE__, __LINE__);
//
//}
//
//
///* 
// * vislib::net::DNS::GetHostEntry
// */
//void vislib::net::DNS::GetHostEntry(IPHostEntryA& outEntry,
//        const IPAddress6& hostAddress) {
//    throw MissingImplementationException("GetHostEntry", __FILE__, __LINE__);
//}
//
//
///* 
// * vislib::net::DNS::GetHostEntry
// */
//void vislib::net::DNS::GetHostEntry(IPHostEntryW& outEntry,
//        const IPAddress& hostAddress) {
//    throw MissingImplementationException("GetHostEntry", __FILE__, __LINE__);
//
//}
//
//
///* 
// * vislib::net::DNS::GetHostEntry
// */
//void vislib::net::DNS::GetHostEntry(IPHostEntryW& outEntry,
//        const IPAddress6& hostAddress) {
//    throw MissingImplementationException("GetHostEntry", __FILE__, __LINE__);
//}


/* 
 * vislib::net::DNS::GetHostEntry
 */
void vislib::net::DNS::GetHostEntry(IPHostEntryA& outEntry, 
        const char *hostNameOrAddress) {
    VLSTACKTRACE("DNS::GetHostEntry", __FILE__, __LINE__);

    struct addrinfo *entries = DNS::getAddrInfo(hostNameOrAddress, AF_UNSPEC);

    try {
        ASSERT(entries != NULL);
        outEntry.set(entries);
    } catch (...) {
        ::freeaddrinfo(entries);
        throw;
    }
    ::freeaddrinfo(entries);
}


/* 
 * vislib::net::DNS::GetHostEntry
 */
void vislib::net::DNS::GetHostEntry(IPHostEntryW& outEntry, 
        const wchar_t *hostNameOrAddress) {
    VLSTACKTRACE("DNS::GetHostEntry", __FILE__, __LINE__);

    ADDRINFOW *entries = DNS::getAddrInfo(hostNameOrAddress, AF_UNSPEC);

    try {
        ASSERT(entries != NULL);
        outEntry.set(entries);
    } catch (...) {
        ::FreeAddrInfoW(entries);
        throw;
    }
    ::FreeAddrInfoW(entries);
}


/*
 * vislib::net::DNS::GetHostEntry
 */
void vislib::net::DNS::GetHostEntry(IPHostEntryA& outEntry, 
        const IPAgnosticAddress& address) {
    VLSTACKTRACE("DNS::GetHostEntry", __FILE__, __LINE__);
    DNS::GetHostEntry(outEntry, address.ToStringA().PeekBuffer());
}


/*
 * vislib::net::DNS::GetHostEntry
 */
void vislib::net::DNS::GetHostEntry(IPHostEntryW& outEntry,  
        const IPAgnosticAddress& address) {
    VLSTACKTRACE("DNS::GetHostEntry", __FILE__, __LINE__);
    DNS::GetHostEntry(outEntry, address.ToStringW().PeekBuffer());
}


/*
 * vislib::net::DNS::GetHostEntry
 */
void vislib::net::DNS::GetHostEntry(IPHostEntryA& outEntry, 
        const IPAddress& address) {
    VLSTACKTRACE("DNS::GetHostEntry", __FILE__, __LINE__);
    DNS::GetHostEntry(outEntry, address.ToStringA().PeekBuffer());
}


/*
 * vislib::net::DNS::GetHostEntry
 */
void vislib::net::DNS::GetHostEntry(IPHostEntryW& outEntry,  
        const IPAddress& address) {
    VLSTACKTRACE("DNS::GetHostEntry", __FILE__, __LINE__);
    DNS::GetHostEntry(outEntry, address.ToStringW().PeekBuffer());
}


/*
 * vislib::net::DNS::GetHostEntry
 */
void vislib::net::DNS::GetHostEntry(IPHostEntryA& outEntry, 
        const IPAddress6& address) {
    VLSTACKTRACE("DNS::GetHostEntry", __FILE__, __LINE__);
    DNS::GetHostEntry(outEntry, address.ToStringA().PeekBuffer());
}


/*
 * vislib::net::DNS::GetHostEntry
 */
void vislib::net::DNS::GetHostEntry(IPHostEntryW& outEntry,  
        const IPAddress6& address) {
    VLSTACKTRACE("DNS::GetHostEntry", __FILE__, __LINE__);
    DNS::GetHostEntry(outEntry, address.ToStringW().PeekBuffer());
}


/*
 * vislib::net::DNS::getAddrInfo
 */
struct addrinfo *vislib::net::DNS::getAddrInfo(const char *hostNameOrAddress,
        const int addressFamily) {
    VLSTACKTRACE("DNS::getAddrInfo", __FILE__, __LINE__);

    struct addrinfo *retval = NULL;     // Receives the address infos.
    struct addrinfo hints;              // The hints about the info we want.
    int err = 0;                        // Receives lookup error codes.

    /*
     * Set the lookup hints:
     * - The input string is either a host name or a human readable IP address.
     * - Return the addresses for any protocol family.
     */
    ::ZeroMemory(&hints, sizeof(hints));
    hints.ai_flags = 0; //AI_CANONNAME | AI_NUMERICHOST;
    hints.ai_family = addressFamily;

    if ((err = ::getaddrinfo(hostNameOrAddress, NULL, &hints, &retval)) != 0) {
        ASSERT(retval == NULL);
#ifdef _WIN32
        throw SocketException(__FILE__, __LINE__);
#else /* _WIN32 */
        throw SocketException(err, ::gai_strerror(err), __FILE__, __LINE__);
#endif /* _WIN32 */
    }

    return retval;
}


/*
 * vislib::net::DNS::getAddrInfo
 */
ADDRINFOW *vislib::net::DNS::getAddrInfo(const wchar_t *hostNameOrAddress,
        const int addressFamily) {
    VLSTACKTRACE("DNS::getAddrInfo", __FILE__, __LINE__);

    ADDRINFOW *retval = NULL;           // Receives the address infos.
    ADDRINFOW hints;                    // The hints about the info we want.
    int err = 0;                        // Receives lookup error codes.

    /*
     * Set the lookup hints:
     * - The input string is either a host name or a human readable IP address.
     * - Return the addresses for any protocol family.
     */
    ::ZeroMemory(&hints, sizeof(hints));
    hints.ai_flags = 0; //AI_CANONNAME | AI_NUMERICHOST;
    hints.ai_family = addressFamily;

#if (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502))
    if (::GetAddrInfoW(hostNameOrAddress, NULL, &hints, &retval) != 0) {
#else /* (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502)) */
    if (::getaddrinfo(W2A(hostNameOrAddress), NULL, &hints, &retval) != 0) {
#endif /* (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502)) */
        ASSERT(retval == NULL);
#ifdef _WIN32
        throw SocketException(__FILE__, __LINE__);
#else /* _WIN32 */
        throw SocketException(err, ::gai_strerror(err), __FILE__, __LINE__);
#endif /* _WIN32 */
    }

    return retval;
}


/*
 * vislib::net::DNS::~DNS
 */
vislib::net::DNS::~DNS(void) {
    VLSTACKTRACE("DNS::~DNS", __FILE__, __LINE__);
    // Nothing to do.
}


/*
 * vislib::net::DNS::DNS
 */
vislib::net::DNS::DNS(void) {
    VLSTACKTRACE("DNS::DNS", __FILE__, __LINE__);
    // Nothing to do.
}


/*
 * vislib::net::DNS::DNS
 */
vislib::net::DNS::DNS(const DNS& rhs) {
    VLSTACKTRACE("DNS::DNS", __FILE__, __LINE__);
    throw UnsupportedOperationException("DNS::DNS", __FILE__, __LINE__);
}


/*
 * vislib::net::DNS::DNS
 */
vislib::net::DNS& vislib::net::DNS::operator =(const DNS& rhs) {
    VLSTACKTRACE("DNS::operator =", __FILE__, __LINE__);

    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}

#if !(defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502))
#undef ADDRINFOW
#undef FreeAddrInfoW
#endif /* !(defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502)) */
