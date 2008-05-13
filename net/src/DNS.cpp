/*
 * DNS.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/DNS.h"

#ifdef _WIN32
#include <ws2tcpip.h>
#endif /* _WIN32 */

#include "vislib/IllegalParamException.h"
#include "vislib/SocketException.h"
#include "vislib/StringConverter.h"
#include "vislib/UnsupportedOperationException.h"

#include "vislib/MissingImplementationException.h"



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
        const char *hostName) {
    struct addrinfo *entries = NULL;    // Receives the address infos.
    struct addrinfo hints;              // The hints about the info we want.

    ::ZeroMemory(&hints, sizeof(struct addrinfo));
    hints.ai_flags = AI_CANONNAME;      // Request canonical name.
    hints.ai_family = AF_UNSPEC;        // Any protocol family.

    if (::getaddrinfo(hostName, NULL, &hints, &entries) != 0) {
        throw SocketException(__FILE__, __LINE__);
    }

    try {
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
        const wchar_t *hostName) {
#if (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502))
    ADDRINFOW *entries = NULL;          // Receives the address infos.
    ADDRINFOW hints;                    // The hints about the info we want.

    ::ZeroMemory(&hints, sizeof(ADDRINFOW));
    hints.ai_flags = AI_CANONNAME;      // Request canonical name.
    hints.ai_family = AF_UNSPEC;        // Any protocol family.

    ::ZeroMemory(&hints, sizeof(struct addrinfo));
    hints.ai_flags = AI_CANONNAME;      // Request canonical name.
    hints.ai_family = AF_UNSPEC;        // Any protocol family.

    if (::GetAddrInfoW(hostName, NULL, &hints, &entries) != 0) {
        throw SocketException(__FILE__, __LINE__);
    }

    try {
        outEntry.set(entries);
    } catch (...) {
        ::FreeAddrInfoW(entries);
        throw;
    }
    ::FreeAddrInfoW(entries);

#else /* (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502)) */
    struct addrinfo *entries = NULL;    // Receives the address infos.
    struct addrinfo hints;              // The hints about the info we want.

    ::ZeroMemory(&hints, sizeof(struct addrinfo));
    hints.ai_flags = AI_CANONNAME;      // Request canonical name.
    hints.ai_family = AF_UNSPEC;        // Any protocol family.

    if (::getaddrinfo(W2A(hostName), NULL, &hints, &entries) != 0) {
        throw SocketException(__FILE__, __LINE__);
    }

    try {
        outEntry.set(entries);
    } catch (...) {
        ::freeaddrinfo(entries);
        throw;
    }
    ::freeaddrinfo(entries);    
#endif /* (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0502)) */
}



/*
 * vislib::net::DNS::~DNS
 */
vislib::net::DNS::~DNS(void) {
    // Nothing to do.
}


/*
 * vislib::net::DNS::DNS
 */
vislib::net::DNS::DNS(void) {
    // Nothing to do.
}


/*
 * vislib::net::DNS::DNS
 */
vislib::net::DNS::DNS(const DNS& rhs) {
    throw UnsupportedOperationException("DNS::DNS", __FILE__, __LINE__);
}


/*
 * vislib::net::DNS::DNS
 */
vislib::net::DNS& vislib::net::DNS::operator =(const DNS& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
