/*
 * NetworkInformation.cpp
 *
 * Copyright (C) 2009 by Christoph Mülller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/NetworkInformation.h"

#include <climits>

#ifdef _WIN32
#include <ws2tcpip.h>

#else /* _WIN32 */
#include <netinet/if_ether.h>
#include <netpacket/packet.h>
#include <net/if.h> 
#include <net/if_arp.h> 
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <errno.h>
#include <ifaddrs.h>
#include <unistd.h>
#endif /* _WIN32 */

#include "vislib/assert.h"
#include "vislib/DNS.h"
#include "vislib/memutils.h"
#include "vislib/IllegalParamException.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/SocketException.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"
#include "vislib/SystemException.h"
#include "vislib/Trace.h"
#include "vislib/UnsupportedOperationException.h"


////////////////////////////////////////////////////////////////////////////////
// NoConfidenceException

/*
 * ...NetworkInformation::NoConfidenceException::NoConfidenceException
 */
vislib::net::NetworkInformation::NoConfidenceException::NoConfidenceException(
        const char *propName, const char *file,  const int line) 
        : Exception(file, line) {
    VLSTACKTRACE("NoConfidenceException::NoConfidenceException", __FILE__, 
        __LINE__);
    this->formatMsg("Property '%s' is invalid.", propName);
}


/*
 * ...NetworkInformation::::NoConfidenceException::NoConfidenceException
 */
vislib::net::NetworkInformation::NoConfidenceException::NoConfidenceException(
        const wchar_t *propName, const char *file, const int line) 
        : Exception(file, line) {
    VLSTACKTRACE("NoConfidenceException::NoConfidenceException", __FILE__, 
        __LINE__);  
    this->formatMsg(L"Property '%s' is invalid.", propName);
}


/*
 * ...NetworkInformation::::NoConfidenceException::NoConfidenceException
 */
vislib::net::NetworkInformation::NoConfidenceException::NoConfidenceException(
        const NoConfidenceException& rhs) 
        : Exception(rhs) {
    VLSTACKTRACE("NoConfidenceException::NoConfidenceException", __FILE__, 
        __LINE__);
}


/*
 * ...NetworkInformation::::NoConfidenceException::~NoConfidenceException
 */
vislib::net::NetworkInformation::NoConfidenceException::~NoConfidenceException(
        void) {
    VLSTACKTRACE("NoConfidenceException::~NoConfidenceException", __FILE__, 
        __LINE__);
}


/*
 * ...NetworkInformation::NoConfidenceException::operator =
 */
vislib::net::NetworkInformation::NoConfidenceException& 
vislib::net::NetworkInformation::NoConfidenceException::operator =(
        const NoConfidenceException& rhs) {
    VLSTACKTRACE("NoConfidenceException::operator =", __FILE__, __LINE__);
    Exception::operator =(rhs);
    return *this;
}

// NoConfidenceException
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// UnicastAddressInformation

/*
 * ...::UnicastAddressInformation::UnicastAddressInformation
 */
vislib::net::NetworkInformation::UnicastAddressInformation::\
UnicastAddressInformation(void) {
    this->prefixLength.SetValue(0);
    this->prefixOrigin.SetValue(PREFIX_ORIGIN_OTHER);
    this->suffixOrigin.SetValue(SUFFIX_ORIGIN_OTHER);
}


/*
 * ...::UnicastAddressInformation::UnicastAddressInformation
 */
vislib::net::NetworkInformation::UnicastAddressInformation::\
UnicastAddressInformation(const UnicastAddressInformation& rhs) {
    *this = rhs;
}


/*
 * .::UnicastAddressInformation::UnicastAddressInformation
 */
vislib::net::NetworkInformation::UnicastAddressInformation::\
~UnicastAddressInformation(void) {
}


/*
 * vislib::net::NetworkInformation::UnicastAddressInformation::operator =
 */
vislib::net::NetworkInformation::UnicastAddressInformation& 
vislib::net::NetworkInformation::UnicastAddressInformation::operator =(
        const UnicastAddressInformation& rhs) {
    if (this != &rhs) {
        this->address = rhs.address;
        this->prefixLength = rhs.prefixLength;
        this->prefixOrigin = rhs.prefixOrigin;
        this->suffixOrigin = rhs.suffixOrigin;
    }

    return *this;
}


/*
 * vislib::net::NetworkInformation::UnicastAddressInformation::operator ==
 */
bool vislib::net::NetworkInformation::UnicastAddressInformation::operator ==(
        const UnicastAddressInformation& rhs) const {
    return ((this->prefixLength == rhs.prefixLength)
            && (this->prefixOrigin == rhs.prefixOrigin)
            && (this->suffixOrigin == rhs.suffixOrigin)
            && (this->address == rhs.address));
}


/*
 * ...::UnicastAddressInformation::UnicastAddressInformation
 */
vislib::net::NetworkInformation::UnicastAddressInformation::\
UnicastAddressInformation(const IPEndPoint endPoint, 
        const ULONG prefixLength, 
        const Confidence prefixLengthConfidence,
        const PrefixOrigin prefixOrigin,
        const Confidence prefixOriginConfidence,
        const SuffixOrigin suffixOrigin,
        const Confidence suffixOriginConfidence) {
    this->address = endPoint.GetIPAddress();
    this->prefixLength.Set(prefixLength, prefixLengthConfidence);
    this->prefixOrigin.Set(prefixOrigin, prefixOriginConfidence);
    this->suffixOrigin.Set(suffixOrigin, suffixOriginConfidence);


}

// UnicastAddressInformation
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Adapter

/*
 * vislib::net::NetworkInformation::Adapter::Adapter
 */
vislib::net::NetworkInformation::Adapter::Adapter(void) {
    VLSTACKTRACE("Adapter::Adapter", __FILE__, __LINE__);
}

/*
 * vislib::net::NetworkInformation::Adapter::Adapter
 */
vislib::net::NetworkInformation::Adapter::Adapter(const Adapter& rhs) {
    VLSTACKTRACE("Adapter::Adapter", __FILE__, __LINE__);
    *this = rhs;
}


/*
 * vislib::net::NetworkInformation::Adapter::~Adapter
 */
vislib::net::NetworkInformation::Adapter::~Adapter(void) {
    VLSTACKTRACE("Adapter::~Adapter", __FILE__, __LINE__);
}


/*
 * vislib::net::NetworkInformation::Adapter::FormatPhysicalAddressA
 */
vislib::StringA 
vislib::net::NetworkInformation::Adapter::FormatPhysicalAddressA(void) const {
    VLSTACKTRACE("Adapter::FormatPhysicalAddressA", __FILE__, __LINE__);
    StringA retval;
    StringA tmp;

    try {
        const vislib::Array<BYTE>& addr = this->GetPhysicalAddress();

        for (SIZE_T i = 0; i < addr.Count(); i++) {
            tmp.Format((i > 0) ? "-%.2X" : "%.2X", addr[i]);
            retval += tmp;
        }
    } catch (...) {
        // Nothing to do, will return empty string now.
    }

    return retval;
}


/*
 * vislib::net::NetworkInformation::Adapter::FormatPhysicalAddressW(
 */
vislib::StringW
vislib::net::NetworkInformation::Adapter::FormatPhysicalAddressW(void) const {
    VLSTACKTRACE("Adapter::FormatPhysicalAddressW", __FILE__, __LINE__);
    StringW retval;
    StringW tmp;

    try {
        const vislib::Array<BYTE>& addr = this->GetPhysicalAddress();

        for (SIZE_T i = 0; i < addr.Count(); i++) {
            tmp.Format((i > 0) ? L"-%.2X" : L"%.2X", addr[i]);
            retval += tmp;
        }
    } catch (...) {
        // Nothing to do, will return empty string now.
    }

    return retval;
}


/*
 * vislib::net::NetworkInformation::Adapter::GetPhysicalAddress
 */
const vislib::Array<BYTE>& 
vislib::net::NetworkInformation::Adapter::GetPhysicalAddress(
        Confidence *outConfidence) const {
    VLSTACKTRACE("Adapter::GetPhysicalAddress", __FILE__, __LINE__);

    if (outConfidence != NULL) {
        *outConfidence = (this->physicalAddress.IsEmpty()) ? INVALID : VALID;
    } else if (this->physicalAddress.IsEmpty()) {
        throw NoConfidenceException("Physical Address", __FILE__, __LINE__);
    }
    return this->physicalAddress;
}


/*
 * vislib::net::NetworkInformation::Adapter::GetUnicastAddress
 */
const vislib::net::IPAgnosticAddress 
vislib::net::NetworkInformation::Adapter::GetUnicastAddress(
        const IPAgnosticAddress::AddressFamily preferredFamily) const {
    VLSTACKTRACE("Adapter::GetUnicastAddress", __FILE__, __LINE__);
    UnicastAddressList addrs = this->GetUnicastAddresses();

    for (SIZE_T i = 0; i < addrs.Count(); i++) {
        if (addrs[i].GetAddressFamily() == preferredFamily) {
            return addrs[i].GetAddress();
        }
    }

    return addrs[0].GetAddress();
}


/*
 * vislib::net::NetworkInformation::Adapter::operator =
 */
vislib::net::NetworkInformation::Adapter& 
vislib::net::NetworkInformation::Adapter::operator =(const Adapter& rhs) {
    VLSTACKTRACE("Adapter::operator =", __FILE__, __LINE__);

    if (this != &rhs) {
        this->anycastAddresses = rhs.anycastAddresses;
        this->broadcastAddress = rhs.broadcastAddress;
        this->description = rhs.description;
        this->id = rhs.id;
        this->mtu = rhs.mtu;
        this->multicastAddresses = rhs.multicastAddresses;
        this->name = rhs.name;
        this->physicalAddress = rhs.physicalAddress;
        this->status = rhs.status;
        this->type = rhs.type;
        this->unicastAddresses = rhs.unicastAddresses;
    }

    return *this;
}


/*
 * vislib::net::NetworkInformation::Adapter::operator ==
 */
bool vislib::net::NetworkInformation::Adapter::operator ==(
        const Adapter& rhs) const {
    VLSTACKTRACE("Adapter::operator ==", __FILE__, __LINE__);
#define CHECK_MEMBER_NEQ(m) if (this->m != rhs.m) { return false; }

    CHECK_MEMBER_NEQ(anycastAddresses);
    CHECK_MEMBER_NEQ(broadcastAddress);
    CHECK_MEMBER_NEQ(description);
    CHECK_MEMBER_NEQ(id);
    CHECK_MEMBER_NEQ(mtu);
    CHECK_MEMBER_NEQ(multicastAddresses);
    CHECK_MEMBER_NEQ(name);
    CHECK_MEMBER_NEQ(physicalAddress);
    CHECK_MEMBER_NEQ(status);
    CHECK_MEMBER_NEQ(type);
    CHECK_MEMBER_NEQ(unicastAddresses);
    return true;

#undef CHECK_MEMBER_EQ
}

// Adapter
////////////////////////////////////////////////////////////////////////////////


/*
 * vislib::net::NetworkInformation::CountAdapters
 */
SIZE_T vislib::net::NetworkInformation::CountAdapters() {
    VLSTACKTRACE("NetworkInformation::CountAdapters", __FILE__, __LINE__);

    NetworkInformation::lockAdapters.Lock();
    try {
        NetworkInformation::initAdapters();
    } catch (...) {
        NetworkInformation::lockAdapters.Unlock();
        throw;
    }

    SIZE_T retval = NetworkInformation::adapters.Count();
    NetworkInformation::lockAdapters.Unlock();
    return retval;
}


/*
 * vislib::net::NetworkInformation::DiscardCache
 */
void vislib::net::NetworkInformation::DiscardCache(const bool reread) {
    VLSTACKTRACE("NetworkInformation::DiscardCache", __FILE__, __LINE__);

    NetworkInformation::lockAdapters.Lock();
    try {
        NetworkInformation::adapters.Clear();

        if (reread) {
            NetworkInformation::initAdapters();
        }
    } catch (...) {
        NetworkInformation::lockAdapters.Unlock();
        throw;
    }
    NetworkInformation::lockAdapters.Unlock();
}


/*
 * vislib::net::NetworkInformation::EnumerateAdapters
 */
void vislib::net::NetworkInformation::EnumerateAdapters(
        EnumerateAdaptersCallback cb, void *userContext) {
    VLSTACKTRACE("NetworkInformation::EnumerateAdapters", __FILE__, __LINE__);

    /* Sanity checks. */
    if (cb == NULL) {
        throw IllegalParamException("cb", __FILE__, __LINE__);
    }

    NetworkInformation::lockAdapters.Lock();
    /* Update adapter list if necessary. */
    try {
        NetworkInformation::initAdapters();
    } catch (...) {
        NetworkInformation::lockAdapters.Unlock();
        throw;
    }

    /* Invoke the callback. */
    for (SIZE_T i = 0; i < NetworkInformation::adapters.Count(); i++) {
        try {
            if (!cb(NetworkInformation::adapters[i], userContext)) {
                break;
            }
        } catch (...) {
            NetworkInformation::lockAdapters.Unlock();
            throw;
        }
    }

    NetworkInformation::lockAdapters.Unlock();
}


/*
 * vislib::net::NetworkInformation::GetAdapter
 */
vislib::net::NetworkInformation::Adapter 
vislib::net::NetworkInformation::GetAdapter(const SIZE_T idx) {
    VLSTACKTRACE("NetworkInformation::GetAdapter", __FILE__, __LINE__);

    NetworkInformation::lockAdapters.Lock();
    try {
        Adapter retval = NetworkInformation::GetAdapterUnsafe(idx);
        NetworkInformation::lockAdapters.Unlock();
        return retval;
    } catch (...) {
        NetworkInformation::lockAdapters.Unlock();
        throw;
    }
}


/*
 * vislib::net::NetworkInformation::GetAdapterForID
 */
bool vislib::net::NetworkInformation::GetAdapterForID(
        Adapter& outAdapter, const char *id) {
    VLSTACKTRACE("NetworkInformation::GetAdapterForID", __FILE__, __LINE__);

    NetworkInformation::lockAdapters.Lock();
    try {
        NetworkInformation::initAdapters();
    } catch (...) {
        NetworkInformation::lockAdapters.Unlock();
        throw;
    }

    for (SIZE_T i = 0; i < NetworkInformation::adapters.Count(); i++) {
        try {
            if (NetworkInformation::adapters[i].GetID().Equals(id)) {
                outAdapter = NetworkInformation::adapters[i];
                NetworkInformation::lockAdapters.Unlock();
                return true;
            } 
        } catch (...) {
            // Ignore that and go on with the next adapter.
        }
    }
    /* Did not find valid adapter with the specified name. */

    NetworkInformation::lockAdapters.Unlock();
    return false;
}


/*
 * vislib::net::NetworkInformation::GetAdaptersForPredicate
 */
SIZE_T vislib::net::NetworkInformation::GetAdaptersForPredicate(
        AdapterList& outAdapters, SelectAdapterCallback cb, void *userContext) {
    VLSTACKTRACE("NetworkInformation::GetAdaptersForPredicate", __FILE__, 
        __LINE__);
 
    outAdapters.Clear();

    NetworkInformation::lockAdapters.Lock();
    try {
        NetworkInformation::initAdapters();
    } catch (...) {
        NetworkInformation::lockAdapters.Unlock();
        throw;
    }

    for (SIZE_T i = 0; i < NetworkInformation::adapters.Count(); i++) {
        try {
            if (cb(NetworkInformation::adapters[i], userContext)) {
                outAdapters.Add(NetworkInformation::adapters[i]);
            }
        } catch (...) {
            VLTRACE(Trace::LEVEL_VL_WARN, "Exception in SelectAdapterCallback "
                "while assessing adapter %u.\n", i);
            // Ignore that and go on with the next adapter.
        }
    }

    NetworkInformation::lockAdapters.Unlock();
    return outAdapters.Count();
}


/*
 * vislib::net::NetworkInformation::GetAdaptersForUnicastAddress
 */
SIZE_T vislib::net::NetworkInformation::GetAdaptersForUnicastAddress(
        AdapterList& outAdapters, const IPAddress& address) {
    VLSTACKTRACE("NetworkInformation::GetAdaptersForUnicastAddress", 
        __FILE__, __LINE__);
    return NetworkInformation::GetAdaptersForPredicate(
        outAdapters, 
        NetworkInformation::selectAdapterByUnicastIPv4, 
        const_cast<void *>(static_cast<const void *>(&address)));
}


/*
 * vislib::net::NetworkInformation::GetAdaptersForUnicastAddress
 */
SIZE_T vislib::net::NetworkInformation::GetAdaptersForUnicastAddress(
        AdapterList& outAdapters, const IPAddress6& address) {
    VLSTACKTRACE("NetworkInformation::GetAdaptersForUnicastAddress", 
        __FILE__, __LINE__);
    return NetworkInformation::GetAdaptersForPredicate(
        outAdapters, 
        NetworkInformation::selectAdapterByUnicastIPv6, 
        const_cast<void *>(static_cast<const void *>(&address)));
}


/*
 * vislib::net::NetworkInformation::GetAdaptersForUnicastAddress
 */
SIZE_T vislib::net::NetworkInformation::GetAdaptersForUnicastAddress(
        AdapterList& outAdapters, const IPAgnosticAddress& address) {
    VLSTACKTRACE("NetworkInformation::GetAdaptersForUnicastAddress", 
        __FILE__, __LINE__);
    return NetworkInformation::GetAdaptersForPredicate(
        outAdapters, 
        NetworkInformation::selectAdapterByUnicastIP, 
        const_cast<void *>(static_cast<const void *>(&address)));
}


/*
 * vislib::net::NetworkInformation::GetAdaptersForType
 */
SIZE_T vislib::net::NetworkInformation::GetAdaptersForType(
        AdapterList& outAdapters, const Adapter::Type type) {
    VLSTACKTRACE("NetworkInformation::GetAdaptersForType", __FILE__, __LINE__);

    return NetworkInformation::GetAdaptersForPredicate(outAdapters,
        NetworkInformation::selectAdapterByType,
        const_cast<void *>(static_cast<const void *>(&type)));
}


/*
 * vislib::net::NetworkInformation::GetAdaptersUnsafe
 */
//const vislib::net::NetworkInformation::AdapterList& 
//vislib::net::NetworkInformation::GetAdaptersUnsafe(const bool forceUpdate) {
//    VLSTACKTRACE("NetworkInformation::GetAdaptersUnsafe", __FILE__, __LINE__);
//    
//    NetworkInformation::lockAdapters.Lock();
//    try {
//        NetworkInformation::initAdapters(forceUpdate);
//        NetworkInformation::lockAdapters.Unlock();
//    } catch (...) {
//        NetworkInformation::lockAdapters.Unlock();
//        throw;
//    }
//
//    return NetworkInformation::adapters;
//}


/*
 * vislib::net::NetworkInformation::GetAdaptersForUnicastPrefix
 */
SIZE_T vislib::net::NetworkInformation::GetAdaptersForUnicastPrefix(
        AdapterList& outAdapters, const IPAgnosticAddress& address, 
        const ULONG prefixLength) {
    VLSTACKTRACE("NetworkInformation::GetAdaptersForSubnet", __FILE__, 
        __LINE__);
    UnicastAddressInformation ai(IPEndPoint(address, 0), prefixLength, VALID,
        UnicastAddressInformation::PREFIX_ORIGIN_OTHER, INVALID,
        UnicastAddressInformation::SUFFIX_ORIGIN_OTHER, INVALID);

    return NetworkInformation::GetAdaptersForPredicate(outAdapters,
        NetworkInformation::selectAdapterByUnicastPrefix,
        static_cast<void *>(&ai));
}


/*
 * vislib::net::NetworkInformation::GetAdapterUnsafe
 */
const vislib::net::NetworkInformation::Adapter& 
vislib::net::NetworkInformation::GetAdapterUnsafe(const SIZE_T idx) {
    VLSTACKTRACE("NetworkInformation::GetAdapterUnsafe", __FILE__, __LINE__);
    SIZE_T cntAdapters = 0;

    NetworkInformation::lockAdapters.Lock();
    try {
        NetworkInformation::initAdapters();
    } catch (...) {
        NetworkInformation::lockAdapters.Unlock();
        throw;
    }

    cntAdapters = NetworkInformation::adapters.Count();
    if ((idx >= 0) && (idx < cntAdapters)) {
        Adapter& retval = NetworkInformation::adapters[idx];
        NetworkInformation::lockAdapters.Unlock();
        return retval;

    } else {
        NetworkInformation::lockAdapters.Unlock();
        throw OutOfRangeException(static_cast<int>(idx), 0,
            static_cast<int>(cntAdapters), __FILE__, __LINE__);
    }
}


/*
 * vislib::net::NetworkInformation::GuessAdapter
 */
float vislib::net::NetworkInformation::GuessAdapter(Adapter& outAdapter, 
        const char *str, const bool invertWildness) {
    VLSTACKTRACE("NetworkInformation::GuessAdapter", __FILE__, __LINE__);
    return NetworkInformation::GuessAdapter(outAdapter, A2W(str), 
        invertWildness);
}



/*
 * vislib::net::NetworkInformation::GuessAdapter
 */
float vislib::net::NetworkInformation::GuessAdapter(Adapter& outAdapter, 
        const wchar_t *str, const bool invertWildness) {
    VLSTACKTRACE("NetworkInformation::GuessAdapter", __FILE__, __LINE__);
    float retval = 1.0f;            // The wildness of the guess.
    UINT32 validMask = 0;           // Valid fields from the input.
    IPAgnosticAddress address;      // The adapter address from the input.
    StringW device;                 // The device name from the input.
    ULONG prefixLen;                // The prefix length from the input.
    USHORT port;                    // The port from the input.

#if (defined(DEBUG) || defined(_DEBUG))
    if (invertWildness) {
        VLTRACE(Trace::LEVEL_VL_WARN, "You have chosen to invert the wildness "
            "of GuessAdapter(). Please be advised that this is not "
            "recommended and severely degrades the Chefmäßigkeit of your "
            "application. It is recommended not to invert the wildness.\n");
    }
#endif /* (defined(DEBUG) || defined(_DEBUG)) */
    VLTRACE(Trace::LEVEL_VL_VERBOSE, "GuessAdapter() trying to guess what "
        " adapter \"%s\" might be ...\n", W2A(str));

    /* Parse the input. */
    validMask = NetworkInformation::wildGuessSplitInput(address, device, 
        prefixLen, port, str);

    /* Do the wild guess. */
    try {
        NetworkInformation::lockAdapters.Lock();
        retval = NetworkInformation::wildGuessAdapter(outAdapter, address, 
            device, prefixLen, validMask);
    } catch (...) {
        NetworkInformation::lockAdapters.Unlock();
        throw;
    }
    NetworkInformation::lockAdapters.Unlock();

    ASSERT(retval >= 0.0f);
    ASSERT(retval <= 1.0f);
    return (invertWildness ? (1.0f - retval) : retval);
}


/*
 *  vislib::net::NetworkInformation::GuessLocalEndPoint
 */
float vislib::net::NetworkInformation::GuessLocalEndPoint(
        IPEndPoint& outEndPoint, const char *str, 
        const IPAgnosticAddress::AddressFamily preferredFamily, 
        const bool invertWildness) {
    VLSTACKTRACE("NetworkInformation::GuessLocalEndPoint", __FILE__, __LINE__);
    return NetworkInformation::guessLocalEndPoint(outEndPoint, A2W(str),
        &preferredFamily, invertWildness);
}


/*
 *  vislib::net::NetworkInformation::GuessLocalEndPoint
 */
float vislib::net::NetworkInformation::GuessLocalEndPoint(
        IPEndPoint& outEndPoint, const wchar_t *str, 
        const IPAgnosticAddress::AddressFamily preferredFamily, 
        const bool invertWildness) {
    VLSTACKTRACE("NetworkInformation::GuessLocalEndPoint", __FILE__, __LINE__);
    return NetworkInformation::guessLocalEndPoint(outEndPoint, str,
        &preferredFamily, invertWildness);
}


/*
 * vislib::net::NetworkInformation::GuessLocalEndPoint
 */
float vislib::net::NetworkInformation::GuessLocalEndPoint(
        IPEndPoint& outEndPoint, const char *str, const bool invertWildness) {
    VLSTACKTRACE("NetworkInformation::GuessLocalEndPoint", __FILE__, __LINE__);
    return NetworkInformation::guessLocalEndPoint(outEndPoint, A2W(str),
        NULL, invertWildness);
}


/*
 * vislib::net::NetworkInformation::GuessLocalEndPoint
 */
float vislib::net::NetworkInformation::GuessLocalEndPoint(
        IPEndPoint& outEndPoint, const wchar_t *str, 
        const bool invertWildness) {
    VLSTACKTRACE("NetworkInformation::GuessLocalEndPoint", __FILE__, __LINE__);
    return NetworkInformation::guessLocalEndPoint(outEndPoint, str,
        NULL, invertWildness);
}


/*
 * vislib::net::NetworkInformation::GuessRemoteEndPoint
 */
float vislib::net::NetworkInformation::GuessRemoteEndPoint(
        IPEndPoint& outEndPoint, 
        const char *str, 
        const IPAgnosticAddress::AddressFamily preferredFamily,
        const bool invertWildness) {
    VLSTACKTRACE("NetworkInformation::GuessRemoteEndPoint", __FILE__, __LINE__);
    return NetworkInformation::guessRemoteEndPoint(outEndPoint, 
        A2W(str), &preferredFamily, invertWildness);
}


/*
 * NetworkInformation::GuessRemoteEndPoint
 */
float vislib::net::NetworkInformation::GuessRemoteEndPoint(
        IPEndPoint& outEndPoint, const char *str, const bool invertWildness) {
    VLSTACKTRACE("NetworkInformation::GuessRemoteEndPoint", __FILE__, __LINE__);
    return NetworkInformation::guessRemoteEndPoint(outEndPoint, 
        A2W(str), NULL, invertWildness);
}



/*
 * vislib::net::NetworkInformation::GuessRemoteEndPoint
 */
float vislib::net::NetworkInformation::GuessRemoteEndPoint(
        IPEndPoint& outEndPoint, 
        const wchar_t *str, 
        const IPAgnosticAddress::AddressFamily preferredFamily,
        const bool invertWildness) {
    VLSTACKTRACE("NetworkInformation::GuessRemoteEndPoint", __FILE__, __LINE__);
    return NetworkInformation::guessRemoteEndPoint(outEndPoint, 
        str, &preferredFamily, invertWildness);
}


/*
 * NetworkInformation::GuessRemoteEndPoint
 */
float vislib::net::NetworkInformation::GuessRemoteEndPoint(
        IPEndPoint& outEndPoint, const wchar_t *str, const bool invertWildness) {
    VLSTACKTRACE("NetworkInformation::GuessRemoteEndPoint", __FILE__, __LINE__);
    return NetworkInformation::guessRemoteEndPoint(outEndPoint, 
        str, NULL, invertWildness);
}


/*
 * vislib::net::NetworkInformation::NetmaskToPrefix
 */
ULONG vislib::net::NetworkInformation::NetmaskToPrefix(
        const IPAgnosticAddress& netmask) {
    VLSTACKTRACE("NetworkInformation::NetmaskToPrefix", __FILE__, __LINE__);

    switch (netmask.GetAddressFamily()) {
        case IPAgnosticAddress::FAMILY_INET:
            return NetworkInformation::NetmaskToPrefix(
                *static_cast<const IPAddress *>(netmask));

        case IPAgnosticAddress::FAMILY_INET6:
            return NetworkInformation::NetmaskToPrefix(
                *static_cast<const IPAddress6 *>(netmask));

        default:
            ASSERT(false);
            return static_cast<ULONG>(ULONG_MAX);
    }
}


#define IMPLEMENT_STRINGISE_CASE(ns, v) case ns::v: return #v

/*
 * vislib::net::NetworkInformation::Stringise
 */
const char *vislib::net::NetworkInformation::Stringise(
        const Adapter::ScopeLevel sl) {
    VLSTACKTRACE("NetworkInformation::Stringise", __FILE__, 
        __LINE__);

    switch (sl) {
        IMPLEMENT_STRINGISE_CASE(Adapter, SCOPE_INTERFACE);
        IMPLEMENT_STRINGISE_CASE(Adapter, SCOPE_LINK);
        IMPLEMENT_STRINGISE_CASE(Adapter, SCOPE_SUBNET);
        IMPLEMENT_STRINGISE_CASE(Adapter, SCOPE_ADMIN);
        IMPLEMENT_STRINGISE_CASE(Adapter, SCOPE_SITE);
        IMPLEMENT_STRINGISE_CASE(Adapter, SCOPE_ORGANISATION);
        IMPLEMENT_STRINGISE_CASE(Adapter, SCOPE_GLOBAL);

        default:
            throw IllegalParamException("sl", __FILE__, __LINE__);
    }
}


/*
 * vislib::net::NetworkInformation::Stringise
 */
const char *vislib::net::NetworkInformation::Stringise(
        const Adapter::OperStatus as) {
    VLSTACKTRACE("NetworkInformation::Stringise", __FILE__, 
        __LINE__);

    switch (as) {
        IMPLEMENT_STRINGISE_CASE(Adapter, OPERSTATUS_UNKNOWN);
        IMPLEMENT_STRINGISE_CASE(Adapter, OPERSTATUS_UP);
        IMPLEMENT_STRINGISE_CASE(Adapter, OPERSTATUS_DOWN);
        IMPLEMENT_STRINGISE_CASE(Adapter, OPERSTATUS_TESTING);
        IMPLEMENT_STRINGISE_CASE(Adapter, OPERSTATUS_DORMANT);
        IMPLEMENT_STRINGISE_CASE(Adapter, OPERSTATUS_NOT_PRESENT);
        IMPLEMENT_STRINGISE_CASE(Adapter, OPERSTATUS_LOWER_LAYER_DOWN);

        default:
            throw IllegalParamException("as", __FILE__, __LINE__);
    }
}


/* 
 * vislib::net::NetworkInformation::Stringise
 */
const char *vislib::net::NetworkInformation::Stringise(
        const Adapter::Type at) {
    VLSTACKTRACE("NetworkInformation::Stringise", __FILE__, 
        __LINE__);

    switch (at) {
        IMPLEMENT_STRINGISE_CASE(Adapter, TYPE_OTHER);
        IMPLEMENT_STRINGISE_CASE(Adapter, TYPE_ETHERNET);
        IMPLEMENT_STRINGISE_CASE(Adapter, TYPE_TOKENRING);
        IMPLEMENT_STRINGISE_CASE(Adapter, TYPE_PPP);
        IMPLEMENT_STRINGISE_CASE(Adapter, TYPE_LOOPBACK);
        IMPLEMENT_STRINGISE_CASE(Adapter, TYPE_ATM);
        IMPLEMENT_STRINGISE_CASE(Adapter, TYPE_IEEE80211);
        IMPLEMENT_STRINGISE_CASE(Adapter, TYPE_TUNNEL);
        IMPLEMENT_STRINGISE_CASE(Adapter, TYPE_IEEE1394);

        default:
            throw IllegalParamException("at", __FILE__, __LINE__);
    }
}


/* 
 * vislib::net::NetworkInformation::Stringise
 */
const char *vislib::net::NetworkInformation::Stringise(
        const Adapter::TunnelType tt) {
    VLSTACKTRACE("NetworkInformation::Stringise", __FILE__, 
        __LINE__);

    switch (tt) {
        IMPLEMENT_STRINGISE_CASE(Adapter, TUNNELTYPE_NONE);
        IMPLEMENT_STRINGISE_CASE(Adapter, TUNNELTYPE_OTHER);
        IMPLEMENT_STRINGISE_CASE(Adapter, TUNNELTYPE_DIRECT);
        IMPLEMENT_STRINGISE_CASE(Adapter, TUNNELTYPE_6TO4);
        IMPLEMENT_STRINGISE_CASE(Adapter, TUNNEL_TYPE_ISATAP);
        IMPLEMENT_STRINGISE_CASE(Adapter, TUNNEL_TYPE_TEREDO);

        default:
            throw IllegalParamException("tt", __FILE__, __LINE__);
    }
}


/*
 * vislib::net::NetworkInformation::Stringise
 */
const char *vislib::net::NetworkInformation::Stringise(
        const UnicastAddressInformation::PrefixOrigin po) {
    VLSTACKTRACE("NetworkInformation::Stringise", __FILE__, 
        __LINE__);

    switch (po) {
        IMPLEMENT_STRINGISE_CASE(UnicastAddressInformation, 
            PREFIX_ORIGIN_OTHER);
        IMPLEMENT_STRINGISE_CASE(UnicastAddressInformation, 
            PREFIX_ORIGIN_MANUAL);
        IMPLEMENT_STRINGISE_CASE(UnicastAddressInformation, 
            PREFIX_ORIGIN_WELL_KNOWN);
        IMPLEMENT_STRINGISE_CASE(UnicastAddressInformation, 
            PREFIX_ORIGIN_DHCP);
        IMPLEMENT_STRINGISE_CASE(UnicastAddressInformation, 
            PREFIX_ORIGIN_ROUTER_ADVERTISEMENT);

        default:
            throw IllegalParamException("po", __FILE__, __LINE__);
    }
}


/*
 * vislib::net::NetworkInformation::Stringise
 */
const char *vislib::net::NetworkInformation::Stringise(
        const UnicastAddressInformation::SuffixOrigin so) {
    VLSTACKTRACE("NetworkInformation::Stringise", __FILE__, 
        __LINE__);

    switch (so) {
        IMPLEMENT_STRINGISE_CASE(UnicastAddressInformation,
            SUFFIX_ORIGIN_OTHER);
        IMPLEMENT_STRINGISE_CASE(UnicastAddressInformation, 
            SUFFIX_ORIGIN_MANUAL);
        IMPLEMENT_STRINGISE_CASE(UnicastAddressInformation, 
            SUFFIX_ORIGIN_WELL_KNOWN);
        IMPLEMENT_STRINGISE_CASE(UnicastAddressInformation, 
            SUFFIX_ORIGIN_DHCP);
        IMPLEMENT_STRINGISE_CASE(UnicastAddressInformation, 
            SUFFIX_ORIGIN_LINK_LAYER_ADDRESS);
        IMPLEMENT_STRINGISE_CASE(UnicastAddressInformation, 
            SUFFIX_ORIGIN_RANDOM);

        default:
            throw IllegalParamException("so", __FILE__, __LINE__);
    }
}

#undef IMPLEMENT_STRINGISE_CASE


/*
 * vislib::net::NetworkInformation::assessAddressAsEndPoint
 */
float vislib::net::NetworkInformation::assessAddressAsEndPoint(
        const UnicastAddressInformation& addrInfo, 
        const GuessLocalEndPointCtx& ctx) {
    VLSTACKTRACE("NetworkInformation::assessAddressAsEndPoint", __FILE__, 
        __LINE__);
    float retval = 1.0f;        // Initialise wildness result.

    if (ctx.Address != NULL) {
        IPAgnosticAddress a = addrInfo.GetAddress();
        VLTRACE(Trace::LEVEL_VL_VERBOSE, "Checking input \"%s\" against \"%s\" "
            "...\n", ctx.Address->ToStringA().PeekBuffer(),  
            a.ToStringA().PeekBuffer());

        if (a == *ctx.Address) {
            /* Identified by exact address. */
            VLTRACE(Trace::LEVEL_VL_VERBOSE, "Found exact address match "
                "\"%s\".\n", a.ToStringA().PeekBuffer());
            retval = 0.0f;

            if (ctx.PrefixLen != NULL) {
                /* Check for expected prefix length. */
                if (addrInfo.GetPrefixLength() != *ctx.PrefixLen) {
                    VLTRACE(Trace::LEVEL_VL_VERBOSE, "Prefix length %u was "
                        "found, but %u was expected.\n", 
                        addrInfo.GetPrefixLength(), *ctx.PrefixLen);
                    retval += NetworkInformation::PENALTY_WRONG_PREFIX;
                }
            }

            if ((a.GetAddressFamily() != IPAgnosticAddress::FAMILY_INET)
                    && ctx.IsIPv4Preferred) {
                VLTRACE(Trace::LEVEL_VL_VERBOSE, "Address \"%s\" is not "
                    "IPv4 as preferred.\n", a.ToStringA().PeekBuffer());
                retval += NetworkInformation::PENALTY_WRONG_ADDRESSFAMILY;
            }

        } else if (ctx.PrefixLen != NULL) {
            VLTRACE(Trace::LEVEL_VL_VERBOSE, "Have no exact address match for "
                "\"%s\", can check for same prefix length of %u.\n", 
                a.ToStringA().PeekBuffer(), *ctx.PrefixLen);

            try {
                a = a.GetPrefix(*ctx.PrefixLen);
                
                if (a == ctx.Address->GetPrefix(*ctx.PrefixLen)) {
                    /* Identified by subnet. */
                    VLTRACE(Trace::LEVEL_VL_VERBOSE, "Found exact prefix "
                        "match \"%s\".\n", a.ToStringA().PeekBuffer());
                    retval = 0.0f;
                }
            } catch (...) {
                // Illegal prefix length, just ignore it and leave wildness.
            }
        } /* end if (a == *ctx.Address) */

    } else if (ctx.PrefixLen != NULL) {
        /* Choose only based on prefix length. This is really wild ... */
        if (addrInfo.GetPrefixLength() == *ctx.PrefixLen) {
            VLTRACE(Trace::LEVEL_VL_VERBOSE, "Found prefix length match "
                "\"%s/%u\".\n", 
                addrInfo.GetAddress().ToStringA().PeekBuffer(),
                *ctx.PrefixLen);
            retval = 0.8f;
        }
    } /* end if (ctx.Address != NULL) */

    VLTRACE(Trace::LEVEL_VL_VERBOSE, "Input \"%s\" was assessed as endpoint "
        "with wildness %f.\n", addrInfo.GetAddress().ToStringA().PeekBuffer(),  
        retval);

    return retval;
}


/*
 * vislib::net::NetworkInformation::consolidateWildness
 */
float vislib::net::NetworkInformation::consolidateWildness(
        Array<float>& inOutWildness, SIZE_T& outIdxFirstBest) {
    VLSTACKTRACE("NetworkInformation::consolidateWildness", __FILE__, __LINE__);
    SIZE_T cntEqualWildness = 1;    // # of equal wildness values.
    float retval = 1.0f;            // The minimal wildness found.

    outIdxFirstBest = 0;

    /* Consolidate to [0, 1] and find the minimum. */
    ASSERT(!inOutWildness.IsEmpty());
    for (SIZE_T i = 0; i < inOutWildness.Count(); i++) {
        if (inOutWildness[i] < 0.0f) {
            inOutWildness[i] = 0.0f;
        } else if (inOutWildness[i] > 1.0f) {
            inOutWildness[i] = 1.0f;
        }

        if (inOutWildness[i] < retval) {
            retval = inOutWildness[i];
            outIdxFirstBest = i;
            cntEqualWildness = 1;

        } else if (inOutWildness[i] == retval) {
            cntEqualWildness++;
        }
    }
    /* We have the candidate now. */
    VLTRACE(Trace::LEVEL_VL_VERBOSE, "Have %u candidates with equal "
        "wildness %f. The first is at index %u.\n", cntEqualWildness, retval,
        outIdxFirstBest);

    /* 
     * If there are multiple equally good candidates, use equal weight to 
     * reflect the wildness of guessing the first one as right candidate.
     *
     * The change is not applied to 'inOutWildness' in order to allow multiple,
     * incremental calls to the method.
     */
    if (cntEqualWildness > 1) {
        if ((retval > 0.0f) && (retval < 1.0f)) {
            retval *= cntEqualWildness;
        } else if (retval == 0.0f) {
            retval = 1.0f / static_cast<float>(cntEqualWildness);
        }
    }

    /* Ensure 'retval' within [0, 1]. There might be rounding errors ... */
    if (retval < 0.0f) {
        retval = 0.0f;
    } else if (retval > 1.0f) {
        retval = 1.0f;
    }

    ASSERT(retval >= 0.0f);
    ASSERT(retval <= 1.0f);
    return retval;
}


/*
 * vislib::net::NetworkInformation::guessBroadcastAddress
 */
vislib::net::IPAddress vislib::net::NetworkInformation::guessBroadcastAddress(
        const IPAddress& address, const IPAddress& netmask) {
    VLSTACKTRACE("NetworkInformation::guessBroadcastAddress", __FILE__, 
        __LINE__);

    UINT32 addr = static_cast<UINT32>(
        static_cast<const struct in_addr *>(address)->s_addr);
    UINT32 mask = static_cast<UINT32>(
        static_cast<const struct in_addr *>(netmask)->s_addr);

    if (addr == 0) {
        throw IllegalParamException("address", __FILE__, __LINE__);

    } else if (mask == 0) {
        throw IllegalParamException("netmask", __FILE__, __LINE__);

    } else {
        struct in_addr retval;
        retval.s_addr = addr | ~mask;
        return IPAddress(retval);
    }
}


/*
 * vislib::net::NetworkInformation::guessLocalEndPoint
 */
float vislib::net::NetworkInformation::guessLocalEndPoint(
        IPEndPoint& outEndPoint, const wchar_t *str, 
        const IPAgnosticAddress::AddressFamily *prefFam, 
        const bool invertWildness) {
    VLSTACKTRACE("NetworkInformation::guessLocalEndPoint", __FILE__, __LINE__);

    Adapter candidate;              // The adapter candidate.
    GuessLocalEndPointCtx ctx;      // The enumeration context.
    Array<float> wildness(0);       // The wildness of multiple candidates.
    float retval = 1.0f;            // The wildness of the guess.
    SIZE_T bestAddressIdx = 0;      // Index of best address after consolidate.
    UINT32 validMask = 0;           // Valid fields from the input.
    IPAgnosticAddress address;      // The adapter address from the input.
    StringW device;                 // The device name from the input.
    ULONG prefixLen;                // The prefix length from the input.
    USHORT port;                    // The port from the input.

#if (defined(DEBUG) || defined(_DEBUG))
    if (invertWildness) {
        VLTRACE(Trace::LEVEL_VL_WARN, "You have chosen to invert the wildness "
            "of GuessLocalEndPoint(). Please be advised that this is not "
            "recommended and severely degrades the Chefmäßigkeit of your "
            "application. It is recommended not to invert the wildness.\n");
    }
#endif /* (defined(DEBUG) || defined(_DEBUG)) */
    VLTRACE(Trace::LEVEL_VL_VERBOSE, "GuessLocalEndPoint() trying to guess "
        " what endpoint \"%s\" might be ...\n", W2A(str));

    /* Parse the input. */
    validMask = NetworkInformation::wildGuessSplitInput(address, device, 
        prefixLen, port, str, prefFam);

    /* Set ephemeral port if no port was specified. */
    if ((validMask & WILD_GUESS_HAS_PORT) == 0) {
        port = 0;
    }

    /* Do the wild guess. */
    try {
        NetworkInformation::lockAdapters.Lock();
        NetworkInformation::initAdapters();

        /* Return any endpoint (with wildness 1) if no adapter is available. */
        if (NetworkInformation::adapters.IsEmpty()) {
            VLTRACE(Trace::LEVEL_VL_VERBOSE, "Suggest any endpoint as we do "
                "not have a network adapter ...\n", W2A(str));
            if ((validMask & WILD_GUESS_HAS_NETMASK) != 0) {
                outEndPoint.SetIPAddress(IPAgnosticAddress::ANY4);
            } else if (prefFam != NULL) {
                outEndPoint.SetIPAddress(IPAgnosticAddress::CreateAny(
                    *prefFam));
            } else {
                outEndPoint.SetIPAddress(IPAgnosticAddress::ANY6);
            }
            retval = 1.0f;

        } else if ((validMask & WILD_GUESS_FROM_EMPTY_ADDRESS) != 0) {
            /*
             * If the address is empty, we assume (for local end points) that 
             * the user actively wants to specify the ANY address.
             */
            outEndPoint.SetIPAddress((prefFam != NULL) 
                ? IPAgnosticAddress::CreateAny(*prefFam) 
                : IPAgnosticAddress(IPAgnosticAddress::ANY6));
            // mueller: IPAgnosticAddress ctor is required for correct 
            // resolution of method polymorphism! Do not remove!
            retval = 0.0f;
             
        } else {
            /* 
             * Try a real guess, first check for ANY address, then parse 
             * directly from the user input, last try to find an adapter
             * that might servet the user input.
             */
            ctx.Address = ((validMask & WILD_GUESS_HAS_ADDRESS) != 0) 
                ? &address : NULL;
            ctx.PrefixLen = (((validMask & WILD_GUESS_HAS_NETMASK) != 0) 
                || ((validMask & WILD_GUESS_HAS_PREFIX_LEN) != 0)) 
                ? &prefixLen : NULL;
            ctx.Wildness = &wildness;
            ctx.IsIPv4Preferred = ((validMask & WILD_GUESS_HAS_NETMASK) != 0)
                || ((prefFam != NULL) 
                && (*prefFam == IPAgnosticAddress::FAMILY_INET));

            if (ctx.Address != NULL) {
                if (ctx.Address->IsAny()) {
                    VLTRACE(Trace::LEVEL_VL_VERBOSE, "User explicity specified "
                        "ANY address.\n");
                    outEndPoint.SetIPAddress(*ctx.Address);
                    retval = 0.0f;
                }
            }

            if (retval == 1.0f) {
                NetworkInformation::EnumerateAdapters(
                    NetworkInformation::processAdapterForLocalEndpointGuess,
                    &ctx);
                retval = NetworkInformation::consolidateWildness(wildness, 
                    bestAddressIdx);
                for (SIZE_T i = 0, j = 0; 
                        i < NetworkInformation::adapters.Count(); i++) {
                    Confidence dummy; // TODO: consider to use in wildness 'calculation'
                    const UnicastAddressList& al = NetworkInformation::adapters[
                        i].GetUnicastAddresses(&dummy);
                    VLTRACE(Trace::LEVEL_VL_VERBOSE, "Confidence for unicast "
                        "address is %d", dummy);
                    if ((j <= bestAddressIdx) 
                            && (bestAddressIdx < j + al.Count())) {
                        outEndPoint.SetIPAddress(
                            al[bestAddressIdx - j].GetAddress());
                        break;
                    }
                    j += al.Count();
                } /* end for for (SIZE_T i = 0, j = 0; ... */
            }
                
            if ((retval == 1.0f) 
                    || ((validMask & WILD_GUESS_HAS_ADDRESS) == 0)) {
                /* 
                 * The guess might be complete nonsense, try to find something 
                 * better by guessing the adapter via the default method.
                 *
                 * At this point we know, that we cannot find a exact address 
                 * match and no exact prefix match. Therefore, the search is
                 * limited to prefix length and address family, if some is
                 * given.
                 */
                float oldGuessWildness = retval;    // preserve this.
                retval = NetworkInformation::wildGuessAdapter(candidate, 
                    address, device, prefixLen, validMask);
                const UnicastAddressList& al = candidate.GetUnicastAddresses();
                
                wildness.Clear();
                for (SIZE_T i = 0; i < al.Count(); i++) {
                    wildness.Add(NetworkInformation::assessAddressAsEndPoint(
                        al[i], ctx) * retval);
                }

                retval = NetworkInformation::consolidateWildness(wildness, 
                    bestAddressIdx);
                if (retval < oldGuessWildness) {
                    // Set only, if meaningful. Otherwise, it is better to
                    // leave previous guess.
                    outEndPoint.SetIPAddress(al[bestAddressIdx].GetAddress());
                }
            } /* end if (retval == 1.0f) */

            if ((retval == 1.0f) 
                    || ((validMask & WILD_GUESS_HAS_ADDRESS) == 0)) {
                /*
                 * If we still did not find an adapter address, there might be
                 * another option for local end points: The address could be a
                 * valid ANY4 or ANY6 address.
                 */
                if (address == IPAgnosticAddress::ANY4) {
                    outEndPoint.SetIPAddress(IPAgnosticAddress::ANY4);
                    retval = 0.0f;
                } else if (address == IPAgnosticAddress::ANY6) {
                    outEndPoint.SetIPAddress(IPAgnosticAddress::ANY6);
                    retval = 0.0f;
                }
            } /* end if (retval == 1.0f) */
        
        } /* end if (NetworkInformation::adapters.IsEmpty()) */

        outEndPoint.SetPort(port);
    } catch (...) {
        NetworkInformation::lockAdapters.Unlock();
        throw;
    }
    NetworkInformation::lockAdapters.Unlock();

    ASSERT(retval >= 0.0f);
    ASSERT(retval <= 1.0f);
    return (invertWildness ? (1.0f - retval) : retval);
}


/*
 * NetworkInformation::guessRemoteEndPoint
 */
float vislib::net::NetworkInformation::guessRemoteEndPoint(
        IPEndPoint& outEndPoint, const wchar_t *str, 
        const IPAgnosticAddress::AddressFamily *prefFam, 
        const bool invertWildness) {
    VLSTACKTRACE("NetworkInformation::guessRemoteEndPoint", __FILE__, __LINE__);

    Array<float> wildness(0);       // The wildness of multiple candidates.
    float retval = 1.0f;            // The wildness of the guess.
    IPHostEntryW hostEntry;         // DNS host entry.
    SIZE_T bestAddressIdx = 0;      // Index of best address after consolidate.
    UINT32 validMask = 0;           // Valid fields from the input.
    IPAgnosticAddress address;      // The adapter address from the input.
    StringW device;                 // The device name from the input.
    ULONG prefixLen;                // The prefix length from the input.
    USHORT port;                    // The port from the input.

#if (defined(DEBUG) || defined(_DEBUG))
    if (invertWildness) {
        VLTRACE(Trace::LEVEL_VL_WARN, "You have chosen to invert the wildness "
            "of GuessRemoteEndPoint(). Please be advised that this is not "
            "recommended and severely degrades the Chefmäßigkeit of your "
            "application. It is recommended not to invert the wildness.\n");
    }
#endif /* (defined(DEBUG) || defined(_DEBUG)) */
    VLTRACE(Trace::LEVEL_VL_VERBOSE, "GuessRemoteEndPoint() trying to guess "
        " what endpoint \"%s\" might be ...\n", W2A(str));

    /* Parse the input. */
    validMask = NetworkInformation::wildGuessSplitInput(address, device, 
        prefixLen, port, str, prefFam);

    /* Set ephemeral port if no port was specified. */
    if ((validMask & WILD_GUESS_HAS_PORT) == 0) {
        port = 0;
    }

    /* Do the wild guess. */
    try {
        DNS::GetHostEntry(hostEntry, address);
        const Array<IPAgnosticAddress>& al = hostEntry.GetAddresses();

        if ((validMask & WILD_GUESS_HAS_ADDRESS) != 0) {
            for (SIZE_T i = 0; i < al.Count(); i++) {
                retval = 1.0f;

                if (al[i] == address) {
                    retval = 0.0f;
                }

                if ((validMask & (WILD_GUESS_HAS_NETMASK 
                        | WILD_GUESS_HAS_PREFIX_LEN)) != 0) {
                    try {
                        if (al[i].GetPrefix(prefixLen) 
                                != address.GetPrefix(prefixLen)) {
                            retval 
                                += NetworkInformation::PENALTY_WRONG_PREFIX;
                        }
                    } catch (...) {
                        retval += NetworkInformation::PENALTY_WRONG_PREFIX;
                    }
                }

                if (al[i].GetAddressFamily() != address.GetAddressFamily()) {
                    retval += NetworkInformation::PENALTY_WRONG_ADDRESSFAMILY;
                }

                if ((validMask & WILD_GUESS_FROM_EMPTY_ADDRESS) != 0) {
                    retval += NetworkInformation::PENALTY_EMPTY_ADDRESS;
                }

                wildness.Add(retval);
            }

        } else {
            /*
             * If we do not have a remote address to lookup, deliberately 
             * generate an exception to enter the catch block.
             */
            throw 1;
        } /* if ((validMask & WILD_GUESS_HAS_ADDRESS) != 0) */

        retval = NetworkInformation::consolidateWildness(wildness, 
            bestAddressIdx);
        outEndPoint.SetIPAddress(al[bestAddressIdx]);
        outEndPoint.SetPort(port);

    } catch (...) {
        /*  
         * Lookup failed, assume remote endpoint on local machine and
         * add a penalty for the
         */
        retval = NetworkInformation::GuessLocalEndPoint(outEndPoint, str, 
            invertWildness) * 2.0f;
        if (retval < 0.5f) {
            retval = 0.5f;
        } else if (retval > 1.0f) {
            retval = 1.0f;
        }

    }

    ASSERT(retval >= 0.0f);
    ASSERT(retval <= 1.0f);
    return (invertWildness ? (1.0f - retval) : retval);
}


/*
 * vislib::net::NetworkInformation::initAdapters
 */
void vislib::net::NetworkInformation::initAdapters(void) {
    VLSTACKTRACE("NetworkInformation::initAdapters", __FILE__, __LINE__);

    /* Check whether there is work to do or not. */
    if (!NetworkInformation::adapters.IsEmpty()) {
        return;
    }

    /* Initialise the socket subsystem. */
    Socket::Startup(); 

#ifdef _WIN32
#define SAFE_CLEAN_RES()                                                       \
    SAFE_FREE(adapterAddresses);                                               \
    SAFE_FREE(ipAddrTable);                                                    \
    try {                                                                      \
        Socket::Cleanup();                                                     \
    } catch (SocketException e) {                                              \
        VLTRACE(Trace::LEVEL_VL_WARN, "Unexpected error while cleaning "       \
            " up socket subsystem after an error: %s", e.GetMsgA());           \
    }

    PIP_ADAPTER_ADDRESSES adapterAddresses = NULL;  // Receives the data.
    MIB_IPADDRTABLE  *ipAddrTable = NULL;           // Receives address table.
    DWORD result = 0;                               // API call result.
    ULONG flags = GAA_FLAG_INCLUDE_PREFIX;          // Get the prefix, too.
    ULONG family = AF_UNSPEC;                       // Get IPv4 and IPv6.
    ULONG bufLenAdapters = 0;                       // Size of adapter data.
    ULONG bufLenIpAddr = 0;                         // Size of IP table.
    IPEndPoint tmpEndPoint;                         // Parsing helper.

    /* Determine and allocate required memory. */
    if ((result = ::GetAdaptersAddresses(family, flags, NULL, adapterAddresses,
            &bufLenAdapters)) != ERROR_BUFFER_OVERFLOW) {
        SAFE_CLEAN_RES();
        throw sys::SystemException(result, __FILE__, __LINE__);
    }

    adapterAddresses = static_cast<PIP_ADAPTER_ADDRESSES>(::malloc(bufLenAdapters));
    if (adapterAddresses == NULL) {
        SAFE_CLEAN_RES();
        throw std::bad_alloc();
    }

    if ((result = ::GetIpAddrTable(NULL, &bufLenIpAddr, 0))
            != ERROR_INSUFFICIENT_BUFFER) {
        SAFE_CLEAN_RES();
        throw sys::SystemException(result, __FILE__, __LINE__);
    }

    ipAddrTable = static_cast<MIB_IPADDRTABLE *>(::malloc(bufLenIpAddr));
    if (ipAddrTable == NULL) {
        SAFE_CLEAN_RES();
        throw std::bad_alloc();
    }

    /* The broadcast address can be retrieved via the IP address table. */
    if ((result = ::GetIpAddrTable(ipAddrTable, &bufLenIpAddr, 0)) 
            != NO_ERROR) {
        SAFE_CLEAN_RES();
        throw sys::SystemException(result, __FILE__, __LINE__);
    }

    /* Get the adapter addresses. */
    if ((result = ::GetAdaptersAddresses(family, flags, NULL, adapterAddresses, 
            &bufLenAdapters)) != NO_ERROR) {
        SAFE_CLEAN_RES();
        throw sys::SystemException(result, __FILE__, __LINE__);
    }

    /* Process all adapter addresses. */
    PIP_ADAPTER_ADDRESSES cur = adapterAddresses;
    while (cur != NULL) {
        NetworkInformation::adapters.Add(Adapter());
        Adapter& adapter = NetworkInformation::adapters.Last();

        /* Retrieve the easy stuff. */
        adapter.id.Set(cur->AdapterName, VALID);
        adapter.description.Set(cur->Description, VALID);
        adapter.name.Set(cur->FriendlyName, VALID);
        adapter.mtu.Set(cur->Mtu, VALID);

        /* Retrieve the unicast addresses. */
        UnicastAddressList& uc = static_cast<UnicastAddressList&>(
            adapter.unicastAddresses);
        for (PIP_ADAPTER_UNICAST_ADDRESS a = cur->FirstUnicastAddress;
                a != NULL; a = a->Next) {
            tmpEndPoint = *reinterpret_cast<sockaddr_in6 *>(
                a->Address.lpSockaddr);
            uc.Add(UnicastAddressInformation(tmpEndPoint,
                a->OnLinkPrefixLength, VALID, 
                NetworkInformation::mapPrefixOrigin(a->PrefixOrigin), VALID,
                NetworkInformation::mapSuffixOrigin(a->SuffixOrigin), VALID));

            /* Retrieve the broadcast address for IPv4. */
            if (tmpEndPoint.GetAddressFamily() == IPEndPoint::FAMILY_INET) {
                // TODO: dwBCastAddr seems to be nonsense.
                //for (DWORD i = 0; i < ipAddrTable->dwNumEntries; i++) {
                //    // Note: MSDN states that addresses are in host byte order,
                //    // which seems to be wrong.
                //    IPAddress tmpAddr(ipAddrTable->table[i].dwAddr, false);
                //    //printf("%s %s %d\n", tmpEndPoint.GetIPAddress4().ToStringA().PeekBuffer(), tmpAddr.ToStringA().PeekBuffer(),
                //        tmpEndPoint.GetIPAddress4() == tmpAddr);
                //    if (tmpEndPoint.GetIPAddress4() == tmpAddr) {
                //        adapter.broadcastAddress.Set(
                //            IPAddress(ipAddrTable->table[i].dwBCastAddr, false),
                //            VALID);
                //        break;
                //    }
                //}

                /* Guess broadcast address if not found. */
                if (adapter.broadcastAddress.GetConfidence() == INVALID) {
                    try { 
                        IPAddress mask = NetworkInformation::PrefixToNetmask4(
                            a->OnLinkPrefixLength);
                        adapter.broadcastAddress.Set(
                            NetworkInformation::guessBroadcastAddress(
                            tmpEndPoint.GetIPAddress4(), mask), GUESSED);
                    } catch (...) {
                    }
                }
            }
        } /* end for (PIP_ADAPTER_UNICAST_ADDRESS a = ... */
        adapter.unicastAddresses.SetConfidence(VALID);

        /* Retrieve the anycast addresses. */
        AddressList& ac = static_cast<AddressList&>(
            adapter.anycastAddresses);
        for (PIP_ADAPTER_ANYCAST_ADDRESS a = cur->FirstAnycastAddress;
                a != NULL; a = a->Next) {
            tmpEndPoint = *reinterpret_cast<sockaddr_in6 *>(
                a->Address.lpSockaddr);
            ac.Add(tmpEndPoint.GetIPAddress());
        }
        adapter.anycastAddresses.SetConfidence(VALID);

        /* Retrieve the multicast addresses. */
        AddressList& mc = static_cast<AddressList&>(
            adapter.multicastAddresses);
        for (PIP_ADAPTER_MULTICAST_ADDRESS a = cur->FirstMulticastAddress;
                a != NULL; 
                a = a->Next) {
            tmpEndPoint = *reinterpret_cast<sockaddr_in6 *>(
                a->Address.lpSockaddr);
            mc.Add(tmpEndPoint.GetIPAddress());
        }
        adapter.multicastAddresses.SetConfidence(VALID);

        /* Retrieve MAC address. */
        adapter.physicalAddress.AssertCapacity(cur->PhysicalAddressLength);
        for (ULONG i = 0; i < cur->PhysicalAddressLength; i++) {
            adapter.physicalAddress.Add(cur->PhysicalAddress[i]);
        }

        /* Retrieve adapter type. */
        try {
            adapter.type.Set(NetworkInformation::mapAdapterType(cur->IfType),
                VALID);
        } catch (IllegalParamException) {
            adapter.type.Set(Adapter::TYPE_OTHER, GUESSED);
        }

        /* Retrieve adapter status. */
        try {
            adapter.status.Set(NetworkInformation::mapOperationStatus(
                cur->OperStatus), VALID);
        } catch (IllegalParamException) {
            adapter.status.Set(Adapter::OPERSTATUS_UNKNOWN, GUESSED);
        }

        //pDnServer = pCurrAddresses->FirstDnsServerAddress;
        //if (pDnServer) {
        //    for (i = 0; pDnServer != NULL; i++) {
        //        IPEndPoint ep(*((sockaddr_in6 *)(pDnServer->Address.lpSockaddr)));
        //        printf("\t\tdns: %s\n", ep.ToStringA().PeekBuffer());
        //        pDnServer = pDnServer->Next;
        //    }
        //    printf("\tNumber of DNS Server Addresses: %d\n", i);
        //} else
        //    printf("\tNo DNS Server Addresses\n");

        //printf("\tDNS Suffix: %wS\n", pCurrAddresses->DnsSuffix);
        //printf("\tDescription: %wS\n", pCurrAddresses->Description);
        //printf("\tFriendly name: %wS\n", pCurrAddresses->FriendlyName);

        cur = cur->Next;
    }

#else  /* _WIN32 */
#define SAFE_CLEAN_RES()                                                       \
    if (addrs != NULL) {                                                       \
        ::freeifaddrs(addrs);                                                  \
    }                                                                          \
    if (handle >= 0) {                                                         \
        ::close(handle);                                                       \
    }                                                                          \
    try {                                                                      \
        Socket::Cleanup();                                                     \
    } catch (SocketException e) {                                              \
        VLTRACE(Trace::LEVEL_VL_WARN, "Unexpected error while cleaning "       \
            " up socket subsystem after an error: %s", e.GetMsgA());           \
    }

#define SAFE_CLEAN_RES_THROWSYS(errorCode, file, line)                         \
    SAFE_CLEAN_RES();                                                          \
    throw vislib::sys::SystemException(errorCode, (file), (line));
    
    int handle = -1;                // Handle for IOCTLs
    struct ifaddrs *addrs = NULL;   // Pointer to address list.
    struct ifreq ifr;               // IOCTL request/response.

    /* 
     * Get a socket for initiating the IOCTL requests to the net subsys. Try to
     * get IPv6 first, fall back to IPv4 if necessary.
     */
    handle = ::socket(PF_INET6, SOCK_DGRAM, 0);
    if (handle < 0) {
        if ((handle = ::socket(PF_INET, SOCK_DGRAM, 0)) < 0) {
            SAFE_CLEAN_RES_THROWSYS(errno, __FILE__, __LINE__);
        }
    }
    
    /* Get the adapter addresses. */
    if (::getifaddrs(&addrs) != 0) {
        SAFE_CLEAN_RES_THROWSYS(errno, __FILE__, __LINE__);
    }

    /* Process all adapter addresses. */
    for (struct ifaddrs *cur = addrs; cur != NULL; cur = cur->ifa_next) {
        ASSERT(cur->ifa_name != NULL);
        
        /* 
         * Search whether we already have the adapter or add a new one.
         */
        Adapter *adapter = NULL;
        for (SIZE_T i = 0; i < NetworkInformation::adapters.Count(); i++) {
            if (NetworkInformation::adapters[i].id.GetValue().Equals(
                    cur->ifa_name)) {
                adapter = &(adapters[i]);
            }
        }
        if (adapter == NULL) {
            NetworkInformation::adapters.Add(Adapter());
            adapter = &(NetworkInformation::adapters.Last());

            /* Retrieve the easy stuff. */
            adapter->id.Set(cur->ifa_name, VALID);
            adapter->name.Set(A2W(cur->ifa_name), VALID);
            adapter->description.Set(A2W(cur->ifa_name), VALID);

            /* Retrieve adapter status. */
            if ((cur->ifa_flags & IFF_UP) != 0) {
                adapter->status.Set(Adapter::OPERSTATUS_UP, VALID);
            } else {
                /* Linux does not tell us whether an adapter is down. */
                adapter->status.Set(Adapter::OPERSTATUS_UNKNOWN, GUESSED);
            }
        }

        /* Get the address. */
        UnicastAddressList& ucal = static_cast<UnicastAddressList&>(
            adapter->unicastAddresses);

        switch (cur->ifa_addr->sa_family) {

            case AF_INET: {
                /* This is an IPv4 address. */
                IPEndPoint addr(*reinterpret_cast<struct sockaddr_in *>(
                    cur->ifa_addr));
                IPEndPoint mask;
                ULONG prefixLen = static_cast<ULONG>(ULONG_MAX);
                Confidence prefixConf = INVALID;

                if (cur->ifa_netmask != NULL) {
                    mask = IPEndPoint(*reinterpret_cast<struct sockaddr_in *>(
                        cur->ifa_netmask));
                    prefixLen = NetworkInformation::NetmaskToPrefix(
                        mask.GetIPAddress());
                    prefixConf = VALID;
                }

                ucal.Add(UnicastAddressInformation(addr, 
                    prefixLen, prefixConf,
                    UnicastAddressInformation::PREFIX_ORIGIN_OTHER, INVALID,
                    UnicastAddressInformation::SUFFIX_ORIGIN_OTHER, INVALID));
                adapter->unicastAddresses.SetConfidence(VALID);

                if (((cur->ifa_flags & IFF_BROADCAST) != 0) 
                        && (cur->ifa_broadaddr != NULL)) {
                    IPEndPoint bcast(*reinterpret_cast<struct sockaddr_in *>(
                        cur->ifa_broadaddr));
                    adapter->broadcastAddress.Set(
                        static_cast<IPAddress>(bcast.GetIPAddress()), VALID);

                } else {
                    try {
                        adapter->broadcastAddress.Set(
                            NetworkInformation::guessBroadcastAddress(
                            static_cast<IPAddress>(addr.GetIPAddress()),
                            static_cast<IPAddress>(mask.GetIPAddress())),
                            GUESSED);
                    } catch (...) {
                        adapter->broadcastAddress.SetConfidence(INVALID);
                    }
                } /* end if (cur->ifa_broadaddr != NULL) */
                } break;


            case AF_INET6: {
                /* This is an IPv6 address. */
                IPEndPoint addr(*reinterpret_cast<struct sockaddr_in6 *>(
                    cur->ifa_addr));
                ULONG prefixLen = static_cast<ULONG>(ULONG_MAX);
                Confidence prefixConf = INVALID;

                if (cur->ifa_netmask != NULL) {
                    IPEndPoint mask(*reinterpret_cast<struct sockaddr_in6 *>(
                        cur->ifa_netmask));
                    prefixLen = NetworkInformation::NetmaskToPrefix(
                        mask.GetIPAddress());
                    prefixConf = VALID;
                }

                ucal.Add(UnicastAddressInformation(addr, 
                    prefixLen, prefixConf,
                    UnicastAddressInformation::PREFIX_ORIGIN_OTHER, INVALID,
                    UnicastAddressInformation::SUFFIX_ORIGIN_OTHER, INVALID));
                adapter->unicastAddresses.SetConfidence(VALID);

                ASSERT(cur->ifa_broadaddr == NULL);
                } break;

            case AF_PACKET: {
                /* This is a layer 2 (MAC) address. */
                sockaddr_ll *sll = reinterpret_cast<sockaddr_ll *>(
                    cur->ifa_addr);
                adapter->physicalAddress.Clear();
                adapter->physicalAddress.AssertCapacity(sll->sll_halen);
                for (int i = 0; i < sll->sll_halen; i++) {
                    adapter->physicalAddress.Add(reinterpret_cast<BYTE *>(
                        sll->sll_addr)[i]);
                }
                } break;

            default:
                assert(true);
        }
    } /* end for (struct ifaddrs *cur = addrs; cur != NULL; ... */

    /* Now that we have all addresses, do some post-processing. */
    for (SIZE_T i = 0; i < NetworkInformation::adapters.Count(); i++) {
        Adapter& adapter = NetworkInformation::adapters[i];

        ///* Set unknown adapter type if we could not retrieve it. */
        //if (adapter.type.GetConfidence() == INVALID) {
        //    adapter.type.Set(Adapter::TYPE_OTHER, GUESSED);
        //}

        /* Get MTU. */
        ::ZeroMemory(&ifr, sizeof(ifr));
        ::strcpy(ifr.ifr_name, adapter.GetID().PeekBuffer());
        if (::ioctl(handle, SIOCGIFMTU, &ifr) >= 0) {
            adapter.mtu.Set(ifr.ifr_mtu , VALID);
        }

        /* 
         * Retrieve the MAC via IOCTL again. We need this to determine the 
         * adapter type.
         */
        // TODO: Somehow, I cannot retrieve the IPoIB MAC any more ...
        ::ZeroMemory(&ifr, sizeof(ifr));
        ::strcpy(ifr.ifr_name, adapter.GetID().PeekBuffer());
        if (::ioctl(handle, SIOCGIFHWADDR, &ifr) >= 0) {
            sockaddr_ll *sll = reinterpret_cast<sockaddr_ll *>(&ifr.ifr_hwaddr);

            if (adapter.physicalAddress.Count() == 0) {
                // If we do not yet have a MAC, set it now ...
                adapter.physicalAddress.Clear();
                adapter.physicalAddress.AssertCapacity(sll->sll_halen);
                for (int i = 0; i < sll->sll_halen; i++) {
                    adapter.physicalAddress.Add(reinterpret_cast<BYTE *>(
                        sll->sll_addr)[i]);
                }
            }

            try {
                adapter.type.Set(NetworkInformation::mapAdapterType(
                    sll->sll_family), VALID);
            } catch (IllegalParamException) {
                adapter.type.SetConfidence(INVALID);
            }
        }

    } /* end for (SIZE_T i = 0; i < NetworkInformation::adapters.Count(); ... */

#undef SAFE_CLEAN_RES_THROWSYS
#endif /* _WIN32 */

    SAFE_CLEAN_RES();
#undef SAFE_CLEAN_RES
}


/*
 * vislib::net::NetworkInformation::mapAdapterType
 */
vislib::net::NetworkInformation::Adapter::Type 
vislib::net::NetworkInformation::mapAdapterType(
#ifdef _WIN32
        const IFTYPE type) {
    VLSTACKTRACE("NetworkInformation::mapAdapterType", __FILE__, __LINE__);

    switch (type) {
        case IF_TYPE_ETHERNET_CSMACD:
            return Adapter::TYPE_ETHERNET;

        case IF_TYPE_ISO88025_TOKENRING:
            return Adapter::TYPE_TOKENRING;

        case IF_TYPE_PPP:
            return Adapter::TYPE_PPP;

        case IF_TYPE_SOFTWARE_LOOPBACK:
            return Adapter::TYPE_LOOPBACK;

        case IF_TYPE_ATM:
            return Adapter::TYPE_ATM;

        case IF_TYPE_IEEE80211:
            return Adapter::TYPE_IEEE80211;

        case IF_TYPE_TUNNEL:
            return Adapter::TYPE_TUNNEL;

        case IF_TYPE_IEEE1394:
            return Adapter::TYPE_IEEE1394;

        case IF_TYPE_OTHER:
            return Adapter::TYPE_OTHER;

        default:
            throw vislib::IllegalParamException("type", __FILE__, __LINE__);
    }
}

#else /* _WIN32 */
        const int type) {
    VLSTACKTRACE("NetworkInformation::mapAdapterType", __FILE__, __LINE__);

    switch (type) {
        case ARPHRD_ETHER:
        case ARPHRD_EETHER:
        case ARPHRD_IEEE802:
            return Adapter::TYPE_ETHERNET;

        case ARPHRD_IEEE802_TR:
            return Adapter::TYPE_TOKENRING;

        case ARPHRD_PPP:
            return Adapter::TYPE_PPP;

        case ARPHRD_LOOPBACK:
            return Adapter::TYPE_LOOPBACK;

        case ARPHRD_IEEE1394:
            return Adapter::TYPE_IEEE1394;

        case ARPHRD_ATM:
            return Adapter::TYPE_ATM;

        case ARPHRD_IEEE80211:
        case ARPHRD_IEEE80211_PRISM:
        case ARPHRD_IEEE80211_RADIOTAP:
            return Adapter::TYPE_IEEE80211;

        case ARPHRD_TUNNEL:
        case ARPHRD_TUNNEL6:
        case ARPHRD_SIT:    /* sit0 device - IPv6-in-IPv4   */
        case ARPHRD_IPDDP:  /* IP over DDP tunneller        */
        case ARPHRD_IPGRE:  /* GRE over IP                  */
            return Adapter::TYPE_TUNNEL;

        case ARPHRD_NETROM:
        case ARPHRD_AX25:
        case ARPHRD_PRONET:
        case ARPHRD_CHAOS:
        case ARPHRD_ARCNET:
        case ARPHRD_APPLETLK:
        case ARPHRD_DLCI:
        case  ARPHRD_METRICOM:
        case ARPHRD_EUI64:
        case ARPHRD_INFINIBAND:
        case ARPHRD_SLIP:
        case ARPHRD_CSLIP:
        case ARPHRD_SLIP6:
        case ARPHRD_CSLIP6:
        case ARPHRD_RSRVD:
        case ARPHRD_ADAPT:
        case ARPHRD_ROSE:
        case ARPHRD_X25:
        case ARPHRD_HWX25:
        case ARPHRD_CISCO:
        //case ARPHRD_HDLC:
        case ARPHRD_LAPB:
        case ARPHRD_DDCMP:
        case ARPHRD_RAWHDLC:
        case ARPHRD_FRAD:
        case ARPHRD_SKIP:
        case ARPHRD_LOCALTLK:
        case ARPHRD_FDDI:
        case ARPHRD_BIF:
        case ARPHRD_PIMREG:
        case ARPHRD_HIPPI: 
        case ARPHRD_ASH:
        case ARPHRD_ECONET:
        case ARPHRD_IRDA:
        case ARPHRD_FCPP:
        case ARPHRD_FCAL:
        case ARPHRD_FCPL:
        case ARPHRD_FCFABRIC:
            return Adapter::TYPE_OTHER;

        default:
            throw vislib::IllegalParamException("type", __FILE__, __LINE__);
    }
}
#endif /* _WIN32 */



#ifdef _WIN32
/*
 * vislib::net::NetworkInformation::mapOperationStatus
 */
vislib::net::NetworkInformation::Adapter::OperStatus
vislib::net::NetworkInformation::mapOperationStatus(
        const IF_OPER_STATUS status) {
    VLSTACKTRACE("NetworkInformation::mapOperationStatus", __FILE__, __LINE__);

    switch (status) {
        case IfOperStatusUp:
            return Adapter::OPERSTATUS_UP;

        case IfOperStatusDown:
            return Adapter::OPERSTATUS_DOWN;

        case IfOperStatusTesting:
            return Adapter::OPERSTATUS_TESTING;

        case IfOperStatusDormant:
            return Adapter::OPERSTATUS_DOWN;

        case IfOperStatusNotPresent:
            return Adapter::OPERSTATUS_NOT_PRESENT;

        case IfOperStatusLowerLayerDown:
            return Adapter::OPERSTATUS_LOWER_LAYER_DOWN;

        case IfOperStatusUnknown:
            return Adapter::OPERSTATUS_UNKNOWN;

        default:
            throw IllegalParamException("status", __FILE__, __LINE__);
    }
}
#endif /* _WIN32 */


#ifdef _WIN32
/*
 * vislib::net::NetworkInformation::mapPrefixOrigin
 */
vislib::net::NetworkInformation::UnicastAddressInformation::PrefixOrigin 
vislib::net::NetworkInformation::mapPrefixOrigin(
        const IP_PREFIX_ORIGIN prefixOrigin) {
    VLSTACKTRACE("NetworkInformation::mapPrefixOrigin", __FILE__, __LINE__);

    switch (prefixOrigin) {
        case IpPrefixOriginOther:
            return UnicastAddressInformation::PREFIX_ORIGIN_OTHER;

        case IpPrefixOriginManual:
            return UnicastAddressInformation::PREFIX_ORIGIN_MANUAL;
    
        case IpPrefixOriginWellKnown:
            return UnicastAddressInformation::PREFIX_ORIGIN_WELL_KNOWN;
    
        case IpPrefixOriginDhcp:
            return UnicastAddressInformation::PREFIX_ORIGIN_DHCP;
    
        case IpPrefixOriginRouterAdvertisement:
            return UnicastAddressInformation::\
                PREFIX_ORIGIN_ROUTER_ADVERTISEMENT;

        default:
            throw IllegalParamException("prefixOrigin", __FILE__, __LINE__);
    }
}


/*
 * vislib::net::NetworkInformation::mapSuffixOrigin
 */
vislib::net::NetworkInformation::UnicastAddressInformation::SuffixOrigin 
vislib::net::NetworkInformation::mapSuffixOrigin(
        const IP_SUFFIX_ORIGIN suffixOrigin) {
    VLSTACKTRACE("NetworkInformation::suffixOrigin", __FILE__, __LINE__);

    switch (suffixOrigin) {
        case IpSuffixOriginOther:
            return UnicastAddressInformation::SUFFIX_ORIGIN_OTHER;
    
        case IpSuffixOriginManual:
            return UnicastAddressInformation::SUFFIX_ORIGIN_MANUAL;

        case IpSuffixOriginWellKnown:
            return UnicastAddressInformation::SUFFIX_ORIGIN_WELL_KNOWN;

        case IpSuffixOriginDhcp:
            return UnicastAddressInformation::SUFFIX_ORIGIN_DHCP;

        case IpSuffixOriginLinkLayerAddress:
            return UnicastAddressInformation::SUFFIX_ORIGIN_LINK_LAYER_ADDRESS;

        case IpSuffixOriginRandom:
            return UnicastAddressInformation::SUFFIX_ORIGIN_RANDOM;

        default:
            throw IllegalParamException("suffixOrigin", __FILE__, __LINE__);
    }
}
#endif /* _WIN32 */


/*
 * vislib::net::NetworkInformation::netmaskToPrefix
 */
ULONG vislib::net::NetworkInformation::netmaskToPrefix(const BYTE *netmask,
                                                       const SIZE_T len) {
    VLSTACKTRACE("NetworkInformation::netmaskToPrefix", __FILE__, __LINE__);
    const BYTE *mask = netmask;
    const BYTE *end = mask + len;
    LONG retval = 0;

    while ((mask < end) && (*mask == 0xFF)) {
        retval += 8;
        mask++;
    }

    if (mask < end) {
        BYTE tmp = *mask;
        while (tmp != 0) {
            retval++;
            tmp <<= 1;
        }
        mask++;
    }

    while (mask < end) {
        if (*mask != 0) {
            throw IllegalParamException("netmask", __FILE__, __LINE__);
        }
        mask++;
    }

    return retval;
}


/*
 * vislib::net::NetworkInformation::prefixToNetmask
 */
void vislib::net::NetworkInformation::prefixToNetmask(BYTE *outNetmask, 
        const SIZE_T len, const ULONG prefix) {
    VLSTACKTRACE("NetworkInformation::prefixToNetmask", __FILE__, __LINE__);
    BYTE *mask = outNetmask;
    LONG remBits = prefix;
    LONG maxPrefix = 8L * static_cast<LONG>(len);
    
    if ((remBits < 0) || (remBits > maxPrefix)) {
        throw vislib::OutOfRangeException(remBits, 0, maxPrefix, __FILE__, 
            __LINE__);
    }

    ::ZeroMemory(mask, len);
    while (remBits > 0) {
        if (remBits >= 8) {
            *mask++ = 0xFF;
            remBits -= 8;
            continue;
        }
        
        *mask |= 1 << (8 - remBits);
        remBits--;
    }
}


/*
 * vislib::net::NetworkInformation::processAdapterForLocalEndpointGuess
 */
bool vislib::net::NetworkInformation::processAdapterForLocalEndpointGuess(
        const Adapter& adapter, void *context) {
    VLSTACKTRACE("NetworkInformation::processAdapterForLocalEndpointGuess", 
        __FILE__, __LINE__);
    GuessLocalEndPointCtx *ctx = static_cast<GuessLocalEndPointCtx *>(context);
    NetworkInformation::Confidence dummy; // TODO: should be accounted in wildness!
    UnicastAddressList al = adapter.GetUnicastAddresses(&dummy);

    ASSERT(ctx != NULL);
    ASSERT(ctx->Wildness != NULL);

    for (SIZE_T i = 0; i < al.Count(); i++) {
        ctx->Wildness->Add(NetworkInformation::assessAddressAsEndPoint(al[i], 
            *ctx));
    }
        
    return true;
}


/*
 * vislib::net::NetworkInformation::selectAdapterByType
 */
bool vislib::net::NetworkInformation::selectAdapterByType(
        const Adapter& adapter, void *type) {
    VLSTACKTRACE("NetworkInformation::selectAdapterByType", __FILE__, __LINE__);
    Adapter::Type t = *static_cast<Adapter::Type *>(type);   
    return (adapter.GetType() == t);
}


/*
 * vislib::net::NetworkInformation::selectAdapterByUnicastIP
 */
bool vislib::net::NetworkInformation::selectAdapterByUnicastIP(
        const Adapter& adapter, void *addr) {
    VLSTACKTRACE("NetworkInformation::selectAdapterByUnicastIP", __FILE__, 
        __LINE__);
    return (NetworkInformation::findAddress(
        adapter.GetUnicastAddresses(), 
        *static_cast<IPAgnosticAddress *>(addr)) >= 0);
}


/*
 * vislib::net::NetworkInformation::selectAdapterByUnicastIPv4
 */
bool vislib::net::NetworkInformation::selectAdapterByUnicastIPv4(
        const Adapter& adapter, void *addr) {
    VLSTACKTRACE("NetworkInformation::selectAdapterByUnicastIPv4", __FILE__, 
        __LINE__);
    return (NetworkInformation::findAddress(
        adapter.GetUnicastAddresses(), 
        *static_cast<IPAddress *>(addr)) >= 0);
}



/*
 * vislib::net::NetworkInformation::selectAdapterByUnicastIPv6
 */
bool vislib::net::NetworkInformation::selectAdapterByUnicastIPv6(
        const Adapter& adapter, void *addr) {
    VLSTACKTRACE("NetworkInformation::selectAdapterByUnicastIPv6", __FILE__, 
        __LINE__);
    return (NetworkInformation::findAddress(
        adapter.GetUnicastAddresses(), 
        *static_cast<IPAddress6 *>(addr)) >= 0);
}


/*
 * vislib::net::NetworkInformation::selectAdapterByUnicastPrefix
 */
bool vislib::net::NetworkInformation::selectAdapterByUnicastPrefix(
        const Adapter& adapter, void *addrInfo) {
    VLSTACKTRACE("NetworkInformation::selectAdapterByUnicastPrefix", __FILE__, 
        __LINE__);
    UnicastAddressInformation *ai = static_cast<UnicastAddressInformation *>(
        addrInfo);
    return (NetworkInformation::findSamePrefix(
        adapter.GetUnicastAddresses(),
        ai->GetAddress(),
        ai->GetPrefixLength()) >= 0);
}


/*
 * vislib::net::NetworkInformation::wildGuessAdapter
 */
float vislib::net::NetworkInformation::wildGuessAdapter(Adapter& outAdapter, 
            const IPAgnosticAddress& address, const StringW& device, 
            const ULONG prefixLen, const UINT32 validMask) {
    VLSTACKTRACE("NetworkInformation::wildGuessAdapter", __FILE__, __LINE__);
    //static const float LEVENSHTEIN_WILDNESS_WEIGHT = 0.05f;
    Array<float> wildness(0);       // The pre-adapter wildness.
    float retval = 1.0f;            // The final wildness of the guess.
    float dist = 0.0f;              // Levenshtein between input and adapter.
    float len1 = 0.0f;              // Temporary length variable.
    float len2 = 0.0f;              // Temporary length variable.
    int matchedIndex = -1;          // Index of the last successful match.
    SIZE_T candidateIdx = 0;        // Index of the guess candidate.

    /* Ensure that we know the adapters and that we have at least one. */
    NetworkInformation::initAdapters();
    if (NetworkInformation::adapters.IsEmpty()) {
        throw NoSuchElementException("There are no network adapters available "
            "for guessing the right one.", __FILE__, __LINE__);
    }
    
    /* Initially, all adapters are candidates with wildness 1. */
    wildness.AssertCapacity(NetworkInformation::adapters.Count());
    for (SIZE_T i = 0; i < NetworkInformation::adapters.Count(); i++) {
        wildness.Add(1.0f);
    }

    for (SIZE_T i = 0; i < NetworkInformation::adapters.Count(); i++) {
        const Adapter& a = NetworkInformation::adapters[i];
        matchedIndex = -1;

        VLTRACE(Trace::LEVEL_VL_VERBOSE, "WILD_GUESS_HAS_DEVICE = %d\n",
            (validMask & WILD_GUESS_HAS_DEVICE));

        /* Check whether we have an adapter with the specified device name. */
        if ((validMask & WILD_GUESS_HAS_DEVICE) != 0) {
        
            /* Check the device ID. */
            try {
                StringW id(a.GetID());
                dist = static_cast<float>(device.LevenshteinDistance(id));
                VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "Levenshtein "
                    "distance of adapter %u's ID \"%s\" and the input \"%s\" "
                    "is %f.\n", i, W2A(id), W2A(device), dist);
                len1 = static_cast<float>(id.Length());
                len2 = static_cast<float>(device.Length());
                dist /= (len1 > len2) ? len1 : len2;
                
                if (dist < wildness[i]) {
                    wildness[i] = dist;
                }
            } catch (...) {
                VLTRACE(Trace::LEVEL_VL_WARN, "Adapter %u has an invalid ID. "
                    "This should never happen.\n", i);
            }

            /* Check the friendly name. */
            try {
                dist = static_cast<float>(device.LevenshteinDistance(
                    a.GetName()));
                VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "Levenshtein "
                    "distance of adapter %u's name \"%s\" and the input \"%s\" "
                    "is %f.\n", i, W2A(a.GetName()), W2A(device), dist);
                len1 = static_cast<float>(a.GetName().Length());
                len2 = static_cast<float>(device.Length());
                dist /= (len1 > len2) ? len1 : len2;

                if (dist < wildness[i]) {
                    wildness[i] = dist;
                }
            } catch (...) {
                VLTRACE(Trace::LEVEL_VL_WARN, "Adapter %u has an invalid name. "
                    "This should never happen.\n", i);
            }
        } /* end if ((validMask & WILD_GUESS_HAS_DEVICE) != 0) */

        VLTRACE(Trace::LEVEL_VL_VERBOSE, "WILD_GUESS_HAS_ADDRESS = %d\n",
            (validMask & WILD_GUESS_HAS_ADDRESS));
        VLTRACE(Trace::LEVEL_VL_VERBOSE, "WILD_GUESS_HAS_PREFIX_LEN = %d\n",
            (validMask & WILD_GUESS_HAS_PREFIX_LEN));
        VLTRACE(Trace::LEVEL_VL_VERBOSE, "WILD_GUESS_HAS_NETMASK = %d\n",
            (validMask & WILD_GUESS_HAS_NETMASK));

        if ((validMask & WILD_GUESS_HAS_ADDRESS) != 0) {

            /*
             * Check whether we have an adapter bound to the specified 
             * unicast address. 
             */
            VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "Checking adapter "
                "%u against unicast address \"%s\" ...\n", i,
                address.ToStringA().PeekBuffer());
            if ((matchedIndex = NetworkInformation::findAddress(
                    a.GetUnicastAddresses(), address)) >= 0) {
                wildness[i] = 0.0f;
            }
        
            if (((validMask & WILD_GUESS_HAS_PREFIX_LEN) != 0)
                    || ((validMask & WILD_GUESS_HAS_NETMASK) != 0)) {
                /* 
                 * Search an adapter with the given prefix or check whether the
                 * address we already found has the correct prefix.
                 */
                VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "Checking "
                        "adapter %u against prefix \"%s/%u\" ...\n", i, 
                        address.ToStringA().PeekBuffer(), prefixLen);
                if (matchedIndex >= 0) {
                    if ((a.GetUnicastAddresses()[matchedIndex])
                            .GetPrefixLength() != prefixLen) {
                        wildness[i] += NetworkInformation::PENALTY_WRONG_PREFIX;
                    }
                } else if ((matchedIndex = NetworkInformation::findSamePrefix(
                        a.GetUnicastAddresses(), address, prefixLen)) >= 0) {
                    wildness[i] = 0.0f;
                }
            } /* end if (((validMask & WILD_GUESS_HAS_PREFIX_LEN) == 0) ... */

            if ((validMask & WILD_GUESS_FROM_EMPTY_ADDRESS) != 0) {
                /* Add penalty for guessing from empty string. */
                wildness[i] += NetworkInformation::PENALTY_EMPTY_ADDRESS;
            }

        } else if (((validMask & WILD_GUESS_HAS_PREFIX_LEN) != 0)
                || ((validMask & WILD_GUESS_HAS_NETMASK) != 0)) {
            /* 
             * Have no address, but prefix length, i. e. the adapter must have
             * been matched via the name. If so, check the prefix and add
             * optional penalty if it does not match.
             */
            VLTRACE(Trace::LEVEL_VL_VERBOSE, "Special check for correct prefix "
                "length %u ...\n", prefixLen);

            const UnicastAddressList& al = a.GetUnicastAddresses();
            
            // TODO: Das könnte man auch anders implementieren, so dass ein 
            // Adapter, der mehrere Addressen im richtigen Subnetz hat besser
            // bewertet wird.
            wildness[i] += NetworkInformation::PENALTY_WRONG_PREFIX;
            for (SIZE_T a = 0; a < al.Count(); a++) {
                if (al[a].GetPrefixLength() == prefixLen) {
                    wildness[i] -= NetworkInformation::PENALTY_WRONG_PREFIX;
                    VLTRACE(Trace::LEVEL_VL_VERBOSE, "Prefix length %u is "
                        "OK.\n", prefixLen);
                    break;
                }
            }

        } /* end if ((validMask & WILD_GUESS_HAS_ADDRESS) != 0) */

        /* Check for connection state. */
        try {
            if ((a.GetStatus() == Adapter::OPERSTATUS_DOWN) || (a.GetStatus() 
                    == Adapter::OPERSTATUS_LOWER_LAYER_DOWN)) {
                wildness[i] += NetworkInformation::PENALTY_ADAPTER_DOWN;
            }
        } catch (NoConfidenceException) {
            wildness[i] += NetworkInformation::PENALTY_ADAPTER_DOWN / 2.0f;
        }
    } /* end for (SIZE_T i = 0; i < NetworkInformation::adapters.Count() ... */

    retval = NetworkInformation::consolidateWildness(wildness, candidateIdx);
    outAdapter = NetworkInformation::adapters[candidateIdx];

    ASSERT(retval >= 0.0f);
    ASSERT(retval <= 1.0f);
    return retval;
}


/*
 * vislib::net::NetworkInformation::wildGuessSplitInput
 */
UINT32 vislib::net::NetworkInformation::wildGuessSplitInput(
        IPAgnosticAddress& outAddress, StringW& outDevice, 
        ULONG& outPrefixLen, USHORT& outPort, const wchar_t *str,
        const IPAgnosticAddress::AddressFamily *prefFam) {
    VLSTACKTRACE("NetworkInformation::wildGuessSplitInput", __FILE__, __LINE__);
    UINT32 retval = 0;          // Bitmask of valid output.
    StringW::Size pos = 0;      // Split positions.
    StringW input(str);     
    StringW prefix;
    IPAgnosticAddress::AddressFamily preferredFamily = (prefFam != NULL)
        ? *prefFam : IPAgnosticAddress::FAMILY_INET6;

    VLTRACE(Trace::LEVEL_VL_VERBOSE, "Splitting wild guess input \"%s\" ...\n",
        W2A(str));

    /* Check whether we have a port in the input. */
    if ((pos = input.FindLast(L':')) != StringW::INVALID_POS) {
        StringW tmp = input.Substring(0, pos);
        StringW port;

        /* Get the potential port. */
        if (pos < input.Length() + 3) {
            port = input.Substring(pos + 1);
            VLTRACE(Trace::LEVEL_VL_VERBOSE, "Found potential port: \"%s\"\n", 
                W2A(port));
        }

        /* 
         * If the address contains colons, we must assume that we have an IPv6 
         * address. In this case, it must be ensured that we do not interpret 
         * a part of the address as the port.
         */
        if (tmp.Contains(L':') && !port.IsEmpty()) {
            VLTRACE(Trace::LEVEL_VL_VERBOSE, "Check whether IPv6 address can "
                "be valid with port: \"%s\"\n", W2A(tmp));
            if (!tmp.Contains(L']')             // Have brackets
                    && !tmp.Contains(L'/')      // Have subnet
                    && !tmp.Contains(L'%')      // Have zone ID
                    ) {
                port = StringW::EMPTY;
            }
        }

        /* Try to parse the port. */
        port.TrimSpaces();
        if (!port.IsEmpty()) {
            input = tmp;
            VLTRACE(Trace::LEVEL_VL_VERBOSE, "Remaining input without port: "
                "\"%s\"\n", W2A(input));

            try {
                outPort = static_cast<USHORT>(CharTraitsW::ParseInt(
                    port.PeekBuffer()));
                retval |= WILD_GUESS_HAS_PORT;
                VLTRACE(Trace::LEVEL_VL_VERBOSE, "Found port: %d\n", outPort);

            } catch (Exception e) {
                VLTRACE(Trace::LEVEL_VL_VERBOSE, "Parsing port failed: %s\n",
                    e.GetMsgA());
                ASSERT((retval & WILD_GUESS_HAS_PORT) == 0);
            }
        }
    } /* end if ((pos = input.FindLast(L':')) != StringW::INVALID_POS) */

    /* Check for a subnet specification via netmask or prefix length. */
    if ((pos = input.FindLast(L'/')) != StringW::INVALID_POS) {
        StringW tmp = input.Substring(0, pos);
        StringW prefix;

        /* Get the potential subnet mask or prefix length. */
        if (pos < input.Length() + 3) {
            prefix = input.Substring(pos + 1);
            VLTRACE(Trace::LEVEL_VL_VERBOSE, "Found potential prefix: \"%s\"\n",
                W2A(prefix));
        }

        /* Try to parse the prefix. */
        prefix.TrimSpaces();
        if (!prefix.IsEmpty()) {
            input = tmp;
            VLTRACE(Trace::LEVEL_VL_VERBOSE, "Remaining input without prefix: "
                "\"%s\"\n", W2A(input));

            /* Parse as netmask first. */
            try {
                IPAgnosticAddress mask = IPAgnosticAddress::Create(
                    prefix.PeekBuffer());
                VLTRACE(Trace::LEVEL_VL_VERBOSE, "Parsed netmask: %s\n",
                    mask.ToStringA().PeekBuffer());

                outPrefixLen = NetworkInformation::NetmaskToPrefix(mask);
                preferredFamily = IPAgnosticAddress::FAMILY_INET;
                retval |= WILD_GUESS_HAS_NETMASK;
                VLTRACE(Trace::LEVEL_VL_VERBOSE, "Converted netmask to prefix "
                    "length: %u\n", outPrefixLen);

            } catch (Exception e) {
                VLTRACE(Trace::LEVEL_VL_VERBOSE, "Parsing netmask failed: %s\n",
                    e.GetMsgA());
                ASSERT((retval & WILD_GUESS_HAS_NETMASK) == 0);
            }

            /* Parse as prefix length if not a netmask. */
            if ((retval & WILD_GUESS_HAS_NETMASK) != WILD_GUESS_HAS_NETMASK) {
                try {
                    outPrefixLen = static_cast<ULONG>(CharTraitsW::ParseInt(
                        prefix.PeekBuffer()));
                    preferredFamily = IPAgnosticAddress::FAMILY_INET6;
                    retval |= WILD_GUESS_HAS_PREFIX_LEN;
                    VLTRACE(Trace::LEVEL_VL_VERBOSE, "Found prefix length: "
                        "%d.\n", outPrefixLen);

                } catch (Exception e) {
                    VLTRACE(Trace::LEVEL_VL_VERBOSE, "Parsing prefix length "
                        "failed: %s.\n", e.GetMsgA());
                    ASSERT((retval & WILD_GUESS_HAS_PREFIX_LEN) == 0);
                }
            } 
        } /* end if (!prefix.IsEmpty()) */
    } /* end if ((pos = input.FindLast(L'/')) != StringW::INVALID_POS) */

    /* Remove potential zone ID. */
    if ((pos = input.FindLast(L'%')) != StringW::INVALID_POS) {
        input.Truncate(pos);
        VLTRACE(Trace::LEVEL_VL_VERBOSE, "Remaining input without zone ID: "
            "\"%s\"\n", W2A(input));
    }
  
    /* The rest is either the adapter address or device name. */
#ifndef _WIN32
    input.Trim(L"[]");  // mueller: I forgot why we do that on Linux ...
#endif /* !_WIN32 */
    input.TrimSpaces();
    try {
        outAddress = IPAgnosticAddress::Create(input.PeekBuffer(), 
            preferredFamily);

        //outDevice = input;
        retval |= WILD_GUESS_HAS_ADDRESS;
        if (input.IsEmpty()) {
            retval |= WILD_GUESS_FROM_EMPTY_ADDRESS;
        }
        VLTRACE(Trace::LEVEL_VL_VERBOSE, "Parsed address: %s\n",
            outAddress.ToStringA().PeekBuffer());

    } catch (Exception e) {
        VLTRACE(Trace::LEVEL_VL_VERBOSE, "Parsing address \"%s\" failed: %s\n",
            W2A(input), e.GetMsgA());
        ASSERT((retval & WILD_GUESS_HAS_ADDRESS) == 0);

        outDevice = input;
        retval |= WILD_GUESS_HAS_DEVICE;
    }

    return retval;
}


/*
 * vislib::net::NetworkInformation::PENALTY_ADAPTER_DOWN
 */
const float vislib::net::NetworkInformation::PENALTY_ADAPTER_DOWN = 0.2f;


/*
 * vislib::net::NetworkInformation::PENALTY_EMPTY_ADDRESS 
 */
const float vislib::net::NetworkInformation::PENALTY_EMPTY_ADDRESS = 0.75f;


/*
 * vislib::net::NetworkInformation::PENALTY_WRONG_ADDRESSFAMILY 
 */
const float vislib::net::NetworkInformation::PENALTY_WRONG_ADDRESSFAMILY 
    = 0.02f;


/*
 * vislib::net::NetworkInformation::PENALTY_NO_PORT
 */
const float vislib::net::NetworkInformation::PENALTY_NO_PORT = 0.01f;


/*
 * vislib::net::NetworkInformation::PENALTY_WRONG_PREFIX
 */
const float vislib::net::NetworkInformation::PENALTY_WRONG_PREFIX = 0.15f;


/*
 * vislib::net::NetworkInformation::WILD_GUESS_FROM_EMPTY_ADDRESS 
 */
const UINT32 vislib::net::NetworkInformation::WILD_GUESS_FROM_EMPTY_ADDRESS
    = 0x20;


/* 
 * vislib::net::NetworkInformation::WILD_GUESS_HAS_ADDRESS 
 */
const UINT32 vislib::net::NetworkInformation::WILD_GUESS_HAS_ADDRESS = 0x02;


/*
 * vislib::net::NetworkInformation::WILD_GUESS_HAS_DEVICE 
 */
const UINT32 vislib::net::NetworkInformation::WILD_GUESS_HAS_DEVICE = 0x01;


/*
 * vislib::net::NetworkInformation::WILD_GUESS_HAS_NETMASK
 */
const UINT32 vislib::net::NetworkInformation::WILD_GUESS_HAS_NETMASK = 0x08;


/*
 * vislib::net::NetworkInformation::WILD_GUESS_HAS_PORT 
 */
const UINT32 vislib::net::NetworkInformation::WILD_GUESS_HAS_PORT = 0x10;


/*
 * vislib::net::NetworkInformation::WILD_GUESS_HAS_PREFIX_LEN 
 */
const UINT32 vislib::net::NetworkInformation::WILD_GUESS_HAS_PREFIX_LEN = 0x04;


/*
 * vislib::net::NetworkInformation::adapters
 */
vislib::net::NetworkInformation::AdapterList 
vislib::net::NetworkInformation::adapters;


/*
 * vislib::net::NetworkInformation::lockAdapters
 */
vislib::sys::CriticalSection vislib::net::NetworkInformation::lockAdapters;


/*
 * vislib::net::NetworkInformation::NetworkInformation
 */
vislib::net::NetworkInformation::NetworkInformation(void) {
    VLSTACKTRACE("NetworkInformation::NetworkInformation", __FILE__, __LINE__);
    throw UnsupportedOperationException(
        "NetworkInformation::NetworkInformation", __FILE__, __LINE__);
}


/*
 * vislib::net::NetworkInformation::NetworkInformation
 */
vislib::net::NetworkInformation::NetworkInformation(
        const NetworkInformation& rhs) {
    VLSTACKTRACE("NetworkInformation::NetworkInformation", __FILE__, __LINE__);
    throw UnsupportedOperationException(
        "NetworkInformation::NetworkInformation", __FILE__, __LINE__);
}


/*
 * vislib::net::NetworkInformation::~NetworkInformation
 */
vislib::net::NetworkInformation::~NetworkInformation(void) {
    VLSTACKTRACE("NetworkInformation::~NetworkInformation", __FILE__, __LINE__);
}
