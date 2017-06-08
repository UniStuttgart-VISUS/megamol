/*
 * GUID.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/GUID.h"

#include <stdexcept>

#ifdef _WIN32
#include <rpc.h>
#endif /* _WIN32 */

#include "vislib/assert.h"
#include "vislib/memutils.h"
#include "vislib/StringConverter.h"


/*
 * vislib::GUID::GUID
 */
vislib::GUID::GUID(void) {
    this->SetZero();
}


/*
 * vislib::GUID::GUID
 */
vislib::GUID::GUID(const BYTE b[16]) {
    ASSERT(b != NULL);
    ASSERT(sizeof(this->guid) == 16 * sizeof(BYTE));
#ifdef _WIN32
    ::memcpy(&this->guid, b, sizeof(this->guid));
#else /* _WIN32 */
    ::memcpy(this->guid, b, sizeof(this->guid));
#endif /* _WIN32 */
}


/*
 * vislib::GUID::GUID
 */
vislib::GUID::GUID(const BYTE b1, const BYTE b2, const BYTE b3, const BYTE b4,
        const BYTE b5, const BYTE b6, const BYTE b7, const BYTE b8,
        const BYTE b9, const BYTE b10, const BYTE b11, const BYTE b12,
        const BYTE b13, const BYTE b14, const BYTE b15, const BYTE b16) {
#ifdef _WIN32
#define ASSIGN_BYTE(i) (reinterpret_cast<BYTE *>(&this->guid))[i - 1] = b##i
#else /* _WIN32 */
#define ASSIGN_BYTE(i) (this->guid)[i - 1] = b##i
#endif /* _WIN32 */
    ASSIGN_BYTE(1);
    ASSIGN_BYTE(2);
    ASSIGN_BYTE(3);
    ASSIGN_BYTE(4);
    ASSIGN_BYTE(5);
    ASSIGN_BYTE(6);
    ASSIGN_BYTE(7);
    ASSIGN_BYTE(8);
    ASSIGN_BYTE(9);
    ASSIGN_BYTE(10);
    ASSIGN_BYTE(11);
    ASSIGN_BYTE(12);
    ASSIGN_BYTE(13);
    ASSIGN_BYTE(14);
    ASSIGN_BYTE(15);
    ASSIGN_BYTE(16);
#undef ASSIGN_BYTE
}


/*
 * vislib::GUID::GUID
 */
vislib::GUID::GUID(const UINT32 i, const UINT16 s1, const UINT16 s2,
        const BYTE b1, const BYTE b2, const BYTE b3, const BYTE b4,
        const BYTE b5, const BYTE b6, const BYTE b7, const BYTE b8) {
#ifdef _WIN32
    BYTE *g = reinterpret_cast<BYTE *>(&this->guid);
#else /* _WIN32 */
    BYTE *g = this->guid;
#endif /* _WIN32 */
    const BYTE *in = reinterpret_cast<const BYTE *>(&i);
    g[0] = in[0];
    g[1] = in[1];
    g[2] = in[2];
    g[3] = in[3];

    in = reinterpret_cast<const BYTE *>(&s1);
    g[4] = in[0];
    g[5] = in[1];

    in = reinterpret_cast<const BYTE *>(&s2);
    g[6] = in[0];
    g[7] = in[1];

    g[8] = b1;
    g[9] = b2;
    g[10] = b3;
    g[11] = b4;
    g[12] = b5;
    g[13] = b6;
    g[14] = b7;
    g[15] = b8;
}



/*
 * vislib::GUID::GUID
 */
vislib::GUID::GUID(const UINT32 i, const UINT16 s1, const UINT16 s2,
        const BYTE b[8]) {
    ASSERT(b != NULL);
#ifdef _WIN32
    BYTE *g = reinterpret_cast<BYTE *>(&this->guid);
#else /* _WIN32 */
    BYTE *g = this->guid;
#endif /* _WIN32 */
    const BYTE *in = reinterpret_cast<const BYTE *>(&i);
    g[0] = in[0];
    g[1] = in[1];
    g[2] = in[2];
    g[3] = in[3];

    in = reinterpret_cast<const BYTE *>(&s1);
    g[4] = in[0];
    g[5] = in[1];

    in = reinterpret_cast<const BYTE *>(&s2);
    g[6] = in[0];
    g[7] = in[1];

    ::memcpy(g + 8, b, 8 * sizeof(BYTE));
}


/*
 * vislib::GUID::~GUID
 */
vislib::GUID::~GUID(void) {
    // Nothing to do.
}


/*
 * vislib::GUID::Create
 */
bool vislib::GUID::Create(const bool doNotUseMacAddress) {
    if (doNotUseMacAddress) {
#ifdef _WIN32
        RPC_STATUS status = ::UuidCreate(&this->guid);
        return ((status == RPC_S_OK) || (status == RPC_S_UUID_LOCAL_ONLY));

#else /* _WIN32 */
        ::uuid_generate_random(this->guid);
        return true;
#endif /* _WIN32 */

    } else {

#ifdef _WIN32
        RPC_STATUS status = ::UuidCreateSequential(&this->guid);
        return ((status == RPC_S_OK) || (status == RPC_S_UUID_LOCAL_ONLY));

#else /* _WIN32 */
        ::uuid_generate_time(this->guid);
        return true;
#endif /* _WIN32 */
    }
}


/*
 * vislib::GUID::IsZero
 */
bool vislib::GUID::IsZero(void) const {
#ifdef _WIN32
    const BYTE *g = reinterpret_cast<const BYTE *>(&this->guid);
#else /* _WIN32 */
    const BYTE *g = this->guid;
#endif /* _WIN32 */

    for (SIZE_T i = 0; i < sizeof(this->guid); i++) {
        if (g[i] != 0) {
            return false;
        }
    }

    return true;
}


/*
 * vislib::GUID::Parse
 */
bool vislib::GUID::Parse(const StringA& str) {
#ifdef _WIN32
    return (::UuidFromStringA(reinterpret_cast<RPC_CSTR>(const_cast<char *>(
        str.PeekBuffer())), &this->guid) == RPC_S_OK);
#else /* _WIN32 */
    return (::uuid_parse(str.PeekBuffer(), this->guid) == 0);
#endif /* _WIN32 */
}


/*
 * vislib::GUID::Parse
 */
bool vislib::GUID::Parse(const StringW& str) {
#ifdef _WIN32
    return (::UuidFromStringW(reinterpret_cast<RPC_WSTR>(const_cast<wchar_t *>(
        str.PeekBuffer())), &this->guid) == RPC_S_OK);
#else /* _WIN32 */
    return this->Parse(W2A(str));
#endif /* _WIN32 */
}


/*
 * vislib::GUID::HashCode
 */
UINT32 vislib::GUID::HashCode(void) const {
    // DJB2 hash function
    UINT32 hash = 0;
    BYTE c;
#ifdef _WIN32
    const BYTE *str = reinterpret_cast<const BYTE *>(&this->guid);
#else /* _WIN32 */
    const BYTE *str = this->guid;
#endif /* _WIN32 */

    for (size_t i = 0; i < sizeof(this->guid); ++i) {
        c = str[i];
        hash = ((hash << 5) + hash) + static_cast<UINT32>(c);
    }

    return hash;
}


/*
 * vislib::GUID::SetZero
 */
void vislib::GUID::SetZero(void) {
#ifdef _WIN32
    ::ZeroMemory(&this->guid, sizeof(this->guid));
#else /* _WIN32 */
    ::ZeroMemory(this->guid, sizeof(this->guid));
#endif /* _WIN32 */
}


/*
 * vislib::GUID::ToStringA
 */
vislib::StringA vislib::GUID::ToStringA(void) const {
#ifdef _WIN32
    RPC_CSTR str;
    if (::UuidToStringA(const_cast<::GUID *>(&this->guid), &str) == RPC_S_OK) {
        StringA retval(reinterpret_cast<char *>(str));
        ::RpcStringFreeA(&str);
        return retval;
    } else {
        throw std::bad_alloc();
    }
#else /* _WIN32 */
    StringA retval;
    ::uuid_unparse(this->guid, retval.AllocateBuffer((36 + 1) * sizeof(char)));
    return retval;
#endif /* _WIN32 */
}


/*
 * vislib::GUID::ToStringW
 */
vislib::StringW vislib::GUID::ToStringW(void) const {
#ifdef _WIN32
    RPC_WSTR str;
    if (::UuidToStringW(const_cast<::GUID *>(&this->guid), &str) == RPC_S_OK) {
        StringW retval(reinterpret_cast<wchar_t *>(str));
        ::RpcStringFreeW(&str);
        return retval;
    } else {
        throw std::bad_alloc();
    }

#else /* _WIN32 */
    return StringW(this->ToStringA());
#endif /* _WIN32 */
}


/*
 * vislib::GUID::operator =
 */
vislib::GUID& vislib::GUID::operator =(const GUID& rhs) {
    if (this != &rhs) {
#ifdef _WIN32
        ::memcpy(&this->guid, &rhs.guid, sizeof(this->guid));
#else /* _WIN32 */
        ::memcpy(this->guid, rhs.guid, sizeof(this->guid));
#endif /* _WIN32 */
    }

    return *this;
}


#ifdef _WIN32
/*
 * vislib::GUID::operator =
 */
vislib::GUID& vislib::GUID::operator =(const ::GUID& rhs) {
    if (&this->guid != &rhs) {
        ::memcpy(&this->guid, &rhs, sizeof(this->guid));
    }

    return *this;
}


/*
 * vislib::GUID::operator =
 */
vislib::GUID& vislib::GUID::operator =(const ::GUID *rhs) {
    if (&this->guid != rhs) {
        ::memcpy(&this->guid, rhs, sizeof(this->guid));
    }

    return *this;
}

#else /* _WIN32 */
/*
 * vislib::GUID::operator =
 */
vislib::GUID& vislib::GUID::operator =(const uuid_t& rhs) {
    if (this->guid != rhs) {
        ::memcpy(&this->guid, rhs, sizeof(this->guid));
    }

    return *this;
}
#endif /* _WIN32 */
