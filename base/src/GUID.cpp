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

#include "the/assert.h"
#include "the/memory.h"
#include "the/text/string_converter.h"


/*
 * vislib::GUID::GUID
 */
vislib::GUID::GUID(void) {
    this->SetZero();
}


/*
 * vislib::GUID::GUID
 */
vislib::GUID::GUID(const uint8_t b[16]) {
    THE_ASSERT(b != NULL);
    THE_ASSERT(sizeof(this->guid) == 16 * sizeof(uint8_t));
#ifdef _WIN32
    ::memcpy(&this->guid, b, sizeof(this->guid));
#else /* _WIN32 */
    ::memcpy(this->guid, b, sizeof(this->guid));
#endif /* _WIN32 */
}


/*
 * vislib::GUID::GUID
 */
vislib::GUID::GUID(const uint8_t b1, const uint8_t b2, const uint8_t b3, const uint8_t b4,
        const uint8_t b5, const uint8_t b6, const uint8_t b7, const uint8_t b8,
        const uint8_t b9, const uint8_t b10, const uint8_t b11, const uint8_t b12,
        const uint8_t b13, const uint8_t b14, const uint8_t b15, const uint8_t b16) {
#ifdef _WIN32
#define ASSIGN_BYTE(i) (reinterpret_cast<uint8_t *>(&this->guid))[i - 1] = b##i
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
vislib::GUID::GUID(const uint32_t i, const uint16_t s1, const uint16_t s2,
        const uint8_t b1, const uint8_t b2, const uint8_t b3, const uint8_t b4,
        const uint8_t b5, const uint8_t b6, const uint8_t b7, const uint8_t b8) {
#ifdef _WIN32
    uint8_t *g = reinterpret_cast<uint8_t *>(&this->guid);
#else /* _WIN32 */
    uint8_t *g = this->guid;
#endif /* _WIN32 */
    const uint8_t *in = reinterpret_cast<const uint8_t *>(&i);
    g[0] = in[0];
    g[1] = in[1];
    g[2] = in[2];
    g[3] = in[3];

    in = reinterpret_cast<const uint8_t *>(&s1);
    g[4] = in[0];
    g[5] = in[1];

    in = reinterpret_cast<const uint8_t *>(&s2);
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
vislib::GUID::GUID(const uint32_t i, const uint16_t s1, const uint16_t s2,
        const uint8_t b[8]) {
    THE_ASSERT(b != NULL);
#ifdef _WIN32
    uint8_t *g = reinterpret_cast<uint8_t *>(&this->guid);
#else /* _WIN32 */
    uint8_t *g = this->guid;
#endif /* _WIN32 */
    const uint8_t *in = reinterpret_cast<const uint8_t *>(&i);
    g[0] = in[0];
    g[1] = in[1];
    g[2] = in[2];
    g[3] = in[3];

    in = reinterpret_cast<const uint8_t *>(&s1);
    g[4] = in[0];
    g[5] = in[1];

    in = reinterpret_cast<const uint8_t *>(&s2);
    g[6] = in[0];
    g[7] = in[1];

    ::memcpy(g + 8, b, 8 * sizeof(uint8_t));
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
    const uint8_t *g = reinterpret_cast<const uint8_t *>(&this->guid);
#else /* _WIN32 */
    const uint8_t *g = this->guid;
#endif /* _WIN32 */

    for (size_t i = 0; i < sizeof(this->guid); i++) {
        if (g[i] != 0) {
            return false;
        }
    }

    return true;
}


/*
 * vislib::GUID::Parse
 */
bool vislib::GUID::Parse(const the::astring& str) {
#ifdef _WIN32
    return (::UuidFromStringA(reinterpret_cast<RPC_CSTR>(const_cast<char *>(
        str.c_str())), &this->guid) == RPC_S_OK);
#else /* _WIN32 */
    return (::uuid_parse(str.c_str(), this->guid) == 0);
#endif /* _WIN32 */
}


/*
 * vislib::GUID::Parse
 */
bool vislib::GUID::Parse(const the::wstring& str) {
#ifdef _WIN32
    return (::UuidFromStringW(reinterpret_cast<RPC_WSTR>(const_cast<wchar_t *>(
        str.c_str())), &this->guid) == RPC_S_OK);
#else /* _WIN32 */
    return this->Parse(THE_W2A(str));
#endif /* _WIN32 */
}


/*
 * vislib::GUID::HashCode
 */
uint32_t vislib::GUID::HashCode(void) const {
    // DJB2 hash function
    uint32_t hash = 0;
    uint8_t c;
#ifdef _WIN32
    const uint8_t *str = reinterpret_cast<const uint8_t *>(&this->guid);
#else /* _WIN32 */
    const uint8_t *str = this->guid;
#endif /* _WIN32 */

    for (size_t i = 0; i < sizeof(this->guid); ++i) {
        c = str[i];
        hash = ((hash << 5) + hash) + static_cast<uint32_t>(c);
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
    the::zero_memory(this->guid, sizeof(this->guid));
#endif /* _WIN32 */
}


/*
 * vislib::GUID::ToStringA
 */
the::astring vislib::GUID::ToStringA(void) const {
#ifdef _WIN32
    RPC_CSTR str;
    if (::UuidToStringA(const_cast<::GUID *>(&this->guid), &str) == RPC_S_OK) {
        the::astring retval(reinterpret_cast<char *>(str));
        ::RpcStringFreeA(&str);
        return retval;
    } else {
        throw std::bad_alloc();
    }
#else /* _WIN32 */
    the::astring retval((36 + 1) * sizeof(char), ' ');
    ::uuid_unparse(this->guid, const_cast<char*>(retval.c_str()));
    return retval;
#endif /* _WIN32 */
}


/*
 * vislib::GUID::ToStringW
 */
the::wstring vislib::GUID::ToStringW(void) const {
#ifdef _WIN32
    RPC_WSTR str;
    if (::UuidToStringW(const_cast<::GUID *>(&this->guid), &str) == RPC_S_OK) {
        the::wstring retval(reinterpret_cast<wchar_t *>(str));
        ::RpcStringFreeW(&str);
        return retval;
    } else {
        throw std::bad_alloc();
    }

#else /* _WIN32 */
    return the::text::string_converter::to_w(this->ToStringA());
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
