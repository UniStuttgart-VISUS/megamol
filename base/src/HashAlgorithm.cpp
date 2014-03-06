/*
 * HashAlgorithm.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/HashAlgorithm.h"

#include "the/assert.h"


/*
 * vislib::HashAlgorithm::HashAlgorithm
 */
vislib::HashAlgorithm::HashAlgorithm(void) {
}


/*
 * vislib::HashAlgorithm::~HashAlgorithm
 */
vislib::HashAlgorithm::~HashAlgorithm(void) {
}


/*
 * vislib::HashAlgorithm::ComputeHash
 */
bool vislib::HashAlgorithm::ComputeHash(uint8_t *outHash, size_t& inOutSize, 
        const uint8_t *input, const size_t cntInput) {
    this->Initialise();
    this->TransformFinalBlock(outHash, inOutSize, input, cntInput);
    // Fix for ticket #64. It is, however, unclear why this is required.
    return this->GetHashValue(outHash, inOutSize);
}


/*
 * vislib::HashAlgorithm::ComputeHash
 */
bool vislib::HashAlgorithm::ComputeHash(uint8_t *outHash, size_t& inOutSize, 
        const char *input) {
    return this->ComputeHash(outHash, inOutSize, 
        reinterpret_cast<const uint8_t *>(input), ::strlen(input));
}


/*
 * vislib::HashAlgorithm::ComputeHash
 */
bool vislib::HashAlgorithm::ComputeHash(uint8_t *outHash, size_t& inOutSize, 
        const wchar_t *input) {
    return this->ComputeHash(outHash, inOutSize, 
        reinterpret_cast<const uint8_t *>(input), 
        ::wcslen(input) * sizeof(wchar_t));
}


/*
 * vislib::HashAlgorithm::GetHashSize
 */ 
size_t vislib::HashAlgorithm::GetHashSize(void) const {
    size_t retval;
    const_cast<HashAlgorithm *>(this)->TransformFinalBlock(NULL, retval, NULL, 0);
    return retval;
}


/*
 * vislib::HashAlgorithm::GetHashValue
 */
bool vislib::HashAlgorithm::GetHashValue(uint8_t *outHash, 
                                         size_t& inOutSize) const {
    return const_cast<HashAlgorithm *>(this)->TransformFinalBlock(outHash, inOutSize, NULL, 0);
}


/*
 * vislib::HashAlgorithm::ToStringA
 */
vislib::StringA vislib::HashAlgorithm::ToStringA(void) const {
    uint8_t *hash = NULL;
    StringA::Char *out = NULL;
    size_t hashSize = 0;
    StringA retval;
    StringA tmp(' ', 2);

    try {
        this->GetHashValue(hash, hashSize);
        hash = new uint8_t[hashSize];
        this->GetHashValue(hash, hashSize);

        out = retval.AllocateBuffer(2 * static_cast<StringA::Size>(hashSize));
        THE_ASSERT(out[2 * hashSize] == 0);

        for (size_t i = 0; i < hashSize; i++) {
            tmp.Format("%02x", hash[i]);
            *out++ = tmp[0];
            *out++ = tmp[1];
        }
    } catch (...) {
    }

    the::safe_array_delete(hash);
    return retval;
}


/*
 * vislib::HashAlgorithm::ToStringW
 */
vislib::StringW vislib::HashAlgorithm::ToStringW(void) const {
    uint8_t *hash = NULL;
    StringW::Char *out = NULL;
    size_t hashSize = 0;
    StringW retval;
    StringW tmp(L' ', 2);

    try {
        this->GetHashValue(hash, hashSize);
        hash = new uint8_t[hashSize];
        this->GetHashValue(hash, hashSize);

        out = retval.AllocateBuffer(2 * static_cast<StringW::Size>(hashSize));
        THE_ASSERT(out[2 * hashSize] == 0);

        for (size_t i = 0; i < hashSize; i++) {
            tmp.Format(L"%02x", hash[i]);
            *out++ = tmp[0];
            *out++ = tmp[1];
        }
    } catch (...) {
    }

    the::safe_array_delete(hash);
    return retval;
}


/*
 * vislib::HashAlgorithm::HashAlgorithm
 */
vislib::HashAlgorithm::HashAlgorithm(const HashAlgorithm& rhs) {
}


/*
 * vislib::HashAlgorithm::operator =
 */
vislib::HashAlgorithm& vislib::HashAlgorithm::operator =(
        const HashAlgorithm& rhs) {
    return *this;
}
