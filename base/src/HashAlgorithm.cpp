/*
 * HashAlgorithm.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/HashAlgorithm.h"

#include "vislib/assert.h"


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
bool vislib::HashAlgorithm::ComputeHash(BYTE *outHash, SIZE_T& inOutSize, 
        const BYTE *input, const SIZE_T cntInput) {
    this->Initialise();
    this->TransformFinalBlock(outHash, inOutSize, input, cntInput);
    // Fix for ticket #64. It is, however, unclear why this is required.
    return this->GetHashValue(outHash, inOutSize);
}


/*
 * vislib::HashAlgorithm::ComputeHash
 */
bool vislib::HashAlgorithm::ComputeHash(BYTE *outHash, SIZE_T& inOutSize, 
        const char *input) {
    return this->ComputeHash(outHash, inOutSize, 
        reinterpret_cast<const BYTE *>(input), ::strlen(input));
}


/*
 * vislib::HashAlgorithm::ComputeHash
 */
bool vislib::HashAlgorithm::ComputeHash(BYTE *outHash, SIZE_T& inOutSize, 
        const wchar_t *input) {
    return this->ComputeHash(outHash, inOutSize, 
        reinterpret_cast<const BYTE *>(input), 
        ::wcslen(input) * sizeof(wchar_t));
}


/*
 * vislib::HashAlgorithm::GetHashSize
 */ 
SIZE_T vislib::HashAlgorithm::GetHashSize(void) const {
    SIZE_T retval;
    const_cast<HashAlgorithm *>(this)->TransformFinalBlock(NULL, retval, NULL, 0);
    return retval;
}


/*
 * vislib::HashAlgorithm::GetHashValue
 */
bool vislib::HashAlgorithm::GetHashValue(BYTE *outHash, 
                                         SIZE_T& inOutSize) const {
    return const_cast<HashAlgorithm *>(this)->TransformFinalBlock(outHash, inOutSize, NULL, 0);
}


/*
 * vislib::HashAlgorithm::ToStringA
 */
vislib::StringA vislib::HashAlgorithm::ToStringA(void) const {
    BYTE *hash = NULL;
    StringA::Char *out = NULL;
    SIZE_T hashSize = 0;
    StringA retval;
    StringA tmp(' ', 2);

    try {
        this->GetHashValue(hash, hashSize);
        hash = new BYTE[hashSize];
        this->GetHashValue(hash, hashSize);

        out = retval.AllocateBuffer(2 * static_cast<StringA::Size>(hashSize));
        ASSERT(out[2 * hashSize] == 0);

        for (SIZE_T i = 0; i < hashSize; i++) {
            tmp.Format("%02x", hash[i]);
            *out++ = tmp[0];
            *out++ = tmp[1];
        }
    } catch (...) {
    }

    ARY_SAFE_DELETE(hash);
    return retval;
}


/*
 * vislib::HashAlgorithm::ToStringW
 */
vislib::StringW vislib::HashAlgorithm::ToStringW(void) const {
    BYTE *hash = NULL;
    StringW::Char *out = NULL;
    SIZE_T hashSize = 0;
    StringW retval;
    StringW tmp(L' ', 2);

    try {
        this->GetHashValue(hash, hashSize);
        hash = new BYTE[hashSize];
        this->GetHashValue(hash, hashSize);

        out = retval.AllocateBuffer(2 * static_cast<StringW::Size>(hashSize));
        ASSERT(out[2 * hashSize] == 0);

        for (SIZE_T i = 0; i < hashSize; i++) {
            tmp.Format(L"%02x", hash[i]);
            *out++ = tmp[0];
            *out++ = tmp[1];
        }
    } catch (...) {
    }

    ARY_SAFE_DELETE(hash);
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
