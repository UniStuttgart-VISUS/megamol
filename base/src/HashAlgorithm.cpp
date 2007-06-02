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
 * vislib::HashAlgorithm::ToStringA
 */
vislib::StringA vislib::HashAlgorithm::ToStringA(void) const {
    BYTE *hash = NULL;
    StringA::Char *out = NULL;
    SIZE_T hashSize = 0;
    StringA retval;
    StringA tmp(' ', 2);

    try {
        this->GetHash(hash, hashSize);
        hash = new BYTE[hashSize];
        this->GetHash(hash, hashSize);

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
        this->GetHash(hash, hashSize);
        hash = new BYTE[hashSize];
        this->GetHash(hash, hashSize);

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
