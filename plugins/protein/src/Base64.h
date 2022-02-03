//
// Base64.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 12, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINPLUGIN_BASE64_H_INCLUDED
#define MMPROTEINPLUGIN_BASE64_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

namespace megamol {
namespace protein {

class Base64 {

public:
    static const char MapEncode[64];

    static const char MapDecode[128];

    /**
     * Encodes a byte array using base 64 encoding.
     *
     * @param input The data to be encoded.
     * @param output The array holding the output
     * @param s The size of the input array in bytes
     */
    static void Encode(const char* input, char* output, size_t s);

    /**
     * Decodes a byte array encoded with base 64 encoding.
     *
     * @param input The data to be decoded.
     * @param output The array holding the output
     * @param s The size of the output array in bytes
     */
    static void Decode(const char* input, char* output, size_t s);
};

} // end namespace protein
} // end namespace megamol

#endif // MMPROTEINPLUGIN_BASE64_H_INCLUDED
