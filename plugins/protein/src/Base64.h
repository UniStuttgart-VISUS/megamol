//
// Base64.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 12, 2013
//     Author: scharnkn
//

#pragma once

#include <cstddef>

namespace megamol::protein {

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

} // namespace megamol::protein
