//
// Base64.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 12, 2013
//     Author: scharnkn
//

#include "Base64.h"
#include "stdafx.h"

using namespace megamol::protein;

typedef unsigned int uint;

/// Encodes numbers between 0 .. 63 into the according one byte representation
const char Base64::MapEncode[64] = {0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E,
    0x4F, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67,
    0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A,
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x2B, 0x2F};

/// Decodes ascii representation of a letter to the respective number
/// between 0 .. 63
const char Base64::MapDecode[128] = {
    // 0 .. 43 are zeros
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00,
    0x3E, // # 43 '+' <=> 62
    // 44 .. 46 are zeros
    0x00, 0x00, 0x00,
    0x3F, // # 47 '/' <=> 63
    0x34, // # 48 '0' <=> 52
    0x35, // # 49 '1' <=> 53
    0x36, // # 50 '2' <=> 54
    0x37, // # 51 '3' <=> 55
    0x38, // # 52 '4' <=> 56
    0x39, // # 53 '5' <=> 57
    0x3A, // # 54 '6' <=> 58
    0x3B, // # 55 '7' <=> 59
    0x3C, // # 56 '8' <=> 60
    0x3D, // # 57 '9' <=> 61
    // 58 .. 64 are zeros
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, //# 65 'A' <=> 0
    0x01, //# 66 'B' <=> 1
    0x02, //# 67 'C' <=> 2
    0x03, //# 68 'D' <=> 3
    0x04, //# 69 'E' <=> 4
    0x05, //# 70 'F' <=> 5
    0x06, //# 71 'G' <=> 6
    0x07, //# 72 'H' <=> 7
    0x08, //# 73 'I' <=> 8
    0x09, //# 74 'J' <=> 9
    0x0A, //# 75 'K' <=> 10
    0x0B, //# 76 'L' <=> 11
    0x0C, //# 77 'M' <=> 12
    0x0D, //# 78 'N' <=> 13
    0x0E, //# 79 'O' <=> 14
    0x0F, //# 80 'P' <=> 15
    0x10, //# 81 'Q' <=> 16
    0x11, //# 82 'R' <=> 17
    0x12, //# 83 'S' <=> 18
    0x13, //# 84 'T' <=> 19
    0x14, //# 85 'U' <=> 20
    0x15, //# 86 'V' <=> 21
    0x16, //# 87 'W' <=> 22
    0x17, //# 88 'X' <=> 23
    0x18, //# 89 'Y' <=> 24
    0x19, //# 90 'Z' <=> 25
    // 91 - 96 are zeros
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x1A, // #097 'a' <=> 26
    0x1B, // #098 'b' <=> 27
    0x1C, // #099 'c' <=> 28
    0x1D, // #100 'd' <=> 29
    0x1E, // #101 'e' <=> 30
    0x1F, // #102 'f' <=> 31
    0x20, // #103 'g' <=> 32
    0x21, // #104 'h' <=> 33
    0x22, // #105 'i' <=> 34
    0x23, // #106 'j' <=> 35
    0x24, // #107 'k' <=> 36
    0x25, // #108 'l' <=> 37
    0x26, // #109 'm' <=> 38
    0x27, // #110 'n' <=> 39
    0x28, // #111 'o' <=> 40
    0x29, // #112 'p' <=> 41
    0x2A, // #113 'q' <=> 42
    0x2B, // #114 'r' <=> 43
    0x2C, // #115 's' <=> 44
    0x2D, // #116 't' <=> 45
    0x2E, // #117 'u' <=> 46
    0x2F, // #118 'v' <=> 47
    0x30, // #119 'w' <=> 48
    0x31, // #120 'x' <=> 49
    0x32, // #121 'y' <=> 50
    0x33, // #122 'z' <=> 51
    // 123 .. 127 are zeros
    0x00, 0x00, 0x00, 0x00, 0x00};

/*
 * Base64::Encode
 */
void Base64::Encode(const char* input, char* output, size_t s) {

    char bytes[3];
    uint nfillers;

    // All byte triples contained in the data
    for (uint cnt = 0; cnt < s / 3; ++cnt) {
        output[4 * cnt + 0] = MapEncode[(input[cnt * 3 + 0] & 0xfc) >> 2];
        output[4 * cnt + 1] = MapEncode[((input[cnt * 3 + 0] & 0x03) << 4) + ((input[cnt * 3 + 1] & 0xf0) >> 4)];
        output[4 * cnt + 2] = MapEncode[((input[cnt * 3 + 1] & 0x0f) << 2) + ((input[cnt * 3 + 2] & 0xc0) >> 6)];
        output[4 * cnt + 3] = MapEncode[input[cnt * 3 + 2] & 0x3f];
    }

    // Last byte triple (potentially containing one or two filler bytes). 6-bit
    // bundles that only contain bits from filler bytes are encoded by '=', the
    // number of '=', therefore, indicates the number of filler bytes used
    nfillers = (3 - s % 3) % 3;
    memcpy(&bytes[0], input + s / 3, s % 3); // Copy valid bytes from the actual data
    memset(&bytes[s % 3], 0, nfillers);      // Set filler bytes
    output[4 * int(s / 3) + 0] = MapEncode[(bytes[0] & 0xfc) >> 2];
    output[4 * int(s / 3) + 1] = MapEncode[((bytes[0] & 0x03) << 4) + ((bytes[1] & 0xf0) >> 4)];
    if (nfillers > 1) {
        output[4 * int(s / 3) + 2] = '=';
    } else {
        output[4 * int(s / 3) + 2] = MapEncode[((bytes[1] & 0x0f) << 2) + ((bytes[2] & 0xc0) >> 6)];
    }
    if (nfillers > 0) {
        output[4 * int(s / 3) + 3] = '=';
    } else {
        output[4 * int(s / 3) + 3] = MapEncode[bytes[2] & 0x3f];
    }
}

/*
 * Base64::Decode
 */
void Base64::Decode(const char* input, char* output, size_t s) {
    char bytes[4];
    uint nFillers;
    // Decode byte triples
    for (unsigned int cnt = 0; cnt < s / 3; cnt++) {
        char bytes[] = {MapDecode[static_cast<int>(input[cnt * 4 + 0])],
            MapDecode[static_cast<int>(input[cnt * 4 + 1])], MapDecode[static_cast<int>(input[cnt * 4 + 2])],
            MapDecode[static_cast<int>(input[cnt * 4 + 3])]};
        output[3 * cnt + 0] = (bytes[0] << 2) + ((bytes[1] & 0x30) >> 4);
        output[3 * cnt + 1] = ((bytes[1] & 0xf) << 4) + ((bytes[2] & 0x3c) >> 2);
        output[3 * cnt + 2] = ((bytes[2] & 0x3) << 6) + bytes[3];
    }
    bytes[0] = MapDecode[static_cast<int>(input[int(s / 3) * 4 + 0])];
    bytes[1] = MapDecode[static_cast<int>(input[int(s / 3) * 4 + 1])];
    bytes[2] = MapDecode[static_cast<int>(input[int(s / 3) * 4 + 2])];
    bytes[3] = MapDecode[static_cast<int>(input[int(s / 3) * 4 + 3])];
    // Decode last one or two bytes
    nFillers = (3 - s % 3) % 3;
    if (nFillers == 2) { // Decode last byte
        output[3 * int(s / 3) + 0] = (bytes[0] << 2) + ((bytes[1] & 0x30) >> 4);
    } else if (nFillers == 1) { // Decode last two bytes
        output[3 * int(s / 3) + 0] = (bytes[0] << 2) + ((bytes[1] & 0x30) >> 4);
        output[3 * int(s / 3) + 1] = ((bytes[1] & 0xf) << 4) + ((bytes[2] & 0x3c) >> 2);
    }
}
