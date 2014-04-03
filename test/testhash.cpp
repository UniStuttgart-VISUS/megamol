/*
 * testfile.h  14.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testhash.h"

#include <iostream>

#include "testhelper.h"
#include "vislib/MD5HashProvider.h"
#include "vislib/SHA1HashProvider.h"
#include "the/string.h"
#include "the/text/string_builder.h"


void TestMD5(void) {
    using namespace vislib;

    std::cout << std::endl << "MD5 ..." << std::endl;
    
    MD5HashProvider hash;
    const char *TEXT = "Horst";
    uint8_t hashValue[16];
    size_t hashSize = 16;

    AssertNoException("Initialisation in ctor",  hash.TransformFinalBlock(hashValue, hashSize, reinterpret_cast<const uint8_t *>(TEXT), ::strlen(TEXT)));

    AssertNoException("Initialise MD5", hash.Initialise());

    // Test reference values from RFC 1321:
    AssertNoException("Reinitialisation succeeds", hash.Initialise());
    TEXT = "";
    AssertNoException("MD5 \"\"", hash.TransformFinalBlock(hashValue, hashSize, reinterpret_cast<const uint8_t *>(TEXT), ::strlen(TEXT)));
    std::cout << "Hash is " << hash.ToStringA() << std::endl;
    std::cout << "As wide string " << the::text::string_converter::to_a(hash.ToStringW()) << std::endl;
    AssertEqualCaseInsensitive("MD5 string equal to reference", hash.ToStringA().c_str(), "d41d8cd98f00b204e9800998ecf8427e");

    AssertNoException("Reinitialisation succeeds", hash.Initialise());
    TEXT = "a";
    AssertNoException("MD5 \"a\"", hash.TransformFinalBlock(hashValue, hashSize, reinterpret_cast<const uint8_t *>(TEXT), ::strlen(TEXT)));
    std::cout << "Hash is " << hash.ToStringA() << std::endl;
    AssertEqualCaseInsensitive("MD5 string equal to reference", hash.ToStringA().c_str(), "0cc175b9c0f1b6a831c399e269772661");

    AssertNoException("Reinitialisation succeeds", hash.Initialise());
    TEXT = "abc";
    AssertNoException("MD5 \"abc\"", hash.TransformFinalBlock(hashValue, hashSize, reinterpret_cast<const uint8_t *>(TEXT), ::strlen(TEXT)));
    std::cout << "Hash is " << hash.ToStringA() << std::endl;
    AssertEqualCaseInsensitive("MD5 string equal to reference", hash.ToStringA().c_str(), "900150983cd24fb0d6963f7d28e17f72");

    AssertNoException("Reinitialisation succeeds", hash.Initialise());
    TEXT = "message digest";
    AssertNoException("MD5 \"message digest\"", hash.TransformFinalBlock(hashValue, hashSize, reinterpret_cast<const uint8_t *>(TEXT), ::strlen(TEXT)));
    std::cout << "Hash is " << hash.ToStringA() << std::endl;
    AssertEqualCaseInsensitive("MD5 string equal to reference", hash.ToStringA().c_str(), "f96b697d7cb7938d525a2f31aaf161d0");

    AssertNoException("Reinitialisation succeeds", hash.Initialise());
    TEXT = "abcdefghijklmnopqrstuvwxyz";
    AssertNoException("MD5 \"abcdefghijklmnopqrstuvwxyz\"", hash.TransformFinalBlock(hashValue, hashSize, reinterpret_cast<const uint8_t *>(TEXT), ::strlen(TEXT)));
    std::cout << "Hash is " << hash.ToStringA() << std::endl;
    AssertEqualCaseInsensitive("MD5 string equal to reference", hash.ToStringA().c_str(), "c3fcd3d76192e4007dfb496cca67e13b");

    AssertNoException("Reinitialisation succeeds", hash.Initialise());
    TEXT = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    AssertNoException("MD5 \"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\"", hash.TransformFinalBlock(hashValue, hashSize, reinterpret_cast<const uint8_t *>(TEXT), ::strlen(TEXT)));
    std::cout << "Hash is " << hash.ToStringA() << std::endl;
    AssertEqualCaseInsensitive("MD5 string equal to reference", hash.ToStringA().c_str(), "d174ab98d277d9f5a5611c2c9f419d9f");

    AssertNoException("Reinitialisation succeeds", hash.Initialise());
    TEXT = "12345678901234567890123456789012345678901234567890123456789012345678901234567890";
    AssertNoException("MD5 \"12345678901234567890123456789012345678901234567890123456789012345678901234567890\"", hash.TransformFinalBlock(hashValue, hashSize, reinterpret_cast<const uint8_t *>(TEXT), ::strlen(TEXT)));
    std::cout << "Hash is " << hash.ToStringA() << std::endl;
    AssertEqualCaseInsensitive("MD5 string equal to reference", hash.ToStringA().c_str(), "57edf4a22be3c955ac49da2e2107b67a");

    TEXT = "12345678901234567890123456789012345678901234567890123456789012345678901234567890";
    AssertNoException("MD5 \"12345678901234567890123456789012345678901234567890123456789012345678901234567890\"", hash.ComputeHash(hashValue, hashSize, reinterpret_cast<const uint8_t *>(TEXT), ::strlen(TEXT)));
    std::cout << "Hash is " << hash.ToStringA() << std::endl;
    AssertEqualCaseInsensitive("MD5 string equal to reference", hash.ToStringA().c_str(), "57edf4a22be3c955ac49da2e2107b67a");

    TEXT = "12345678901234567890123456789012345678901234567890123456789012345678901234567890";
    AssertNoException("MD5 \"12345678901234567890123456789012345678901234567890123456789012345678901234567890\"", hash.ComputeHash(hashValue, hashSize, TEXT));
    std::cout << "Hash is " << hash.ToStringA() << std::endl;
    AssertEqualCaseInsensitive("MD5 string equal to reference", hash.ToStringA().c_str(), "57edf4a22be3c955ac49da2e2107b67a");


    // strange code™
    MD5HashProvider hash2;
    hashSize = hash2.GetHashSize();
    uint8_t *hashVal = new uint8_t[hashSize];
    TEXT = "12345678901234567890123456789012345678901234567890123456789012345678901234567890";
    hash2.TransformFinalBlock(hashVal, hashSize, reinterpret_cast<const uint8_t*>(TEXT), ::strlen(TEXT));
//    hash2.ComputeHash(hashVal, hashSize, reinterpret_cast<const uint8_t*>(TEXT), ::strlen(TEXT));
//    hash2.GetHashValue(hashVal, hashSize); //<= with this line the code works! Why?
    the::astring hashStr, tmp;
    for (size_t i =0; i < hashSize; i++) {
        the::text::astring_builder::format_to(tmp, "%.2x", hashVal[i]);
        hashStr += tmp;
    }
    delete[] hashVal;
    //std::cout << "hash bytes " << hashStr << " hash.ToString " << hash.ToStringA().c_str();
    //AssertEqualCaseInsensitive("MD5 string equal to reference", hashStr, hash.ToStringA());
    AssertEqualCaseInsensitive("MD5 string equal to reference", hashStr.c_str(), "57edf4a22be3c955ac49da2e2107b67a");
}


void TestSHA1(void) {
    using namespace vislib;

    std::cout << std::endl << "SHA-1 ..." << std::endl;
    
    SHA1HashProvider hash;
    uint8_t hashValue[20];
    size_t hashSize = 20;

    // SHA-1 tests from RFC 3174
    const char *TEST1 = "abc";
    const char *TEST2 = "abcdbcdecdefdefgefghfghighijhi""jkijkljklmklmnlmnomnopnopq";
    const char *TEST3 = "a";
    const char *TEST4 = "01234567012345670123456701234567" "01234567012345670123456701234567";
    const char *testarray[4] = { TEST1, TEST2, TEST3, TEST4 };
    long int repeatcount[4] = { 1, 1, 1000000, 10 };
    const char *resultarray[4] = { 
        "A9993E364706816ABA3E25717850C26C9CD0D89D",
        "84983E441C3BD26EBAAE4AA1F95129E5E54670F1",
        "34AA973CD4C4DAA4F61EEB2BDBAD27316534016F",
        "DEA356A2CDDD90C7A7ECEDC5EBB563934F460452" 
    };

    for (int j = 0; j < 4; ++j) {
        AssertNoException("Reinitialisation succeeds", hash.Initialise());

        for (int i = 0; i < repeatcount[j]; ++i) {
            //AssertNoException("Transform block", hash.TransformBlock(reinterpret_cast<uint8_t *>(testarray[j]), ::strlen(testarray[j])));
            hash.TransformBlock(reinterpret_cast<const uint8_t *>(testarray[j]), ::strlen(testarray[j]));
        }

        AssertNoException("Transform final NULL block", hash.TransformFinalBlock(hashValue, hashSize, NULL, 0));
        std::cout << "Hash is " << hash.ToStringA() << std::endl;
        AssertEqualCaseInsensitive("SHA-1 string equal to reference", hash.ToStringA().c_str(), resultarray[j]);
    }

    // strange code™
    SHA1HashProvider hash2;
    hashSize = hash2.GetHashSize();
    uint8_t *hashVal = new uint8_t[hashSize];
    const char *TEXT = "01234567012345670123456701234567" "01234567012345670123456701234567";
    hash2.TransformFinalBlock(hashVal, hashSize, reinterpret_cast<const uint8_t*>(TEXT), ::strlen(TEXT));
//    hash2.ComputeHash(hashVal, hashSize, reinterpret_cast<const uint8_t*>(TEXT), ::strlen(TEXT));
//    hash2.GetHashValue(hashVal, hashSize); //<= with this line the code works! Why?
    the::astring hashStr, tmp;
    for (size_t i = 0; i < hashSize; i++) {
        the::text::astring_builder::format_to(tmp, "%.2x", hashVal[i]);
        hashStr += tmp;
    }
    delete[] hashVal;
    //std::cout << "hash bytes " << hashStr << " hash.ToString " << hash.ToStringA().c_str();
    //AssertEqualCaseInsensitive("MD5 string equal to reference", hashStr, hash.ToStringA());
    AssertEqualCaseInsensitive("SHA-1 string equal to reference", hashStr.c_str(), "DEA356A2CDDD90C7A7ECEDC5EBB563934F460452");
}


void TestHash(void) {
    ::TestMD5();
    ::TestSHA1();
}
