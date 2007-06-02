/*
 * testfile.h  14.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testhash.h"

#include <iostream>

#include "testhelper.h"
#include "vislib/MD5HashProvider.h"
#include "vislib/IllegalStateException.h"


void TestHash(void) {
    using namespace vislib;
    
    MD5HashProvider hash;
    char *TEXT = "Horst";

    AssertException("Cannot use uninitialised hash",  hash.TransformFinalBlock(reinterpret_cast<const BYTE *>(TEXT), ::strlen(TEXT)), IllegalStateException);

    AssertNoException("Initialise MD5", hash.Initialise());

    // Test reference values from RFC 1321:
    AssertNoException("Reinitialisation succeeds", hash.Initialise());
    TEXT = "";
    AssertNoException("MD5 \"\"", hash.TransformFinalBlock(reinterpret_cast<const BYTE *>(TEXT), ::strlen(TEXT)));
    std::cout << "Hash is " << hash.ToStringA() << std::endl;
    std::cout << "As wide string " << StringA(hash.ToStringW()) << std::endl;
    AssertEqualCaseInsensitive("MD5 string equal to reference", hash.ToStringA(), "d41d8cd98f00b204e9800998ecf8427e");

    AssertException("Cannot update finalised hash",  hash.TransformFinalBlock(reinterpret_cast<const BYTE *>(TEXT), ::strlen(TEXT)), IllegalStateException);

    AssertNoException("Reinitialisation succeeds", hash.Initialise());
    TEXT = "a";
    AssertNoException("MD5 \"a\"", hash.TransformFinalBlock(reinterpret_cast<const BYTE *>(TEXT), ::strlen(TEXT)));
    std::cout << "Hash is " << hash.ToStringA() << std::endl;
    AssertEqualCaseInsensitive("MD5 string equal to reference", hash.ToStringA(), "0cc175b9c0f1b6a831c399e269772661");

    AssertNoException("Reinitialisation succeeds", hash.Initialise());
    TEXT = "abc";
    AssertNoException("MD5 \"abc\"", hash.TransformFinalBlock(reinterpret_cast<const BYTE *>(TEXT), ::strlen(TEXT)));
    std::cout << "Hash is " << hash.ToStringA() << std::endl;
    AssertEqualCaseInsensitive("MD5 string equal to reference", hash.ToStringA(), "900150983cd24fb0d6963f7d28e17f72");

    AssertNoException("Reinitialisation succeeds", hash.Initialise());
    TEXT = "message digest";
    AssertNoException("MD5 \"message digest\"", hash.TransformFinalBlock(reinterpret_cast<const BYTE *>(TEXT), ::strlen(TEXT)));
    std::cout << "Hash is " << hash.ToStringA() << std::endl;
    AssertEqualCaseInsensitive("MD5 string equal to reference", hash.ToStringA(), "f96b697d7cb7938d525a2f31aaf161d0");

    AssertNoException("Reinitialisation succeeds", hash.Initialise());
    TEXT = "abcdefghijklmnopqrstuvwxyz";
    AssertNoException("MD5 \"abcdefghijklmnopqrstuvwxyz\"", hash.TransformFinalBlock(reinterpret_cast<const BYTE *>(TEXT), ::strlen(TEXT)));
    std::cout << "Hash is " << hash.ToStringA() << std::endl;
    AssertEqualCaseInsensitive("MD5 string equal to reference", hash.ToStringA(), "c3fcd3d76192e4007dfb496cca67e13b");

    AssertNoException("Reinitialisation succeeds", hash.Initialise());
    TEXT = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    AssertNoException("MD5 \"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\"", hash.TransformFinalBlock(reinterpret_cast<const BYTE *>(TEXT), ::strlen(TEXT)));
    std::cout << "Hash is " << hash.ToStringA() << std::endl;
    AssertEqualCaseInsensitive("MD5 string equal to reference", hash.ToStringA(), "d174ab98d277d9f5a5611c2c9f419d9f");

    AssertNoException("Reinitialisation succeeds", hash.Initialise());
    TEXT = "12345678901234567890123456789012345678901234567890123456789012345678901234567890";
    AssertNoException("MD5 \"12345678901234567890123456789012345678901234567890123456789012345678901234567890\"", hash.TransformFinalBlock(reinterpret_cast<const BYTE *>(TEXT), ::strlen(TEXT)));
    std::cout << "Hash is " << hash.ToStringA() << std::endl;
    AssertEqualCaseInsensitive("MD5 string equal to reference", hash.ToStringA(), "57edf4a22be3c955ac49da2e2107b67a");

}
