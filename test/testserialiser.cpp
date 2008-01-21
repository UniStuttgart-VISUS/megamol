/*
 * testserialiser.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "testserialiser.h"
#include "testhelper.h"

#include "vislib/Serialiser.h"
#include "vislib/RegistrySerialiser.h"



void TestSerialiser(void) {
#ifdef _WIN32
    using vislib::sys::RegistrySerialiser;
    using vislib::StringA;
    using vislib::StringW;
#define TEST_SIMPLE_SERIALISE(type) AssertNoException("Serialise " #type, rs1.Serialise(in##type, #type))
#define TEST_SIMPLE_DESERIALISE(type) AssertNoException("Deserialise " #type, rs1.Deserialise(out##type, #type));\
    AssertEqual("Deserialisation of " #type  " returned same data", in##type, out##type)

    RegistrySerialiser rs1("SOFTWARE\\VIS\\VISlib\\Test");

    bool inbool = true, outbool;
    wchar_t inwchar_t = L'v', outwchar_t;
    float infloat = 346.6443f, outfloat;
    double indouble = 4634561245.13421345, outdouble;
    INT8 inINT8 = 34, outINT8;
    INT16 inINT16 = 654, outINT16;
    INT32 inINT32 = -564, outINT32;
    INT64 inINT64 = -65462, outINT64;
    UINT8 inUINT8 = 31, outUINT8;
    UINT16 inUINT16 = 642, outUINT16;
    UINT32 inUINT32 = 56332, outUINT32;
    UINT64 inUINT64 = 324645, outUINT64;
    StringA inStringA = "Ich mach mich zum Horst", outStringA;
    StringW inStringW = L"Ich mach mich zum Hugo", outStringW;

    TEST_SIMPLE_SERIALISE(bool);
    TEST_SIMPLE_SERIALISE(wchar_t);
    TEST_SIMPLE_SERIALISE(float);
    TEST_SIMPLE_SERIALISE(double);
    TEST_SIMPLE_SERIALISE(INT8);
    TEST_SIMPLE_SERIALISE(INT16);
    TEST_SIMPLE_SERIALISE(INT32);
    TEST_SIMPLE_SERIALISE(INT64);
    TEST_SIMPLE_SERIALISE(UINT8);
    TEST_SIMPLE_SERIALISE(UINT16);
    TEST_SIMPLE_SERIALISE(UINT32);
    TEST_SIMPLE_SERIALISE(UINT64);
    TEST_SIMPLE_SERIALISE(StringA);
    TEST_SIMPLE_SERIALISE(StringW);

    TEST_SIMPLE_DESERIALISE(bool);
    TEST_SIMPLE_DESERIALISE(wchar_t);
    TEST_SIMPLE_DESERIALISE(float);
    TEST_SIMPLE_DESERIALISE(double);
    TEST_SIMPLE_DESERIALISE(INT8);
    TEST_SIMPLE_DESERIALISE(INT16);
    TEST_SIMPLE_DESERIALISE(INT32);
    TEST_SIMPLE_DESERIALISE(INT64);
    TEST_SIMPLE_DESERIALISE(UINT8);
    TEST_SIMPLE_DESERIALISE(UINT16);
    TEST_SIMPLE_DESERIALISE(UINT32);
    TEST_SIMPLE_DESERIALISE(UINT64);
    TEST_SIMPLE_DESERIALISE(StringA);
    TEST_SIMPLE_DESERIALISE(StringW);

#undef TEST_SIMPLE_SERIALISE
#undef TEST_SIMPLE_DESERIALISE
#endif /* _WIN32 */
}
