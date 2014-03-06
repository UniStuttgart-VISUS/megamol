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
    int8_t inint8_t = 34, outint8_t;
    int16_t inint16_t = 654, outint16_t;
    int32_t inint32_t = -564, outint32_t;
    int64_t inint64_t = -65462, outint64_t;
    uint8_t inuint8_t = 31, outuint8_t;
    uint16_t inuint16_t = 642, outuint16_t;
    uint32_t inuint32_t = 56332, outuint32_t;
    uint64_t inuint64_t = 324645, outuint64_t;
    StringA inStringA = "Ich mach mich zum Horst", outStringA;
    StringW inStringW = L"Ich mach mich zum Hugo", outStringW;

    TEST_SIMPLE_SERIALISE(bool);
    TEST_SIMPLE_SERIALISE(wchar_t);
    TEST_SIMPLE_SERIALISE(float);
    TEST_SIMPLE_SERIALISE(double);
    TEST_SIMPLE_SERIALISE(int8_t);
    TEST_SIMPLE_SERIALISE(int16_t);
    TEST_SIMPLE_SERIALISE(int32_t);
    TEST_SIMPLE_SERIALISE(int64_t);
    TEST_SIMPLE_SERIALISE(uint8_t);
    TEST_SIMPLE_SERIALISE(uint16_t);
    TEST_SIMPLE_SERIALISE(uint32_t);
    TEST_SIMPLE_SERIALISE(uint64_t);
    TEST_SIMPLE_SERIALISE(StringA);
    TEST_SIMPLE_SERIALISE(StringW);

    TEST_SIMPLE_DESERIALISE(bool);
    TEST_SIMPLE_DESERIALISE(wchar_t);
    TEST_SIMPLE_DESERIALISE(float);
    TEST_SIMPLE_DESERIALISE(double);
    TEST_SIMPLE_DESERIALISE(int8_t);
    TEST_SIMPLE_DESERIALISE(int16_t);
    TEST_SIMPLE_DESERIALISE(int32_t);
    TEST_SIMPLE_DESERIALISE(int64_t);
    TEST_SIMPLE_DESERIALISE(uint8_t);
    TEST_SIMPLE_DESERIALISE(uint16_t);
    TEST_SIMPLE_DESERIALISE(uint32_t);
    TEST_SIMPLE_DESERIALISE(uint64_t);
    TEST_SIMPLE_DESERIALISE(StringA);
    TEST_SIMPLE_DESERIALISE(StringW);

#undef TEST_SIMPLE_SERIALISE
#undef TEST_SIMPLE_DESERIALISE
#endif /* _WIN32 */
}
