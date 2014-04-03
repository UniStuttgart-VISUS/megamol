/*
 * testwinreg.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "testwinreg.h"
#ifdef _WIN32
#include "testhelper.h"
#include "vislib/RegistryKey.h"

void TestWinReg(void) {
    using vislib::sys::RegistryKey;

    // testing key creation/deletion/query
    RegistryKey curw(RegistryKey::HKeyCurrentUser(), KEY_ALL_ACCESS);
    ::AssertTrue("HKeyCurrentUser opened for writing", curw.IsValid());

    RegistryKey vl;
    if (curw.OpenSubKey(vl, "SOFTWARE\\vislib\\test") != ERROR_SUCCESS) {
        if (curw.CreateSubKey(vl, "SOFTWARE\\vislib\\test") != ERROR_SUCCESS) {
            ::AssertTrue("HKCU\\SOFTWARE\\vislib\\test Created/Opened", false);
        } else {
            ::AssertTrue("HKCU\\SOFTWARE\\vislib\\test Created", true);
        }
    } else {
        ::AssertTrue("HKCU\\SOFTWARE\\vislib\\test Opened", true);
    }

    RegistryKey sc1;
    if (vl.OpenSubKey(sc1, "testA") != ERROR_SUCCESS) {
        if (vl.CreateSubKey(sc1, "testA") != ERROR_SUCCESS) {
            ::AssertTrue("HKCU\\SOFTWARE\\vislib\\test\\testA Created/Opened", false);
        } else {
            ::AssertTrue("HKCU\\SOFTWARE\\vislib\\test\\testA Created", true);
        }
    } else {
        ::AssertTrue("HKCU\\SOFTWARE\\vislib\\test\\testA Opened", true);
    }

    RegistryKey sc2;
    if (sc1.OpenSubKey(sc2, L"testB") != ERROR_SUCCESS) {
        if (sc1.CreateSubKey(sc2, L"testB") != ERROR_SUCCESS) {
            ::AssertTrue("HKCU\\SOFTWARE\\vislib\\test\\testA\\testB Created/Opened", false);
        } else {
            ::AssertTrue("HKCU\\SOFTWARE\\vislib\\test\\testA\\testB Created", true);
        }
    } else {
        ::AssertTrue("HKCU\\SOFTWARE\\vislib\\test\\testA\\testB Opened", true);
    }

    sc2.Close();
    sc1.Close();

    ::AssertEqual<unsigned int>("HKCU\\SOFTWARE\\vislib\\test\\testA Deleted", vl.DeleteSubKey(L"testA"), ERROR_SUCCESS);
    ::AssertNotEqual<unsigned int>("HKCU\\SOFTWARE\\vislib\\test\\testA Deleted", vl.OpenSubKey(sc1, "testA"), ERROR_SUCCESS);
    ::AssertEqual<unsigned int>("HKCU\\SOFTWARE\\vislib\\test\\testA Re-Created", vl.CreateSubKey(sc1, L"testA"), ERROR_SUCCESS);
    ::AssertEqual<size_t>("testA now has no subkeys", sc1.GetSubKeysA().Count(), 0);

    // testing integer value set/get/query
    ::AssertEqual<size_t>("testA now has no values", sc1.GetValueNamesW().Count(), 0);
    uint32_t u32_1, u32_2;
    uint64_t u64_1;
    u32_1 = 0x12345678;
    ::AssertEqual<unsigned int>("Value u1 set", sc1.SetValue("u1", u32_1), ERROR_SUCCESS);
    ::AssertEqual<size_t>("testA now has one value", sc1.GetValueNamesW().Count(), 1);
    ::AssertTrue("testA now has one value", sc1.GetValueNamesW()[0].compare(L"u1") == 0);
    ::AssertEqual<RegistryKey::RegValueType>("u1 is of right type", sc1.GetValueType(L"u1"), RegistryKey::REGVAL_DWORD);
    ::AssertEqual<unsigned int>("Getting value u1", sc1.GetValue("u1", u64_1), ERROR_SUCCESS);
    u32_2 = static_cast<uint32_t>(u64_1);
    ::AssertEqual("Value u1 correct", u32_1, u32_2);
    ::AssertEqual<unsigned int>("Getting value u1", sc1.GetValue("u1", reinterpret_cast<void*>(&u32_2), sizeof(uint32_t)), ERROR_SUCCESS);
    ::AssertEqual("Value u1 correct", u32_1, u32_2);
    ::AssertEqual<size_t>("Size of u1 correct", sc1.GetValueSize("u1"), sizeof(uint32_t));
    ::AssertEqual<unsigned int>("Value u1 removed", sc1.DeleteValue("u1"), ERROR_SUCCESS);
    ::AssertEqual<size_t>("testA now has no values", sc1.GetValueNamesW().Count(), 0);

    // testing string value set/get/query
    the::astring strA("Teschtingteschtvalue");
    the::wstring strW;
    the::multi_sza multiSzA;
    vislib::Array<the::wstring> aryStrW;
    ::AssertEqual<unsigned int>("Value s1 set", sc1.SetValue("s1", strA), ERROR_SUCCESS);
    ::AssertEqual<size_t>("testA now has one value", sc1.GetValueNamesW().Count(), 1);
    ::AssertTrue("testA now has one value", sc1.GetValueNamesW()[0].compare(L"s1") == 0);
    ::AssertEqual<RegistryKey::RegValueType>("s1 is of right type", sc1.GetValueType(L"s1"), RegistryKey::REGVAL_STRING);
    ::AssertEqual<unsigned int>("Value s1 get", sc1.GetValue("s1", strW), ERROR_SUCCESS);
    ::AssertTrue("Value s1 correct", strW.compare(the::text::string_converter::to_w(strA)) == 0);
    multiSzA.add("Test1");
    multiSzA.add("Test2");
    multiSzA.add("Test3");
    multiSzA.add("Test4");
    ::AssertEqual<unsigned int>("Value s2 set", sc1.SetValue("s2", multiSzA), ERROR_SUCCESS);
    ::AssertEqual<size_t>("testA now has two values", sc1.GetValueNamesW().Count(), 2);
    ::AssertEqual<RegistryKey::RegValueType>("s2 is of right type", sc1.GetValueType(L"s2"), RegistryKey::REGVAL_MULTI_SZ);
    aryStrW.Add(L"Test1");
    aryStrW.Add(L"");
    aryStrW.Add(L"Test2");
    aryStrW.Add(L"Test3");
    aryStrW.Add(L"");
    aryStrW.Add(L"Test4");
    ::AssertEqual<unsigned int>("Value s3 set", sc1.SetValue(L"s3", aryStrW), ERROR_SUCCESS);
    ::AssertEqual<size_t>("testA now has three values", sc1.GetValueNamesW().Count(), 3);
    ::AssertEqual<RegistryKey::RegValueType>("s3 is of right type", sc1.GetValueType(L"s3"), RegistryKey::REGVAL_MULTI_SZ);
    ::AssertEqual<unsigned int>("Value s3 get", sc1.GetValue("s3", multiSzA), ERROR_SUCCESS);
    ::AssertEqual<unsigned int>("Value s2 get", sc1.GetValue("s2", aryStrW), ERROR_SUCCESS);
    ::AssertEqual("s2 and s3 equal", multiSzA.size(), aryStrW.Count());
    size_t cnt = vislib::math::Min(multiSzA.size(), aryStrW.Count());
    for (size_t i = 0; i < cnt; i++) {
        ::AssertEqual("s2 and s3 equal", the::text::string_converter::to_w(multiSzA[i]), aryStrW[i]);
    }

    sc1.Close();
    ::AssertFalse("testA closed", sc1.IsValid());
    ::AssertEqual<unsigned int>("HKCU\\SOFTWARE\\vislib\\test\\testA Deleted", vl.DeleteSubKey(L"testA"), ERROR_SUCCESS);

}

#endif /* _WIN32 */
