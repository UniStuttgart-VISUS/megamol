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

    ::AssertEqual<DWORD>("HKCU\\SOFTWARE\\vislib\\test\\testA Deleted", vl.DeleteSubKey(L"testA"), ERROR_SUCCESS);
    ::AssertNotEqual<DWORD>("HKCU\\SOFTWARE\\vislib\\test\\testA Deleted", vl.OpenSubKey(sc1, "testA"), ERROR_SUCCESS);
    ::AssertEqual<DWORD>("HKCU\\SOFTWARE\\vislib\\test\\testA Re-Created", vl.CreateSubKey(sc1, L"testA"), ERROR_SUCCESS);
    ::AssertEqual<SIZE_T>("testA now has no subkeys", sc1.GetSubKeysA().Count(), 0);

    // testing integer value set/get/query
    ::AssertEqual<SIZE_T>("testA now has no values", sc1.GetValueNamesW().Count(), 0);
    UINT32 u32_1, u32_2;
    UINT64 u64_1;
    u32_1 = 0x12345678;
    ::AssertEqual<DWORD>("Value u1 set", sc1.SetValue("u1", u32_1), ERROR_SUCCESS);
    ::AssertEqual<SIZE_T>("testA now has one value", sc1.GetValueNamesW().Count(), 1);
    ::AssertTrue("testA now has one value", sc1.GetValueNamesW()[0].Equals(L"u1"));
    ::AssertEqual<RegistryKey::RegValueType>("u1 is of right type", sc1.GetValueType(L"u1"), RegistryKey::REGVAL_DWORD);
    ::AssertEqual<DWORD>("Getting value u1", sc1.GetValue("u1", u64_1), ERROR_SUCCESS);
    u32_2 = static_cast<UINT32>(u64_1);
    ::AssertEqual("Value u1 correct", u32_1, u32_2);
    ::AssertEqual<DWORD>("Getting value u1", sc1.GetValue("u1", reinterpret_cast<void*>(&u32_2), sizeof(UINT32)), ERROR_SUCCESS);
    ::AssertEqual("Value u1 correct", u32_1, u32_2);
    ::AssertEqual<SIZE_T>("Size of u1 correct", sc1.GetValueSize("u1"), sizeof(UINT32));
    ::AssertEqual<DWORD>("Value u1 removed", sc1.DeleteValue("u1"), ERROR_SUCCESS);
    ::AssertEqual<SIZE_T>("testA now has no values", sc1.GetValueNamesW().Count(), 0);

    // testing string value set/get/query
    vislib::StringA strA("Teschtingteschtvalue");
    vislib::StringW strW;
    vislib::MultiSzA multiSzA;
    vislib::Array<vislib::StringW> aryStrW;
    ::AssertEqual<DWORD>("Value s1 set", sc1.SetValue("s1", strA), ERROR_SUCCESS);
    ::AssertEqual<SIZE_T>("testA now has one value", sc1.GetValueNamesW().Count(), 1);
    ::AssertTrue("testA now has one value", sc1.GetValueNamesW()[0].Equals(L"s1"));
    ::AssertEqual<RegistryKey::RegValueType>("s1 is of right type", sc1.GetValueType(L"s1"), RegistryKey::REGVAL_STRING);
    ::AssertEqual<DWORD>("Value s1 get", sc1.GetValue("s1", strW), ERROR_SUCCESS);
    ::AssertTrue("Value s1 correct", strW.Equals(vislib::StringW(strA)));
    multiSzA.Append("Test1");
    multiSzA.Append("Test2");
    multiSzA.Append("Test3");
    multiSzA.Append("Test4");
    ::AssertEqual<DWORD>("Value s2 set", sc1.SetValue("s2", multiSzA), ERROR_SUCCESS);
    ::AssertEqual<SIZE_T>("testA now has two values", sc1.GetValueNamesW().Count(), 2);
    ::AssertEqual<RegistryKey::RegValueType>("s2 is of right type", sc1.GetValueType(L"s2"), RegistryKey::REGVAL_MULTI_SZ);
    aryStrW.Add(L"Test1");
    aryStrW.Add(L"");
    aryStrW.Add(L"Test2");
    aryStrW.Add(L"Test3");
    aryStrW.Add(L"");
    aryStrW.Add(L"Test4");
    ::AssertEqual<DWORD>("Value s3 set", sc1.SetValue(L"s3", aryStrW), ERROR_SUCCESS);
    ::AssertEqual<SIZE_T>("testA now has three values", sc1.GetValueNamesW().Count(), 3);
    ::AssertEqual<RegistryKey::RegValueType>("s3 is of right type", sc1.GetValueType(L"s3"), RegistryKey::REGVAL_MULTI_SZ);
    ::AssertEqual<DWORD>("Value s3 get", sc1.GetValue("s3", multiSzA), ERROR_SUCCESS);
    ::AssertEqual<DWORD>("Value s2 get", sc1.GetValue("s2", aryStrW), ERROR_SUCCESS);
    ::AssertEqual("s2 and s3 equal", multiSzA.Count(), aryStrW.Count());
    SIZE_T cnt = vislib::math::Min(multiSzA.Count(), aryStrW.Count());
    for (SIZE_T i = 0; i < cnt; i++) {
        ::AssertEqual("s2 and s3 equal", vislib::StringW(multiSzA[i]), aryStrW[i]);
    }

    sc1.Close();
    ::AssertFalse("testA closed", sc1.IsValid());
    ::AssertEqual<DWORD>("HKCU\\SOFTWARE\\vislib\\test\\testA Deleted", vl.DeleteSubKey(L"testA"), ERROR_SUCCESS);

}

#endif /* _WIN32 */
