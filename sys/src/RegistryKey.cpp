/*
 * RegistryKey.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/RegistryKey.h"

#define _WINXP32_LEGACY_SUPPORT

#ifdef _WIN32
#include "vislib/MissingImplementationException.h"
#ifdef _WINXP32_LEGACY_SUPPORT
#include "DynamicFunctionPointer.h"
#endif /* _WINXP32_LEGACY_SUPPORT */


#ifdef _WINXP32_LEGACY_SUPPORT
namespace vislib {
namespace sys {

/*
 * Dynamic function pointer for RegDeleteKeyExA
 */
static DynamicFunctionPointer<LSTATUS (APIENTRY *)(HKEY hKey, LPCSTR lpSubKey,
        REGSAM samDesired, DWORD Reserved )>
        dynRegDeleteKeyExA("Advapi32.dll", "RegDeleteKeyExA");

/*
 * Dynamic function pointer for RegDeleteKeyExA
 */
static DynamicFunctionPointer<LSTATUS (APIENTRY *)(HKEY hKey, LPCWSTR lpSubKey,
        REGSAM samDesired, DWORD Reserved )>
        dynRegDeleteKeyExW("Advapi32.dll", "RegDeleteKeyExW");

} /* end namespace sys */
} /* end namespace vislib */
#endif /* _WINXP32_LEGACY_SUPPORT */


/*
 * vislib::sys::RegistryKey::INVALID_HKEY
 */
const HKEY vislib::sys::RegistryKey::INVALID_HKEY(
    reinterpret_cast<HKEY>(INVALID_HANDLE_VALUE));


/*
 * vislib::sys::RegistryKey::HKeyClassesRoot
 */
const vislib::sys::RegistryKey&
vislib::sys::RegistryKey::HKeyClassesRoot(void) {
    static vislib::sys::RegistryKey keyCR(HKEY_CLASSES_ROOT, true);
    return keyCR;
}


/*
 * vislib::sys::RegistryKey::HKeyCurrentUser
 */
const vislib::sys::RegistryKey&
vislib::sys::RegistryKey::HKeyCurrentUser(void) {
    static vislib::sys::RegistryKey keyCU(
        vislib::sys::RegistryKey::INVALID_HKEY, false);
    if (!keyCU.IsValid()) {
        LONG errcode = ::RegOpenCurrentUser(KEY_READ, &keyCU.key);
        if (errcode == ERROR_SUCCESS) {
            keyCU.sam = KEY_READ;
        } else {
            keyCU.key = vislib::sys::RegistryKey::INVALID_HKEY;
        }
    }
    return keyCU;
}


/*
 * vislib::sys::RegistryKey::HKeyLocalMachine
 */
const vislib::sys::RegistryKey&
vislib::sys::RegistryKey::HKeyLocalMachine(void) {
    static vislib::sys::RegistryKey keyLM(HKEY_LOCAL_MACHINE, true);
    return keyLM;
}


/*
 * vislib::sys::RegistryKey::HKeyUsers
 */
const vislib::sys::RegistryKey&
vislib::sys::RegistryKey::HKeyUsers(void) {
    static vislib::sys::RegistryKey keyUs(HKEY_USERS, true);
    return keyUs;
}


/*
 * vislib::sys::RegistryKey::RegistryKey
 */
vislib::sys::RegistryKey::RegistryKey(void) : key(INVALID_HKEY), sam(0) {
    // intentionally empty
}


/*
 * vislib::sys::RegistryKey::RegistryKey
 */
vislib::sys::RegistryKey::RegistryKey(HKEY key, bool duplicate, bool write)
        : key(INVALID_HKEY), sam(write ? KEY_ALL_ACCESS : KEY_READ) {
    if (duplicate && (key != INVALID_HKEY)) {
        DWORD errcode = ::RegOpenKeyExA(key, NULL, 0, sam, &this->key);
        if (errcode != ERROR_SUCCESS) {
            this->key = INVALID_HKEY;
            this->sam = 0;
        }
    } else {
        this->key = key;
    }
}


/*
 * vislib::sys::RegistryKey::RegistryKey
 */
vislib::sys::RegistryKey::RegistryKey(const vislib::sys::RegistryKey& src,
        REGSAM sam) : key(INVALID_HKEY), sam(0) {
    if (sam == 0) {
        *this = src;
    } else {
        this->ReopenKey(src, sam);
    }
}


/*
 * vislib::sys::RegistryKey::~RegistryKey
 */
vislib::sys::RegistryKey::~RegistryKey(void) {
    this->Close();
}


/*
 * vislib::sys::RegistryKey::Close
 */
void vislib::sys::RegistryKey::Close(void) {
    if (this->key != INVALID_HKEY) {
        ::RegCloseKey(this->key);
        this->key = INVALID_HKEY;
    }
    this->sam = 0;
}


/*
 * vislib::sys::RegistryKey::CreateSubKey
 */
DWORD vislib::sys::RegistryKey::CreateSubKey(
        vislib::sys::RegistryKey& outKey, const vislib::StringA& name,
        REGSAM sam) {
    outKey.Close();
    outKey.sam = (sam == 0) ? this->sam : sam; // TODO: WOW64-Stuff
    DWORD errcode = ::RegCreateKeyExA(this->key, name, 0, NULL,
        REG_OPTION_NON_VOLATILE, outKey.sam, NULL, &outKey.key, NULL);
    if (errcode != NULL) {
        outKey.key = INVALID_HKEY;
        outKey.sam = 0;
    }
    return errcode;
}


/*
 * vislib::sys::RegistryKey::CreateSubKey
 */
DWORD vislib::sys::RegistryKey::CreateSubKey(
        vislib::sys::RegistryKey& outKey, const vislib::StringW& name,
        REGSAM sam) {
    outKey.Close();
    outKey.sam = (sam == 0) ? this->sam : sam; // TODO: WOW64-Stuff
    DWORD errcode = ::RegCreateKeyExW(this->key, name, 0, NULL,
        REG_OPTION_NON_VOLATILE, outKey.sam, NULL, &outKey.key, NULL);
    if (errcode != NULL) {
        outKey.key = INVALID_HKEY;
        outKey.sam = 0;
    }
    return errcode;
}


/*
 * vislib::sys::RegistryKey::DeleteSubKey
 */
DWORD vislib::sys::RegistryKey::DeleteSubKey(
        const vislib::StringA& name) {
    DWORD errcode;
    RegistryKey sc;

    errcode = this->OpenSubKey(sc, name);
    if (errcode != ERROR_SUCCESS) return errcode;
    Array<StringA> subkeys(sc.GetSubKeysA());
    for (SIZE_T i = 0; i < subkeys.Count(); i++) {
        errcode = sc.DeleteSubKey(subkeys[i]);
        if (errcode != ERROR_SUCCESS) return errcode;
    }
    sc.Close();

#ifdef _WINXP32_LEGACY_SUPPORT
    if (dynRegDeleteKeyExA.IsValid()) {
        errcode = dynRegDeleteKeyExA(this->key, name,
            this->sam & (KEY_WOW64_32KEY | KEY_WOW64_64KEY), 0);
    } else {
        errcode = ::RegDeleteKeyA(this->key, name);
    }
#else /* _WINXP32_LEGACY_SUPPORT */
    errcode = ::RegDeleteKeyExA(this->key, name,
        this->sam & (KEY_WOW64_32KEY | KEY_WOW64_64KEY), 0);
#endif /* _WINXP32_LEGACY_SUPPORT */
    return errcode;
}


/*
 * vislib::sys::RegistryKey::DeleteSubKey
 */
DWORD vislib::sys::RegistryKey::DeleteSubKey(
        const vislib::StringW& name) {
    DWORD errcode;
    RegistryKey sc;

    errcode = this->OpenSubKey(sc, name);
    if (errcode != ERROR_SUCCESS) return errcode;
    Array<StringW> subkeys(sc.GetSubKeysW());
    for (SIZE_T i = 0; i < subkeys.Count(); i++) {
        errcode = sc.DeleteSubKey(subkeys[i]);
        if (errcode != ERROR_SUCCESS) return errcode;
    }
    sc.Close();

#ifdef _WINXP32_LEGACY_SUPPORT
    if (dynRegDeleteKeyExW.IsValid()) {
        errcode = dynRegDeleteKeyExW(this->key, name,
            this->sam & (KEY_WOW64_32KEY | KEY_WOW64_64KEY), 0);
    } else {
        errcode = ::RegDeleteKeyW(this->key, name);
    }
#else /* _WINXP32_LEGACY_SUPPORT */
    errcode = ::RegDeleteKeyExW(this->key, name,
        this->sam & (KEY_WOW64_32KEY | KEY_WOW64_64KEY), 0);
#endif /* _WINXP32_LEGACY_SUPPORT */
    return errcode;
}


/*
 * vislib::sys::RegistryKey::DeleteValue
 */
DWORD vislib::sys::RegistryKey::DeleteValue(
        const vislib::StringA& name) {
    return ::RegDeleteValueA(this->key, name);
}


/*
 * vislib::sys::RegistryKey::DeleteValue
 */
DWORD vislib::sys::RegistryKey::DeleteValue(
        const vislib::StringW& name) {
    return ::RegDeleteValueW(this->key, name);
}


/*
 * vislib::sys::RegistryKey::GetSubKeysA
 */
vislib::Array<vislib::StringA>
vislib::sys::RegistryKey::GetSubKeysA(void) const {
    Array<StringA> rv;
    DWORD errcode;
    DWORD subKeyCount;
    DWORD subKeyMaxNameLen;
    StringA str;
    char *strptr;
    errcode = ::RegQueryInfoKeyA(this->key, NULL, NULL, NULL, &subKeyCount,
        &subKeyMaxNameLen, NULL, NULL, NULL, NULL, NULL, NULL);
    if (errcode != ERROR_SUCCESS) {
        rv.Clear();
        return rv;
    }
    strptr = str.AllocateBuffer(subKeyMaxNameLen);

    for (DWORD i = 0; i < subKeyCount; i++) {
        DWORD strptrlen = subKeyMaxNameLen + 1;
        errcode = ::RegEnumKeyExA(this->key, i, strptr, &strptrlen, 0, NULL,
            NULL, NULL);
        if (errcode != ERROR_SUCCESS) {
            rv.Clear();
            return rv;
        }
        rv.Add(str);
    }

    return rv;
}


/*
 * vislib::sys::RegistryKey::GetSubKeysW
 */
vislib::Array<vislib::StringW>
vislib::sys::RegistryKey::GetSubKeysW(void) const {
    Array<StringW> rv;
    DWORD errcode;
    DWORD subKeyCount;
    DWORD subKeyMaxNameLen;
    StringW str;
    wchar_t *strptr;
    errcode = ::RegQueryInfoKeyW(this->key, NULL, NULL, NULL, &subKeyCount,
        &subKeyMaxNameLen, NULL, NULL, NULL, NULL, NULL, NULL);
    if (errcode != ERROR_SUCCESS) {
        rv.Clear();
        return rv;
    }
    strptr = str.AllocateBuffer(subKeyMaxNameLen);

    for (DWORD i = 0; i < subKeyCount; i++) {
        DWORD strptrlen = subKeyMaxNameLen + 1;
        errcode = ::RegEnumKeyExW(this->key, i, strptr, &strptrlen, 0, NULL,
            NULL, NULL);
        if (errcode != ERROR_SUCCESS) {
            rv.Clear();
            return rv;
        }
        rv.Add(str);
    }

    return rv;
}


/*
 * vislib::sys::RegistryKey::GetValueNamesA
 */
vislib::Array<vislib::StringA>
vislib::sys::RegistryKey::GetValueNamesA(void) const {
    Array<StringA> rv;
    DWORD errcode;
    DWORD valuesCount;
    DWORD valuesMaxNameLen;
    StringA str;
    char *strptr;
    errcode = ::RegQueryInfoKeyA(this->key, NULL, NULL, NULL, NULL, NULL,
        NULL, &valuesCount, &valuesMaxNameLen, NULL, NULL, NULL);
    if (errcode != ERROR_SUCCESS) {
        rv.Clear();
        return rv;
    }
    strptr = str.AllocateBuffer(valuesMaxNameLen);

    for (DWORD i = 0; i < valuesCount; i++) {
        DWORD strptrlen = valuesMaxNameLen + 1;
        errcode = ::RegEnumValueA(this->key, i, strptr, &strptrlen, 0, NULL,
            NULL, NULL);
        if (errcode != ERROR_SUCCESS) {
            rv.Clear();
            return rv;
        }
        rv.Add(str);
    }

    return rv;
}


/*
 * vislib::sys::RegistryKey::GetValueNamesW
 */
vislib::Array<vislib::StringW>
vislib::sys::RegistryKey::GetValueNamesW(void) const {
    Array<StringW> rv;
    DWORD errcode;
    DWORD valuesCount;
    DWORD valuesMaxNameLen;
    StringW str;
    wchar_t *strptr;
    errcode = ::RegQueryInfoKeyW(this->key, NULL, NULL, NULL, NULL, NULL,
        NULL, &valuesCount, &valuesMaxNameLen, NULL, NULL, NULL);
    if (errcode != ERROR_SUCCESS) {
        rv.Clear();
        return rv;
    }
    strptr = str.AllocateBuffer(valuesMaxNameLen);

    for (DWORD i = 0; i < valuesCount; i++) {
        DWORD strptrlen = valuesMaxNameLen + 1;
        errcode = ::RegEnumValueW(this->key, i, strptr, &strptrlen, 0, NULL,
            NULL, NULL);
        if (errcode != ERROR_SUCCESS) {
            rv.Clear();
            return rv;
        }
        rv.Add(str);
    }

    return rv;
}


/*
 * vislib::sys::RegistryKey::GetValueType
 */
vislib::sys::RegistryKey::RegValueType vislib::sys::RegistryKey::GetValueType(
        const vislib::StringA& name) const {
    DWORD valtype;
    DWORD errcode = ::RegQueryValueExA(this->key, name, NULL, &valtype,
        NULL, NULL);
    if (errcode != ERROR_SUCCESS) return REGVAL_NONE;
    switch (valtype) {
        case REG_BINARY: return REGVAL_BINARY;
        case REG_DWORD: return REGVAL_DWORD;
        case REG_EXPAND_SZ: return REGVAL_EXPAND_SZ;
        case REG_MULTI_SZ: return REGVAL_MULTI_SZ;
        case REG_NONE : return REGVAL_NONE;
        case REG_QWORD: return REGVAL_QWORD;
        case REG_SZ: return REGVAL_STRING;
        default: return REGVAL_BINARY;
    }
    return REGVAL_NONE;
}


/*
 * vislib::sys::RegistryKey::GetValueType
 */
vislib::sys::RegistryKey::RegValueType vislib::sys::RegistryKey::GetValueType(
        const vislib::StringW& name) const {
    DWORD valtype;
    DWORD errcode = ::RegQueryValueExW(this->key, name, NULL, &valtype,
        NULL, NULL);
    if (errcode != ERROR_SUCCESS) return REGVAL_NONE;
    switch (valtype) {
        case REG_BINARY: return REGVAL_BINARY;
        case REG_DWORD: return REGVAL_DWORD;
        case REG_EXPAND_SZ: return REGVAL_EXPAND_SZ;
        case REG_MULTI_SZ: return REGVAL_MULTI_SZ;
        case REG_NONE : return REGVAL_NONE;
        case REG_QWORD: return REGVAL_QWORD;
        case REG_SZ: return REGVAL_STRING;
        default: return REGVAL_BINARY;
    }
    return REGVAL_NONE;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const vislib::StringA& name, vislib::RawStorage& outData) const {
    DWORD size;
    DWORD errcode = ::RegQueryValueExA(this->key, name, NULL, NULL, NULL,
        &size);
    if (errcode != ERROR_SUCCESS) return errcode;
    outData.AssertSize(size);
    errcode = ::RegQueryValueExA(this->key, name, NULL, NULL,
        outData.As<BYTE>(), &size);
    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const vislib::StringW& name, vislib::RawStorage& outData) const {
    DWORD size;
    DWORD errcode = ::RegQueryValueExW(this->key, name, NULL, NULL, NULL,
        &size);
    if (errcode != ERROR_SUCCESS) return errcode;
    outData.AssertSize(size);
    errcode = ::RegQueryValueExW(this->key, name, NULL, NULL,
        outData.As<BYTE>(), &size);
    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const vislib::StringA& name, void* outData,
        SIZE_T dataSize) const {
    DWORD size = static_cast<DWORD>(dataSize);
    DWORD errcode = ::RegQueryValueExA(this->key, name, NULL, NULL,
        static_cast<BYTE*>(outData), &size);
    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const vislib::StringW& name, void* outData,
        SIZE_T dataSize) const {
    DWORD size = static_cast<DWORD>(dataSize);
    DWORD errcode = ::RegQueryValueExW(this->key, name, NULL, NULL,
        static_cast<BYTE*>(outData), &size);
    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const vislib::StringA& name, vislib::StringA& outStr) const {
    DWORD size;
    DWORD type;
    DWORD errcode = ::RegQueryValueExA(this->key, name, NULL, &type, NULL,
        &size);
    if (errcode != ERROR_SUCCESS) return errcode;
    if ((type != REG_SZ) && (type != REG_EXPAND_SZ)) {
        return ERROR_INVALID_DATATYPE;
    }
    errcode = ::RegQueryValueExA(this->key, name, NULL, NULL,
        reinterpret_cast<BYTE*>(outStr.AllocateBuffer(size)), &size);
    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const vislib::StringW& name, vislib::StringW& outStr) const {
    DWORD size;
    DWORD type;
    DWORD errcode = ::RegQueryValueExW(this->key, name, NULL, &type, NULL,
        &size);
    if (errcode != ERROR_SUCCESS) return errcode;
    if ((type != REG_SZ) && (type != REG_EXPAND_SZ)) {
        return ERROR_INVALID_DATATYPE;
    }
    errcode = ::RegQueryValueExW(this->key, name, NULL, NULL,
        reinterpret_cast<BYTE*>(outStr.AllocateBuffer(size)), &size);
    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const vislib::StringA& name, vislib::StringW& outStr) const {
    vislib::StringA v;
    DWORD rv = this->GetValue(name, v);
    if (rv == ERROR_SUCCESS) {
        outStr = v;
    }
    return rv;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const vislib::StringW& name, vislib::StringA& outStr) const {
    vislib::StringW v;
    DWORD rv = this->GetValue(name, v);
    if (rv == ERROR_SUCCESS) {
        outStr = v;
    }
    return rv;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const vislib::StringA& name, vislib::MultiSzA& outStrs) const {
    DWORD size;
    DWORD type;
    DWORD errcode = ::RegQueryValueExA(this->key, name, NULL, &type, NULL,
        &size);
    if (errcode != ERROR_SUCCESS) return errcode;
    if ((type == REG_SZ) || (type == REG_EXPAND_SZ)) {
        StringA val;
        errcode = this->GetValue(name, val);
        if (errcode != ERROR_SUCCESS) return errcode;
        outStrs.Clear();
        outStrs.Append(val);

    } else if (type == REG_MULTI_SZ) {
        errcode = ::RegQueryValueExA(this->key, name, NULL, NULL,
            reinterpret_cast<BYTE*>(outStrs.AllocateBuffer(size)), &size);

    } else {
        errcode = ERROR_INVALID_DATATYPE;
    }

    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const vislib::StringW& name, vislib::MultiSzW& outStrs) const {
    DWORD size;
    DWORD type;
    DWORD errcode = ::RegQueryValueExW(this->key, name, NULL, &type, NULL,
        &size);
    if (errcode != ERROR_SUCCESS) return errcode;
    if ((type == REG_SZ) || (type == REG_EXPAND_SZ)) {
        StringW val;
        errcode = this->GetValue(name, val);
        if (errcode != ERROR_SUCCESS) return errcode;
        outStrs.Clear();
        outStrs.Append(val);

    } else if (type == REG_MULTI_SZ) {
        errcode = ::RegQueryValueExW(this->key, name, NULL, NULL,
            reinterpret_cast<BYTE*>(outStrs.AllocateBuffer(size)), &size);

    } else {
        errcode = ERROR_INVALID_DATATYPE;
    }

    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const vislib::StringA& name, vislib::MultiSzW& outStrs) const {
    return this->GetValue(StringW(name), outStrs);
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const vislib::StringW& name, vislib::MultiSzA& outStrs) const {
    return this->GetValue(StringA(name), outStrs);
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const vislib::StringA& name,
        vislib::Array<vislib::StringA>& outStrs) const {
    MultiSzA strs;
    DWORD errcode = this->GetValue(name, strs);
    if (errcode != ERROR_SUCCESS) return errcode;
    outStrs.SetCount(strs.Count());
    for (SIZE_T i = 0; i < outStrs.Count(); i++) {
        outStrs[i] = strs[i];
    }
    return ERROR_SUCCESS;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const vislib::StringW& name,
        vislib::Array<vislib::StringW>& outStrs) const {
    MultiSzW strs;
    DWORD errcode = this->GetValue(name, strs);
    if (errcode != ERROR_SUCCESS) return errcode;
    outStrs.SetCount(strs.Count());
    for (SIZE_T i = 0; i < outStrs.Count(); i++) {
        outStrs[i] = strs[i];
    }
    return ERROR_SUCCESS;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const vislib::StringA& name,
        vislib::Array<vislib::StringW>& outStrs) const {
    MultiSzA strs;
    DWORD errcode = this->GetValue(name, strs);
    if (errcode != ERROR_SUCCESS) return errcode;
    outStrs.SetCount(strs.Count());
    for (SIZE_T i = 0; i < outStrs.Count(); i++) {
        outStrs[i] = strs[i];
    }
    return ERROR_SUCCESS;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const vislib::StringW& name,
        vislib::Array<vislib::StringA>& outStrs) const {
    MultiSzW strs;
    DWORD errcode = this->GetValue(name, strs);
    if (errcode != ERROR_SUCCESS) return errcode;
    outStrs.SetCount(strs.Count());
    for (SIZE_T i = 0; i < outStrs.Count(); i++) {
        outStrs[i] = strs[i];
    }
    return ERROR_SUCCESS;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const vislib::StringA& name, UINT32& outVal) const {
    DWORD size;
    DWORD type;
    DWORD errcode = ::RegQueryValueExA(this->key, name, NULL, &type, NULL,
        NULL);
    if (errcode != ERROR_SUCCESS) return errcode;
    if (type == REG_DWORD) {
        size = sizeof(UINT32);
        errcode = ::RegQueryValueExA(this->key, name, NULL, NULL,
            reinterpret_cast<BYTE*>(&outVal), &size);

    } else if (type == REG_QWORD) {
        UINT64 val;
        size = sizeof(UINT64);
        errcode = ::RegQueryValueExA(this->key, name, NULL, NULL,
            reinterpret_cast<BYTE*>(&val), &size);
        if (errcode == ERROR_SUCCESS) {
            outVal = static_cast<UINT32>(val);
        }

    } else {
        errcode = ERROR_INVALID_DATATYPE;
    }

    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const vislib::StringW& name, UINT32& outVal) const {
    DWORD size;
    DWORD type;
    DWORD errcode = ::RegQueryValueExW(this->key, name, NULL, &type, NULL,
        NULL);
    if (errcode != ERROR_SUCCESS) return errcode;
    if (type == REG_DWORD) {
        size = sizeof(UINT32);
        errcode = ::RegQueryValueExW(this->key, name, NULL, NULL,
            reinterpret_cast<BYTE*>(&outVal), &size);

    } else if (type == REG_QWORD) {
        UINT64 val;
        size = sizeof(UINT64);
        errcode = ::RegQueryValueExW(this->key, name, NULL, NULL,
            reinterpret_cast<BYTE*>(&val), &size);
        if (errcode == ERROR_SUCCESS) {
            outVal = static_cast<UINT32>(val);
        }

    } else {
        errcode = ERROR_INVALID_DATATYPE;
    }

    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const vislib::StringA& name, UINT64& outVal) const {
    DWORD size;
    DWORD type;
    DWORD errcode = ::RegQueryValueExA(this->key, name, NULL, &type, NULL,
        NULL);
    if (errcode != ERROR_SUCCESS) return errcode;
    if (type == REG_DWORD) {
        UINT32 val;
        size = sizeof(UINT32);
        errcode = ::RegQueryValueExA(this->key, name, NULL, NULL,
            reinterpret_cast<BYTE*>(&val), &size);
        if (errcode == ERROR_SUCCESS) {
            outVal = static_cast<UINT64>(val);
        }

    } else if (type == REG_QWORD) {
        size = sizeof(UINT64);
        errcode = ::RegQueryValueExA(this->key, name, NULL, NULL,
            reinterpret_cast<BYTE*>(&outVal), &size);

    } else {
        errcode = ERROR_INVALID_DATATYPE;
    }

    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const vislib::StringW& name, UINT64& outVal) const {
    DWORD size;
    DWORD type;
    DWORD errcode = ::RegQueryValueExW(this->key, name, NULL, &type, NULL,
        NULL);
    if (errcode != ERROR_SUCCESS) return errcode;
    if (type == REG_DWORD) {
        UINT32 val;
        size = sizeof(UINT32);
        errcode = ::RegQueryValueExW(this->key, name, NULL, NULL,
            reinterpret_cast<BYTE*>(&val), &size);
        if (errcode == ERROR_SUCCESS) {
            outVal = static_cast<UINT64>(val);
        }

    } else if (type == REG_QWORD) {
        size = sizeof(UINT64);
        errcode = ::RegQueryValueExW(this->key, name, NULL, NULL,
            reinterpret_cast<BYTE*>(&outVal), &size);

    } else {
        errcode = ERROR_INVALID_DATATYPE;
    }

    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValueSize
 */
SIZE_T
vislib::sys::RegistryKey::GetValueSize(const vislib::StringA& name) const {
    DWORD size;
    DWORD errcode = ::RegQueryValueExA(this->key, name, NULL, NULL, NULL,
        &size);
    if (errcode != ERROR_SUCCESS) return 0;
    return static_cast<SIZE_T>(size);
}


/*
 * vislib::sys::RegistryKey::GetValueSize
 */
SIZE_T
vislib::sys::RegistryKey::GetValueSize(const vislib::StringW& name) const {
    DWORD size;
    DWORD errcode = ::RegQueryValueExW(this->key, name, NULL, NULL, NULL,
        &size);
    if (errcode != ERROR_SUCCESS) return 0;
    return static_cast<SIZE_T>(size);
}


/*
 * vislib::sys::RegistryKey::OpenSubKey
 */
DWORD vislib::sys::RegistryKey::OpenSubKey(
        vislib::sys::RegistryKey& outKey, const vislib::StringA& name,
        REGSAM sam) const {
    outKey.Close();
    outKey.sam = (sam == 0) ? this->sam : sam; // TODO: WOW64-Stuff
    DWORD errcode = ::RegOpenKeyExA(this->key, name, 0,
        outKey.sam, &outKey.key);
    if (errcode != ERROR_SUCCESS) {
        outKey.sam = 0;
        outKey.key = INVALID_HKEY;
    }
    return errcode;
}


/*
 * vislib::sys::RegistryKey::OpenSubKey
 */
DWORD vislib::sys::RegistryKey::OpenSubKey(
        vislib::sys::RegistryKey& outKey, const vislib::StringW& name,
        REGSAM sam) const {
    outKey.Close();
    outKey.sam = (sam == 0) ? this->sam : sam; // TODO: WOW64-Stuff
    DWORD errcode = ::RegOpenKeyExW(this->key, name, 0,
        outKey.sam, &outKey.key);
    if (errcode != ERROR_SUCCESS) {
        outKey.sam = 0;
        outKey.key = INVALID_HKEY;
    }
    return errcode;
}


/*
 * vislib::sys::RegistryKey::ReopenKey
 */
DWORD vislib::sys::RegistryKey::ReopenKey(
        const vislib::sys::RegistryKey& key, REGSAM sam) {
    if ((this == &key) && (this->sam == sam)) return ERROR_SUCCESS;
    RegistryKey src(key);
    this->Close();
    DWORD errcode = ::RegOpenKeyExW(src.key, NULL, 0, sam, &this->key);
    if (errcode == ERROR_SUCCESS) {
        this->sam = sam;
    } else {
        this->key = INVALID_HKEY;
        this->sam = 0;
    }
    return errcode;
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringA& name, const vislib::RawStorage& data) {
    return this->SetValue(name, data.As<void>(), data.GetSize());
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringW& name, const vislib::RawStorage& data) {
    return this->SetValue(name, data.As<void>(), data.GetSize());
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringA& name, const void* data, SIZE_T size) {
    return ::RegSetValueExA(this->key, name, NULL, REG_BINARY,
        reinterpret_cast<const BYTE*>(data), static_cast<DWORD>(size));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringW& name, const void* data, SIZE_T size) {
    return ::RegSetValueExW(this->key, name, NULL, REG_BINARY,
        reinterpret_cast<const BYTE*>(data), static_cast<DWORD>(size));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringA& name, const vislib::StringA& str,
        bool expandable) {
    return ::RegSetValueExA(this->key, name, NULL,
        expandable ? REG_EXPAND_SZ : REG_SZ,
        reinterpret_cast<const BYTE*>(str.PeekBuffer()), str.Length() + 1);
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringW& name, const vislib::StringW& str,
        bool expandable) {
    return ::RegSetValueExW(this->key, name, NULL,
        expandable ? REG_EXPAND_SZ : REG_SZ,
        reinterpret_cast<const BYTE*>(str.PeekBuffer()),
        (str.Length() + 1) * sizeof(wchar_t));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringA& name, const vislib::StringW& str,
        bool expandable) {
    return this->SetValue(name, vislib::StringA(str), expandable);
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringW& name, const vislib::StringA& str,
        bool expandable) {
    return this->SetValue(name, vislib::StringW(str), expandable);
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringA& name, const vislib::MultiSzA& strs) {
    return ::RegSetValueExA(this->key, name, NULL, REG_MULTI_SZ,
        reinterpret_cast<const BYTE*>(strs.PeekBuffer()),
        static_cast<DWORD>(strs.Length()));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringW& name, const vislib::MultiSzW& strs) {
    return ::RegSetValueExW(this->key, name, NULL, REG_MULTI_SZ,
        reinterpret_cast<const BYTE*>(strs.PeekBuffer()),
        static_cast<DWORD>(strs.Length() * sizeof(wchar_t)));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringA& name, const vislib::MultiSzW& strs) {
    vislib::MultiSzA msz;
    for (SIZE_T i = 0; i < strs.Count(); i++) {
        msz.Append(vislib::StringA(strs[i]));
    }
    return this->SetValue(name, msz);
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringW& name, const vislib::MultiSzA& strs) {
    vislib::MultiSzW msz;
    for (SIZE_T i = 0; i < strs.Count(); i++) {
        msz.Append(vislib::StringW(strs[i]));
    }
    return this->SetValue(name, msz);
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringA& name,
        const vislib::Array<vislib::StringA>& strs) {
    vislib::MultiSzA msz;
    for (SIZE_T i = 0; i < strs.Count(); i++) {
        if (!strs[i].IsEmpty()) msz.Append(strs[i]);
    }
    return this->SetValue(name, msz);
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringW& name,
        const vislib::Array<vislib::StringW>& strs) {
    vislib::MultiSzW msz;
    for (SIZE_T i = 0; i < strs.Count(); i++) {
        if (!strs[i].IsEmpty()) msz.Append(strs[i]);
    }
    return this->SetValue(name, msz);
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringA& name,
        const vislib::Array<vislib::StringW>& strs) {
    vislib::MultiSzA msz;
    for (SIZE_T i = 0; i < strs.Count(); i++) {
        if (!strs[i].IsEmpty()) msz.Append(vislib::StringA(strs[i]));
    }
    return this->SetValue(name, msz);
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringW& name,
        const vislib::Array<vislib::StringA>& strs) {
    vislib::MultiSzW msz;
    for (SIZE_T i = 0; i < strs.Count(); i++) {
        if (!strs[i].IsEmpty()) msz.Append(vislib::StringW(strs[i]));
    }
    return this->SetValue(name, msz);
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringA& name, INT32 val) {
    return this->SetValue(name, static_cast<UINT32>(val));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringW& name, INT32 val) {
    return this->SetValue(name, static_cast<UINT32>(val));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringA& name, UINT32 val) {
    return ::RegSetValueExA(this->key, name, NULL, REG_DWORD,
        reinterpret_cast<const BYTE*>(&val), sizeof(UINT32));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringW& name, UINT32 val) {
    return ::RegSetValueExW(this->key, name, NULL, REG_DWORD,
        reinterpret_cast<const BYTE*>(&val), sizeof(UINT32));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringA& name, INT64 val) {
    return this->SetValue(name, static_cast<UINT64>(val));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringW& name, INT64 val) {
    return this->SetValue(name, static_cast<UINT64>(val));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringA& name, UINT64 val) {
    return ::RegSetValueExA(this->key, name, NULL, REG_QWORD,
        reinterpret_cast<const BYTE*>(&val), sizeof(UINT64));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const vislib::StringW& name, UINT64 val) {
    return ::RegSetValueExW(this->key, name, NULL, REG_QWORD,
        reinterpret_cast<const BYTE*>(&val), sizeof(UINT64));
}


/*
 * vislib::sys::RegistryKey::operator=
 */
vislib::sys::RegistryKey& vislib::sys::RegistryKey::operator=(
        const vislib::sys::RegistryKey& rhs) {
    if (this->key != INVALID_HKEY) {
        ::RegCloseKey(this->key);
    }
    if (rhs.key == INVALID_HKEY) {
        this->key = INVALID_HKEY;
        this->sam = 0;
    } else {
        DWORD errcode = ::RegOpenKeyExA(rhs.key, NULL, 0, rhs.sam, &this->key);
        if (errcode != ERROR_SUCCESS) {
            this->key = INVALID_HKEY;
        } else {
            this->sam = rhs.sam;
        }
    }
    return *this;
}

#endif /* _WIN32 */
