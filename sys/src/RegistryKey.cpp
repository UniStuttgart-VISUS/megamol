/*
 * RegistryKey.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/RegistryKey.h"

#define _WINXP32_LEGACY_SUPPORT

#ifdef _WIN32
#include "the/not_implemented_exception.h"
#ifdef _WINXP32_LEGACY_SUPPORT
#include "DynamicFunctionPointer.h"
#endif /* _WINXP32_LEGACY_SUPPORT */
#include "the/text/string_buffer.h"


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
        vislib::sys::RegistryKey& outKey, const the::astring& name,
        REGSAM sam) {
    outKey.Close();
    outKey.sam = (sam == 0) ? this->sam : sam; // TODO: WOW64-Stuff
    DWORD errcode = ::RegCreateKeyExA(this->key, name.c_str(), 0, NULL,
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
        vislib::sys::RegistryKey& outKey, const the::wstring& name,
        REGSAM sam) {
    outKey.Close();
    outKey.sam = (sam == 0) ? this->sam : sam; // TODO: WOW64-Stuff
    DWORD errcode = ::RegCreateKeyExW(this->key, name.c_str(), 0, NULL,
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
        const the::astring& name) {
    DWORD errcode;
    RegistryKey sc;

    errcode = this->OpenSubKey(sc, name);
    if (errcode != ERROR_SUCCESS) return errcode;
    Array<the::astring> subkeys(sc.GetSubKeysA());
    for (size_t i = 0; i < subkeys.Count(); i++) {
        errcode = sc.DeleteSubKey(subkeys[i]);
        if (errcode != ERROR_SUCCESS) return errcode;
    }
    sc.Close();

#ifdef _WINXP32_LEGACY_SUPPORT
    if (dynRegDeleteKeyExA.IsValid()) {
        errcode = dynRegDeleteKeyExA(this->key, name.c_str(),
            this->sam & (KEY_WOW64_32KEY | KEY_WOW64_64KEY), 0);
    } else {
        errcode = ::RegDeleteKeyA(this->key, name.c_str());
    }
#else /* _WINXP32_LEGACY_SUPPORT */
    errcode = ::RegDeleteKeyExA(this->key, name.c_str(),
        this->sam & (KEY_WOW64_32KEY | KEY_WOW64_64KEY), 0);
#endif /* _WINXP32_LEGACY_SUPPORT */
    return errcode;
}


/*
 * vislib::sys::RegistryKey::DeleteSubKey
 */
DWORD vislib::sys::RegistryKey::DeleteSubKey(
        const the::wstring& name) {
    DWORD errcode;
    RegistryKey sc;

    errcode = this->OpenSubKey(sc, name);
    if (errcode != ERROR_SUCCESS) return errcode;
    Array<the::wstring> subkeys(sc.GetSubKeysW());
    for (size_t i = 0; i < subkeys.Count(); i++) {
        errcode = sc.DeleteSubKey(subkeys[i]);
        if (errcode != ERROR_SUCCESS) return errcode;
    }
    sc.Close();

#ifdef _WINXP32_LEGACY_SUPPORT
    if (dynRegDeleteKeyExW.IsValid()) {
        errcode = dynRegDeleteKeyExW(this->key, name.c_str(),
            this->sam & (KEY_WOW64_32KEY | KEY_WOW64_64KEY), 0);
    } else {
        errcode = ::RegDeleteKeyW(this->key, name.c_str());
    }
#else /* _WINXP32_LEGACY_SUPPORT */
    errcode = ::RegDeleteKeyExW(this->key, name.c_str(),
        this->sam & (KEY_WOW64_32KEY | KEY_WOW64_64KEY), 0);
#endif /* _WINXP32_LEGACY_SUPPORT */
    return errcode;
}


/*
 * vislib::sys::RegistryKey::DeleteValue
 */
DWORD vislib::sys::RegistryKey::DeleteValue(
        const the::astring& name) {
    return ::RegDeleteValueA(this->key, name.c_str());
}


/*
 * vislib::sys::RegistryKey::DeleteValue
 */
DWORD vislib::sys::RegistryKey::DeleteValue(
        const the::wstring& name) {
    return ::RegDeleteValueW(this->key, name.c_str());
}


/*
 * vislib::sys::RegistryKey::GetSubKeysA
 */
vislib::Array<the::astring>
vislib::sys::RegistryKey::GetSubKeysA(void) const {
    Array<the::astring> rv;
    DWORD errcode;
    DWORD subKeyCount;
    DWORD subKeyMaxNameLen;
    the::astring str;
    char *strptr;
    errcode = ::RegQueryInfoKeyA(this->key, NULL, NULL, NULL, &subKeyCount,
        &subKeyMaxNameLen, NULL, NULL, NULL, NULL, NULL, NULL);
    if (errcode != ERROR_SUCCESS) {
        rv.Clear();
        return rv;
    }
    str = the::astring(subKeyMaxNameLen, '\0');
    strptr = const_cast<char*>(str.c_str());

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
vislib::Array<the::wstring>
vislib::sys::RegistryKey::GetSubKeysW(void) const {
    Array<the::wstring> rv;
    DWORD errcode;
    DWORD subKeyCount;
    DWORD subKeyMaxNameLen;
    the::wstring str;
    wchar_t *strptr;
    errcode = ::RegQueryInfoKeyW(this->key, NULL, NULL, NULL, &subKeyCount,
        &subKeyMaxNameLen, NULL, NULL, NULL, NULL, NULL, NULL);
    if (errcode != ERROR_SUCCESS) {
        rv.Clear();
        return rv;
    }
    str = the::wstring(subKeyMaxNameLen, L'\0');
    strptr = const_cast<wchar_t*>(str.c_str());

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
vislib::Array<the::astring>
vislib::sys::RegistryKey::GetValueNamesA(void) const {
    Array<the::astring> rv;
    DWORD errcode;
    DWORD valuesCount;
    DWORD valuesMaxNameLen;
    the::astring str;
    char *strptr;
    errcode = ::RegQueryInfoKeyA(this->key, NULL, NULL, NULL, NULL, NULL,
        NULL, &valuesCount, &valuesMaxNameLen, NULL, NULL, NULL);
    if (errcode != ERROR_SUCCESS) {
        rv.Clear();
        return rv;
    }
    str = the::astring(valuesMaxNameLen, '\0');
    strptr = const_cast<char*>(str.c_str());

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
vislib::Array<the::wstring>
vislib::sys::RegistryKey::GetValueNamesW(void) const {
    Array<the::wstring> rv;
    DWORD errcode;
    DWORD valuesCount;
    DWORD valuesMaxNameLen;
    the::wstring str;
    wchar_t *strptr;
    errcode = ::RegQueryInfoKeyW(this->key, NULL, NULL, NULL, NULL, NULL,
        NULL, &valuesCount, &valuesMaxNameLen, NULL, NULL, NULL);
    if (errcode != ERROR_SUCCESS) {
        rv.Clear();
        return rv;
    }
    str = the::wstring(valuesMaxNameLen, L'\0');
    strptr = const_cast<wchar_t*>(str.c_str());

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
        const the::astring& name) const {
    DWORD valtype;
    DWORD errcode = ::RegQueryValueExA(this->key, name.c_str(), NULL, &valtype,
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
        const the::wstring& name) const {
    DWORD valtype;
    DWORD errcode = ::RegQueryValueExW(this->key, name.c_str(), NULL, &valtype,
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
        const the::astring& name, vislib::RawStorage& outData) const {
    DWORD size;
    DWORD errcode = ::RegQueryValueExA(this->key, name.c_str(), NULL, NULL, NULL,
        &size);
    if (errcode != ERROR_SUCCESS) return errcode;
    outData.AssertSize(size);
    errcode = ::RegQueryValueExA(this->key, name.c_str(), NULL, NULL,
        outData.As<uint8_t>(), &size);
    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const the::wstring& name, vislib::RawStorage& outData) const {
    DWORD size;
    DWORD errcode = ::RegQueryValueExW(this->key, name.c_str(), NULL, NULL, NULL,
        &size);
    if (errcode != ERROR_SUCCESS) return errcode;
    outData.AssertSize(size);
    errcode = ::RegQueryValueExW(this->key, name.c_str(), NULL, NULL,
        outData.As<uint8_t>(), &size);
    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const the::astring& name, void* outData,
        size_t dataSize) const {
    DWORD size = static_cast<DWORD>(dataSize);
    DWORD errcode = ::RegQueryValueExA(this->key, name.c_str(), NULL, NULL,
        static_cast<uint8_t*>(outData), &size);
    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const the::wstring& name, void* outData,
        size_t dataSize) const {
    DWORD size = static_cast<DWORD>(dataSize);
    DWORD errcode = ::RegQueryValueExW(this->key, name.c_str(), NULL, NULL,
        static_cast<uint8_t*>(outData), &size);
    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const the::astring& name, the::astring& outStr) const {
    DWORD size;
    DWORD type;
    DWORD errcode = ::RegQueryValueExA(this->key, name.c_str(), NULL, &type, NULL,
        &size);
    if (errcode != ERROR_SUCCESS) return errcode;
    if ((type != REG_SZ) && (type != REG_EXPAND_SZ)) {
        return ERROR_INVALID_DATATYPE;
    }
    errcode = ::RegQueryValueExA(this->key, name.c_str(), NULL, NULL,
        reinterpret_cast<uint8_t*>(the::text::string_buffer_allocate(outStr, size).operator char *()),
        &size);
    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const the::wstring& name, the::wstring& outStr) const {
    DWORD size;
    DWORD type;
    DWORD errcode = ::RegQueryValueExW(this->key, name.c_str(), NULL, &type, NULL,
        &size);
    if (errcode != ERROR_SUCCESS) return errcode;
    if ((type != REG_SZ) && (type != REG_EXPAND_SZ)) {
        return ERROR_INVALID_DATATYPE;
    }
    errcode = ::RegQueryValueExW(this->key, name.c_str(), NULL, NULL,
        reinterpret_cast<uint8_t*>(the::text::string_buffer_allocate(outStr, size).operator wchar_t *()),
        &size);
    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const the::astring& name, the::wstring& outStr) const {
    the::astring v;
    DWORD rv = this->GetValue(name, v);
    if (rv == ERROR_SUCCESS) {
        the::text::string_converter::convert(outStr, v);
    }
    return rv;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const the::wstring& name, the::astring& outStr) const {
    the::wstring v;
    DWORD rv = this->GetValue(name, v);
    if (rv == ERROR_SUCCESS) {
        the::text::string_converter::convert(outStr, v);
    }
    return rv;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const the::astring& name, the::multi_sza& outStrs) const {
    DWORD size;
    DWORD type;
    DWORD errcode = ::RegQueryValueExA(this->key, name.c_str(), NULL, &type, NULL,
        &size);
    if (errcode != ERROR_SUCCESS) return errcode;
    if ((type == REG_SZ) || (type == REG_EXPAND_SZ)) {
        the::astring val;
        errcode = this->GetValue(name, val);
        if (errcode != ERROR_SUCCESS) return errcode;
        outStrs.clear();
        outStrs.add(val.c_str());

    } else if (type == REG_MULTI_SZ) {
        errcode = ::RegQueryValueExA(this->key, name.c_str(), NULL, NULL,
            reinterpret_cast<uint8_t*>(outStrs.allocate(size)), &size);

    } else {
        errcode = ERROR_INVALID_DATATYPE;
    }

    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const the::wstring& name, the::multi_szw& outStrs) const {
    DWORD size;
    DWORD type;
    DWORD errcode = ::RegQueryValueExW(this->key, name.c_str(), NULL, &type, NULL,
        &size);
    if (errcode != ERROR_SUCCESS) return errcode;
    if ((type == REG_SZ) || (type == REG_EXPAND_SZ)) {
        the::wstring val;
        errcode = this->GetValue(name, val);
        if (errcode != ERROR_SUCCESS) return errcode;
        outStrs.clear();
        outStrs.add(val);

    } else if (type == REG_MULTI_SZ) {
        errcode = ::RegQueryValueExW(this->key, name.c_str(), NULL, NULL,
            reinterpret_cast<uint8_t*>(outStrs.allocate(size)), &size);

    } else {
        errcode = ERROR_INVALID_DATATYPE;
    }

    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const the::astring& name, the::multi_szw& outStrs) const {
    return this->GetValue(THE_A2W(name), outStrs);
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const the::wstring& name, the::multi_sza& outStrs) const {
    return this->GetValue(THE_W2A(name), outStrs);
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const the::astring& name,
        vislib::Array<the::astring>& outStrs) const {
    the::multi_sza strs;
    DWORD errcode = this->GetValue(name, strs);
    if (errcode != ERROR_SUCCESS) return errcode;
    outStrs.SetCount(strs.count());
    for (size_t i = 0; i < outStrs.Count(); i++) {
        outStrs[i] = strs[i];
    }
    return ERROR_SUCCESS;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const the::wstring& name,
        vislib::Array<the::wstring>& outStrs) const {
    the::multi_szw strs;
    DWORD errcode = this->GetValue(name, strs);
    if (errcode != ERROR_SUCCESS) return errcode;
    outStrs.SetCount(strs.count());
    for (size_t i = 0; i < outStrs.Count(); i++) {
        outStrs[i] = strs[i];
    }
    return ERROR_SUCCESS;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const the::astring& name,
        vislib::Array<the::wstring>& outStrs) const {
    the::multi_sza strs;
    DWORD errcode = this->GetValue(name, strs);
    if (errcode != ERROR_SUCCESS) return errcode;
    outStrs.SetCount(strs.count());
    for (size_t i = 0; i < outStrs.Count(); i++) {
        the::text::string_converter::convert(outStrs[i], strs[i]);
    }
    return ERROR_SUCCESS;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const the::wstring& name,
        vislib::Array<the::astring>& outStrs) const {
    the::multi_szw strs;
    DWORD errcode = this->GetValue(name, strs);
    if (errcode != ERROR_SUCCESS) return errcode;
    outStrs.SetCount(strs.count());
    for (size_t i = 0; i < outStrs.Count(); i++) {
        the::text::string_converter::convert(outStrs[i], strs[i]);
    }
    return ERROR_SUCCESS;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const the::astring& name, uint32_t& outVal) const {
    DWORD size;
    DWORD type;
    DWORD errcode = ::RegQueryValueExA(this->key, name.c_str(), NULL, &type, NULL,
        NULL);
    if (errcode != ERROR_SUCCESS) return errcode;
    if (type == REG_DWORD) {
        size = sizeof(uint32_t);
        errcode = ::RegQueryValueExA(this->key, name.c_str(), NULL, NULL,
            reinterpret_cast<uint8_t*>(&outVal), &size);

    } else if (type == REG_QWORD) {
        uint64_t val;
        size = sizeof(uint64_t);
        errcode = ::RegQueryValueExA(this->key, name.c_str(), NULL, NULL,
            reinterpret_cast<uint8_t*>(&val), &size);
        if (errcode == ERROR_SUCCESS) {
            outVal = static_cast<uint32_t>(val);
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
        const the::wstring& name, uint32_t& outVal) const {
    DWORD size;
    DWORD type;
    DWORD errcode = ::RegQueryValueExW(this->key, name.c_str(), NULL, &type, NULL,
        NULL);
    if (errcode != ERROR_SUCCESS) return errcode;
    if (type == REG_DWORD) {
        size = sizeof(uint32_t);
        errcode = ::RegQueryValueExW(this->key, name.c_str(), NULL, NULL,
            reinterpret_cast<uint8_t*>(&outVal), &size);

    } else if (type == REG_QWORD) {
        uint64_t val;
        size = sizeof(uint64_t);
        errcode = ::RegQueryValueExW(this->key, name.c_str(), NULL, NULL,
            reinterpret_cast<uint8_t*>(&val), &size);
        if (errcode == ERROR_SUCCESS) {
            outVal = static_cast<uint32_t>(val);
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
        const the::astring& name, uint64_t& outVal) const {
    DWORD size;
    DWORD type;
    DWORD errcode = ::RegQueryValueExA(this->key, name.c_str(), NULL, &type, NULL,
        NULL);
    if (errcode != ERROR_SUCCESS) return errcode;
    if (type == REG_DWORD) {
        uint32_t val;
        size = sizeof(uint32_t);
        errcode = ::RegQueryValueExA(this->key, name.c_str(), NULL, NULL,
            reinterpret_cast<uint8_t*>(&val), &size);
        if (errcode == ERROR_SUCCESS) {
            outVal = static_cast<uint64_t>(val);
        }

    } else if (type == REG_QWORD) {
        size = sizeof(uint64_t);
        errcode = ::RegQueryValueExA(this->key, name.c_str(), NULL, NULL,
            reinterpret_cast<uint8_t*>(&outVal), &size);

    } else {
        errcode = ERROR_INVALID_DATATYPE;
    }

    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValue
 */
DWORD vislib::sys::RegistryKey::GetValue(
        const the::wstring& name, uint64_t& outVal) const {
    DWORD size;
    DWORD type;
    DWORD errcode = ::RegQueryValueExW(this->key, name.c_str(), NULL, &type, NULL,
        NULL);
    if (errcode != ERROR_SUCCESS) return errcode;
    if (type == REG_DWORD) {
        uint32_t val;
        size = sizeof(uint32_t);
        errcode = ::RegQueryValueExW(this->key, name.c_str(), NULL, NULL,
            reinterpret_cast<uint8_t*>(&val), &size);
        if (errcode == ERROR_SUCCESS) {
            outVal = static_cast<uint64_t>(val);
        }

    } else if (type == REG_QWORD) {
        size = sizeof(uint64_t);
        errcode = ::RegQueryValueExW(this->key, name.c_str(), NULL, NULL,
            reinterpret_cast<uint8_t*>(&outVal), &size);

    } else {
        errcode = ERROR_INVALID_DATATYPE;
    }

    return errcode;
}


/*
 * vislib::sys::RegistryKey::GetValueSize
 */
size_t
vislib::sys::RegistryKey::GetValueSize(const the::astring& name) const {
    DWORD size;
    DWORD errcode = ::RegQueryValueExA(this->key, name.c_str(), NULL, NULL, NULL,
        &size);
    if (errcode != ERROR_SUCCESS) return 0;
    return static_cast<size_t>(size);
}


/*
 * vislib::sys::RegistryKey::GetValueSize
 */
size_t
vislib::sys::RegistryKey::GetValueSize(const the::wstring& name) const {
    DWORD size;
    DWORD errcode = ::RegQueryValueExW(this->key, name.c_str(), NULL, NULL, NULL,
        &size);
    if (errcode != ERROR_SUCCESS) return 0;
    return static_cast<size_t>(size);
}


/*
 * vislib::sys::RegistryKey::OpenSubKey
 */
DWORD vislib::sys::RegistryKey::OpenSubKey(
        vislib::sys::RegistryKey& outKey, const the::astring& name,
        REGSAM sam) const {
    outKey.Close();
    outKey.sam = (sam == 0) ? this->sam : sam; // TODO: WOW64-Stuff
    DWORD errcode = ::RegOpenKeyExA(this->key, name.c_str(), 0,
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
        vislib::sys::RegistryKey& outKey, const the::wstring& name,
        REGSAM sam) const {
    outKey.Close();
    outKey.sam = (sam == 0) ? this->sam : sam; // TODO: WOW64-Stuff
    DWORD errcode = ::RegOpenKeyExW(this->key, name.c_str(), 0,
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
        const the::astring& name, const vislib::RawStorage& data) {
    return this->SetValue(name, data.As<void>(), data.GetSize());
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::wstring& name, const vislib::RawStorage& data) {
    return this->SetValue(name, data.As<void>(), data.GetSize());
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::astring& name, const void* data, size_t size) {
    return ::RegSetValueExA(this->key, name.c_str(), NULL, REG_BINARY,
        reinterpret_cast<const uint8_t*>(data), static_cast<DWORD>(size));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::wstring& name, const void* data, size_t size) {
    return ::RegSetValueExW(this->key, name.c_str(), NULL, REG_BINARY,
        reinterpret_cast<const uint8_t*>(data), static_cast<DWORD>(size));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::astring& name, const the::astring& str,
        bool expandable) {
    return ::RegSetValueExA(this->key, name.c_str(), NULL,
        expandable ? REG_EXPAND_SZ : REG_SZ,
        reinterpret_cast<const uint8_t*>(str.c_str()),
        static_cast<DWORD>(str.size() + 1));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::wstring& name, const the::wstring& str,
        bool expandable) {
    return ::RegSetValueExW(this->key, name.c_str(), NULL,
        expandable ? REG_EXPAND_SZ : REG_SZ,
        reinterpret_cast<const uint8_t*>(str.c_str()),
        static_cast<DWORD>((str.size() + 1) * sizeof(wchar_t)));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::astring& name, const the::wstring& str,
        bool expandable) {
    return this->SetValue(name, the::text::string_converter::to_a(str),
        expandable);
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::wstring& name, const the::astring& str,
        bool expandable) {
    return this->SetValue(name, the::text::string_converter::to_w(str),
        expandable);
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::astring& name, const the::multi_sza& strs) {
    return ::RegSetValueExA(this->key, name.c_str(), NULL, REG_MULTI_SZ,
        reinterpret_cast<const uint8_t*>(strs.data()),
        static_cast<DWORD>(strs.size()));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::wstring& name, const the::multi_szw& strs) {
    return ::RegSetValueExW(this->key, name.c_str(), NULL, REG_MULTI_SZ,
        reinterpret_cast<const uint8_t*>(strs.data()),
        static_cast<DWORD>(strs.size() * sizeof(wchar_t)));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::astring& name, const the::multi_szw& strs) {
    the::multi_sza msz;
    for (size_t i = 0; i < strs.size(); i++) {
        msz.add(the::text::string_converter::to_a(strs[i]));
    }
    return this->SetValue(name, msz);
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::wstring& name, const the::multi_sza& strs) {
    the::multi_szw msz;
    for (size_t i = 0; i < strs.size(); i++) {
        msz.add(the::text::string_converter::to_w(strs[i]));
    }
    return this->SetValue(name, msz);
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::astring& name,
        const vislib::Array<the::astring>& strs) {
    the::multi_sza msz;
    for (size_t i = 0; i < strs.Count(); i++) {
        if (!strs[i].empty()) msz.add(strs[i]);
    }
    return this->SetValue(name, msz);
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::wstring& name,
        const vislib::Array<the::wstring>& strs) {
    the::multi_szw msz;
    for (size_t i = 0; i < strs.Count(); i++) {
        if (!strs[i].empty()) msz.add(strs[i]);
    }
    return this->SetValue(name, msz);
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::astring& name,
        const vislib::Array<the::wstring>& strs) {
    the::multi_sza msz;
    for (size_t i = 0; i < strs.Count(); i++) {
        if (!strs[i].empty()) msz.add(the::text::string_converter::to_a(strs[i]));
    }
    return this->SetValue(name, msz);
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::wstring& name,
        const vislib::Array<the::astring>& strs) {
    the::multi_szw msz;
    for (size_t i = 0; i < strs.Count(); i++) {
        if (!strs[i].empty()) msz.add(the::text::string_converter::to_w(strs[i]));
    }
    return this->SetValue(name, msz);
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::astring& name, int32_t val) {
    return this->SetValue(name, static_cast<uint32_t>(val));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::wstring& name, int32_t val) {
    return this->SetValue(name, static_cast<uint32_t>(val));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::astring& name, uint32_t val) {
    return ::RegSetValueExA(this->key, name.c_str(), NULL, REG_DWORD,
        reinterpret_cast<const uint8_t*>(&val), sizeof(uint32_t));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::wstring& name, uint32_t val) {
    return ::RegSetValueExW(this->key, name.c_str(), NULL, REG_DWORD,
        reinterpret_cast<const uint8_t*>(&val), sizeof(uint32_t));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::astring& name, int64_t val) {
    return this->SetValue(name, static_cast<uint64_t>(val));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::wstring& name, int64_t val) {
    return this->SetValue(name, static_cast<uint64_t>(val));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::astring& name, uint64_t val) {
    return ::RegSetValueExA(this->key, name.c_str(), NULL, REG_QWORD,
        reinterpret_cast<const uint8_t*>(&val), sizeof(uint64_t));
}


/*
 * vislib::sys::RegistryKey::SetValue
 */
DWORD vislib::sys::RegistryKey::SetValue(
        const the::wstring& name, uint64_t val) {
    return ::RegSetValueExW(this->key, name.c_str(), NULL, REG_QWORD,
        reinterpret_cast<const uint8_t*>(&val), sizeof(uint64_t));
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
