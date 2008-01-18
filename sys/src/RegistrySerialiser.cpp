/*
 * RegistrySerialiser.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/RegistrySerialiser.h"

#include "vislib/IllegalParamException.h"
#include "vislib/SystemException.h"
#include "vislib/UnsupportedOperationException.h"


#ifdef _WIN32

namespace vislib {
namespace sys {

    /**
     *
     */
    template<class T, DWORD R, class C, LONG (APIENTRY* F)(HKEY, const C *, 
        DWORD *, DWORD *, BYTE *, DWORD *)>
    static void IntegralDeserialise(T& outValue, HKEY hKey, const C *name) {
        DWORD size = 0;                 // Size of stored value in bytes.
        DWORD type = 0;                 // Type of stored value.
        LONG result = ERROR_SUCCESS;    // API call return values.

        /* Sanity check. */
        if (name == NULL) {
            throw IllegalParamException("name", __FILE__, __LINE__);
        }

        /* Retrieve size of registry value. */
        if ((result = F(hKey, name, 0, &type, NULL, &size)) 
                != ERROR_SUCCESS) {
            throw SystemException(result, __FILE__, __LINE__);
        }

        /* Sanity checks. */
        if ((type != R) && (size != sizeof(T))) {
            throw IllegalParamException("outValue", __FILE__, __LINE__);
        }

        /* Get data. */
        if ((result = F(hKey, name, 0, &type, reinterpret_cast<BYTE *>(
                &outValue), &size)) != ERROR_SUCCESS) {
            throw SystemException(result, __FILE__, __LINE__);
        }
    }


    /**
     *
     */
    template<class T, DWORD R, class C, LONG (APIENTRY* F)(HKEY, const C *, 
        DWORD, DWORD, const BYTE *, DWORD)>
    static void IntegralSerialise(const T& value, HKEY hKey, const C *name) {
        LONG result = F(hKey, name, 0, R, reinterpret_cast<const BYTE *>(
            &value), sizeof(T));

        if (result != ERROR_SUCCESS) {
            throw SystemException(result, __FILE__, __LINE__);
        }
    }


    /**
     *
     */
    template<class T, LONG (APIENTRY* F)(HKEY, const typename T::Char *,
        DWORD *, DWORD *, BYTE *, DWORD *), class C>
    static void StringDeserialise(T& outValue, HKEY hKey, const C *name) {
        DWORD size = 0;                 // Size of stored value in bytes.
        DWORD type = 0;                 // Type of stored value.
        LONG result = ERROR_SUCCESS;    // API call return values.
        BYTE *outPtr = NULL;            // Pointer to buffer of 'outValue'.
        T tName(name);                  // Copy of name with correct charset.

        /* Sanity check. */
        if (name == NULL) {
            throw IllegalParamException("name", __FILE__, __LINE__);
        }

        /* Retrieve size of registry value. */
        if ((result = F(hKey, tName.PeekBuffer(), 0, &type, NULL, &size)) 
                != ERROR_SUCCESS) {
            throw SystemException(result, __FILE__, __LINE__);
        }

        /* Sanity checks. */
        if (type != REG_SZ) {
            throw IllegalParamException("outValue", __FILE__, __LINE__);
        }

        outPtr = reinterpret_cast<BYTE *>(outValue.AllocateBuffer(size));

        /* Get data. */
        if ((result = F(hKey, tName.PeekBuffer(), 0, &type, outPtr, &size)) 
                != ERROR_SUCCESS) {
            throw SystemException(result, __FILE__, __LINE__);
        }
    }


    /**
     *
     */
    template<class T, LONG (APIENTRY* F)(HKEY, const typename T::Char *,
        DWORD, DWORD, const BYTE *, DWORD), class C>
    static void StringSerialise(const T& value, HKEY hKey, const C *name) {
        T tName(name);                  // Copy of name with correct charset.
        LONG result = F(hKey, tName.PeekBuffer(), 0, REG_SZ,
            reinterpret_cast<const BYTE *>(value.PeekBuffer()),
            value.Length() + 1);

        if (result != ERROR_SUCCESS) {
            throw SystemException(result, __FILE__, __LINE__);
        }
    }

} /* end namespace sys */
} /* end namespace vislib */


/*
 * vislib::sys::RegistrySerialiser::RegistrySerialiser
 */
vislib::sys::RegistrySerialiser::RegistrySerialiser(const char *subKey, 
        HKEY hKey) : hBaseKey(NULL) {
    DWORD createDisp = 0;
    LONG result = ::RegCreateKeyExA(hKey, subKey, 0, NULL, 
        REG_OPTION_NON_VOLATILE, KEY_READ | KEY_WRITE, NULL, &this->hBaseKey, 
        &createDisp);

    if (result != ERROR_SUCCESS) {
        // Note: This is safe because not resources that must be deleted in the
        // dtor have been allocated.
        throw SystemException(result, __FILE__, __LINE__);
    }
}


/*
 * vislib::sys::RegistrySerialiser::RegistrySerialiser
 */
vislib::sys::RegistrySerialiser::RegistrySerialiser(const wchar_t *subKey, 
        HKEY hKey) : hBaseKey(NULL) {
    DWORD createDisp = 0;
    LONG result = ::RegCreateKeyExW(hKey, subKey, 0, NULL, 
        REG_OPTION_NON_VOLATILE, KEY_READ | KEY_WRITE, NULL, &this->hBaseKey, 
        &createDisp);

    if (result != ERROR_SUCCESS) {
        // Note: This is safe because not resources that must be deleted in the
        // dtor have been allocated.
        throw SystemException(result, __FILE__, __LINE__);
    }
}


/*
 * vislib::sys::RegistrySerialiser::~RegistrySerialiser
 */
vislib::sys::RegistrySerialiser::~RegistrySerialiser(void) {
    if (this->hBaseKey != NULL) {
        ::RegCloseKey(this->hBaseKey);
        this->hBaseKey = NULL;
    }
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(bool& outValue, 
        const char *name) {
    this->deserialiseAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(bool& outValue, 
        const wchar_t *name) {
    this->deserialiseAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(char& outValue, 
        const char *name) {
    this->deserialiseAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(char& outValue, 
        const wchar_t *name) {
    this->deserialiseAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(wchar_t& outValue, 
        const char *name) {
    this->deserialiseAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(wchar_t& outValue, 
        const wchar_t *name) {
    this->deserialiseAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(INT8& outValue, 
        const char *name) {
    this->deserialiseAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(INT8& outValue, 
        const wchar_t *name) {
    this->deserialiseAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(UINT8& outValue, 
        const char *name) {
    this->deserialiseAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(UINT8& outValue, 
        const wchar_t *name) {
    this->deserialiseAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(INT16& outValue, 
        const char *name) {
    this->deserialiseAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(INT16& outValue, 
        const wchar_t *name) {
    this->deserialiseAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(UINT16& outValue, 
        const char *name) {
    this->deserialiseAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(UINT16& outValue, 
        const wchar_t *name) {
    this->deserialiseAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(INT32& outValue, 
        const char *name) {
    this->deserialiseAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(INT32& outValue, 
        const wchar_t *name) {
    this->deserialiseAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(UINT32& outValue, 
        const char *name) {
    IntegralDeserialise<UINT32, REG_DWORD, char, ::RegQueryValueExA>(
        outValue, this->hBaseKey, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(UINT32& outValue, 
        const wchar_t *name) {
    IntegralDeserialise<UINT32, REG_DWORD, wchar_t, ::RegQueryValueExW>(
        outValue, this->hBaseKey, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(INT64& outValue, 
        const char *name) {
    UINT64 value;
    this->Deserialise(value, name);
    outValue = static_cast<INT64>(value);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(INT64& outValue, 
        const wchar_t *name) {
    UINT64 value;
    this->Deserialise(value, name);
    outValue = static_cast<INT64>(value);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(UINT64& outValue, 
        const char *name) {
    IntegralDeserialise<UINT64, REG_QWORD, char, ::RegQueryValueExA>(
        outValue, this->hBaseKey, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(UINT64& outValue, 
        const wchar_t *name) {
    IntegralDeserialise<UINT64, REG_QWORD, wchar_t, ::RegQueryValueExW>(
        outValue, this->hBaseKey, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(StringA& outValue, 
        const char *name) {
    StringDeserialise<StringA, ::RegQueryValueExA, char>(outValue, 
        this->hBaseKey, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(StringA& outValue, 
        const wchar_t *name) {
    StringDeserialise<StringA, ::RegQueryValueExA, wchar_t>(outValue, 
        this->hBaseKey, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(StringW& outValue, 
        const char *name) {
    StringDeserialise<StringW, ::RegQueryValueExW, char>(outValue, 
        this->hBaseKey, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(StringW& outValue, 
        const wchar_t *name) {
    StringDeserialise<StringW, ::RegQueryValueExW, wchar_t>(outValue, 
        this->hBaseKey, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const bool value, 
        const char *name) {
    this->serialiseAsDword(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const bool value, 
        const wchar_t *name) {
    this->serialiseAsDword(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const char value,
        const char *name) {
    this->serialiseAsDword(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const char value,
        const wchar_t *name) {
    this->serialiseAsDword(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const wchar_t value,
        const char *name) {
    this->serialiseAsDword(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const wchar_t value,
        const wchar_t *name) {
    this->serialiseAsDword(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const INT8 value,
        const char *name) {
    this->serialiseAsDword(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const INT8 value,
        const wchar_t *name) {
    this->serialiseAsDword(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const UINT8 value,
        const char *name) {
    this->serialiseAsDword(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const UINT8 value,
        const wchar_t *name) {
    this->serialiseAsDword(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const INT16 value,
        const char *name) {
    this->serialiseAsDword(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const INT16 value,
        const wchar_t *name) {
    this->serialiseAsDword(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const UINT16 value,
        const char *name) {
    this->serialiseAsDword(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const UINT16 value,
        const wchar_t *name) {
    this->serialiseAsDword(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const INT32 value,
        const char *name) {
    this->serialiseAsDword(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const INT32 value,
        const wchar_t *name) {
    this->serialiseAsDword(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const UINT32 value,
        const char *name) {
    IntegralSerialise<UINT32, REG_DWORD, char, ::RegSetValueExA>(
        value, this->hBaseKey, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const UINT32 value,
        const wchar_t *name) {
    IntegralSerialise<UINT32, REG_DWORD, wchar_t, ::RegSetValueExW>(
        value, this->hBaseKey, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const INT64 value,
        const char *name) {
    IntegralSerialise<INT64, REG_QWORD, char, ::RegSetValueExA>(
        value, this->hBaseKey, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const INT64 value,
        const wchar_t *name) {
    IntegralSerialise<INT64, REG_QWORD, wchar_t, ::RegSetValueExW>(
        value, this->hBaseKey, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const UINT64 value,
        const char *name) {
    IntegralSerialise<UINT64, REG_QWORD, char, ::RegSetValueExA>(
        value, this->hBaseKey, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const UINT64 value,
        const wchar_t *name) {
    IntegralSerialise<UINT64, REG_QWORD, wchar_t, ::RegSetValueExW>(
        value, this->hBaseKey, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const StringA& value,
        const char *name) {
    StringSerialise<StringA, ::RegSetValueExA, char>(value, this->hBaseKey, 
        name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const StringA& value,
        const wchar_t *name) {
    StringSerialise<StringA, ::RegSetValueExA, wchar_t>(value, this->hBaseKey, 
        name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const StringW& value,
        const char *name) {
    StringSerialise<StringW, ::RegSetValueExW, char>(value, this->hBaseKey, 
        name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const StringW& value,
        const wchar_t *name) {
    StringSerialise<StringW, ::RegSetValueExW, wchar_t>(value, this->hBaseKey, 
        name);
}


/*
 * vislib::sys::RegistrySerialiser::RegistrySerialiser
 */
vislib::sys::RegistrySerialiser::RegistrySerialiser(
        const RegistrySerialiser& rhs) : vislib::Serialiser(rhs) {
    throw UnsupportedOperationException("RegistrySerialiser", __FILE__, 
        __LINE__);
}


/*
 * vislib::sys::RegistrySerialiser::operator =
 */
vislib::sys::RegistrySerialiser& vislib::sys::RegistrySerialiser::operator =(
        const RegistrySerialiser& rhs) {
    Serialiser::operator =(rhs);

    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}

#endif /* _WIN32 */
