/*
 * RegistrySerialiser.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/RegistrySerialiser.h"

#include "vislib/IllegalParamException.h"
#include "vislib/NoSuchElementException.h"
#include "vislib/SystemException.h"
#include "vislib/UnsupportedOperationException.h"


#ifdef _WIN32

namespace vislib {
namespace sys {

    /**
     * This function implements the generic registry deserialisation mechanism
     * for integral types.
     *
     * The template parameters are the following:
     * T: The type of the variable to deserialise. This must be an integral 
     *    type, which of the size can be determined at compile-time.
     * R: The registry value type of T, e.g. REG_DWORD.
     * C: The character type of the name.
     * F: The RegQueryValueEx* function that is compatible with C.
     *
     * @param outValue Receives the deserialised value.
     * @param name     The name of the value to retrieve.
     *
     * @throws IllegalParamException If 'name' is NULL or the type of the value 
     *                               parameter does not match the type of the 
     *                               stored element.
     * @throws SystemException If the registry access failed.
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
     * This function implements the generic registry serialisation mechanism
     * for integral types.
     *
     * The template parameters are the following:
     * T: The type of the variable to serialise. This must be an integral 
     *    type, which of the size can be determined at compile-time.
     * R: The registry value type of T, e.g. REG_DWORD.
     * C: The character type of the name.
     * F: The RegSetValueEx* function that is compatible with C.
     *
     * @param value The variable to be serialised.
     * @param hKey  The registry key to write the value to.
     * @param name  The name of the value.
     *
     * @throws SystemException If the registry access failed.
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
     * This function implements string deserialisation.
     *
     * The template parameters are the following:
     * T: The type of the variable to deserialise. This must be a VISlib String
     *    instantiation.
     * F: The RegQueryValueEx* function which is compatible with T::Char.
     * C: The character type of the name.
     *
     * @param outValue Receives the deserialised value.
     * @param name     The name of the value to retrieve.
     *
     * @throws IllegalParamException If 'name' is NULL or the type of the value 
     *                               parameter does not match the type of the 
     *                               stored element.
     * @throws SystemException If the registry access failed.
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
     * This function implements string serialisation.
     *
     * The template parameters are the following:
     * T: The type of the variable to serialise. This must be a VISlib String
     *    instantiation.
     * F: The RegSetValueEx* function which is compatible with T::Char.
     * C: The character type of the name.
     *
     * @param value The variable to be serialised.
     * @param hKey  The registry key to write the value to.
     * @param name  The name of the value.
     *
     * @throws SystemException If the registry access failed.
     */
    template<class T, LONG (APIENTRY* F)(HKEY, const typename T::Char *,
        DWORD, DWORD, const BYTE *, DWORD), class C>
    static void StringSerialise(const T& value, HKEY hKey, const C *name) {
        T tName(name);                  // Copy of name with correct charset.
        LONG result = F(hKey, tName.PeekBuffer(), 0, REG_SZ,
            reinterpret_cast<const BYTE *>(value.PeekBuffer()),
            (value.Length() + 1) * sizeof(typename T::Char));

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
        HKEY hKey) : Serialiser(SERIALISER_REQUIRES_NAMES), hBaseKey(NULL) {
    this->initialise(subKey, hKey);
}


/*
 * vislib::sys::RegistrySerialiser::RegistrySerialiser
 */
vislib::sys::RegistrySerialiser::RegistrySerialiser(const wchar_t *subKey, 
        HKEY hKey) : Serialiser(SERIALISER_REQUIRES_NAMES), hBaseKey(NULL) {
    this->initialise(subKey, hKey);
}


/*
 * vislib::sys::RegistrySerialiser::RegistrySerialiser
 */
vislib::sys::RegistrySerialiser::RegistrySerialiser(
        const vislib::StringA& subKey, HKEY hKey) 
        : Serialiser(SERIALISER_REQUIRES_NAMES), hBaseKey(NULL) {
    this->initialise(subKey, hKey);
}


/*
 * vislib::sys::RegistrySerialiser::RegistrySerialiser
 */
vislib::sys::RegistrySerialiser::RegistrySerialiser(
        const vislib::StringW& subKey, HKEY hKey) 
        : Serialiser(SERIALISER_REQUIRES_NAMES), hBaseKey(NULL) {
    this->initialise(subKey, hKey);
}


/*
 * vislib::sys::RegistrySerialiser::RegistrySerialiser
 */
vislib::sys::RegistrySerialiser::RegistrySerialiser(
        const RegistrySerialiser& rhs) 
        : vislib::Serialiser(rhs), hBaseKey(NULL) {
    *this = rhs;
    ASSERT(this->keyStack.Top() == this->hBaseKey);
}


/*
 * vislib::sys::RegistrySerialiser::~RegistrySerialiser
 */
vislib::sys::RegistrySerialiser::~RegistrySerialiser(void) {
    this->closeAllRegistry();
}


/*
 * vislib::sys::RegistrySerialiser::ClearKey
 */
void vislib::sys::RegistrySerialiser::ClearKey(const bool includeValues) {
    RegistryKey key(this->keyStack.Top());
    
    Array<StringW> keys = key.GetSubKeysW();
    for (SIZE_T i = 0; i < keys.Count(); i++) {
        key.DeleteSubKey(keys[i]);
    }

    if (includeValues) {
        Array<StringW> values = key.GetValueNamesW();
        for (SIZE_T i = 0; i < values.Count(); i++) {
            key.DeleteValue(values[i]);
        }
    }
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(bool& outValue, 
        const char *name) {
    ASSERT(sizeof(bool) <= sizeof(DWORD));
    DWORD tmp;
    IntegralDeserialise<DWORD, REG_DWORD, char, ::RegQueryValueExA>(
        tmp, this->keyStack.Top(), name);
    outValue = (tmp != 0);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(bool& outValue, 
        const wchar_t *name) {
    ASSERT(sizeof(bool) <= sizeof(DWORD));
    DWORD tmp;
    IntegralDeserialise<DWORD, REG_DWORD, wchar_t, ::RegQueryValueExW>(
        tmp, this->keyStack.Top(), name);
    outValue = (tmp != 0);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(wchar_t& outValue, 
        const char *name) {
    this->deserialiseSignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(wchar_t& outValue, 
        const wchar_t *name) {
    this->deserialiseSignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(INT8& outValue, 
        const char *name) {
    this->deserialiseSignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(INT8& outValue, 
        const wchar_t *name) {
    this->deserialiseSignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(UINT8& outValue, 
        const char *name) {
    this->deserialiseUnsignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(UINT8& outValue, 
        const wchar_t *name) {
    this->deserialiseUnsignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(INT16& outValue, 
        const char *name) {
    this->deserialiseSignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(INT16& outValue, 
        const wchar_t *name) {
    this->deserialiseSignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(UINT16& outValue, 
        const char *name) {
    this->deserialiseUnsignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(UINT16& outValue, 
        const wchar_t *name) {
    this->deserialiseUnsignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(INT32& outValue, 
        const char *name) {
    this->deserialiseSignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(INT32& outValue, 
        const wchar_t *name) {
    this->deserialiseSignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(UINT32& outValue, 
        const char *name) {
    IntegralDeserialise<UINT32, REG_DWORD, char, ::RegQueryValueExA>(
        outValue, this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(UINT32& outValue, 
        const wchar_t *name) {
    IntegralDeserialise<UINT32, REG_DWORD, wchar_t, ::RegQueryValueExW>(
        outValue, this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(INT64& outValue, 
        const char *name) {
    UINT64 value;
    this->Deserialise(value, name);
    outValue = *reinterpret_cast<INT64 *>(&value);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(INT64& outValue, 
        const wchar_t *name) {
    UINT64 value;
    this->Deserialise(value, name);
    outValue = *reinterpret_cast<INT64 *>(&value);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(UINT64& outValue, 
        const char *name) {
    IntegralDeserialise<UINT64, REG_QWORD, char, ::RegQueryValueExA>(
        outValue, this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(UINT64& outValue, 
        const wchar_t *name) {
    IntegralDeserialise<UINT64, REG_QWORD, wchar_t, ::RegQueryValueExW>(
        outValue, this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(float& outValue, 
        const char *name) {
    this->deserialiseAsDword<float, float, char>(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(float& outValue, 
        const wchar_t *name) {
    this->deserialiseAsDword<float, float, wchar_t>(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(double& outValue, 
        const char *name) {
    UINT64 value;
    this->Deserialise(value, name);
    outValue = *reinterpret_cast<double *>(&value);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(double& outValue, 
        const wchar_t *name) {
    UINT64 value;
    this->Deserialise(value, name);
    outValue = *reinterpret_cast<double *>(&value);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(StringA& outValue, 
        const char *name) {
    StringDeserialise<StringA, ::RegQueryValueExA, char>(outValue, 
        this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(StringA& outValue, 
        const wchar_t *name) {
    StringDeserialise<StringA, ::RegQueryValueExA, wchar_t>(outValue, 
        this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(StringW& outValue, 
        const char *name) {
    StringDeserialise<StringW, ::RegQueryValueExW, char>(outValue, 
        this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(StringW& outValue, 
        const wchar_t *name) {
    StringDeserialise<StringW, ::RegQueryValueExW, wchar_t>(outValue, 
        this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::PopKey
 */
void vislib::sys::RegistrySerialiser::PopKey(const bool isSilent) {
    HKEY hKey = this->keyStack.Pop();

    if (!this->keyStack.IsEmpty()) {
        ::RegCloseKey(hKey);

    } else {
        /* The last key must not be removed by the user. */
        ASSERT(hKey == this->hBaseKey);
        this->keyStack.Push(hKey);

        /* Throw if not disabled. */
        if (!isSilent) {
            throw NoSuchElementException("There must be at least one key in "
                "the stack.", __FILE__, __LINE__);
        }
    }
}


/*
 * vislib::sys::RegistrySerialiser::PushKey
 */
void vislib::sys::RegistrySerialiser::PushKey(const char *name) {
    DWORD createDisp = 0;
    HKEY hKey = NULL;
    LONG result = ::RegCreateKeyExA(this->keyStack.Top(), name, 0, NULL,
        REG_OPTION_NON_VOLATILE, KEY_READ | KEY_WRITE, NULL, &hKey,
        &createDisp);

    if (result != ERROR_SUCCESS) {
        throw SystemException(result, __FILE__, __LINE__);
    }

    this->keyStack.Push(hKey);
}


/*
 * vislib::sys::RegistrySerialiser::PushKey
 */
void vislib::sys::RegistrySerialiser::PushKey(const wchar_t *name) {
    DWORD createDisp = 0;
    HKEY hKey = NULL;
    LONG result = ::RegCreateKeyExW(this->keyStack.Top(), name, 0, NULL,
        REG_OPTION_NON_VOLATILE, KEY_READ | KEY_WRITE, NULL, &hKey,
        &createDisp);

    if (result != ERROR_SUCCESS) {
        throw SystemException(result, __FILE__, __LINE__);
    }

    this->keyStack.Push(hKey);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const bool value, 
        const char *name) {
    IntegralSerialise<UINT32, REG_DWORD, char, ::RegSetValueExA>(
        value, this->keyStack.Top(), name);
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
    this->serialiseAsDword0<INT8, unsigned char, char>(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const INT8 value,
        const wchar_t *name) {
    this->serialiseAsDword0<INT8, unsigned char, wchar_t>(value, name);
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
    this->serialiseAsDword0<INT16, UINT16, char>(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const INT16 value,
        const wchar_t *name) {
    this->serialiseAsDword0<INT16, UINT16, wchar_t>(value, name);
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
    this->serialiseAsDword0<INT32, UINT32, char>(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const INT32 value,
        const wchar_t *name) {
    this->serialiseAsDword0<INT32, UINT32, wchar_t>(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const UINT32 value,
        const char *name) {
    IntegralSerialise<UINT32, REG_DWORD, char, ::RegSetValueExA>(
        value, this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const UINT32 value,
        const wchar_t *name) {
    IntegralSerialise<UINT32, REG_DWORD, wchar_t, ::RegSetValueExW>(
        value, this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const INT64 value,
        const char *name) {
    IntegralSerialise<UINT64, REG_QWORD, char, ::RegSetValueExA>(
        *reinterpret_cast<const UINT64 *>(&value), this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const INT64 value,
        const wchar_t *name) {
    IntegralSerialise<UINT64, REG_QWORD, wchar_t, ::RegSetValueExW>(
        *reinterpret_cast<const UINT64 *>(&value), this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const UINT64 value,
        const char *name) {
    IntegralSerialise<UINT64, REG_QWORD, char, ::RegSetValueExA>(
        value, this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const UINT64 value,
        const wchar_t *name) {
    IntegralSerialise<UINT64, REG_QWORD, wchar_t, ::RegSetValueExW>(
        value, this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const float value, 
        const char *name) {
    this->serialiseAsDword0<float, UINT32, char>(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const float value,
        const wchar_t *name) {
this->serialiseAsDword0<float, UINT32, wchar_t>(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const double value, 
        const char *name) {
    ASSERT(sizeof(double) == sizeof(UINT64));
    IntegralSerialise<UINT64, REG_QWORD, char, ::RegSetValueExA>(
        *reinterpret_cast<const UINT64 *>(&value), this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const double value, 
        const wchar_t *name) {
    ASSERT(sizeof(double) == sizeof(UINT64));
    IntegralSerialise<UINT64, REG_QWORD, wchar_t, ::RegSetValueExW>(
        *reinterpret_cast<const UINT64 *>(&value), this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const StringA& value,
        const char *name) {
    StringSerialise<StringA, ::RegSetValueExA, char>(value, this->keyStack.Top(), 
        name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const StringA& value,
        const wchar_t *name) {
    StringSerialise<StringA, ::RegSetValueExA, wchar_t>(value, 
        this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const StringW& value,
        const char *name) {
    StringSerialise<StringW, ::RegSetValueExW, char>(value, 
        this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const StringW& value,
        const wchar_t *name) {
    StringSerialise<StringW, ::RegSetValueExW, wchar_t>(value, 
        this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::operator =
 */
vislib::sys::RegistrySerialiser& vislib::sys::RegistrySerialiser::operator =(
        const RegistrySerialiser& rhs) {
    Serialiser::operator =(rhs);

    if (this != &rhs) {
        this->closeAllRegistry();
        ASSERT(this->keyStack.IsEmpty());
        ASSERT(this->hBaseKey == NULL);

        // MSDN: DuplicateHandle can duplicate handles to the following types 
        // of objects: The handle is returned by the RegCreateKey, 
        // RegCreateKeyEx, RegOpenKey, or RegOpenKeyEx function. Note that 
        // registry key handles returned by the RegConnectRegistry function 
        // cannot be used in a call to DuplicateHandle. Windows Me/98/95: 
        // You cannot use DuplicateHandle to duplicate registry key handles.

        // Note: ::GetCurrentProcess() returns a pseudo-handle which needs not 
        // to be closed.
        if (!::DuplicateHandle(::GetCurrentProcess(), rhs.hBaseKey, 
                ::GetCurrentProcess(), reinterpret_cast<HANDLE *>(
                &this->hBaseKey), 0, FALSE, DUPLICATE_SAME_ACCESS)) {
            this->hBaseKey = NULL;
            SystemException(__FILE__, __LINE__);
        }

        this->keyStack.Push(this->hBaseKey);
        ASSERT(this->keyStack.Top() == this->hBaseKey);
    }

    return *this;
}


/*
 * vislib::sys::RegistrySerialiser::closeAllRegistry
 */
void vislib::sys::RegistrySerialiser::closeAllRegistry(void) {
    while (this->keyStack.IsEmpty()) {
        HKEY hKey = this->keyStack.Pop();
        ASSERT(hKey != NULL);
        ::RegCloseKey(hKey);

        ASSERT((this->keyStack.Count() != 1) 
            || (this->keyStack.Top() == this->hBaseKey));
    }

    this->hBaseKey = NULL;  // Must have been bottom element in stakc.
}


/*
 * vislib::sys::RegistrySerialiser::initialise
 */
void vislib::sys::RegistrySerialiser::initialise(const char *subKey, 
        HKEY hKey) {
    // Assert initialisations that ctor must make in initialisation list:
    ASSERT((this->GetProperties() & SERIALISER_REQUIRES_NAMES) != 0);
    ASSERT(this->hBaseKey == NULL);

    DWORD createDisp = 0;
    LONG result = ::RegCreateKeyExA(hKey, subKey, 0, NULL, 
        REG_OPTION_NON_VOLATILE, KEY_READ | KEY_WRITE, NULL, &this->hBaseKey, 
        &createDisp);

    if (result != ERROR_SUCCESS) {
        // Note: This is safe because not resources that must be deleted in the
        // dtor have been allocated.
        throw SystemException(result, __FILE__, __LINE__);
    }

    this->keyStack.Push(this->hBaseKey);
    ASSERT(this->keyStack.Top() == this->hBaseKey);
}


/* 
 * vislib::sys::RegistrySerialiser::initialise
 */
void vislib::sys::RegistrySerialiser::initialise(const wchar_t *subKey, 
        HKEY hKey) {
    // Assert initialisations that ctor must make in initialisation list:
    ASSERT((this->GetProperties() & SERIALISER_REQUIRES_NAMES) != 0);
    ASSERT(this->hBaseKey == NULL);

    DWORD createDisp = 0;
    LONG result = ::RegCreateKeyExW(hKey, subKey, 0, NULL, 
        REG_OPTION_NON_VOLATILE, KEY_READ | KEY_WRITE, NULL, &this->hBaseKey, 
        &createDisp);

    if (result != ERROR_SUCCESS) {
        // Note: This is safe because not resources that must be deleted in the
        // dtor have been allocated.
        throw SystemException(result, __FILE__, __LINE__);
    }

    this->keyStack.Push(this->hBaseKey);
    ASSERT(this->keyStack.Top() == this->hBaseKey);
}

#endif /* _WIN32 */
