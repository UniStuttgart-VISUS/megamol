/*
 * RegistrySerialiser.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/RegistrySerialiser.h"

#include "the/argument_exception.h"
#include "the/no_such_element_exception.h"
#include "the/system/system_exception.h"
#include "the/not_supported_exception.h"


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
     * @throws argument_exception If 'name' is NULL or the type of the value 
     *                               parameter does not match the type of the 
     *                               stored element.
     * @throws the::system::system_exception If the registry access failed.
     */
    template<class T, DWORD R, class C, LONG (APIENTRY* F)(HKEY, const C *, 
        DWORD *, DWORD *, uint8_t *, DWORD *)>
    static void IntegralDeserialise(T& outValue, HKEY hKey, const C *name) {
        DWORD size = 0;                 // Size of stored value in bytes.
        DWORD type = 0;                 // Type of stored value.
        LONG result = ERROR_SUCCESS;    // API call return values.

        /* Sanity check. */
        if (name == NULL) {
            throw the::argument_exception("name", __FILE__, __LINE__);
        }

        /* Retrieve size of registry value. */
        if ((result = F(hKey, name, 0, &type, NULL, &size)) 
                != ERROR_SUCCESS) {
            throw the::system::system_exception(result, __FILE__, __LINE__);
        }

        /* Sanity checks. */
        if ((type != R) && (size != sizeof(T))) {
            throw the::argument_exception("outValue", __FILE__, __LINE__);
        }

        /* Get data. */
        if ((result = F(hKey, name, 0, &type, reinterpret_cast<uint8_t *>(
                &outValue), &size)) != ERROR_SUCCESS) {
            throw the::system::system_exception(result, __FILE__, __LINE__);
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
     * @throws the::system::system_exception If the registry access failed.
     */
    template<class T, DWORD R, class C, LONG (APIENTRY* F)(HKEY, const C *, 
        DWORD, DWORD, const uint8_t *, DWORD)>
    static void IntegralSerialise(const T& value, HKEY hKey, const C *name) {
        LONG result = F(hKey, name, 0, R, reinterpret_cast<const uint8_t *>(
            &value), sizeof(T));

        if (result != ERROR_SUCCESS) {
            throw the::system::system_exception(result, __FILE__, __LINE__);
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
     * @throws argument_exception If 'name' is NULL or the type of the value 
     *                               parameter does not match the type of the 
     *                               stored element.
     * @throws the::system::system_exception If the registry access failed.
     */
    template<class T, LONG (APIENTRY* F)(HKEY, const typename T::value_type *,
        DWORD *, DWORD *, uint8_t *, DWORD *), class C>
    static void StringDeserialise(T& outValue, HKEY hKey, const C *name) {
        DWORD size = 0;                 // Size of stored value in bytes.
        DWORD type = 0;                 // Type of stored value.
        LONG result = ERROR_SUCCESS;    // API call return values.
        uint8_t *outPtr = NULL;            // Pointer to buffer of 'outValue'.
        T tName;                  // Copy of name with correct charset.
        the::text::string_converter::convert(tName, name);

        /* Sanity check. */
        if (name == NULL) {
            throw the::argument_exception("name", __FILE__, __LINE__);
        }

        /* Retrieve size of registry value. */
        if ((result = F(hKey, tName.c_str(), 0, &type, NULL, &size)) 
                != ERROR_SUCCESS) {
            throw the::system::system_exception(result, __FILE__, __LINE__);
        }

        /* Sanity checks. */
        if (type != REG_SZ) {
            throw the::argument_exception("outValue", __FILE__, __LINE__);
        }

        outValue = T(size, static_cast<T::value_type>('\0'));
        outPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t *>(outValue.c_str()));

        /* Get data. */
        if ((result = F(hKey, tName.c_str(), 0, &type, outPtr, &size)) 
                != ERROR_SUCCESS) {
            throw the::system::system_exception(result, __FILE__, __LINE__);
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
     * @throws the::system::system_exception If the registry access failed.
     */
    template<class T, LONG (APIENTRY* F)(HKEY, const typename T::value_type *,
        DWORD, DWORD, const uint8_t *, DWORD), class C>
    static void StringSerialise(const T& value, HKEY hKey, const C *name) {
        T tName;                  // Copy of name with correct charset.
        the::text::string_converter::convert(tName, name);
        LONG result = F(hKey, tName.c_str(), 0, REG_SZ,
            reinterpret_cast<const uint8_t *>(value.c_str()),
            static_cast<DWORD>(value.size() * sizeof(typename T::value_type)));

        if (result != ERROR_SUCCESS) {
            throw the::system::system_exception(result, __FILE__, __LINE__);
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
        const the::astring& subKey, HKEY hKey) 
        : Serialiser(SERIALISER_REQUIRES_NAMES), hBaseKey(NULL) {
    this->initialise(subKey.c_str(), hKey);
}


/*
 * vislib::sys::RegistrySerialiser::RegistrySerialiser
 */
vislib::sys::RegistrySerialiser::RegistrySerialiser(
        const the::wstring& subKey, HKEY hKey) 
        : Serialiser(SERIALISER_REQUIRES_NAMES), hBaseKey(NULL) {
    this->initialise(subKey.c_str(), hKey);
}


/*
 * vislib::sys::RegistrySerialiser::RegistrySerialiser
 */
vislib::sys::RegistrySerialiser::RegistrySerialiser(
        const RegistrySerialiser& rhs) 
        : vislib::Serialiser(rhs), hBaseKey(NULL) {
    *this = rhs;
    THE_ASSERT(this->keyStack.Top() == this->hBaseKey);
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
    
    Array<the::wstring> keys = key.GetSubKeysW();
    for (size_t i = 0; i < keys.Count(); i++) {
        key.DeleteSubKey(keys[i]);
    }

    if (includeValues) {
        Array<the::wstring> values = key.GetValueNamesW();
        for (size_t i = 0; i < values.Count(); i++) {
            key.DeleteValue(values[i]);
        }
    }
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(bool& outValue, 
        const char *name) {
    THE_ASSERT(sizeof(bool) <= sizeof(DWORD));
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
    THE_ASSERT(sizeof(bool) <= sizeof(DWORD));
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
void vislib::sys::RegistrySerialiser::Deserialise(int8_t& outValue, 
        const char *name) {
    this->deserialiseSignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(int8_t& outValue, 
        const wchar_t *name) {
    this->deserialiseSignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(uint8_t& outValue, 
        const char *name) {
    this->deserialiseUnsignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(uint8_t& outValue, 
        const wchar_t *name) {
    this->deserialiseUnsignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(int16_t& outValue, 
        const char *name) {
    this->deserialiseSignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(int16_t& outValue, 
        const wchar_t *name) {
    this->deserialiseSignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(uint16_t& outValue, 
        const char *name) {
    this->deserialiseUnsignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(uint16_t& outValue, 
        const wchar_t *name) {
    this->deserialiseUnsignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(int32_t& outValue, 
        const char *name) {
    this->deserialiseSignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(int32_t& outValue, 
        const wchar_t *name) {
    this->deserialiseSignedAsDword(outValue, name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(uint32_t& outValue, 
        const char *name) {
    IntegralDeserialise<uint32_t, REG_DWORD, char, ::RegQueryValueExA>(
        outValue, this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(uint32_t& outValue, 
        const wchar_t *name) {
    IntegralDeserialise<uint32_t, REG_DWORD, wchar_t, ::RegQueryValueExW>(
        outValue, this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(int64_t& outValue, 
        const char *name) {
    uint64_t value;
    this->Deserialise(value, name);
    outValue = *reinterpret_cast<int64_t *>(&value);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(int64_t& outValue, 
        const wchar_t *name) {
    uint64_t value;
    this->Deserialise(value, name);
    outValue = *reinterpret_cast<int64_t *>(&value);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(uint64_t& outValue, 
        const char *name) {
    IntegralDeserialise<uint64_t, REG_QWORD, char, ::RegQueryValueExA>(
        outValue, this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(uint64_t& outValue, 
        const wchar_t *name) {
    IntegralDeserialise<uint64_t, REG_QWORD, wchar_t, ::RegQueryValueExW>(
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
    uint64_t value;
    this->Deserialise(value, name);
    outValue = *reinterpret_cast<double *>(&value);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(double& outValue, 
        const wchar_t *name) {
    uint64_t value;
    this->Deserialise(value, name);
    outValue = *reinterpret_cast<double *>(&value);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(the::astring& outValue, 
        const char *name) {
    StringDeserialise<the::astring, ::RegQueryValueExA, char>(outValue, 
        this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(the::astring& outValue, 
        const wchar_t *name) {
    StringDeserialise<the::astring, ::RegQueryValueExA, wchar_t>(outValue, 
        this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(the::wstring& outValue, 
        const char *name) {
    StringDeserialise<the::wstring, ::RegQueryValueExW, char>(outValue, 
        this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(the::wstring& outValue, 
        const wchar_t *name) {
    StringDeserialise<the::wstring, ::RegQueryValueExW, wchar_t>(outValue, 
        this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::PopKey
 */
void vislib::sys::RegistrySerialiser::PopKey(const bool isSilent) {
    HKEY hKey = this->keyStack.Pop();

    if (!this->keyStack.empty()) {
        ::RegCloseKey(hKey);

    } else {
        /* The last key must not be removed by the user. */
        THE_ASSERT(hKey == this->hBaseKey);
        this->keyStack.Push(hKey);

        /* Throw if not disabled. */
        if (!isSilent) {
            throw the::no_such_element_exception("There must be at least one key in "
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
        throw the::system::system_exception(result, __FILE__, __LINE__);
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
        throw the::system::system_exception(result, __FILE__, __LINE__);
    }

    this->keyStack.Push(hKey);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const bool value, 
        const char *name) {
    IntegralSerialise<uint32_t, REG_DWORD, char, ::RegSetValueExA>(
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
void vislib::sys::RegistrySerialiser::Serialise(const int8_t value,
        const char *name) {
    this->serialiseAsDword0<int8_t, unsigned char, char>(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const int8_t value,
        const wchar_t *name) {
    this->serialiseAsDword0<int8_t, unsigned char, wchar_t>(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const uint8_t value,
        const char *name) {
    this->serialiseAsDword(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const uint8_t value,
        const wchar_t *name) {
    this->serialiseAsDword(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const int16_t value,
        const char *name) {
    this->serialiseAsDword0<int16_t, uint16_t, char>(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const int16_t value,
        const wchar_t *name) {
    this->serialiseAsDword0<int16_t, uint16_t, wchar_t>(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const uint16_t value,
        const char *name) {
    this->serialiseAsDword(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const uint16_t value,
        const wchar_t *name) {
    this->serialiseAsDword(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const int32_t value,
        const char *name) {
    this->serialiseAsDword0<int32_t, uint32_t, char>(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const int32_t value,
        const wchar_t *name) {
    this->serialiseAsDword0<int32_t, uint32_t, wchar_t>(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const uint32_t value,
        const char *name) {
    IntegralSerialise<uint32_t, REG_DWORD, char, ::RegSetValueExA>(
        value, this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const uint32_t value,
        const wchar_t *name) {
    IntegralSerialise<uint32_t, REG_DWORD, wchar_t, ::RegSetValueExW>(
        value, this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const int64_t value,
        const char *name) {
    IntegralSerialise<uint64_t, REG_QWORD, char, ::RegSetValueExA>(
        *reinterpret_cast<const uint64_t *>(&value), this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const int64_t value,
        const wchar_t *name) {
    IntegralSerialise<uint64_t, REG_QWORD, wchar_t, ::RegSetValueExW>(
        *reinterpret_cast<const uint64_t *>(&value), this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const uint64_t value,
        const char *name) {
    IntegralSerialise<uint64_t, REG_QWORD, char, ::RegSetValueExA>(
        value, this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const uint64_t value,
        const wchar_t *name) {
    IntegralSerialise<uint64_t, REG_QWORD, wchar_t, ::RegSetValueExW>(
        value, this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const float value, 
        const char *name) {
    this->serialiseAsDword0<float, uint32_t, char>(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const float value,
        const wchar_t *name) {
this->serialiseAsDword0<float, uint32_t, wchar_t>(value, name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const double value, 
        const char *name) {
    THE_ASSERT(sizeof(double) == sizeof(uint64_t));
    IntegralSerialise<uint64_t, REG_QWORD, char, ::RegSetValueExA>(
        *reinterpret_cast<const uint64_t *>(&value), this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const double value, 
        const wchar_t *name) {
    THE_ASSERT(sizeof(double) == sizeof(uint64_t));
    IntegralSerialise<uint64_t, REG_QWORD, wchar_t, ::RegSetValueExW>(
        *reinterpret_cast<const uint64_t *>(&value), this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const the::astring& value,
        const char *name) {
    StringSerialise<the::astring, ::RegSetValueExA, char>(value, this->keyStack.Top(), 
        name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const the::astring& value,
        const wchar_t *name) {
    StringSerialise<the::astring, ::RegSetValueExA, wchar_t>(value, 
        this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const the::wstring& value,
        const char *name) {
    StringSerialise<the::wstring, ::RegSetValueExW, char>(value, 
        this->keyStack.Top(), name);
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(const the::wstring& value,
        const wchar_t *name) {
    StringSerialise<the::wstring, ::RegSetValueExW, wchar_t>(value, 
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
        THE_ASSERT(this->keyStack.empty());
        THE_ASSERT(this->hBaseKey == NULL);

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
            the::system::system_exception(__FILE__, __LINE__);
        }

        this->keyStack.Push(this->hBaseKey);
        THE_ASSERT(this->keyStack.Top() == this->hBaseKey);
    }

    return *this;
}


/*
 * vislib::sys::RegistrySerialiser::closeAllRegistry
 */
void vislib::sys::RegistrySerialiser::closeAllRegistry(void) {
    while (this->keyStack.empty()) {
        HKEY hKey = this->keyStack.Pop();
        THE_ASSERT(hKey != NULL);
        ::RegCloseKey(hKey);

        THE_ASSERT((this->keyStack.Count() != 1) 
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
    THE_ASSERT((this->GetProperties() & SERIALISER_REQUIRES_NAMES) != 0);
    THE_ASSERT(this->hBaseKey == NULL);

    DWORD createDisp = 0;
    LONG result = ::RegCreateKeyExA(hKey, subKey, 0, NULL, 
        REG_OPTION_NON_VOLATILE, KEY_READ | KEY_WRITE, NULL, &this->hBaseKey, 
        &createDisp);

    if (result != ERROR_SUCCESS) {
        // Note: This is safe because not resources that must be deleted in the
        // dtor have been allocated.
        throw the::system::system_exception(result, __FILE__, __LINE__);
    }

    this->keyStack.Push(this->hBaseKey);
    THE_ASSERT(this->keyStack.Top() == this->hBaseKey);
}


/* 
 * vislib::sys::RegistrySerialiser::initialise
 */
void vislib::sys::RegistrySerialiser::initialise(const wchar_t *subKey, 
        HKEY hKey) {
    // Assert initialisations that ctor must make in initialisation list:
    THE_ASSERT((this->GetProperties() & SERIALISER_REQUIRES_NAMES) != 0);
    THE_ASSERT(this->hBaseKey == NULL);

    DWORD createDisp = 0;
    LONG result = ::RegCreateKeyExW(hKey, subKey, 0, NULL, 
        REG_OPTION_NON_VOLATILE, KEY_READ | KEY_WRITE, NULL, &this->hBaseKey, 
        &createDisp);

    if (result != ERROR_SUCCESS) {
        // Note: This is safe because not resources that must be deleted in the
        // dtor have been allocated.
        throw the::system::system_exception(result, __FILE__, __LINE__);
    }

    this->keyStack.Push(this->hBaseKey);
    THE_ASSERT(this->keyStack.Top() == this->hBaseKey);
}

#endif /* _WIN32 */
