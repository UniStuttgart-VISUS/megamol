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
void vislib::sys::RegistrySerialiser::Deserialise(
        Serialisable<CharTraitsA>& inOutSerialisable) {
    DWORD type = 0;
    DWORD size = 0;
    LONG result = ::RegQueryValueExA(this->hBaseKey, 
        inOutSerialisable.GetName().PeekBuffer(),
        0,
        &type,
        NULL,
        &size);

    if (result != ERROR_SUCCESS) {
        throw SystemException(result, __FILE__, __LINE__);
    }

    if (type != inOutSerialisable.getType()) {
        throw SystemException(ERROR_INVALID_PARAMETER, __FILE__, __LINE__);
    }

    // TODO: String must resize here.

    result = ::RegQueryValueExA(this->hBaseKey, 
        inOutSerialisable.GetName().PeekBuffer(),
        0,
        &type,
        inOutSerialisable.peekData(),
        &size);

    if (result != ERROR_SUCCESS) {
        throw SystemException(result, __FILE__, __LINE__);
    }
}


/*
 * vislib::sys::RegistrySerialiser::Deserialise
 */
void vislib::sys::RegistrySerialiser::Deserialise(
        Serialisable<CharTraitsW>& inOutSerialisable) {
    DWORD type = 0;
    DWORD size = 0;
    LONG result = ::RegQueryValueExW(this->hBaseKey, 
        inOutSerialisable.GetName().PeekBuffer(),
        0,
        &type,
        NULL,
        &size);

    if (result != ERROR_SUCCESS) {
        throw SystemException(result, __FILE__, __LINE__);
    }

    if (type != inOutSerialisable.getType()) {
        throw SystemException(ERROR_INVALID_PARAMETER, __FILE__, __LINE__);
    }

    // TODO: String must resize here.

    result = ::RegQueryValueExW(this->hBaseKey, 
        inOutSerialisable.GetName().PeekBuffer(),
        0,
        &type,
        inOutSerialisable.peekData(),
        &size);

    if (result != ERROR_SUCCESS) {
        throw SystemException(result, __FILE__, __LINE__);
    }
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(
        const Serialisable<CharTraitsA>& serialisable) {
    LONG result = ::RegSetValueExA(this->hBaseKey, 
        serialisable.GetName().PeekBuffer(),
        0, 
        serialisable.getType(),
        serialisable.peekData(),
        serialisable.getSize());

    if (result != ERROR_SUCCESS) {
        throw SystemException(result, __FILE__, __LINE__);
    }
}


/*
 * vislib::sys::RegistrySerialiser::Serialise
 */
void vislib::sys::RegistrySerialiser::Serialise(
        const Serialisable<CharTraitsW>& serialisable) {
    LONG result = ::RegSetValueExW(this->hBaseKey, 
        serialisable.GetName().PeekBuffer(),
        0, 
        serialisable.getType(),
        serialisable.peekData(),
        serialisable.getSize());

    if (result != ERROR_SUCCESS) {
        throw SystemException(result, __FILE__, __LINE__);
    }
}


/*
 * vislib::sys::RegistrySerialiser::RegistrySerialiser
 */
vislib::sys::RegistrySerialiser::RegistrySerialiser(
        const RegistrySerialiser& rhs) {
    throw UnsupportedOperationException("RegistrySerialiser", __FILE__, 
        __LINE__);
}


/*
 * vislib::sys::RegistrySerialiser::operator =
 */
vislib::sys::RegistrySerialiser& vislib::sys::RegistrySerialiser::operator =(
        const RegistrySerialiser& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}

#endif /* _WIN32 */
