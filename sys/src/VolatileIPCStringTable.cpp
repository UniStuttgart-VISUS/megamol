/*
 * VolatileIPCStringTable.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/VolatileIPCStringTable.h"
#include "vislib/AlreadyExistsException.h"
#include "vislib/assert.h"
#include "vislib/MissingImplementationException.h"
#include "vislib/StringConverter.h"
#include "vislib/SystemException.h"
#include "vislib/UnsupportedOperationException.h"


#ifdef _WIN32
#define VIPCST_ROOT_KEY HKEY_CURRENT_USER
#define VIPCST_BASE_KEY_NAME L"Software\\VIS\\vislib\\VIPCStrTab\\"
#else /* _WIN32 */
#endif /* _WIN32 */


/*
 * vislib::sys::VolatileIPCStringTable
 *     ::EntryImplementation::EntryImplementation
 */
vislib::sys::VolatileIPCStringTable::EntryImplementation::EntryImplementation(
        const char* name) 
#ifdef _WIN32
        : name(A2W(name)), key(NULL) {
    // intentionally empty

#else /* _WIN32 */
        : name(name) {
    // TODO: Implement
    throw MissingImplementationException("EntryImplementation::Ctor", __FILE__, __LINE__);
#endif /* _WIN32 */
}


/*
 * vislib::sys::VolatileIPCStringTable
 *     ::EntryImplementation::EntryImplementation
 */
vislib::sys::VolatileIPCStringTable::EntryImplementation::EntryImplementation(
        const wchar_t* name) 
#ifdef _WIN32
        : name(name), key(NULL) {
    // intentionally empty

#else /* _WIN32 */
        : name(W2A(name)) {
    // TODO: Implement
    throw MissingImplementationException("EntryImplementation::Ctor", __FILE__, __LINE__);
#endif /* _WIN32 */
}


/*
 * vislib::sys::VolatileIPCStringTable
 *     ::EntryImplementation::~EntryImplementation
 */
vislib::sys::VolatileIPCStringTable::EntryImplementation::~EntryImplementation(
        void) {
#ifdef _WIN32
    if (this->key != NULL) {
        RegCloseKey(this->key);
        StringW keyname(VIPCST_BASE_KEY_NAME);
        keyname.Append(this->name);
        RegDeleteKeyW(VIPCST_ROOT_KEY, keyname);
        this->key = NULL;
    }

#else /* _WIN32 */
    // TODO: Implement
    throw MissingImplementationException("EntryImplementation::Dtor", __FILE__, __LINE__);
#endif /* _WIN32 */
}


/*
 * vislib::sys::VolatileIPCStringTable::EntryImplementation::Create
 */
void vislib::sys::VolatileIPCStringTable::EntryImplementation::Create(void) {
#ifdef _WIN32
    ASSERT(this->key == NULL);
    StringW keyname(VIPCST_BASE_KEY_NAME);
    keyname.Append(this->name);
    DWORD disposition;
    LONG errorVal = RegCreateKeyExW(VIPCST_ROOT_KEY, keyname, 0, NULL, 
        REG_OPTION_VOLATILE, KEY_ALL_ACCESS, NULL, &this->key, &disposition);

    if (errorVal != ERROR_SUCCESS) {
        throw SystemException(errorVal, __FILE__, __LINE__);
    }

    /*if (disposition == REG_OPENED_EXISTING_KEY)*/ {
        RegCloseKey(this->key);

        // try to delete the key (if this is possible the key should not be 
        // open any more, so the entry should be outdated).


        RegDeleteKeyW(VIPCST_ROOT_KEY, keyname);
        this->key = NULL;
        throw AlreadyExistsException("Entry exists", __FILE__, __LINE__);
    }

#else /* _WIN32 */
    // TODO: Implement
    throw MissingImplementationException("EntryImplementation::Create", __FILE__, __LINE__);
#endif /* _WIN32 */
}


/*
 * vislib::sys::VolatileIPCStringTable::EntryImplementation::SetValue
 */
void vislib::sys::VolatileIPCStringTable::EntryImplementation::SetValue(
        const char* value) {
#ifdef _WIN32
    // TODO: Implement
    throw MissingImplementationException("EntryImplementation::SetValue", __FILE__, __LINE__);
#else /* _WIN32 */
    // TODO: Implement
    throw MissingImplementationException("EntryImplementation::SetValue", __FILE__, __LINE__);
#endif /* _WIN32 */
}


/*
 * vislib::sys::VolatileIPCStringTable::EntryImplementation::SetValue
 */
void vislib::sys::VolatileIPCStringTable::EntryImplementation::SetValue(
        const wchar_t* value) {
#ifdef _WIN32
    // TODO: Implement
    throw MissingImplementationException("EntryImplementation::SetValue", __FILE__, __LINE__);
#else /* _WIN32 */
    // TODO: Implement
    throw MissingImplementationException("EntryImplementation::SetValue", __FILE__, __LINE__);
#endif /* _WIN32 */
}


/*
 * vislib::sys::VolatileIPCStringTable::EntryImplementation::GetName
 */
void vislib::sys::VolatileIPCStringTable::EntryImplementation::GetName(
        StringA& outName) const {
#ifdef _WIN32
    outName = W2A(this->name);
#else /* _WIN32 */
    outName = this->name;
#endif /* _WIN32 */
}


/*
 * vislib::sys::VolatileIPCStringTable::EntryImplementation::GetName
 */
void vislib::sys::VolatileIPCStringTable::EntryImplementation::GetName(
        StringW& outName) const {
#ifdef _WIN32
    outName = this->name;
#else /* _WIN32 */
    outName = A2W(this->name);
#endif /* _WIN32 */
}

/*****************************************************************************/


/*
 * vislib::sys::VolatileIPCStringTable::Entry::Entry
 */
vislib::sys::VolatileIPCStringTable::Entry::Entry(
        const vislib::sys::VolatileIPCStringTable::Entry& rhs) 
        : impl(rhs.impl) {
    // intentionally empty
}


/*
 * vislib::sys::VolatileIPCStringTable::Entry::~Entry
 */
vislib::sys::VolatileIPCStringTable::Entry::~Entry() {
    // intentionally empty
}


/*
 * vislib::sys::VolatileIPCStringTable::Entry::SetValue
 */
void vislib::sys::VolatileIPCStringTable::Entry::SetValue(
        const char *value) {
    ASSERT(this->impl.IsNull() == false);
    this->impl->SetValue(value);
}


/*
 * vislib::sys::VolatileIPCStringTable::Entry::SetValue
 */
void vislib::sys::VolatileIPCStringTable::Entry::SetValue(
        const wchar_t *value) {
    ASSERT(this->impl.IsNull() == false);
    this->impl->SetValue(value);
}


/*
 * vislib::sys::VolatileIPCStringTable::Entry::NameA
 */
vislib::StringA vislib::sys::VolatileIPCStringTable::Entry::NameA() const {
    ASSERT(this->impl.IsNull() == false);
    vislib::StringA retval;
    this->impl->GetName(retval);
    return retval;
}


/*
 * vislib::sys::VolatileIPCStringTable::Entry::NameW
 */
vislib::StringW vislib::sys::VolatileIPCStringTable::Entry::NameW() const {
    ASSERT(this->impl.IsNull() == false);
    vislib::StringW retval;
    this->impl->GetName(retval);
    return retval;
}


/*
 * vislib::sys::VolatileIPCStringTable::Entry::operator=
 */
vislib::sys::VolatileIPCStringTable::Entry& 
vislib::sys::VolatileIPCStringTable::Entry::operator=(
        const vislib::sys::VolatileIPCStringTable::Entry& rhs) {
    this->impl = rhs.impl;
    return *this;
}


/*
 * vislib::sys::VolatileIPCStringTable::Entry::operator==
 */
bool vislib::sys::VolatileIPCStringTable::Entry::operator==(
        const vislib::sys::VolatileIPCStringTable::Entry& rhs) {
    return this->impl == rhs.impl;
}


/*
 * vislib::sys::VolatileIPCStringTable::Entry::Entry
 */
vislib::sys::VolatileIPCStringTable::Entry::Entry(void) : impl() {
    // intentionally empty
}

/*****************************************************************************/


/*
 * vislib::sys::VolatileIPCStringTable::GetValue
 */
vislib::StringA vislib::sys::VolatileIPCStringTable::GetValue(
        const char *name) {
    // TODO: Implement
    throw MissingImplementationException("GetValue", __FILE__, __LINE__);
}


/*
 * vislib::sys::VolatileIPCStringTable::GetValue
 */
vislib::StringW vislib::sys::VolatileIPCStringTable::GetValue(
        const wchar_t *name) {
    // TODO: Implement
    throw MissingImplementationException("GetValue", __FILE__, __LINE__);
}


/*
 * vislib::sys::VolatileIPCStringTable::Create
 */
vislib::sys::VolatileIPCStringTable::Entry 
vislib::sys::VolatileIPCStringTable::Create(const char *name, 
        const char *value) {
    Entry entry;
    entry.impl = new EntryImplementation(name);
    entry.impl->Create(); // throws exceptions on error
    entry.impl->SetValue(value);
    return entry;
}


/*
 * vislib::sys::VolatileIPCStringTable::Create
 */
vislib::sys::VolatileIPCStringTable::Entry 
vislib::sys::VolatileIPCStringTable::Create(const wchar_t *name, 
        const wchar_t *value) {
    Entry entry;
    entry.impl = new EntryImplementation(name);
    entry.impl->Create(); // throws exceptions on error
    entry.impl->SetValue(value);
    return entry;
}


/*
 * vislib::sys::VolatileIPCStringTable::VolatileIPCStringTable
 */
vislib::sys::VolatileIPCStringTable::VolatileIPCStringTable(void) {
    throw UnsupportedOperationException("VolatileIPCStringTable::Ctor", 
        __FILE__, __LINE__);
}
