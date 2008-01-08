/*
 * VolatileIPCStringTable.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/VolatileIPCStringTable.h"
#include "vislib/AlreadyExistsException.h"
#include "vislib/MissingImplementationException.h"
#include "vislib/UnsupportedOperationException.h"

        
/*
 * vislib::sys::VolatileIPCStringTable::Entry::Entry
 */
vislib::sys::VolatileIPCStringTable::Entry::Entry(
        const vislib::sys::VolatileIPCStringTable::Entry& rhs) {
    // TODO: Implement
    throw MissingImplementationException("Entry::Copy Ctor", __FILE__, __LINE__);
}


/*
 * vislib::sys::VolatileIPCStringTable::Entry::~Entry
 */
vislib::sys::VolatileIPCStringTable::Entry::~Entry() {
    // TODO: Implement
    throw MissingImplementationException("Entry::Dtor", __FILE__, __LINE__);
}


/*
 * vislib::sys::VolatileIPCStringTable::Entry::SetValue
 */
void vislib::sys::VolatileIPCStringTable::Entry::SetValue(
        const char *value) {
    // TODO: Implement
    throw MissingImplementationException("Entry::Copy Ctor", __FILE__, __LINE__);
}


/*
 * vislib::sys::VolatileIPCStringTable::Entry::SetValue
 */
void vislib::sys::VolatileIPCStringTable::Entry::SetValue(
        const wchar_t *value) {
    // TODO: Implement
    throw MissingImplementationException("Entry::Copy Ctor", __FILE__, __LINE__);
}


/*
 * vislib::sys::VolatileIPCStringTable::Entry::NameA
 */
vislib::StringA vislib::sys::VolatileIPCStringTable::Entry::NameA() const {
    // TODO: Implement
    throw MissingImplementationException("Entry::Copy Ctor", __FILE__, __LINE__);
}


/*
 * vislib::sys::VolatileIPCStringTable::Entry::NameW
 */
vislib::StringW vislib::sys::VolatileIPCStringTable::Entry::NameW() const {
    // TODO: Implement
    throw MissingImplementationException("Entry::Copy Ctor", __FILE__, __LINE__);
}


/*
 * vislib::sys::VolatileIPCStringTable::Entry::operator=
 */
vislib::sys::VolatileIPCStringTable::Entry& 
vislib::sys::VolatileIPCStringTable::Entry::operator=(
        const vislib::sys::VolatileIPCStringTable::Entry& rhs) {
    // TODO: Implement
    throw MissingImplementationException("Entry::Copy Ctor", __FILE__, __LINE__);
}


/*
 * vislib::sys::VolatileIPCStringTable::Entry::operator==
 */
bool vislib::sys::VolatileIPCStringTable::Entry::operator==(
        const vislib::sys::VolatileIPCStringTable::Entry& rhs) {
    // TODO: Implement
    throw MissingImplementationException("Entry::Copy Ctor", __FILE__, __LINE__);
}


/*
 * vislib::sys::VolatileIPCStringTable::Entry::Entry
 */
vislib::sys::VolatileIPCStringTable::Entry::Entry(void) {
    // TODO: Implement
    throw MissingImplementationException("Entry::Ctor", __FILE__, __LINE__);
}


/*
 * vislib::sys::VolatileIPCStringTable::GetValue
 */
vislib::StringA vislib::sys::VolatileIPCStringTable::GetValue(
        const char *name) {
    // TODO: Implement
    throw MissingImplementationException("Entry::Copy Ctor", __FILE__, __LINE__);
}


/*
 * vislib::sys::VolatileIPCStringTable::GetValue
 */
vislib::StringW vislib::sys::VolatileIPCStringTable::GetValue(
        const wchar_t *name) {
    // TODO: Implement
    throw MissingImplementationException("Entry::Copy Ctor", __FILE__, __LINE__);
}


/*
 * vislib::sys::VolatileIPCStringTable::Create
 */
vislib::sys::VolatileIPCStringTable::Entry 
vislib::sys::VolatileIPCStringTable::Create(const char *name, 
        const char *value) {
    // TODO: Implement
    throw MissingImplementationException("Entry::Copy Ctor", __FILE__, __LINE__);
}


/*
 * vislib::sys::VolatileIPCStringTable::Create
 */
vislib::sys::VolatileIPCStringTable::Entry 
vislib::sys::VolatileIPCStringTable::Create(const wchar_t *name, 
        const wchar_t *value) {
    // TODO: Implement
    throw MissingImplementationException("Entry::Copy Ctor", __FILE__, __LINE__);
}


/*
 * vislib::sys::VolatileIPCStringTable::VolatileIPCStringTable
 */
vislib::sys::VolatileIPCStringTable::VolatileIPCStringTable(void) {
    throw UnsupportedOperationException("VolatileIPCStringTable::Ctor", 
        __FILE__, __LINE__);
}
