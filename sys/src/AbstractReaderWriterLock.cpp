/*
 * AbstractReaderWriterLock.cpp
 *
 * Copyright (C) 2006 - 2011 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractReaderWriterLock.h"
#include "the/not_supported_exception.h"


/*
 * vislib::sys::AbstractReaderWriterLock::AbstractReaderWriterLock
 */
vislib::sys::AbstractReaderWriterLock::AbstractReaderWriterLock(void)
        : SyncObject() {
    // Intentionally empty
}


/*
 * vislib::sys::AbstractReaderWriterLock::~AbstractReaderWriterLock
 */
vislib::sys::AbstractReaderWriterLock::~AbstractReaderWriterLock(void) {
    // Intentionally empty
}


/*
 * vislib::sys::AbstractReaderWriterLock::Lock
 */
void vislib::sys::AbstractReaderWriterLock::Lock(void) {
    this->LockExclusive();
}


/*
 * vislib::sys::AbstractReaderWriterLock::Unlock
 */
void vislib::sys::AbstractReaderWriterLock::Unlock(void) {
    this->UnlockExclusive();
}


/*
 * vislib::sys::AbstractReaderWriterLock::AbstractReaderWriterLock
 */
vislib::sys::AbstractReaderWriterLock::AbstractReaderWriterLock(
        const vislib::sys::AbstractReaderWriterLock& src) {
    throw the::not_supported_exception("AbstractReaderWriterLock::CopyCtor",
        __FILE__, __LINE__);
}


/*
 * vislib::sys::AbstractReaderWriterLock::operator=
 */
vislib::sys::AbstractReaderWriterLock&
vislib::sys::AbstractReaderWriterLock::operator=(
        const vislib::sys::AbstractReaderWriterLock& rhs) {
    throw the::not_supported_exception("AbstractReaderWriterLock::operator=",
        __FILE__, __LINE__);
}
