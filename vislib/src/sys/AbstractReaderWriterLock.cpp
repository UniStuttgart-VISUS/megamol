/*
 * AbstractReaderWriterLock.cpp
 *
 * Copyright (C) 2006 - 2011 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/sys/AbstractReaderWriterLock.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::sys::AbstractReaderWriterLock::AbstractReaderWriterLock
 */
vislib::sys::AbstractReaderWriterLock::AbstractReaderWriterLock() : SyncObject() {
    // Intentionally empty
}


/*
 * vislib::sys::AbstractReaderWriterLock::~AbstractReaderWriterLock
 */
vislib::sys::AbstractReaderWriterLock::~AbstractReaderWriterLock() {
    // Intentionally empty
}


/*
 * vislib::sys::AbstractReaderWriterLock::Lock
 */
void vislib::sys::AbstractReaderWriterLock::Lock() {
    this->LockExclusive();
}


/*
 * vislib::sys::AbstractReaderWriterLock::Unlock
 */
void vislib::sys::AbstractReaderWriterLock::Unlock() {
    this->UnlockExclusive();
}


/*
 * vislib::sys::AbstractReaderWriterLock::AbstractReaderWriterLock
 */
vislib::sys::AbstractReaderWriterLock::AbstractReaderWriterLock(const vislib::sys::AbstractReaderWriterLock& src) {
    throw UnsupportedOperationException("AbstractReaderWriterLock::CopyCtor", __FILE__, __LINE__);
}


/*
 * vislib::sys::AbstractReaderWriterLock::operator=
 */
vislib::sys::AbstractReaderWriterLock& vislib::sys::AbstractReaderWriterLock::operator=(
    const vislib::sys::AbstractReaderWriterLock& rhs) {
    throw UnsupportedOperationException("AbstractReaderWriterLock::operator=", __FILE__, __LINE__);
}
