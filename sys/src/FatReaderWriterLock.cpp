/*
 * FatReaderWriterLock.cpp
 *
 * Copyright (C) 2006 - 2011 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/FatReaderWriterLock.h"
#include "vislib/Thread.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::sys::FatReaderWriterLock::FatReaderWriterLock
 */
vislib::sys::FatReaderWriterLock::FatReaderWriterLock(void)
        : AbstractReaderWriterLock()/*, exThread(0), exThreadCnt(0),
        exThreadWait(true, true), lock(), shThreads,
        shThreadWait(true, true)*/ {
    // Intentionally empty
}


/*
 * vislib::sys::FatReaderWriterLock::~FatReaderWriterLock
 */
vislib::sys::FatReaderWriterLock::~FatReaderWriterLock(void) {
    // Make sure nothing is locked any more!
    // TODO: Implement
}


/*
 * vislib::sys::FatReaderWriterLock::HasExclusiveLock
 */
bool vislib::sys::FatReaderWriterLock::HasExclusiveLock(void) {

    // TODO: Implement

    return false;
}


/*
 * vislib::sys::FatReaderWriterLock::HasSharedLock
 */
bool vislib::sys::FatReaderWriterLock::HasSharedLock(void) {

    // TODO: Implement

    return false;
}


/*
 * vislib::sys::FatReaderWriterLock::LockExclusive
 */
void vislib::sys::FatReaderWriterLock::LockExclusive(void) {

    // TODO: Implement

}


/*
 * vislib::sys::FatReaderWriterLock::LockShared
 */
void vislib::sys::FatReaderWriterLock::LockShared(void) {

    // TODO: Implement

}


/*
 * vislib::sys::FatReaderWriterLock::UnlockExclusive
 */
void vislib::sys::FatReaderWriterLock::UnlockExclusive(void) {

    // TODO: Implement

}


/*
 * vislib::sys::FatReaderWriterLock::UnlockShared
 */
void vislib::sys::FatReaderWriterLock::UnlockShared(void) {

    // TODO: Implement

}


/*
 * vislib::sys::FatReaderWriterLock::FatReaderWriterLock
 */
vislib::sys::FatReaderWriterLock::FatReaderWriterLock(
        const vislib::sys::FatReaderWriterLock& src) {
    throw UnsupportedOperationException("FatReaderWriterLock::CopyCtor",
        __FILE__, __LINE__);
}


/*
 * vislib::sys::FatReaderWriterLock::operator=
 */
vislib::sys::FatReaderWriterLock&
vislib::sys::FatReaderWriterLock::operator=(
        const vislib::sys::FatReaderWriterLock& rhs) {
    throw UnsupportedOperationException("FatReaderWriterLock::operator=",
        __FILE__, __LINE__);
}
