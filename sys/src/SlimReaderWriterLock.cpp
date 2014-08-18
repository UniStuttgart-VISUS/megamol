/*
 * SlimReaderWriterLock.cpp
 *
 * Copyright (C) 2006 - 2011 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/SlimReaderWriterLock.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::sys::SlimReaderWriterLock::SlimReaderWriterLock
 */
vislib::sys::SlimReaderWriterLock::SlimReaderWriterLock(void)
        : AbstractReaderWriterLock(),
//#ifdef _WIN32
//#if 0
//#else
        exclusiveLock(), sharedCntLock(), sharedCnt(0), exclusiveWait(true, true)
//#endif
//#else /* _WIN32 */
//#endif /* _WIN32 */
        {
    // Intentionally empty
}


/*
 * vislib::sys::SlimReaderWriterLock::~SlimReaderWriterLock
 */
vislib::sys::SlimReaderWriterLock::~SlimReaderWriterLock(void) {
//#ifdef _WIN32
//#if 0
//#else
    this->exclusiveWait.Wait();
//#endif
//#else /* _WIN32 */
//#endif /* _WIN32 */
}


/*
 * vislib::sys::SlimReaderWriterLock::LockExclusive
 */
void vislib::sys::SlimReaderWriterLock::LockExclusive(void) {
//#ifdef _WIN32
//#if 0
//#else
    this->exclusiveLock.Lock();
    this->exclusiveWait.Wait();
//#endif
//#else /* _WIN32 */
//#endif /* _WIN32 */
}


/*
 * vislib::sys::SlimReaderWriterLock::LockShared
 */
void vislib::sys::SlimReaderWriterLock::LockShared(void) {
//#ifdef _WIN32
//#if 0
//#else
    this->exclusiveLock.Lock();
    this->sharedCntLock.Lock();
    if (++this->sharedCnt == 1) {
        this->exclusiveWait.Reset();
    }
    this->sharedCntLock.Unlock();
    this->exclusiveLock.Unlock();
//#endif
//#else /* _WIN32 */
//#endif /* _WIN32 */
}


/*
 * vislib::sys::SlimReaderWriterLock::UnlockExclusive
 */
void vislib::sys::SlimReaderWriterLock::UnlockExclusive(void) {
//#ifdef _WIN32
//#if 0
//#else
    this->exclusiveLock.Unlock();
//#endif
//#else /* _WIN32 */
//#endif /* _WIN32 */
}


/*
 * vislib::sys::SlimReaderWriterLock::UnlockShared
 */
void vislib::sys::SlimReaderWriterLock::UnlockShared(void) {
//#ifdef _WIN32
//#if 0
//#else
    this->exclusiveLock.Lock();
    this->sharedCntLock.Lock();
    if (--this->sharedCnt == 0) {
        this->exclusiveWait.Set();
    }
    this->sharedCntLock.Unlock();
    this->exclusiveLock.Unlock();
//#endif
//#else /* _WIN32 */
//#endif /* _WIN32 */
}


/*
 * vislib::sys::SlimReaderWriterLock::SlimReaderWriterLock
 */
vislib::sys::SlimReaderWriterLock::SlimReaderWriterLock(
        const vislib::sys::SlimReaderWriterLock& src) {
    throw UnsupportedOperationException("SlimReaderWriterLock::CopyCtor",
        __FILE__, __LINE__);
}


/*
 * vislib::sys::SlimReaderWriterLock::operator=
 */
vislib::sys::SlimReaderWriterLock&
vislib::sys::SlimReaderWriterLock::operator=(
        const vislib::sys::SlimReaderWriterLock& rhs) {
    throw UnsupportedOperationException("SlimReaderWriterLock::operator=",
        __FILE__, __LINE__);
}
