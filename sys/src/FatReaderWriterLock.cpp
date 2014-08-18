/*
 * FatReaderWriterLock.cpp
 *
 * Copyright (C) 2006 - 2011 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/FatReaderWriterLock.h"
#include "vislib/IllegalStateException.h"
#include "vislib/Thread.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::sys::FatReaderWriterLock::FatReaderWriterLock
 */
vislib::sys::FatReaderWriterLock::FatReaderWriterLock(void)
        : AbstractReaderWriterLock(), exclusiveLock(), exThread(0),
        exThreadCnt(0), exclusiveWait(true, true), sharedLock(), shThreads() {
    // Intentionally empty
}


/*
 * vislib::sys::FatReaderWriterLock::~FatReaderWriterLock
 */
vislib::sys::FatReaderWriterLock::~FatReaderWriterLock(void) {
    this->exclusiveWait.Wait();
}


/*
 * vislib::sys::FatReaderWriterLock::HasExclusiveLock
 */
bool vislib::sys::FatReaderWriterLock::HasExclusiveLock(void) {
    return (this->exThread == Thread::CurrentID());
}


/*
 * vislib::sys::FatReaderWriterLock::HasSharedLock
 */
bool vislib::sys::FatReaderWriterLock::HasSharedLock(void) {
    bool rv = false;

    this->sharedLock.Lock();
    rv = this->shThreads.Contains(Thread::CurrentID());
    this->sharedLock.Unlock();

    // value of rv cannot change, because it only could be changed by the
    // current thread

    return rv;
}


/*
 * vislib::sys::FatReaderWriterLock::LockExclusive
 */
void vislib::sys::FatReaderWriterLock::LockExclusive(void) {

    this->exclusiveLock.Lock();

    for (SIZE_T i = 0; i < this->shThreads.Count(); i++) {
        if (this->shThreads[i] == Thread::CurrentID()) {
            this->exclusiveLock.Unlock();
            throw IllegalStateException("LockExclusive Upgrade not allowed",
                __FILE__, __LINE__);
        }
    }

    this->exclusiveWait.Wait();

    if (++this->exThreadCnt == 1) {
        this->exThread = Thread::CurrentID();
    }

}


/*
 * vislib::sys::FatReaderWriterLock::LockShared
 */
void vislib::sys::FatReaderWriterLock::LockShared(void) {

    this->exclusiveLock.Lock(); // for down-grade this is reentrant
    this->sharedLock.Lock();

    if ((this->shThreads.IsEmpty()) && (this->exThread != Thread::CurrentID())) {
        this->exclusiveWait.Reset();
    }
    this->shThreads.Add(Thread::CurrentID());

    this->sharedLock.Unlock();
    this->exclusiveLock.Unlock();

}


/*
 * vislib::sys::FatReaderWriterLock::UnlockExclusive
 */
void vislib::sys::FatReaderWriterLock::UnlockExclusive(void) {

    if (this->exThread != Thread::CurrentID()) {
        throw IllegalStateException("UnlockExclusive illegal", __FILE__, __LINE__);
    }

    if (--this->exThreadCnt == 0) {
        this->exThread = 0;

        this->sharedLock.Lock();
        if (!this->shThreads.IsEmpty()) {
            // last exclusive lock closed
            // disallow new exclusives because this would be an upgrade
            this->exclusiveWait.Reset();
        }
        this->sharedLock.Unlock();
    }

    this->exclusiveLock.Unlock();

}


/*
 * vislib::sys::FatReaderWriterLock::UnlockShared
 */
void vislib::sys::FatReaderWriterLock::UnlockShared(void) {

    this->exclusiveLock.Lock(); // reentrant for down-graded locks
    this->sharedLock.Lock();

    INT_PTR pos = this->shThreads.IndexOf(Thread::CurrentID());

    if (pos == Array<DWORD>::INVALID_POS) {
        this->sharedLock.Unlock();
        this->exclusiveLock.Unlock();
        throw IllegalStateException("UnlockShared illegal", __FILE__, __LINE__);
    }

    this->shThreads.RemoveAt(pos);

    if (this->shThreads.IsEmpty()) {
        this->exclusiveWait.Set();
    }

    this->sharedLock.Unlock();
    this->exclusiveLock.Unlock();

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
