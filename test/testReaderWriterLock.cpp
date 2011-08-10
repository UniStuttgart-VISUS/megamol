/*
 * testbezier.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "testReaderWriterLock.h"

#include "testhelper.h"
#include "vislib/Event.h"
#include "vislib/FatReaderWriterLock.h"
#include "vislib/IllegalStateException.h"
#include "vislib/Thread.h"


/* using classes for shorter names */
using vislib::sys::Thread;
using vislib::sys::Event;
using vislib::sys::FatReaderWriterLock;
using vislib::IllegalStateException;


/** locking auto-reset event 1 */
static Event e1;

/** locking auto-reset event 2 */
static Event e2;


/**
 * Simple thread synchronization barrier to be called from the main thread
 */
static void sync1(void) {
    e1.Set();
    e2.Wait();
}


/**
 * Simple thread synchronization barrier to be called from the second thread
 */
static void sync2(void) {
    e1.Wait();
    e2.Set();
}


/**
 * Second test thread
 *
 * @param userData Pointer to the FatReaderWriterLock
 *
 * @return 0
 */
DWORD testThread(void * userData) {
    FatReaderWriterLock& lock = *static_cast<FatReaderWriterLock*>(userData);
    sync2();

    lock.LockShared();
    sync2();
    AssertFalse("Lock Ex open", lock.HasExclusiveLock());
    AssertTrue("Lock Sh locked", lock.HasSharedLock());

    sync2();

    lock.UnlockShared();
    sync2();
    AssertFalse("Lock Ex open", lock.HasExclusiveLock());
    AssertFalse("Lock Sh open", lock.HasSharedLock());

    sync2();
    return 0;
}


/*
 * TestFatReaderWriterLock
 */
void TestFatReaderWriterLock(void) {
    FatReaderWriterLock lock;
    Thread test(&testThread);


    // start condition
    AssertFalse("Lock Ex open", lock.HasExclusiveLock());
    AssertFalse("Lock Sh open", lock.HasSharedLock());


    // simple locking
    lock.LockExclusive();
    AssertTrue("Lock Ex locked", lock.HasExclusiveLock());
    AssertFalse("Lock Sh open", lock.HasSharedLock());

    lock.UnlockExclusive();
    AssertFalse("Lock Ex open", lock.HasExclusiveLock());
    AssertFalse("Lock Sh open", lock.HasSharedLock());

    lock.LockShared();
    AssertFalse("Lock Ex open", lock.HasExclusiveLock());
    AssertTrue("Lock Sh locked", lock.HasSharedLock());

    lock.UnlockShared();
    AssertFalse("Lock Ex open", lock.HasExclusiveLock());
    AssertFalse("Lock Sh open", lock.HasSharedLock());


    // reentrant homogen locking
    lock.LockExclusive();
    AssertTrue("Lock Ex locked", lock.HasExclusiveLock());
    AssertFalse("Lock Sh open", lock.HasSharedLock());

    lock.LockExclusive();
    AssertTrue("Lock Ex locked", lock.HasExclusiveLock());
    AssertFalse("Lock Sh open", lock.HasSharedLock());

    lock.UnlockExclusive();
    AssertTrue("Lock Ex locked", lock.HasExclusiveLock());
    AssertFalse("Lock Sh open", lock.HasSharedLock());

    lock.UnlockExclusive();
    AssertFalse("Lock Ex open", lock.HasExclusiveLock());
    AssertFalse("Lock Sh open", lock.HasSharedLock());


    lock.LockShared();
    AssertFalse("Lock Ex open", lock.HasExclusiveLock());
    AssertTrue("Lock Sh locked", lock.HasSharedLock());

    lock.LockShared();
    AssertFalse("Lock Ex open", lock.HasExclusiveLock());
    AssertTrue("Lock Sh locked", lock.HasSharedLock());

    lock.UnlockShared();
    AssertFalse("Lock Ex open", lock.HasExclusiveLock());
    AssertTrue("Lock Sh locked", lock.HasSharedLock());

    lock.UnlockShared();
    AssertFalse("Lock Ex open", lock.HasExclusiveLock());
    AssertFalse("Lock Sh open", lock.HasSharedLock());


    // downgrade locking
    lock.LockExclusive();
    AssertTrue("Lock Ex locked", lock.HasExclusiveLock());
    AssertFalse("Lock Sh open", lock.HasSharedLock());

    lock.LockShared();
    AssertTrue("Lock Ex locked", lock.HasExclusiveLock());
    AssertTrue("Lock Sh locked", lock.HasSharedLock());

    lock.UnlockShared();
    AssertTrue("Lock Ex locked", lock.HasExclusiveLock());
    AssertFalse("Lock Sh open", lock.HasSharedLock());

    lock.UnlockExclusive();
    AssertFalse("Lock Ex open", lock.HasExclusiveLock());
    AssertFalse("Lock Sh open", lock.HasSharedLock());


    lock.LockExclusive();
    AssertTrue("Lock Ex locked", lock.HasExclusiveLock());
    AssertFalse("Lock Sh open", lock.HasSharedLock());

    lock.LockShared();
    AssertTrue("Lock Ex locked", lock.HasExclusiveLock());
    AssertTrue("Lock Sh locked", lock.HasSharedLock());

    lock.UnlockExclusive();
    AssertFalse("Lock Ex open", lock.HasExclusiveLock());
    AssertTrue("Lock Sh locked", lock.HasSharedLock());

    lock.UnlockShared();
    AssertFalse("Lock Ex open", lock.HasExclusiveLock());
    AssertFalse("Lock Sh open", lock.HasSharedLock());


    // trigger exceptions
    AssertException("Cannot release open Sh lock", lock.UnlockShared(), IllegalStateException);
    AssertException("Cannot release open Ex lock", lock.UnlockExclusive(), IllegalStateException);

    lock.LockShared();
    AssertFalse("Lock Ex open", lock.HasExclusiveLock());
    AssertTrue("Lock Sh locked", lock.HasSharedLock());

    AssertException("Cannot upgrade to Ex lock", lock.LockExclusive(), IllegalStateException);

    lock.UnlockShared();
    AssertFalse("Lock Ex open", lock.HasExclusiveLock());
    AssertFalse("Lock Sh open", lock.HasSharedLock());


    // test with second thread
    test.Start(&lock);
    sync1();

    lock.LockShared();
    AssertFalse("Lock Ex open", lock.HasExclusiveLock());
    AssertTrue("Lock Sh locked", lock.HasSharedLock());
    sync1();

    sync1();

    lock.UnlockShared();
    AssertFalse("Lock Ex open", lock.HasExclusiveLock());
    AssertFalse("Lock Sh open", lock.HasSharedLock());
    sync1();

    sync1();
    test.Join();

}
