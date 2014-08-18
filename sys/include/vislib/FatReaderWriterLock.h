/*
 * FatReaderWriterLock.h
 *
 * Copyright (C) 2006 - 2011 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_FATREADERWRITERLOCK_H_INCLUDED
#define VISLIB_FATREADERWRITERLOCK_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/AbstractReaderWriterLock.h"
#include "vislib/Array.h"
#include "vislib/CriticalSection.h"
#include "vislib/Event.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {


    /**
     * This class implements a reader-writer lock in software. This
     * implementation is reentrant and supports down-grading the lock. I. e.
     * a thread holding the exclusive lock may aquire additional shared
     * locks. This is performed by storing the thread IDs currently locking.
     * Thus this implementation is rather in-efficient (O(n) per operation!)
     * and should NOT BE USED in time-critical applications.
     *
     * Upgrading the lock, i. e. aquiring an exclusive lock while already
     * holding a shared lock is not supported and will raise an exception, as
     * this behaviour would result in a possible deadlock.
     * This issue is discussed here:
     * http://www.codeproject.com/KB/threads/ReentrantReaderWriterLock.aspx
     *   "Sorry to burst your bubble, but there is a reason you can't just
     * upgrade a reader-writer lock. If two threads call ClaimReader and then
     * call ClaimWriter at the same time they will deadlock EVERY time (both
     * have a reader lock and attempt to get a writer lock without releasing
     * their reader lock--both have to wait for each other to finish before
     * they can proceed). The .NET documentation on this is misleading
     * (actually the name of the function itself is misleading), probably
     * because they didn't realize this when they initially wrote it. When
     * "upgrading", the .NET reader-write lock actually RELEASES the read
     * lock before acquiring the writer lock--meaning that it wasn't really
     * "upgraded" at all, it is EXACTLY the same as a call to
     * ReleaseReaderLock followed by AcquireWriterLock, except that everyone
     * thinks that it switched to a writer lock without releasing the reader
     * lock first. When you call UpgradeToWriterLock in .NET, it is entirely
     * possible that another writer acquired thew lock (and CHANGED the data
     * you were protecting) BEFORE UpgradeToWriterLock returns. If you google
     * for it, you'll see where the MONO guys ran into this and had to fix
     * their implementation to match Microsoft's."
     */
    class FatReaderWriterLock : public AbstractReaderWriterLock {

    public:

        /** Ctor. */
        FatReaderWriterLock(void);

        /** Dtor. */
        virtual ~FatReaderWriterLock(void);

        /**
         * Asnwer whether or not the calling thread holds the exclusive lock
         *
         * @return True if the calling thread holds the exclusive lock
         */
        bool HasExclusiveLock(void);

        /**
         * Answer whether or not the calling thread holds an shared lock
         *
         * @return True if the calling thread holds an shared lock
         */
        bool HasSharedLock(void);

        /**
         * Aquires an exclusive lock
         *
         * If any other thread holds the exclusive lock or a shared lock
         * the method blocks until these threads release their locks.
         *
         * The calling thread may hold the exclusive lock already.
         *
         * When a thread is waiting to aquire an exclusive lock, no other
         * thread can aquire any further shared locks, unless that thread
         * currently holds the exclusive lock.
         *
         * @throws IllegalStateException if the calling thread holds a shared
         *                               lock
         */
        virtual void LockExclusive(void);

        /**
         * Aquires a shared lock
         *
         * If any other thread holds the exclusive lock or is waiting for to
         * aquire the exclusive lock this methods blocks until the exclusive
         * lock is released.
         *
         * If the calling thread holds the exclusive lock this method
         * succeeds.
         */
        virtual void LockShared(void);

        /**
         * Release an exclusive lock
         *
         * This method must be called for each time 'LockExclusive' was called
         * to release the exclusive lock.
         *
         * @throws IllegalStateException if the current thread does not hold
         *                               the exclusive lock
         */
        virtual void UnlockExclusive(void);

        /**
         * Release a shared lock
         *
         * This method must be called for each time 'LockShared' was called to
         * release all shared locks.
         *
         * @throws IllegalStateException if the current thread does not hold
         *                               any shared lock
         */
        virtual void UnlockShared(void);

    private:

        /** Forbidden copy ctor. */
        FatReaderWriterLock(const FatReaderWriterLock& src);

        /** Forbidden assignment operator */
        FatReaderWriterLock& operator=(const FatReaderWriterLock& rhs);

        /** The lock used for exclusive locking */
        vislib::sys::CriticalSection exclusiveLock;

        /** ID of the thread currently locked exclusively */
        DWORD exThread;

        /** Number of exclusive locks issued */
        unsigned int exThreadCnt;

        /**
         * Event to make the exclusive lock wait until all shared locks have
         * returned
         */
        vislib::sys::Event exclusiveWait;

        /** The lock for the counter of shared locks */
        vislib::sys::CriticalSection sharedLock;

        /** List of IDs of the threads that have aquired shared locks */
        Array<DWORD> shThreads;

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_FATREADERWRITERLOCK_H_INCLUDED */

