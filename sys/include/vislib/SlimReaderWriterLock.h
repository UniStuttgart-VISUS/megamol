/*
 * SlimReaderWriterLock.h
 *
 * Copyright (C) 2006 - 2011 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SLIMREADERWRITERLOCK_H_INCLUDED
#define VISLIB_SLIMREADERWRITERLOCK_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/AbstractReaderWriterLock.h"
//#ifdef _WIN32
//#if 0
//#else
#include "vislib/CriticalSection.h"
#include "vislib/Event.h"
//#endif
//#else /* _WIN32 */
//#endif /* _WIN32 */


namespace vislib {
namespace sys {


    /**
     * Implements a reader-writer lock which allows for multiple concurrent
     * readers to lock the object or a single exclusive writer
     *
     * This lock is NOT reentrant !
     * Re-entering the lock may result in a dead-lock but behaviour is
     * undefined. It is not guaranteed to throw an exception!
     *
     * You must call the corresponding Unlock methods.
     * Calling non-matching Unlock after a Lock (e. g. 'UnlockShared' after
     * 'LockExclusive' results in undefined behaviour).
     *
     * Enumlation implementation by Glenn Slayden (glenn@glennslayden.com)
     * http://www.glennslayden.com/code/win32/reader-writer-lock
     */
    class SlimReaderWriterLock : public AbstractReaderWriterLock {

    public:

        /** Ctor. */
        SlimReaderWriterLock(void);

        /** Dtor. */
        virtual ~SlimReaderWriterLock(void);

        /**
         * Aquires an exclusive lock
         */
        virtual void LockExclusive(void);

        /**
         * Aquires a shared lock
         */
        virtual void LockShared(void);

        /**
         * Release an exclusive lock
         */
        virtual void UnlockExclusive(void);

        /**
         * Release a shared lock
         */
        virtual void UnlockShared(void);

    private:

        /** Forbidden copy ctor. */
        SlimReaderWriterLock(const SlimReaderWriterLock& src);

        /** Forbidden assignment operator */
        SlimReaderWriterLock& operator=(const SlimReaderWriterLock& rhs);

//#ifdef _WIN32
//#if 0
//        // use 'InitializeSRWLock' supported since Vista
//#else
        // emulate with CriticalSection, Event, and Counter

        /** The lock used for exclusive locking */
        vislib::sys::CriticalSection exclusiveLock;

        /** The lock for the counter of shared locks */
        vislib::sys::CriticalSection sharedCntLock;

        /** The counter of shared locks */
        unsigned long sharedCnt;

        /**
         * Event to make the exclusive lock wait until all shared locks have
         * returned
         */
        vislib::sys::Event exclusiveWait;

//#endif
//#else /* _WIN32 */
//        // use 'pthread_rwlock_t'
//#endif /* _WIN32 */

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SLIMREADERWRITERLOCK_H_INCLUDED */

