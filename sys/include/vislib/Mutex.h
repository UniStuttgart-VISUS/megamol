/*
 * Mutex.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_MUTEX_H_INCLUDED
#define VISLIB_MUTEX_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifdef _WIN32
#include <windows.h>
#else /* _WIN32 */
#include <pthread.h>
#endif /* _WIN32 */

#include "vislib/Lockable.h"
#include "vislib/SyncObject.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {

    /**
     * A platform independent mutex wrapper.
     *
     * Implementation notes: On Windows systems, this mutex can be used for
     * inter-process synchronisation tasks. If you just need to synchronise 
     * threads of a single process, consider using the critical section as it
     * is faster.
     *
     * @author Christoph Mueller
     */
    class Mutex : public SyncObject {

    public:

        /**
         * Create a new mutex, which is initially not locked.
         */
        Mutex(void);

        /** Dtor. */
        virtual ~Mutex(void);

        /**
         * Acquire a lock on the mutex for the calling thread. The method blocks
         * until the lock is acquired. 
         *
         * @throws SystemException If the lock could not be acquired.
         */
        virtual void Lock(void);

        /**
         * Try to acquire a lock on the mutex for the calling thread. If the 
         * mutex is already locked by another thread, the method will return
         * after the specified timeout and the return value is false. The 
         * method is non-blocking if the timeout is set zero.
         *
         * On Linux, the timeout is always zero.
         *
         * @param timeout The timeout for acquiring the mutex.
         *
         * @return true, if the lock was acquired, false, if not.
         *
         * @throws SystemException If an error occurred when trying to acquire
         *                         the lock.
         */
        bool TryLock(const DWORD timeout = 0);

        /**
         * Release the mutex.
         *
         * @throw SystemException If the lock could not be released.
         */
        virtual void Unlock(void);

    private:

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        Mutex(const Mutex& rhs);

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If (this != &rhs).
         */
        Mutex& operator =(const Mutex& rhs);

#ifdef _WIN32

        /** The handle for the OS mutex. */
        HANDLE handle;

#else /* _WIN32 */
        /** The mutex attributes. */
        pthread_mutexattr_t attr;

        /** The mutex object. */
        pthread_mutex_t mutex;

#endif /* _WIN32 */
    };


    /** name typedef for Lockable with this SyncObject */
    typedef Lockable<Mutex> MutexLockable;


} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_MUTEX_H_INCLUDED */
