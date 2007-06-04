/*
 * IPCSemaphore.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_IPCSEMAPHORE_H_INCLUDED
#define VISLIB_IPCSEMAPHORE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifdef _WIN32
#include "vislib/Semaphore.h"
#else /* _WIN32 */
#include <sys/sem.h>

#include "SyncObject.h"
#endif /* _WIN32 */


namespace vislib {
namespace sys {


    /**
     * This class implements a semaphore which is suitable for inter-process
     * synchronisation on all platforms. 
     * 
     * On Windows systems, this just maps on the standard semaphore of the 
     * system. You should name your semaphore in order to use it for 
     * inter-process synchronisation.
     *
     * On Linux systems, the class wraps a System V semaphore, which 
     * implemented by the system kernel. Only the first character of the name
     * will be used on Linux. The string ctor is only intended for compatibility
     * with Windows.
     *
     * NOTE: YOU SHOULD NOT USE THESE SEMAPHORES FOR SYNCHRONISING THREADS!
     */
#ifdef _WIN32
    typedef Semaphore IPCSemaphore;
#else /* _WIN32 */
    class IPCSemaphore : public SyncObject {

    public:

        /** 
         * Ctor. 
         * TODO: Doku
         */
        IPCSemaphore(const char name, const long initialCount = 1, 
            const long maxCount = 1);

        // TODO: Doku
        IPCSemaphore(const char *name, const long initialCount = 1, 
            const long maxCount = 1);

        /** Dtor. */
        ~IPCSemaphore(void);

        /**
         * Acquire a lock on the semaphore. This method blocks until the lock is
		 * acquired.
         *
         * The lock can be acquired, if the state of the semaphore is signaled, 
         * i. e. the counter is greater than zero. If a lock has been 
         * successfully acquired, the counter is decremented by one and if the
         * counter reaches zero, the state of the semaphore becomes nonsignaled.
         *
         * @throws SystemException If the lock could not be acquired.
         */
        virtual void Lock(void);

        /**
         * Try to acquire a lock. If the semaphore is already locked, the method 
         * will return immediately and the return value is false. The method is 
         * therefore non-blocking.
         *
         * @return true, if the lock was acquired, false, if not.
         *
         * @throws SystemException If an error occurred when trying to acquire
         *                         the lock.
         */
        virtual bool TryLock(void);

        /**
         * Release the semaphore.
         *
         * The counter is incremented by one. The state of the semaphore becomes 
         * signaled, as the counter becomes greater than zero, if the semaphore
         * was successfully released.
         *
         * @throw SystemException If the lock could not be released.
         */
        virtual void Unlock(void);

    private:

        /** The index of this member in the semaphore group. */
        static const int MEMBER_IDX;

        /** The default permissions assigned to the semaphore. */
        static const int DFT_PERMS;

        /** Answer the current value of the semaphore. */
        int getCount(void);

        // TODO Doku
        void init(const char name, const long initialCount, 
            const long maxCount);

        /** ID of the semaphore set we use for this semaphore. */
        int id;

    };
#endif /* _WIN32 */
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_IPCSEMAPHORE_H_INCLUDED */

