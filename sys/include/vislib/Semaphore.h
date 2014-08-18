/*
 * Semaphore.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SEMAPHORE_H_INCLUDED
#define VISLIB_SEMAPHORE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifdef _WIN32
#include <windows.h>
#else /* _WIN32 */
#include <semaphore.h>
#endif /* _WIN32 */

#include "Lockable.h"
#include "vislib/SyncObject.h"
#include "vislib/String.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {

    /**
     * A platform-independent semaphore.
     *
     * Named instances of these semaphores can be used for inter-process 
     * synchronisation tasks. These system-wide semaphores are destroyed once
     * the last object is destroyed.
     *
     * Note: Maximum count is not supported on Linux and ignored.
     *
     * @author Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de)
     */
    class Semaphore : public SyncObject {

    public:

        /**
         * Create a new semaphore.
         *
         * If no initial count it set, the semaphore is created with a count of 1, 
         * i. e. the semaphore is initially in signaled state.
         *
         * @param initialCount The initial count for the semaphore object. This 
         *                     value must be within [0, maxCount]. If the value 
         *                     is not within this range, it will be clamped to 
         *                     be valid.
         * @param maxCount     The maximum count for the semaphore object, which
         *                     must be greater than zero. If the value is less 
         *                     than 1, it will be corrected to be 1.
         */
        Semaphore(long initialCount = 1, long maxCount = 1);

        /** 
         * Open or create a new semaphore with the specified name. The ctor 
         * first tries to open an existing semaphore and creates a new one, if
         * such a semaphore does not exist.
         *
         * If no initial count it set, the semaphore is created with a count of 1, 
         * i. e. the semaphore is initially in signaled state.
         *
         * @param name         The name of the semaphore.
         * @param initialCount The initial count for the semaphore object. This 
         *                     value must be within [0, maxCount]. If the value 
         *                     is not within this range, it will be clamped to 
         *                     be valid.
         * @param maxCount     The maximum count for the semaphore object, which
         *                     must be greater than zero. If the value is less 
         *                     than 1, it will be corrected to be 1.
         * @param outIsNew     If not NULL, the ctor returns whether the 
         *                     semaphore was created (true) or opened (false).
         */
        Semaphore(const char *name, long initialCount = 1, long maxCount = 1,
            bool *outIsNew = NULL);

        /** 
         * Open or create a new semaphore with the specified name. The ctor 
         * first tries to open an existing semaphore and creates a new one, if
         * such a semaphore does not exist.
         *
         * If no initial count it set, the semaphore is created with a count of 1, 
         * i. e. the semaphore is initially in signaled state.
         *
         * @param name         The name of the semaphore.
         * @param initialCount The initial count for the semaphore object. This 
         *                     value must be within [0, maxCount]. If the value 
         *                     is not within this range, it will be clamped to 
         *                     be valid.
         * @param maxCount     The maximum count for the semaphore object, which
         *                     must be greater than zero. If the value is less 
         *                     than 1, it will be corrected to be 1.
         * @param outIsNew     If not NULL, the ctor returns whether the 
         *                     semaphore was created (true) or opened (false).
         */
        Semaphore(const wchar_t *name, long initialCount = 1, long maxCount = 1,
            bool *outIsNew = NULL);

        /** Dtor. */
        virtual ~Semaphore(void);

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
         * Try to acquire a lock on the semaphore for the calling thread. If the 
         * semaphore is already locked by another thread, the method will return
         * immediately and the return value is false. The method is therefore 
         * non-blocking.
         *
         * @return true, if the lock was acquired, false, if not.
         *
         * @throws SystemException If an error occurred when trying to acquire
         *                         the lock.
         */
        virtual bool TryLock(void);

        /**
         * Try to acquire a lock on the semaphore for the calling thread for the
         * specified number of milliseconds. If the semaphore could not be locked
         * within the specified amount of time, the method returns and the return 
         * value is false.
         *
         * @return true, if the lock was acquired, false, if not.
         *
         * @throws SystemException If an error occurred when trying to acquire
         *                         the lock.
         */
        virtual bool TryLock(const DWORD timeout);

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

    protected:

        /**
         * Check the initialisation parameters and ensure that
         *
         * - 'inOutMaxCount' is at least 1
         * - 'inOutInitialCount' is within [0, inOutMaxCount]
         */
        static void enforceParamAssertions(long& inOutInitialCount, 
            long& inOutMaxCount);

#ifndef _WIN32
        /** The default permissions assigned to the semaphore. */
        static const int DFT_PERMS;
#endif /* !_WIN32 */

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        Semaphore(const Semaphore& rhs);

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If (this != &rhs).
         */
        Semaphore& operator =(const Semaphore& rhs);

#ifdef _WIN32

        /** The handle for the OS semaphore. */
        HANDLE handle;

#else /* _WIN32 */
        /** The POSIX semaphore handle. */
        sem_t *handle;

        /** The name of the semaphore if it is named. */
        vislib::StringA name;

#endif /* _WIN32 */
    };


    /** name typedef for Lockable with this SyncObject */
    typedef Lockable<Semaphore> SemaphoreLockable;


} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SEMAPHORE_H_INCLUDED */
