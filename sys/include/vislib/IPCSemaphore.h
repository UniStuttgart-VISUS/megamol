/*
 * IPCSemaphore.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_IPCSEMAPHORE_H_INCLUDED
#define VISLIB_IPCSEMAPHORE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifdef _WIN32
#include "vislib/Semaphore.h"
#else /* _WIN32 */
#include <sys/sem.h>

#include "vislib/SyncObject.h"
#endif /* _WIN32 */
#include "Lockable.h"


namespace vislib {
namespace sys {


    /**
     * NOTE: vislib::sys::IPCSemaphore has been superseded by the named version
     * of vislib::sys:Semaphore. Users are stronly encouraged to use
     * vislib::sys::Semaphore for both, thread and process synchronisation.
     *
     * Rationale: vislib::sys::Semaphore is implemented using POSIX semaphores
     * while vislib::sys::IPCSemaphores use System V kernel semaphores. The
     * naming mechanism and the destruction of abandoned semaphores is less
     * reliable for System V semaphores. Additionally, POSIX semaphores are said
     * to be more performant than System V semaphores.
     *
     * -------------------------------------------------------------------------
     *
     * This class implements a semaphore which is suitable for inter-process
     * synchronisation on all platforms. 
     * 
     * On Windows systems, this just maps on the standard semaphore of the 
     * system. You should name your semaphore in order to use it for 
     * inter-process synchronisation. Note that semaphores for IPC must be 
     * named.
     *
     * On Linux systems, the class wraps a System V semaphore, which 
     * implemented by the system kernel. Only the first character of the name
     * will be used on Linux. The string ctor is only intended for compatibility
     * with Windows. If the name of the semaphore starts with the Windows kernel
     * namespaces "Global\" or "Local\", these will be removed and the first
     * character after the namespace is used for generating the name. The name
     * of the semaphore is created using ftok and the user home directory.
     *
     * Note that the semaphore is removed from the system when the object that
     * originally created it is destroyed!
     *
     * NOTE: YOU SHOULD NOT USE THESE SEMAPHORES FOR SYNCHRONISING THREADS!
     */
#ifdef _WIN32
    class IPCSemaphore : public Semaphore {
#else /* _WIN32 */
    class IPCSemaphore : public SyncObject {
#endif /* _WIN32 */

    public:

        /** 
         * Open or create a new semaphore with the specified name. The ctor 
         * first tries to open an existing semaphore and creates a new one, if
         * such a semaphore does not exist.
         *
         * @param name         The name of the semaphore.
         * @param initialCount The initial count for the semaphore object. This value
         *                     must be within [0, maxCount]. If the value is not 
         *                     within this range, it will be clamped to be valid.
         * @param maxCount     The maximum count for the semaphore object, which must
         *                     be greater than zero. If the value is less than 1, it
         *                     will be corrected to be 1.
         */
        IPCSemaphore(const char name, const long initialCount = 1, 
            const long maxCount = 1);

        /** 
         * Open or create a new semaphore with the specified name. The ctor 
         * first tries to open an existing semaphore and creates a new one, if
         * such a semaphore does not exist.
         *
         * @param name         The name of the semaphore.
         * @param initialCount The initial count for the semaphore object. This value
         *                     must be within [0, maxCount]. If the value is not 
         *                     within this range, it will be clamped to be valid.
         * @param maxCount     The maximum count for the semaphore object, which must
         *                     be greater than zero. If the value is less than 1, it
         *                     will be corrected to be 1.
         */
        IPCSemaphore(const char *name, const long initialCount = 1, 
            const long maxCount = 1);

        /** Dtor. */
        virtual ~IPCSemaphore(void);

#ifndef _WIN32

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

#endif /* !_WIN32 */

    private:

#ifndef _WIN32

        /** The index of this member in the semaphore group. */
        static const int MEMBER_IDX;

        /** The default permissions assigned to the semaphore. */
        static const int DFT_PERMS;

        /** Answer the current value of the semaphore. */
        int getCount(void);

#endif /* !_WIN32 */

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        IPCSemaphore(const IPCSemaphore& rhs);

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If (this != &rhs).
         */
        IPCSemaphore& operator =(const IPCSemaphore& rhs);

        /**
         * Initialise the semaphore.
         *
         * All parameters can be directly passed from the ctor, all possible
         * corrections and the processing of the name is done here.
         *
         * On Windows, the assumption that the semaphore has not yet been 
         * created and the handle is therefore NULL is made.
         *
         * @param name         The name (processing occurrs here).
         * @param initialCount The inital count (possible correction here).
         * @param maxCount     The maximum count (possible correction here).
         */
        void init(const char *name, const long initialCount, 
            const long maxCount);

#ifndef _WIN32

        /** ID of the semaphore set we use for this semaphore. */
        int id;

        /** Remember whether object is the owner that destroys the semaphore. */
        bool isOwner;

        /** Maximum count the semaphore can get. */
        int maxCount;

#endif /* !_WIN32 */

    };


    /** name typedef for Lockable with this SyncObject */
    typedef Lockable<IPCSemaphore> IPCSemaphoreLockable;


} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_IPCSEMAPHORE_H_INCLUDED */
