/*
 * CriticalSection.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CRITICALSECTION_H_INCLUDED
#define VISLIB_CRITICALSECTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifdef _WIN32
#include <windows.h>
#else /* _WIN32 */
#include "Mutex.h"
#endif /* _WIN32 */

#include "Lockable.h"
#include "SyncObject.h"


namespace vislib {
namespace sys {

    /**
     * Implements a critical section.
     *
     * Implementation notes: On Windows, the implementation uses a critical 
     * section, which cannot be used for inter-process synchronisation. Only
     * threads of a single process can be synchronised using this class. Use
     * Mutex or Semaphore, if you need inter-process synchronisation or a
     * TryLock() method on systems less than Windows NT 4. Note, 
     * that critical sections are faster than Mutexes or Semaphores.
     *
     * You must compile your program with _WIN32_WINNT defined as 0x0400 or 
     * later to use TryLock on the critical section. TryLock() will always fail
     * otherwise.
     *
     * On Linux systems, the critical section is emulated using a system mutex.
     *
     * @author Christoph Mueller
     */
    class CriticalSection : public SyncObject {

    public:

        /**
         * Create a new critical section, which is initially not locked.
         */
        CriticalSection(void);

        /** Dtor. */
        virtual ~CriticalSection(void);

        /**
         * Enter the crititcal section for the calling thread. The method blocks
         * until the lock is acquired. 
         *
         * @throws SystemException If the lock could not be acquired.
         */
        virtual void Lock(void);

        /**
         * Try to enter the critical section. If another thread is already in 
         * the critical section, the method will return immediately and the r
         * eturn value is false. The method is therefore non-blocking.
         *
         * NOTE: This method will always return fail on Windows systems prior
         * to Windows NT 4. Only programs that are compiled with _WIN32_WINNT 
         * defined as 0x0400 support this method.
         *
         * @return true, if the lock was acquired, false, if not.
         *
         * @throws UnsupportedOperationException On system prior to Windows 
         *                                       NT 4.
         * @throws SystemException If an error occurred when trying to acquire
         *                         the lock.
         *
         */
        bool TryLock(void);

        /**
         * Leave the critical section.
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
        CriticalSection(const CriticalSection& rhs);

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If (this != &rhs).
         */
        CriticalSection& operator =(const CriticalSection& rhs);

#ifdef _WIN32

        /** The OS critical section. */
        CRITICAL_SECTION critSect;

#else /* _WIN32 */

        /** The mutex used for protecting the critical section. */
        Mutex mutex;

#endif /* _WIN32 */
    };


    /** name typedef for Lockable with this SyncObject */
    typedef Lockable<CriticalSection> CriticalSectionLockable;


} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CRITICALSECTION_H_INCLUDED */
