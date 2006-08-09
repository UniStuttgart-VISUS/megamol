/*
 * CriticalSection.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CRITICALSECTION_H_INCLUDED
#define VISLIB_CRITICALSECTION_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#ifdef _WIN32
#include <windows.h>
#else /* _WIN32 */
#include "Mutex.h"
#endif /* _WIN32 */

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
        ~CriticalSection(void);

        /**
         * Enter the crititcal section for the calling thread. The method blocks
		 * until the lock is acquired. 
         *
         * @return true, if the lock was acquired, false, if an error occured.
         */
        virtual bool Lock(void);

        /**
         * Try to enter the critical section. If another thread is already in 
		 * the critical section, the method will return immediately and the r
		 * eturn value is false. The method is therefore non-blocking.
		 *
		 * NOTE: This method will always return false on Windows systems prior
		 * to Windows NT 4. Only programs that are compiled with _WIN32_WINNT 
		 * defined as 0x0400 support this method.
         *
         * @return true, if the lock was acquired, false, if not.
         */
        bool TryLock(void);

        /**
         * Leave the critical section.
         *
         * @return true in case of success, false otherwise.
         */
        virtual bool Unlock(void);

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

} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_CRITICALSECTION_H_INCLUDED */
