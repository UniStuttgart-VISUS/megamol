/*
 * Semaphore.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SEMAPHORE_H_INCLUDED
#define VISLIB_SEMAPHORE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifdef _WIN32
#include <windows.h>
#else /* _WIN32 */
#include "Mutex.h"
#endif /* _WIN32 */

#include "SyncObject.h"


namespace vislib {
namespace sys {

    /**
     * A platform independent semaphore.
	 *
	 * Implementation notes: On Windows systems, this synchronisation object can
     * be used for inter-process synchronisation tasks. The implementation uses 
     * the semaphore that Windows provides. On Linux systems, the semaphore is 
     * emulated using mutexes.
     *
     * @author Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de)
     */
	class Semaphore : public SyncObject {

    public:

        /**
         * Create a new semaphore. 
		 *
		 * @param initialCount The initial count for the semaphore object. This value
		 *                     must be within [0, maxCount]. If the value is not 
		 *                     within this range, it will be clamped to be valid.
		 * @param maxCount     The maximum count for the semaphore object, which must
		 *                     be greater than zero. If the value is less than 1, it
		 *                     will be corrected to be 1.
         */
        Semaphore(const long initialCount = 1, const long maxCount = 1);

        /** Dtor. */
        ~Semaphore(void);

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
		/** The actual count for the sempahore object. */
		long count;

		/** The maximum count for the semaphore object. */
		long maxCount;

		/** The mutex that is used for protecting the attributed. */
		Mutex mutex;

		/** 
		 * This mutex is used to block the calling thread, if the count
		 * reached zero.
		 */
		Mutex waitMutex;

#endif /* _WIN32 */
	};

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SEMAPHORE_H_INCLUDED */
