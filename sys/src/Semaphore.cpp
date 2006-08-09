/*
 * Semaphore.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
#include "vislib/Semaphore.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::sys::Semaphore::Semaphore
 */ 
vislib::sys::Semaphore::Semaphore(const long initialCount, const long maxCount) {
	long m = (maxCount > 0) ? maxCount : 1;
	long i = (initialCount < 0) ? 0 : ((initialCount > m) ? m : initialCount);

	ASSERT(m > 0);
	ASSERT(i >= 0);
	ASSERT(i <= m);

#ifdef _WIN32
    this->handle = ::CreateSemaphore(NULL, i, m, NULL);
    ASSERT(this->handle != NULL);

#else /* _WIN32 */
	this->count = i;
	this->maxCount = m;

	if (this->count == 0) {
		this->waitMutex.Lock();
	}

#endif /* _WIN32 */
}


/*
 * vislib::sys::Semaphore::~Semaphore(void) 
 */
vislib::sys::Semaphore::~Semaphore(void) {
#ifdef _WIN32
    ::CloseHandle(this->handle);

#else /* _WIN32 */
	// Do nothing.

#endif /* _WIN32 */
}


/*
 * vislib::sys::Semaphore::Lock
 */
bool vislib::sys::Semaphore::Lock(void) {
#ifdef _WIN32
    switch (::WaitForSingleObject(this->handle, INFINITE)) {

        case WAIT_OBJECT_0:
            return true;

        case WAIT_TIMEOUT:
            /* falls through. */
        case WAIT_ABANDONED:
            /* falls through. */
        default:
            return false;
    }

#else /* _WIN32 */
    this->waitMutex.Lock();			// Wait to be signaled.
	ASSERT(this->count > 0);
	this->mutex.Lock();				// Prevent other calls to Lock()/Unlock().
	this->waitMutex.Unlock();
	
	if (--this->count == 0) {
		this->waitMutex.Lock();
	}

	this->mutex.Unlock();			// Allow other calls to Lock()/Unlock().
	return true;
#endif /* _WIN32 */
}


/*
 * vislib::sys::Semaphore::Unlock
 */
bool vislib::sys::Semaphore::Unlock(void) {
#ifdef _WIN32
    return (::ReleaseSemaphore(this->handle, 1, NULL) == TRUE);

#else /* _WIN32 */
    this->mutex.Lock();				// Prevent other calls to Lock()/Unlock().
	
	if (++this->count > this->maxCount) {
		this->count = this->maxCount;
	}

	if (this->count == 1) {
		this->waitMutex.Unlock();	// Signal a waiting thread.
	}

	this->mutex.Unlock();			// Allow other calls to Lock()/Unlock().
	return true;
#endif /* _WIN32 */
}


/*
 * vislib::sys::Semaphore::Semaphore
 */
vislib::sys::Semaphore::Semaphore(const Semaphore& rhs) {
    throw UnsupportedOperationException(_T("vislib::sys::Semaphore::Semaphore"),
        __FILE__, __LINE__);
}


/*
 * vislib::sys::Semaphore::operator =
 */
vislib::sys::Semaphore& vislib::sys::Semaphore::operator =(
        const Semaphore& rhs) {
    if (this != &rhs) {
        throw IllegalParamException(_T("rhs"), __FILE__, __LINE__);
    }

    return *this;
}
