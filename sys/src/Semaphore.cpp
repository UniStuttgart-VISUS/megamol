/*
 * Semaphore.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/Semaphore.h"

#include "vislib/assert.h"
#include "vislib/error.h"
#include "vislib/IllegalParamException.h"
#include "vislib/SystemException.h"
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
    ::sem_init(&this->handle, 0, i); 

#endif /* _WIN32 */
}


/*
 * vislib::sys::Semaphore::~Semaphore(void) 
 */
vislib::sys::Semaphore::~Semaphore(void) {
#ifdef _WIN32
    ::CloseHandle(this->handle);

#else /* _WIN32 */
    ::sem_destroy(&this->handle);

#endif /* _WIN32 */
}


/*
 * vislib::sys::Semaphore::Lock
 */
void vislib::sys::Semaphore::Lock(void) {
#ifdef _WIN32
    switch (::WaitForSingleObject(this->handle, INFINITE)) {

        case WAIT_OBJECT_0:
            /* falls through. */
        case WAIT_ABANDONED:
            /* Does nothing. */
            break;

        case WAIT_TIMEOUT:
            /* Waiting infinitely should not timeout. */
            ASSERT(false);
            break;

        default:
            throw SystemException(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    if (::sem_wait(&this->handle) == -1) {
        throw SystemException(__FILE__, __LINE__);
    }
#endif /* _WIN32 */
}


/*
 * vislib::sys::Semaphore::TryLock
 */
bool vislib::sys::Semaphore::TryLock(void) {
#ifdef _WIN32
    switch (::WaitForSingleObject(this->handle, 0)) {

        case WAIT_OBJECT_0:
            /* falls through. */
        case WAIT_ABANDONED:
            return true;

        case WAIT_TIMEOUT:
            return false;

        default:
            throw SystemException(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    if (::sem_trywait(&this->handle) == -1) {
        int error = ::GetLastError(); 
        if (error == EAGAIN) {
            return false;
        } else {
            throw SystemException(error, __FILE__, __LINE__);
        }
    }

    return true;
#endif /* _WIN32 */
}


/*
 * vislib::sys::Semaphore::Unlock
 */
void vislib::sys::Semaphore::Unlock(void) {
#ifdef _WIN32
    if (::ReleaseSemaphore(this->handle, 1, NULL) != TRUE) {
        throw SystemException(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    if (::sem_post(&this->handle) == -1) {
        throw SystemException(__FILE__, __LINE__);
    }
#endif /* _WIN32 */
}


/*
 * vislib::sys::Semaphore::Semaphore
 */
vislib::sys::Semaphore::Semaphore(const Semaphore& rhs) {
    throw UnsupportedOperationException("vislib::sys::Semaphore::Semaphore",
        __FILE__, __LINE__);
}


/*
 * vislib::sys::Semaphore::operator =
 */
vislib::sys::Semaphore& vislib::sys::Semaphore::operator =(
        const Semaphore& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
