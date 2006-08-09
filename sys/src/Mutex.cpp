/*
 * Mutex.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#include "vislib/assert.h"
#include "vislib/error.h"
#include "vislib/Mutex.h"
#include "vislib/IllegalParamException.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::sys::Mutex::Mutex
 */ 
vislib::sys::Mutex::Mutex(void) {
#ifdef _WIN32
    this->handle = ::CreateMutex(NULL, FALSE, NULL);
    ASSERT(this->handle != NULL);

#else /* _WIN32 */
    ::pthread_mutexattr_init(&attr);
    ::pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE_NP);

    ::pthread_mutex_init(&this->mutex, &attr);

#endif /* _WIN32 */
}


/*
 * vislib::sys::Mutex::~Mutex(void) 
 */
vislib::sys::Mutex::~Mutex(void) {
#ifdef _WIN32
    ::CloseHandle(this->handle);

#else /* _WIN32 */
	if (::pthread_mutex_destroy(&this->mutex) == EBUSY) {
		this->Unlock();
		::pthread_mutex_destroy(&this->mutex);
	}

    ::pthread_mutexattr_destroy(&attr);

#endif /* _WIN32 */
}


/*
 * vislib::sys::Mutex::Lock
 */
bool vislib::sys::Mutex::Lock(void) {
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
    return (::pthread_mutex_lock(&this->mutex) != 0);

#endif /* _WIN32 */
}


/*
 * vislib::sys::Mutex::TryLock
 */
bool vislib::sys::Mutex::TryLock(void) {
#ifdef _WIN32
    switch (::WaitForSingleObject(this->handle, 0)) {

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
    return (::pthread_mutex_trylock(&this->mutex) != 0);

#endif /* _WIN32 */
}


/*
 * vislib::sys::Mutex::Unlock
 */
bool vislib::sys::Mutex::Unlock(void) {
#ifdef _WIN32
    return (::ReleaseMutex(this->handle) == TRUE);

#else /* _WIN32 */
    return (::pthread_mutex_unlock(&this->mutex) != 0);

#endif /* _WIN32 */
}


/*
 * vislib::sys::Mutex::Mutex
 */
vislib::sys::Mutex::Mutex(const Mutex& rhs) {
    throw UnsupportedOperationException(_T("vislib::sys::Mutex::Mutex"), 
        __FILE__, __LINE__);
}


/*
 * vislib::sys::Mutex::operator =
 */
vislib::sys::Mutex& vislib::sys::Mutex::operator =(const Mutex& rhs) {
    if (this != &rhs) {
        throw IllegalParamException(_T("rhs"), __FILE__, __LINE__);
    }

    return *this;
}
