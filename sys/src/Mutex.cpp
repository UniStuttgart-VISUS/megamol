/*
 * Mutex.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#include "vislib/Mutex.h"

#include "vislib/assert.h"
#include "vislib/error.h"
#include "vislib/IllegalParamException.h"
#include "vislib/SystemException.h"
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
void vislib::sys::Mutex::Lock(void) {
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
    int returncode = ::pthread_mutex_lock(&this->mutex);
    if (returncode != 0) {
        throw SystemException(returncode, __FILE__, __LINE__);
    }

#endif /* _WIN32 */
}


/*
 * vislib::sys::Mutex::TryLock
 */
bool vislib::sys::Mutex::TryLock(const DWORD timeout) {
#ifdef _WIN32
    switch (::WaitForSingleObject(this->handle, timeout)) {

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
    // TODO: Change implementation to use pthread_mutex_timedlock().
    int returncode = ::pthread_mutex_trylock(&this->mutex);

    switch (returncode) {
        case 0:
            return true;

        case EBUSY:
            return false;

        default:
            throw SystemException(returncode, __FILE__, __LINE__);
    }

#endif /* _WIN32 */
}


/*
 * vislib::sys::Mutex::Unlock
 */
void vislib::sys::Mutex::Unlock(void) {
#ifdef _WIN32
    if (::ReleaseMutex(this->handle) != TRUE) {
        throw SystemException(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    int returncode = ::pthread_mutex_unlock(&this->mutex);
    if (returncode != 0) {
        throw SystemException(returncode, __FILE__, __LINE__);
    }

#endif /* _WIN32 */
}


/*
 * vislib::sys::Mutex::Mutex
 */
vislib::sys::Mutex::Mutex(const Mutex& rhs) {
    throw UnsupportedOperationException("vislib::sys::Mutex::Mutex", 
        __FILE__, __LINE__);
}


/*
 * vislib::sys::Mutex::operator =
 */
vislib::sys::Mutex& vislib::sys::Mutex::operator =(const Mutex& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
