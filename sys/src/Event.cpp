/*
 * Event.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/Event.h"

#ifndef _WIN32
#include <ctime>
#endif /* !_WIN32 */

#include "vislib/assert.h"
#include "vislib/error.h"
#include "vislib/IllegalParamException.h"
#include "vislib/SystemException.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::sys::Event::TIMEOUT_INFINITE
 */
#ifdef _WIN32
const DWORD vislib::sys::Event::TIMEOUT_INFINITE = INFINITE;
#else /* _WIN32 */
const DWORD vislib::sys::Event::TIMEOUT_INFINITE = UINT_MAX;
#endif /* _WIN32 */

/*
 * vislib::sys::Event::Event
 */
vislib::sys::Event::Event(const bool isManualReset) {
#ifdef _WIN32
    this->handle = ::CreateEvent(NULL, isManualReset ? TRUE : FALSE, FALSE, 
        NULL);
    ASSERT(this->handle != NULL);

#else /* _WIN32 */
    ::pthread_condattr_init(&this->attr);
    ::pthread_cond_init(&this->cont, this->attr);

#endif /* _WIN32 */
}


/*
 * vislib::sys::Event::~Event
 */
vislib::sys::Event::~Event(void) {
#ifdef _WIN32
    ::CloseHandle(this->handle);
#else /* _WIN32 */
    while (::pthread_cond_destory(&this->cont) == EBUSY) {
        ::usleep(1000);
    }

    ::pthread_condattr_destroy(&this->attr);

#endif /* _WIN32 */
}


/*
 * vislib::sys::Event::Reset
 */
void vislib::sys::Event::Reset(void) {
#ifdef _WIN32
    if (!::ResetEvent(this->handle)) {
        throw SystemException(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    this->stateMutex.Lock();
    this->isSignaled = false;
    this->stateMutex.Unlock();

#endif /* _WIN32 */
}


/*
 * vislib::sys::Event::Set
 */
void vislib::sys::Event::Set(void) {
#ifdef _WIN32
    if (!::SetEvent(this->handle)) {
        throw SystemException(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    /* Make the event signaled. */
    this->stateMutex.Lock();
    this->isSignaled = true;
    this->stateMutex.Unlock();
    
    /* Release all thread to check the event state. */
    int returnCode = ::pthread_cond_broadcast(&this->cond);
    if (returnCode != 0) {
        throw SystemException(returnCode, __FILE__, __LINE__);
    }

#endif /* _WIN32 */
}


/*
 * vislib::sys::Event::Wait
 */
bool vislib::sys::Event::Wait(const DWORD timeout) {
#ifdef _WIN32
    switch (::WaitForSingleObject(this->handle, timeout)) {

        case WAIT_OBJECT_0:
            /* falls through. */
        case WAIT_ABANDONED:
            return true;
            /* Unreachable. */

        case WAIT_TIMEOUT:
            return false;
            /* Unreachable. */

        default:
            throw SystemException(__FILE__, __LINE__);
            /* Unreachable. */
    }

#else /* _WIN32 */
    bool isLocked = false;      // State of condition mutex.
    bool isSignaled = false;    // Local copy of signaled state.
    int returnCode = 0;         // Return code of wait operation.
    struct timespec tsEnd;      // The timestamp when the operation times out.
    struct timespec tsNow;      // The current timestamp when waiting.

    if (timeout == TIMEOUT_INFINITE) {
        this->condMutex.Lock();

        while (!isSignaled) {
            returnCode = ::pthread_cond_wait(&this->cond, 
                &this->condMutex.mutex);
            if (returnCode != 0) {
                this->mutex.Unlock();
                ::throw SystemException(returnCode, __FILE__, __LINE);
            }

            /*
             * Check this->isSignaled as we always relase all threads once
             * the event is signaled. However, in case the event is auto-reset
             * only one of these may be released. Checking the signaled state
             * and reseting it must be atomic, i. e. protected by the same
             * 'stateMutex' lock operation. 
             *
             * In case 'isSignaled' is false, another thread was released by 
             * the auto-reset event before this one. We must wait for the 
             * condition to be come signaled again.
             */
            this->stateMutex.Lock();
            if ((isSignaled = this->isSignaled)) {
                if (!this->isManualReset) {
                    this->isSignaled = false;
                }
            }
            this->stateMutex.Unlock();
        }
        
        this->condMutex.Unlock();
        return true;

    } else {
        ::clock_gettime(CLOCK_REALTIME, &tsEnd);
        ::memcpy(&tsNow, &tsEnd, sizeof(struct timespec));
        tsEnd.tv_sec += timeout / 1000;
        tsEnd.tv_nsec += (timeout % 1000) * 1000;

        /* Simulate timeout for pthread_mutex_lock. */
        while (!(isLocked = this->condMutex.TryLock()) 
                && ((tsEnd.tv_sec > tsNow.tv_sec)
                || (tsEnd.tv_nsec >= tsNow.tv_nsec))) {
            ::usleep(10);
            ::clock_gettime(CLOCK_REALTIME, &tsNow);
        }

        if (!isLocked) {
            /* Lock on mutex timed out. */
            return false;
        }

        /* Wait for the condition. */
        returnCode = ::pthread_cond_timedwait(&this->cond, 
            &this->contMutex.mutex, &tsEnd);
        this->condMutex.Unlock();
        switch (returnCode) {
            case 0:
                /* Do nothing. */
                break;

            case ETIMEDOUT:
                return false;
                /* Unreachable. */

            default:
                throw SystemException(returnCode, __FILE__, __LINE__);
                /* Unreachable. */
        }
        ASSERT(isLocked);

        this->stateMutex.Lock();
        isSignaled = this->isSignaled;
        this->stateMutex.Unlock();

        return this->isSignaled;
    } /* end if (timeout == TIMEOUT_INFINITE) */
#endif /* _WIN32 */
}


/*
 * vislib::sys::Event::Event
 */
vislib::sys::Event::Event(const Event& rhs) {
    throw UnsupportedOperationException("vislib::sys::Event::Event", 
        __FILE__, __LINE__);
}


/*
 * vislib::sys::Event::operator =
 */
vislib::sys::Event& vislib::sys::Event::operator =(const Event& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
