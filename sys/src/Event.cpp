/*
 * Event.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/Event.h"

#ifndef _WIN32
#include <climits>
#endif /* !_WIN32 */

#include "vislib/assert.h"
#include "vislib/error.h"
#include "vislib/IllegalParamException.h"
#include "vislib/SystemException.h"
#include "vislib/Trace.h"
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
vislib::sys::Event::Event(const bool isManualReset, 
                          const bool isInitiallySignaled) 
#ifndef _WIN32
        : isManualReset(isManualReset), 
        semaphore(isInitiallySignaled ? 1 : 0, 1)
#endif /* _WIN32 */
{
#ifdef _WIN32
    this->handle = ::CreateEventA(NULL, isManualReset ? TRUE : FALSE,
        isInitiallySignaled ? TRUE : FALSE, 
        NULL);
    ASSERT(this->handle != NULL);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Event::Event
 */
vislib::sys::Event::Event(const char *name, const bool isManualReset, 
        const bool isInitiallySignaled, bool *outIsNew) 
#ifndef _WIN32
        : isManualReset(isManualReset), 
        semaphore(name, isInitiallySignaled ? 1 : 0, 1, outIsNew)
#endif /* _WIN32 */
{
#ifdef _WIN32
    if (outIsNew != NULL) {
        *outIsNew = false;
    }

    /* Try to open existing event first. */
    if ((this->handle = ::OpenEventA(SYNCHRONIZE | EVENT_MODIFY_STATE,
            FALSE, name)) == NULL) {
        this->handle = ::CreateEventA(NULL, isManualReset ? TRUE : FALSE,
            isInitiallySignaled ? TRUE : FALSE,
            name);
        if (outIsNew != NULL) {
            *outIsNew = true;
        }
    }
    ASSERT(this->handle != NULL);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Event::Event
 */
vislib::sys::Event::Event(const wchar_t *name, const bool isManualReset, 
        const bool isInitiallySignaled, bool *outIsNew) 
#ifndef _WIN32
        : isManualReset(isManualReset), 
        semaphore(name, isInitiallySignaled ? 1 : 0, 1, outIsNew)
#endif /* _WIN32 */
{
#ifdef _WIN32
    if (outIsNew != NULL) {
        *outIsNew = false;
    }

    /* Try to open existing event first. */
    if ((this->handle = ::OpenEventW(SYNCHRONIZE | EVENT_MODIFY_STATE,
            FALSE, name)) == NULL) {
        this->handle = ::CreateEventW(NULL, isManualReset ? TRUE : FALSE,
            isInitiallySignaled ? TRUE : FALSE,
            name);
        if (outIsNew != NULL) {
            *outIsNew = true;
        }
    }
    ASSERT(this->handle != NULL);
#endif /* _WIN32 */
}


/*
 * vislib::sys::Event::~Event
 */
vislib::sys::Event::~Event(void) {
#ifdef _WIN32
    ::CloseHandle(this->handle);

#else /* _WIN32 */
    /* Nothing to do. */

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
    VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Event::Reset\n");
    
    this->semaphore.TryLock();
    ASSERT(!this->semaphore.TryLock());

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
    VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Event::Set\n");

    this->semaphore.TryLock();
    this->semaphore.Unlock();

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
    VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Event::Wait\n");
    bool retval = false;

    if (timeout == TIMEOUT_INFINITE) {
        this->semaphore.Lock();
        retval = true;

    } else {
        retval = this->semaphore.TryLock(timeout);
    }

    if (retval && this->isManualReset) {
        VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Event::Wait signal again\n");
        this->semaphore.Unlock();
    }

    return retval;
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
