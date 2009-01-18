/*
 * AbstractAsyncContext.cpp
 *
 * Copyright (C) 2006 - 2009 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/AbstractAsyncContext.h"

#include "vislib/assert.h"
#include "vislib/memutils.h"


/*
 * vislib::sys::AbstractAsyncContext::~AbstractAsyncContext
 */
vislib::sys::AbstractAsyncContext::~AbstractAsyncContext(void) {
    // Nothing to do.
}


/*
 * vislib::sys::AbstractAsyncContext::Reset
 */
void vislib::sys::AbstractAsyncContext::Reset(void) {
#ifdef _WIN32
    this->callback = NULL;
    ::ZeroMemory(&this->overlapped, sizeof(OVERLAPPED));
    ASSERT(sizeof(this->overlapped.hEvent) >= sizeof(this));
    this->overlapped.hEvent = reinterpret_cast<HANDLE>(this);
#endif /* _WIN32 */
}


/*
 * vislib::sys::AbstractAsyncContext::AbstractAsyncContext
 */
vislib::sys::AbstractAsyncContext::AbstractAsyncContext(
        AsyncCallback callback, void *userContext) 
        : callback(callback), userContext(userContext) {
    this->Reset();
}


/*
 * vislib::sys::AbstractAsyncContext::notifyCompleted
 */
bool vislib::sys::AbstractAsyncContext::notifyCompleted(void) {
    if (this->callback != NULL) {
        this->callback(this);
        return true;
    } else {
        return false;
    }
}


/*
 * vislib::sys::AbstractAsyncContext::operator =
 */
vislib::sys::AbstractAsyncContext& 
vislib::sys::AbstractAsyncContext::operator =(const AbstractAsyncContext& rhs) {

    if (this != &rhs) {
        this->callback = rhs.callback;

#ifdef _WIN32
        ::memcpy(&this->overlapped, &rhs.overlapped, sizeof(OVERLAPPED));
        ASSERT(sizeof(this->overlapped.hEvent) >= sizeof(this));
        this->overlapped.hEvent = reinterpret_cast<HANDLE>(this);
#endif /* _WIN32 */

        this->userContext = rhs.userContext;
    }

    return *this;
}
