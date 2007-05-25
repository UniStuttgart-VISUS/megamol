/*
 * AutoLock.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/AutoLock.h"

#include "vislib/IllegalParamException.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::sys::AutoLock::AutoLock
 */
vislib::sys::AutoLock::AutoLock(SyncObject& lock) : lock(lock) {
    this->lock.Lock();
}


/*
 * vislib::sys::AutoLock::~AutoLock
 */
vislib::sys::AutoLock::~AutoLock(void) {
    this->lock.Unlock();
}


/*
 * vislib::sys::AutoLock::AutoLock
 */
vislib::sys::AutoLock::AutoLock(const AutoLock& rhs) : lock(rhs.lock) {
    throw UnsupportedOperationException("vislib::sys::AutoLock::AutoLock",
        __FILE__, __LINE__);
}


/*
 * vislib::sys::AutoLock::operator =
 */
vislib::sys::AutoLock& vislib::sys::AutoLock::operator =(const AutoLock& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
