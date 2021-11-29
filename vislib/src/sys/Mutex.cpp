/*
 * Mutex.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 * Copyright 2019 MegaMol Dev Team
 */

#include <chrono>

#include "vislib/sys/Mutex.h"

#include "vislib/IllegalParamException.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/sys/error.h"


/*
 * vislib::sys::Mutex::Lock
 */
void vislib::sys::Mutex::Lock(void) {
    mutex.lock();
}


/*
 * vislib::sys::Mutex::TryLock
 */
bool vislib::sys::Mutex::TryLock(const DWORD timeout) {
    return mutex.try_lock_for(std::chrono::milliseconds(timeout));
}


/*
 * vislib::sys::Mutex::Unlock
 */
void vislib::sys::Mutex::Unlock(void) {
    mutex.unlock();
}


/*
 * vislib::sys::Mutex::Mutex
 */
vislib::sys::Mutex::Mutex(const Mutex& rhs) {
    throw UnsupportedOperationException("vislib::sys::Mutex::Mutex", __FILE__, __LINE__);
}


/*
 * vislib::sys::Mutex::operator =
 */
vislib::sys::Mutex& vislib::sys::Mutex::operator=(const Mutex& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
