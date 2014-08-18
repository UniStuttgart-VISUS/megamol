/*
 * LamportClock.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/LamportClock.h"

#include "vislib/IllegalParamException.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::sys::LamportClock::LamportClock
 */
vislib::sys::LamportClock::LamportClock(void) : value(0) {
}


/*
 * vislib::sys::LamportClock::~LamportClock
 */
vislib::sys::LamportClock::~LamportClock(void) {
}


/*
 * vislib::sys::LamportClock::StepLocal
 */
UINT64 vislib::sys::LamportClock::StepLocal(void) {
    AutoLock l(this->lock);
    return ++this->value;
}


/*
 * vislib::sys::LamportClock::StepReceive
 */
UINT64 vislib::sys::LamportClock::StepReceive(UINT64 timestamp) {
    AutoLock l(this->lock);
    this->value = (this->value > timestamp) ? this->value : timestamp;
    return ++this->value;
}


/*
 * vislib::sys::LamportClock::operator ++
 */
UINT64 vislib::sys::LamportClock::operator ++(int) {
    AutoLock l(this->lock);
    return this->value++;
}


/*
 * vislib::sys::LamportClock::LamportClock
 */
vislib::sys::LamportClock::LamportClock(const LamportClock& rhs) {
    throw UnsupportedOperationException("LamportClock::LamportClock", __FILE__, 
        __LINE__);
}


/*
 * vislib::sys::LamportClock::operator =
 */
vislib::sys::LamportClock& vislib::sys::LamportClock::operator =(
        const LamportClock& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
