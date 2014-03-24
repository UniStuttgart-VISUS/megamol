/*
 * LamportClock.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/LamportClock.h"

#include "the/argument_exception.h"
#include "the/not_supported_exception.h"


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
uint64_t vislib::sys::LamportClock::StepLocal(void) {
    AutoLock l(this->lock);
    return ++this->value;
}


/*
 * vislib::sys::LamportClock::StepReceive
 */
uint64_t vislib::sys::LamportClock::StepReceive(uint64_t timestamp) {
    AutoLock l(this->lock);
    this->value = (this->value > timestamp) ? this->value : timestamp;
    return ++this->value;
}


/*
 * vislib::sys::LamportClock::operator ++
 */
uint64_t vislib::sys::LamportClock::operator ++(int) {
    AutoLock l(this->lock);
    return this->value++;
}


/*
 * vislib::sys::LamportClock::LamportClock
 */
vislib::sys::LamportClock::LamportClock(const LamportClock& rhs) {
    throw the::not_supported_exception("LamportClock::LamportClock", __FILE__, 
        __LINE__);
}


/*
 * vislib::sys::LamportClock::operator =
 */
vislib::sys::LamportClock& vislib::sys::LamportClock::operator =(
        const LamportClock& rhs) {
    if (this != &rhs) {
        throw the::argument_exception("rhs", __FILE__, __LINE__);
    }

    return *this;
}
