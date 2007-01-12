/*
 * DateTimeSpan.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/DateTimeSpan.h"


/*
 * vislib::sys::DateTimeSpan::~DateTimeSpan
 */
vislib::sys::DateTimeSpan::~DateTimeSpan(void) {
}


/*
 * vislib::sys::DateTimeSpan::Set
 */
void vislib::sys::DateTimeSpan::Set(const INT days, const INT hours, 
        const INT minutes, const INT seconds) {
    this->value = ONE_DAY * days + ONE_HOUR * hours + ONE_MINUTE * minutes 
        + ONE_SECOND * seconds;
}


/*
 * vislib::sys::DateTimeSpan::ONE_DAY
 */
const INT64 vislib::sys::DateTimeSpan::ONE_DAY = 24L * 60 * 60 * 1000;


/*
 * vislib::sys::DateTimeSpan::ONE_HOUR
 */
const INT64 vislib::sys::DateTimeSpan::ONE_HOUR = 60L * 60 * 1000;


/*
 * vislib::sys::DateTimeSpan::ONE_MINUTE
 */
const INT64 vislib::sys::DateTimeSpan::ONE_MINUTE = 60L * 1000;
  

/*
 * vislib::sys::DateTimeSpan::ONE_SECOND
 */
const INT64 vislib::sys::DateTimeSpan::ONE_SECOND = 1000L;
