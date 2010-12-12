/*
 * DateTimeSpan.cpp
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/DateTimeSpan.h"

#include <limits>

#include "vislib/mathfunctions.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"


/*
 * vislib::sys::DateTimeSpan::EMPTY
 */
const vislib::sys::DateTimeSpan vislib::sys::DateTimeSpan::EMPTY(0L);


/*
 * vislib::sys::DateTimeSpan::MILLISECONDS_PER_DAY
 */
const INT64 vislib::sys::DateTimeSpan::MILLISECONDS_PER_DAY 
    = 24L * 60L * 60L * 1000L;


/*
 * vislib::sys::DateTimeSpan::MILLISECONDS_PER_HOUR
 */
const INT64 vislib::sys::DateTimeSpan::MILLISECONDS_PER_HOUR 
    = 60L * 60L * 1000L;


/*
 * vislib::sys::DateTimeSpan::MILLISECONDS_PER_MINUTE
 */
const INT64 vislib::sys::DateTimeSpan::MILLISECONDS_PER_MINUTE 
    = 60L * 1000L;
  

/*
 * vislib::sys::DateTimeSpan::MILLISECONDS_PER_SECOND
 */
const INT64 vislib::sys::DateTimeSpan::MILLISECONDS_PER_SECOND 
    = 1000L;


#ifdef _MSC_VER
#pragma push_macro("max")
#undef max
#endif /* _MSC_VER */
/*
 * vislib::sys::DateTimeSpan::MAXIMUM
 */
const vislib::sys::DateTimeSpan vislib::sys::DateTimeSpan::MAXIMUM(
    std::numeric_limits<INT64>::max());
#ifdef _MSC_VER
#pragma pop_macro("max")
#endif /* _MSC_VER */


#ifdef _MSC_VER
#pragma push_macro("min")
#undef min
#endif /* _MSC_VER */
/*
 * vislib::sys::DateTimeSpan::MINIMUM
 */
const vislib::sys::DateTimeSpan vislib::sys::DateTimeSpan::MINIMUM(
    std::numeric_limits<INT64>::min());
#ifdef _MSC_VER
#pragma pop_macro("min")
#endif /* _MSC_VER */


/*
 * vislib::sys::DateTimeSpan::ONE_DAY
 */
const vislib::sys::DateTimeSpan vislib::sys::DateTimeSpan::ONE_DAY(
    vislib::sys::DateTimeSpan::MILLISECONDS_PER_DAY);


/*
 * vislib::sys::DateTimeSpan::ONE_HOUR
 */
const vislib::sys::DateTimeSpan vislib::sys::DateTimeSpan::ONE_HOUR(
    vislib::sys::DateTimeSpan::MILLISECONDS_PER_HOUR);


/*
 * vislib::sys::DateTimeSpan::ONE_MILLISECOND
 */
const vislib::sys::DateTimeSpan vislib::sys::DateTimeSpan::ONE_MILLISECOND(1L);


/*
 * vislib::sys::DateTimeSpan::ONE_MINUTE
 */
const vislib::sys::DateTimeSpan vislib::sys::DateTimeSpan::ONE_MINUTE(
    vislib::sys::DateTimeSpan::MILLISECONDS_PER_MINUTE);


/*
 * vislib::sys::DateTimeSpan::ONE_SECOND
 */
const vislib::sys::DateTimeSpan vislib::sys::DateTimeSpan::ONE_SECOND(
    vislib::sys::DateTimeSpan::MILLISECONDS_PER_SECOND);


/*
 * vislib::sys::DateTimeSpan::DateTimeSpan
 */
vislib::sys::DateTimeSpan::DateTimeSpan(const INT days, const INT hours, 
        const INT minutes, const INT seconds, const INT milliseconds) 
        : value(0L) {
    VLSTACKTRACE("DateTimeSpan::~DateTimeSpan", __FILE__, __LINE__);
    try {
        this->Set(days, hours, minutes, seconds, milliseconds);
    } catch (...) {
        this->~DateTimeSpan();
        throw;
    }
}


/*
 * vislib::sys::DateTimeSpan::~DateTimeSpan
 */
vislib::sys::DateTimeSpan::~DateTimeSpan(void) {
    VLSTACKTRACE("DateTimeSpan::~DateTimeSpan", __FILE__, __LINE__);

}


/*
 * vislib::sys::DateTimeSpan::Set
 */
void vislib::sys::DateTimeSpan::Set(const INT days, const INT hours, 
        const INT minutes, const INT seconds, const INT milliseconds) {
    VLSTACKTRACE("DateTimeSpan::Set", __FILE__, __LINE__);
    // TODO: This must be checked for overflows, too.
    this->value = MILLISECONDS_PER_DAY * days 
        + MILLISECONDS_PER_HOUR * hours 
        + MILLISECONDS_PER_MINUTE * minutes 
        + MILLISECONDS_PER_SECOND * seconds
        + milliseconds;
}


/*
 * vislib::sys::DateTimeSpan::ToStringA
 */
vislib::StringA vislib::sys::DateTimeSpan::ToStringA(void) const {
    VLSTACKTRACE("DateTimeSpan::ToStringA", __FILE__, __LINE__);
    StringA retval;
    retval.Format("%lld:%02lld:%02lld:%02lld.%04lld", this->GetDays(), 
        math::Abs(this->GetHours()), math::Abs(this->GetMinutes()), 
        math::Abs(this->GetSeconds()), math::Abs(this->GetMilliseconds()));
    return retval;
}


/*
 * vislib::sys::DateTimeSpan::ToStringW
 */
vislib::StringW vislib::sys::DateTimeSpan::ToStringW(void) const {
    VLSTACKTRACE("DateTimeSpan::ToStringW", __FILE__, __LINE__);
    StringW retval;
    retval.Format(L"%lld:%02lld:%02lld:%02lld.%04lld", this->GetDays(), 
        math::Abs(this->GetHours()), math::Abs(this->GetMinutes()), 
        math::Abs(this->GetSeconds()), math::Abs(this->GetMilliseconds()));
    return retval;
}


/*
 * vislib::sys::DateTimeSpan::operator -=
 */
vislib::sys::DateTimeSpan& vislib::sys::DateTimeSpan::operator -=(
        const DateTimeSpan& rhs) {
    VLSTACKTRACE("DateTimeSpan::operator -=", __FILE__, __LINE__);
    try {
        this->add((-rhs).value);
        return *this;
    } catch (IllegalStateException) {
        // Repackage error into IllegalStateException, because any other
        // behaviour would be inconsistent.
        // TODO: Does this implementation make sense?
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }
}


/*
 * vislib::sys::DateTimeSpan::operator -
 */
vislib::sys::DateTimeSpan vislib::sys::DateTimeSpan::operator -(void) const {
    VLSTACKTRACE("DateTimeSpan::operator -", __FILE__, __LINE__);
    if (this->value == DateTimeSpan::MINIMUM) {
        throw IllegalStateException("DateTimeSpan::MINIMUM", __FILE__, 
            __LINE__);
    }
    
    DateTimeSpan retval(-this->value);
    return retval;
}


/*
 * vislib::sys::DateTimeSpan::operator =
 */
vislib::sys::DateTimeSpan& vislib::sys::DateTimeSpan::operator =(
        const DateTimeSpan& rhs) throw() {
    VLSTACKTRACE("DateTimeSpan::operator =", __FILE__, __LINE__);
    if (this != &rhs) {
        this->value = rhs.value;
    }
    return *this;
}


/*
 * vislib::sys::DateTimeSpan::add
 */
void vislib::sys::DateTimeSpan::add(const INT64 millis) {
    VLSTACKTRACE("DateTimeSpan::add", __FILE__, __LINE__);
    INT64 max = static_cast<INT64>(DateTimeSpan::MAXIMUM);
    INT64 min = static_cast<INT64>(DateTimeSpan::MINIMUM);
    
    /* Sanity checks. */
    if (math::Signum(this->value) == math::Signum(millis)) {
        if (this->value > 0) {
            /* Both values are positive. Result might exceed 'max'. */
            if (this->value > max - millis) {
                throw IllegalParamException("rhs", __FILE__, __LINE__);
            }
        } else {
            /* Both values are negative. Result might exceed 'min'. */
            if (this->value < min - millis) {
                throw IllegalParamException("rhs", __FILE__, __LINE__);
            }
        }
    }
    /* Operation is safe at this point. */

    this->value += millis;
}