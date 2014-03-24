/*
 * DateTime.cpp
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller. Alle Rechte vorbehalten.
 */


#include "vislib/DateTime.h"

#include <climits>

#include "the/assert.h"
#include "vislib/error.h"
#include "vislib/mathfunctions.h"
#include "the/system/system_exception.h"



/*
 * vislib::sys::DateTime::IsLeapYear
 */
bool vislib::sys::DateTime::IsLeapYear(const int year) {
    THE_STACK_TRACE;
    return ((year % 4) == 0) && (((year % 100) != 0) || ((year % 400) == 0));
}


/*
 * vislib::sys::DateTime::Now
 */
vislib::sys::DateTime vislib::sys::DateTime::Now(void) {
    THE_STACK_TRACE;
    DateTime retval;
    return retval;
}


/*
 * vislib::sys::DateTime::Today
 */
vislib::sys::DateTime vislib::sys::DateTime::Today(void) {
    THE_STACK_TRACE;
    DateTime retval;
    retval.SetTime(0, 0, 0, 0);
    return retval;
}


/*
 * vislib::sys::DateTime::EMPTY
 */
vislib::sys::DateTime vislib::sys::DateTime::EMPTY(0, 0);


/*
 * vislib::sys::DateTime::DateTime
 */
vislib::sys::DateTime::DateTime(void) : ticks(0L) {
    THE_STACK_TRACE;
#ifdef _WIN32
    SYSTEMTIME systemTime;
    ::GetLocalTime(&systemTime);
    this->Set(systemTime, false);

#else /* _WIN32 */
    time_t utc = ::time(NULL);
    this->Set(utc);
#endif /* _WIN32 */
}
        

/*
 * vislib::sys::DateTime::DateTime
 */
vislib::sys::DateTime::DateTime(const int year, const int month, const int day,
        const int hours, const int minutes, const int seconds, 
        const int milliseconds) {
    THE_STACK_TRACE;
    try {
        this->Set(year, month, day, hours, minutes, seconds, milliseconds);
    } catch (...) {
        this->~DateTime();
        throw;
    }
}


/*
 * vislib::sys::DateTime::~DateTime
 */
vislib::sys::DateTime::~DateTime(void) {
    THE_STACK_TRACE;
}


// TODO: This does not work any more with our input interpretation
///*
// * cml::DateTime::AddMonths
// */
//void cml::DateTime::AddMonths(const int months) {
//    int year, month, day;
//    this->GetDate(year, month, day);
//    this->SetDate(year, month + months, day);
//}
//
//
///*
// * cml::DateTime::AddYears
// */
//void cml::DateTime::AddYears(const int years) {
//    int year, month, day;
//    this->GetDate(year, month, day);
//    this->SetDate(year + years, month, day);
//}


/*
 * vislib::sys::DateTime::GetDate
 */
void vislib::sys::DateTime::GetDate(int& outYear, 
                                    int& outMonth, 
                                    int& outDay) const {
    outYear = static_cast<int>(this->get(DATE_PART_YEAR));
    outMonth = static_cast<int>(this->get(DATE_PART_MONTH));
    outDay = static_cast<int>(this->get(DATE_PART_DAY));
//    //THE_ASSERT(this->value >= 0); // TODO: Implementation does not yet work for BC
//    int64_t days = this->GetDays();           // Full days.
//    int64_t cnt400Years = 0;                  // # of full 400 year blocks.
//    int64_t cnt100Years = 0;                  // # of full 100 year blocks.
//    int64_t cnt4Years = 0;                    // # of full 4 year blocks.
//    int64_t cntYears = 0;                     // # of remaining years.
//    bool containsLeapYear = true;           // Contains 4 year block leap year?
//    int64_t divisor = 0;                      // Divisor for n-year blocks.
//
//    /* 
//     * Determine 400 year blocks lying behind and make 'days' relative to
//     * active 400 year block.
//     * A 400 year block has 400 full years. Every 4th year is a leap 
//     * year except for 3, which are divisable by 100 but not by 400.
//     */
//    divisor = (400 * ONE_YEAR) + (400 / 4) - 3;
//    cnt400Years = days / divisor;
//    days %= divisor;
//
//    /*
//     * Determine 100 year blocks within 400 year block lying behind the 
//     * active 100 year block. There can be at most 3 inactive 100 year blocks,
//     * because one of four must always be the active one. If the result
//     * is 0, the first 100 year block is the active one. This is the one which
//     * contains the year divisable by 100 and 400. This year is the exception
//     * that is a leap year although divisable by 100.
//     * A 100 year block has 100 full years. Every 4th year is a leap year
//     * except for the one that is divisable by 100. Whether this one is also
//     * divisable by 400 can be determined via the value of 'cnt100Years': If
//     * this value indicates the first block, it is a leap year.
//     */
//    // mueller: I do not understand that any more. Tests succeed with normal
//    // impl. until now. Check this in future!
//    // Subtract 1 because first century of 4 has 366 days.
//    //cnt100Years = (days - 1+1) / (100 * ONE_YEAR + (100 / 4) - 1);
//    divisor = (100 * ONE_YEAR) + (100 / 4) - 1;
//    cnt100Years = days / divisor;
//    days %= divisor;
//    //THE_ASSERT(cnt100Years > -4);
//    //THE_ASSERT(cnt100Years < 4);
//
//    //days -= cnt100Years;
//
//    /*
//     * Like for 400 and 100 year blocks, determine 4 year blocks in the active
//     * 100 year block which lie behind the current date.
//     * A 4 year block has the days of four full years plus one leap day.
//     */
//    divisor = (4 * ONE_YEAR) + (4 / 4);
//    cnt4Years = days / divisor;
//    days %= divisor;
//    THE_ASSERT(cnt4Years > -(100 / 4));
//    THE_ASSERT(cnt4Years < (100 / 4));
//
//    /*
//     * Divide a last time to determine the active year in the 4 year block.
//     */
//    divisor = ONE_YEAR;// + ((days <= ONE_YEAR) ? 1 : 0);
//    //if ((cnt400Years == 0) || ((cnt4Years == 0) && (cnt100Years != 0))) {
//    //    divisor++;
//    //}
//    cntYears = days / divisor;
//    days %= divisor;
//    THE_ASSERT(cntYears > -4);
//    THE_ASSERT(cntYears < 4);
//
//    /* At this point, we can compute the year. */
//    outYear = static_cast<int>(400 * cnt400Years + 100 * cnt100Years
//         + 4 * cnt4Years + cntYears);
////    if (this->value < 0) {
////        //outYear = -outYear;
////        //days = 365 - days - 1;
//////        THE_ASSERT(days >= 0);
////    }
//    //if (days < 0) {
//    //    days = 365 + IsLeapYear(outYear - 1) + days + 1;
//    //    //days = -days;
//    //}
//
//    if (days < 0) {
//        days = ONE_YEAR + days;
//    } else {
//        outYear++;
//    }
//    if (days == 0) days = 1;
//
//    if (DateTime::IsLeapYear(outYear) && (days + 1 < DAYS_AFTER_MONTH[2])) {
//        days++;
//    }
//
//    /* Compute month. */
//    // Month must be greater than or than ('days' / 32).
//    for (outMonth = static_cast<int>((days >> 5) + 1);
//            days > DAYS_AFTER_MONTH[outMonth]; outMonth++);
//
//    /* Compute day in month. */
//    outDay = static_cast<int>(days - DAYS_AFTER_MONTH[outMonth - 1]);
//
//    //if ((cntYears == 0) && containsLeapYear && (days == 59)) {
//    //    /* On 29.02. */ 
//    //    outMonth = 2;
//    //    outDay = 29;
//
//    //} else {
//    //    /* 
//    //     * Manipulate 'days' to look like a non-leap year for month lookup
//    //     * in the DAYS_AFTER_MONTH array.
//    //     */
//
//    //    if ((cntYears == 0) && containsLeapYear && (days >= 59)) {
//    //        /* After 29.02. */
//    //        days--;
//    //    }
//
//    //    days++;
//
//    //    // Month must be greater than or equal to ('days' / 32).
//    //    for (outMonth = static_cast<int>((days >> 5) + 1); 
//    //        days > DAYS_AFTER_MONTH[outMonth]; outMonth++);
//
//    //    outDay = static_cast<int>(days - DAYS_AFTER_MONTH[outMonth - 1]);
//    //}
//
//    /* Handle years B. C. */
//    //if (this->value < 0) {
//    //    outYear *= -1;
//    //}
}


/*
 * vislib::sys::DateTime::GetTime
 */
void vislib::sys::DateTime::GetTime(int& outHours, 
                                    int& outMinutes, 
                                    int& outSeconds,
                                    int& outMilliseconds) const {
    int64_t time = this->ticks % DateTimeSpan::TICKS_PER_DAY;
    if (time < 0) {
        time += DateTimeSpan::TICKS_PER_DAY;
    }

    outMilliseconds = static_cast<int>(time % DateTimeSpan::TICKS_PER_SECOND);
    time /= DateTimeSpan::TICKS_PER_SECOND;
    outSeconds = static_cast<int>(time % 60L);
    time /= 60L;
    outMinutes = static_cast<int>(time % 60L);
    time /= 60L;
    outHours = static_cast<int>(time);
}


/*
 * vislib::sys::DateTime::Set
 */
void vislib::sys::DateTime::Set(const int year, const int month, const int day,
        const int hours, const int minutes, const int seconds, 
        const int milliseconds, const int ticks ) {
    int64_t y = (year != 0) ? year : 1;       // The possibly corrected year.
    int64_t m = (month != 0) ? month : 1;     // The possibly corrected month.
    int64_t d = (day != 0) ? day : 1;         // The possibly corrected day.
    int64_t t = 0;                            // Auxiliary variable.
    int64_t days = 0;                         // The total days.

    /* Correct possible invalid months by rolling the year. */
    if (m < 0) {
        /* Roll backwards. */
        t = ++m;                            // Must reflect missing month 0.
        m = 12 + m % 12;
        if ((y += t / 12 - 1) == 0) {
            y = -1;
        }

    } else if (m > 12) {
        /* Roll forward. */
        t = m;
        m = m % 12;
        if ((y += t / 12) == 0) {
            y = 1;
        }
    }
    THE_ASSERT(m >= 1);
    THE_ASSERT(m <= 12);

    /* 
     * Determine how many days are in each month of the input year. The year
     * must not yet be zero-based for this operation!
     */
    THE_ASSERT(y != 0);
    const int64_t *daysAfterMonth = DateTime::IsLeapYear(static_cast<int>(y)) 
        ? DAYS_AFTER_MONTH_LY : DAYS_AFTER_MONTH;

    /* Positive years are zero-based from now on (year 0 does not exist!). */
    THE_ASSERT(y != 0);
    if (y > 0) {
        y--;
    }

    /* Compute the days since 01.01.0001. */
    days = y * DAYS_PER_YEAR 
        + (y / 4) - (y / 100) + (y / 400)   // Add leap day for leap years.
        + daysAfterMonth[m - 1] 
        + ((d > 0) ? (d - 1) : d);          // Days now zero-based. Also fix 
                                            // day 0 to be day 1 of month.

    /* Convert the days to ticks. */
    this->ticks = DateTimeSpan::DaysToTicks(days);

    /* Add the time. */
    this->ticks += DateTimeSpan::TimeToTicks(hours, minutes, seconds, 
        milliseconds, ticks);
}


/*
 * vislib::sys::DateTime::Set
 */
void vislib::sys::DateTime::Set(const struct tm& tm) {
    THE_STACK_TRACE;
    this->Set(tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
        tm.tm_hour, tm.tm_min, tm.tm_sec, 0);
}


/*
 * vislib::sys::DateTime::Set
 */
void vislib::sys::DateTime::Set(const time_t time) {
    THE_STACK_TRACE;
#if (defined(_MSC_VER) && (_MSC_VER >= 1400))
    struct tm tm;
    if (::localtime_s(&tm, &time) == 0) {
        this->Set(tm);
    } else {
        throw the::system::system_exception(ERROR_INVALID_DATA, __FILE__, __LINE__);
    }

#elif defined(_WIN32)
    struct tm *tm = ::localtime(&time);
    if (tm != NULL) {
        this->Set(*tm);
    } else {
        throw the::system::system_exception(ERROR_INVALID_DATA, __FILE__, __LINE__);
    }

#else  /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */
    struct tm tm;
    if (::localtime_r(&time, &tm) != NULL) {
        this->Set(tm);
    } else {
        throw the::system::system_exception(EINVAL, __FILE__, __LINE__);
    }
#endif /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */
}


#ifdef _WIN32
/*
 * vislib::sys::DateTime::Set
 */
void vislib::sys::DateTime::Set(const FILETIME& fileTime) {
    THE_STACK_TRACE;
    SYSTEMTIME systemTime;
    if (!::FileTimeToSystemTime(&fileTime, &systemTime)) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }
    this->Set(systemTime, true);
}


/*
 * vislib::sys::DateTime::Set
 */
void vislib::sys::DateTime::Set(const SYSTEMTIME& systemTime, 
        const bool isUTC) {
    THE_STACK_TRACE;
    SYSTEMTIME lSystemTime;

    if (isUTC) {
        if (!::SystemTimeToTzSpecificLocalTime(NULL, 
                const_cast<SYSTEMTIME *>(&systemTime), &lSystemTime)) {
            throw the::system::system_exception(__FILE__, __LINE__);
        }
        this->Set(lSystemTime.wYear, lSystemTime.wMonth, lSystemTime.wDay,
            lSystemTime.wHour, lSystemTime.wMinute, lSystemTime.wSecond, 
            lSystemTime.wMilliseconds);

    } else {
        this->Set(systemTime.wYear, systemTime.wMonth, systemTime.wDay,
            systemTime.wHour, systemTime.wMinute, systemTime.wSecond,
            systemTime.wMilliseconds);
    }
}
#endif /* _WIN32 */


/*
 * vislib::sys::DateTime::SetDate
 */
void vislib::sys::DateTime::SetDate(const int year, 
                                    const int month, 
                                    const int day) {
    THE_STACK_TRACE;
    int64_t time = this->ticks % DateTimeSpan::TICKS_PER_DAY;
    this->Set(year, month, day, 0, 0, 0);
    this->ticks += time;
}


/*
 * vislib::sys::DateTime::SetTime
 */
void vislib::sys::DateTime::SetTime(const int hours, 
                                    const int minutes, 
                                    const int seconds,
                                    const int milliseconds,
                                    const int ticks) {
    THE_STACK_TRACE;
    this->ticks -= this->ticks % DateTimeSpan::TICKS_PER_DAY;
    this->ticks += DateTimeSpan::TimeToTicks(hours, minutes, seconds,
        milliseconds, ticks);
}


/*
 * vislib::sys::DateTime::operator =
 */
vislib::sys::DateTime& vislib::sys::DateTime::operator =(const DateTime& rhs) {
    THE_STACK_TRACE;
    if (this != &rhs) {
        this->ticks = rhs.ticks;
    }
    return *this;
}


/*
 * vislib::sys::DateTime::operator +=
 */
vislib::sys::DateTime& vislib::sys::DateTime::operator +=(
        const DateTimeSpan& rhs) {
    THE_STACK_TRACE;
    DateTimeSpan tmp(this->ticks);
    tmp += rhs;
    this->ticks = static_cast<int64_t>(tmp);
    return *this;
}


/*
 * vislib::sys::DateTime::operator -=
 */
vislib::sys::DateTime& vislib::sys::DateTime::operator -=(
        const DateTimeSpan& rhs) {
    THE_STACK_TRACE;
    DateTimeSpan tmp(this->ticks);
    tmp -= rhs;
    this->ticks = static_cast<int64_t>(tmp);
    return *this;
}


/*
 * vislib::sys::DateTime::operator -
 */
vislib::sys::DateTimeSpan vislib::sys::DateTime::operator -(
        const DateTime& rhs) const {
    THE_STACK_TRACE;
    DateTimeSpan retval(this->ticks);
    DateTimeSpan r(rhs.ticks);
    retval -= r;
    return retval;
}


/*
 * vislib::sys::DateTime::operator struct tm
 */
vislib::sys::DateTime::operator struct tm(void) const {
    struct tm retval;
    int year, month, day, hour, minute, second;
    
    this->Get(year, month, day, hour, minute, second);

    retval.tm_year = year - 1900;
    retval.tm_mon = month - 1;
    retval.tm_mday = day;
    retval.tm_hour = hour;
    retval.tm_min = minute;
    retval.tm_sec = second;
    retval.tm_isdst = -1;   // Mark DST state as unknown.
    retval.tm_wday = 0;     // TODO
    retval.tm_yday = 0;     // TODO

    return retval;
}


/*
 * vislib::sys::DateTime::operator time_t
 */
vislib::sys::DateTime::operator time_t(void) const {
    struct tm tmp = static_cast<struct tm>(*this);
    return ::mktime(&tmp);
}


#ifdef _WIN32
/* 
 * vislib::sys::DateTime::operator FILETIME
 */
vislib::sys::DateTime::operator FILETIME(void) const {
    FILETIME retval;
    SYSTEMTIME systemTime = static_cast<SYSTEMTIME>(*this);
    
    if (!::SystemTimeToFileTime(&systemTime, &retval)) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    return retval;
}


/*
 * vislib::sys::DateTime::operator SYSTEMTIME
 */
vislib::sys::DateTime::operator SYSTEMTIME(void) const {
    SYSTEMTIME localTime, retval;
    int year, month, day, hours, minutes, seconds, millis;
    
    this->Get(year, month, day, hours, minutes, seconds, millis);

    localTime.wYear = year;
    localTime.wMonth = month;
    localTime.wDay = day;
    localTime.wHour = hours;
    localTime.wMinute = minutes;
    localTime.wSecond = seconds;
    localTime.wMilliseconds = millis;
    localTime.wDayOfWeek = 0;          // TODO

    ::TzSpecificLocalTimeToSystemTime(NULL, &localTime, &retval);
    
    return retval;
}
#endif /* _WIN32 */


/*
 * vislib::sys::DateTime::DAYS_AFTER_MONTH
 */
const int64_t vislib::sys::DateTime::DAYS_AFTER_MONTH[13] = {
    0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365 
};


/*
 * vislib::sys::DateTime::DAYS_AFTER_MONTH
 */
const int64_t vislib::sys::DateTime::DAYS_AFTER_MONTH_LY[13] = {
    0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366
};


/*
 * vislib::sys::DateTime::DAYS_PER_YEAR
 */
const int64_t vislib::sys::DateTime::DAYS_PER_YEAR = static_cast<int64_t>(365);


/*
 * vislib::sys::DateTime::DAYS_PER_4YEARS
 */
const int64_t vislib::sys::DateTime::DAYS_PER_4YEARS 
    = static_cast<int64_t>(365) * 4 + 1;


/*
 * vislib::sys::DateTime::DAYS_PER_100YEARS
 */
const int64_t vislib::sys::DateTime::DAYS_PER_100YEARS 
    = static_cast<int64_t>(365) * 100 + (100 / 4) - 1;


/*
 * vislib::sys::DateTime::DAYS_PER_400YEARS
 */
const int64_t vislib::sys::DateTime::DAYS_PER_400YEARS 
    = static_cast<int64_t>(365) * 400 + (400 / 4) - 3;


/*
 * vislib::sys::DateTime::get
 */
int64_t vislib::sys::DateTime::get(const DatePart datePart) const {
    THE_STACK_TRACE;
    // This implementation is shamelessly inspired by the DateTime of
    // Singularity: https://singularity.svn.codeplex.com/svn/base/Applications/Runtime/Full/System/DateTime.cs

    int64_t sign = (this->ticks < 0) ? -1 : 1;

    // n = number of days since 1/1/0001
    int64_t n = this->ticks / DateTimeSpan::TICKS_PER_DAY;

    // y400 = number of whole 400-year periods since 1/1/0001
    int64_t y400 = n / DAYS_PER_400YEARS;

    // n = day number within 400-year period
    n -= y400 * DAYS_PER_400YEARS;

    // y100 = number of whole 100-year periods within 400-year period
    int64_t y100 = n / DAYS_PER_100YEARS;

    // Last 100-year period has an extra day, so decrement result if 4
    if (math::Abs(y100) == 4) {
        y100 = sign * 3;
    }

    // n = day number within 100-year period
    n -= y100 * DAYS_PER_100YEARS;

    // y4 = number of whole 4-year periods within 100-year period
    int64_t y4 = n / DAYS_PER_4YEARS;

    // n = day number within 4-year period
    n -= y4 * DAYS_PER_4YEARS;

    // y1 = number of whole years within 4-year period
    int64_t y1 = n / DAYS_PER_YEAR;

    // Last year has an extra day, so decrement result if 4
    if (y1 == 4) {
        y1 = 3;
    }
    else if (y1 == -4) {
        THE_ASSERT(0);
        y1 = -3;
    }

    //if (sign == -1 && n - y1 * DAYS_PER_YEAR == 0 && math::Abs(y1) != 3) {
    //    y1 = 0;
    //}

    // If year was requested, compute and return it
    if (datePart == DATE_PART_YEAR) {
        return y400 * 400 + y100 * 100 + y4 * 4 + y1 + sign * 1;
    }

    // n = day number within year
    n -= y1 * DAYS_PER_YEAR;

    // If day-of-year was requested, return it
    if (datePart == DATE_PART_DAY_OF_YEAR) {
        return n + 1;
    }

    // Leap year calculation looks different from IsLeapYear since y1, y4,
    // and y100 are relative to year 1, not year 0
    bool leapYear = (y1 == 3) && ((y4 != 24) || (y100 == 3));
    const int64_t *daysAfterMonth = leapYear 
        ? DAYS_AFTER_MONTH_LY : DAYS_AFTER_MONTH;

    // All months have less than 32 days, so n >> 5 is a good conservative
    // estimate for the month
    int64_t m = (n >> 5) + 1;
    
    // m = 1-based month number
    while (n >= daysAfterMonth[m]) {
        m++;
    }
            
    // If month was requested, return it
    if (datePart == DATE_PART_MONTH) {
        return m;
    }

    THE_ASSERT(datePart == DATE_PART_DAY);
    // Return 1-based day-of-month
    return n - daysAfterMonth[m - 1] + 1;
}
