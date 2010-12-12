/*
 * DateTime.cpp
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller. Alle Rechte vorbehalten.
 */


#include "vislib/DateTime.h"

#include <climits>

#include "vislib/assert.h"
#include "vislib/error.h"
#include "vislib/SystemException.h"



/*
 * vislib::sys::DateTime::IsLeapYear
 */
bool vislib::sys::DateTime::IsLeapYear(const INT year) {
    VLSTACKTRACE("DateTime::IsLeapYear", __FILE__, __LINE__);
    return ((year % 4) == 0) && (((year % 100) != 0) || ((year % 400) == 0));
}


/*
 * vislib::sys::DateTime::Now
 */
vislib::sys::DateTime vislib::sys::DateTime::Now(void) {
    VLSTACKTRACE("DateTime::Now", __FILE__, __LINE__);
    DateTime retval;
    return retval;
}


/*
 * vislib::sys::DateTime::Today
 */
vislib::sys::DateTime vislib::sys::DateTime::Today(void) {
    VLSTACKTRACE("DateTime::Today", __FILE__, __LINE__);
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
vislib::sys::DateTime::DateTime(void) : value(0L) {
    VLSTACKTRACE("DateTime::DateTime", __FILE__, __LINE__);
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
vislib::sys::DateTime::DateTime(const INT year, const INT month, const INT day,
        const INT hours, const INT minutes, const INT seconds, 
        const INT milliseconds) {
    VLSTACKTRACE("DateTime::DateTime", __FILE__, __LINE__);
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
    VLSTACKTRACE("DateTime::~DateTime", __FILE__, __LINE__);
}


// TODO: This does not work any more with our input interpretation
///*
// * cml::DateTime::AddMonths
// */
//void cml::DateTime::AddMonths(const INT months) {
//    INT year, month, day;
//    this->GetDate(year, month, day);
//    this->SetDate(year, month + months, day);
//}
//
//
///*
// * cml::DateTime::AddYears
// */
//void cml::DateTime::AddYears(const INT years) {
//    INT year, month, day;
//    this->GetDate(year, month, day);
//    this->SetDate(year + years, month, day);
//}


/*
 * vislib::sys::DateTime::GetDate
 */
void vislib::sys::DateTime::GetDate(INT& outYear, 
                                    INT& outMonth, 
                                    INT& outDay) const {
    ASSERT(this->value >= 0); // TODO: Implementation does not yet work for BC
    INT64 days = this->value / ONE_DAY;     // Full days.
    INT64 bcExtraOffset = (this->value % ONE_DAY == 0) ? 1 : 0;
    INT64 cnt400Years = 0;                  // # of 400 year blocks.
    INT64 cnt100Years = 0;                  // # of 100 year blocks.
    INT64 cnt4Years = 0;                    // # of 4 year blocks.
    INT64 cntYears = 0;                     // # of remaining years.
    bool containsLeapYear = true;           // Contains 4 year block leap year?

    //if (days >= 0) {
    //    /*
    //     * The subsequent iterative divisions and modulo operations require a 
    //     * year 0 to work. However, the Gregorian calendar does not have such
    //     * a year and therefore a value of 0 represents 01.01.0001 in this 
    //     * class. Therefore, we add the non-existing year 0 for the following 
    //     * steps here. Note, that we have to add 366 days because the 
    //     * non-existing year 0 would be a leap year according to the rules of
    //     * the Gregorian calendar.
    //     */
    //    days += 366;

    //} else {
    //    days = -days + 365;
    //}

    /* 
     * Determine 400 year blocks lying behind and make 'days' relative to
     * active 400 year block.
     */
    cnt400Years = days / (400 * ONE_YEAR + (400 / 4) - 3);
    days %= (400 * ONE_YEAR + (400 / 4) - 3);

    /* Determine 100 year blocks within 400 year block lying behind. */
    // Subtract 1 because first century of 4 has 366 days.
    cnt100Years = (days - 1) / (100 * ONE_YEAR + (100 / 4) - 1);
    ASSERT(cnt100Years >= -3);
    ASSERT(cnt100Years <= 3);

    if (cnt100Years == 0) {
        /*
         * First century within 400 year block, which has one day more than
         * the following centuries.
         */

        /* 
         * Determine 4 year blocks lying behind and make 'days' relative to the
         * active 4 year block.
         */
        cnt4Years = days / (4 * ONE_YEAR + 1);
        days %= (4 * ONE_YEAR + 1);

    } else {
        /*
         * "Normal" century, must make 'days' relative to century first and 
         * handle special case of a year that is divisible by 4 and 100 is
         * not a leap year.
         */
        
        days = (days - 1) % (100 * ONE_YEAR + (100 / 4) - 1); 

        /* Determine 4 year block within century lying behind. */
        // Add 1, because one 4 year block is non-leap.
        cnt4Years = (days + 1) / (4 * ONE_YEAR + 1);
        ASSERT(cnt4Years >= -24);
        ASSERT(cnt4Years <= 24);

        if (cnt4Years == 0) {
            /* 
             * We are in first 4 year block of a non-leap century, so this block
             * is the exception (divisible by 100, but not by 400) having no
             * leap year.
             */
            containsLeapYear = false;

        } else {
            /* Other 4 year block, make 'days' relative. */
            days = (days + 1) % (4 * ONE_YEAR + 1);
        }
    } /* end if (cnt100Years == 0) */
    
    /* Compute years in 4 year block lying behind. */
    if (containsLeapYear) {
        /* Special case: We have one leap year in the 4 year block. */

        cntYears = (days - 1) / ONE_YEAR;   // One year has 366 days.
        
        if (cntYears != 0) {
            days = (days - 1) % ONE_YEAR;   // Make relative to year.
        }

    } else {
        /*
         * No leap year, compute full years and remaining days using the "normal"
         * ONE_YEAR length.
         */

        cntYears = days / ONE_YEAR;
        days %= ONE_YEAR;
    } /* if (hasLeap) */
    ASSERT(cntYears >= -3);
    ASSERT(cntYears <= 3);

    /* At this point, we can compute the year. */
    outYear = static_cast<INT>(400 * cnt400Years + 100 * cnt100Years
         + 4 * cnt4Years + cntYears);
    if (this->value < 0) {
        //outYear = -outYear;
        //days = 365 - days - 1;
//        ASSERT(days >= 0);
    }
    //if (days < 0) {
    //    days = 365 + IsLeapYear(outYear - 1) + days + 1;
    //    //days = -days;
    //}


     /* Compute month and day. */
    if ((cntYears == 0) && containsLeapYear && (days == 59)) {
        /* On 29.02. */ 
        outMonth = 2;
        outDay = 29;

    } else {
        /* 
         * Manipulate 'days' to look like a non-leap year for month lookup
         * in the DAYS_AFTER_MONTH array.
         */

        if ((cntYears == 0) && containsLeapYear && (days >= 59)) {
            /* After 29.02. */
            days--;
        }

        days++;

        // Month must be greater than or equal to ('days' / 32).
        for (outMonth = static_cast<INT>((days >> 5) + 1); 
            days > DAYS_AFTER_MONTH[outMonth]; outMonth++);

        outDay = static_cast<INT>(days - DAYS_AFTER_MONTH[outMonth - 1]);
    }

    /* Handle years B. C. */
    //if (this->value < 0) {
    //    outYear *= -1;
    //}
}


/*
 * vislib::sys::DateTime::GetTime
 */
void vislib::sys::DateTime::GetTime(INT& outHours, 
                                    INT& outMinutes, 
                                    INT& outSeconds,
                                    INT& outMilliseconds) const {
    INT64 time = this->value % ONE_DAY;
    if (time < 0) {
        time += ONE_DAY;
    }

    outMilliseconds = static_cast<INT>(time % ONE_SECOND);
    time /= ONE_SECOND;
    outSeconds = static_cast<INT>(time % 60L);
    time /= 60L;
    outMinutes = static_cast<INT>(time % 60L);
    time /= 60L;
    outHours = static_cast<INT>(time);
}


/*
 * vislib::sys::DateTime::Set
 */
void vislib::sys::DateTime::Set(const INT year, const INT month, const INT day,
        const INT hours, const INT minutes, const INT seconds, 
        const INT milliseconds) {
    INT64 y = year;                         // The possibly corrected year.
    INT64 m = (month != 0) ? month : 1;     // The possibly corrected month.
    INT64 d = (day != 0) ? day : 1;         // The possibly corrected day.
    INT64 t = 0;                            // Auxiliary variable.

    /* Correct possible invalid months by rolling the year. */
    if (m < 0) {
        /* Roll backwards. */
        t = ++m;                            // Must reflect missing month 0.
        m = 12 + m % 12;
        if ((y += t / 12 - 1) == 0) {       // TODO: Check this!
            y = -1;                         // TODO: Check this!
        }

    } else if (m > 12) {
        /* Roll forward. */
        t = m;
        m = m % 12;
        if ((y += t / 12) == 0) {
            y = 0;
        }
    }
    ASSERT(m >= 1);
    ASSERT(m <= 12);

    /* Compute the days since 01.01.0000. */
    this->value = y * ONE_YEAR 
        + ((y >= 0) ? 1 : 0)                // Add leap day for year 0.
        + (y / 4) - (y / 100) + (y / 400)   // Add leap day for leap years.
        + DAYS_AFTER_MONTH[m - 1] 
        + ((d > 0) ? (d - 1) : d);

    /* 
     * If we are before march in a leap year, we added too much leap days in the
     * computation before.
     */
    if ((m <= 2) && DateTime::IsLeapYear(static_cast<INT>(y))) {
        this->value--;
    }

    /* Convert the days to milliseconds. */
    this->value *= ONE_DAY;

    /* Add the time now. */
    this->value += hours * ONE_HOUR 
        + minutes * ONE_MINUTE 
        + seconds * ONE_SECOND
        + milliseconds;
}


/*
 * vislib::sys::DateTime::Set
 */
void vislib::sys::DateTime::Set(const struct tm& tm) {
    VLSTACKTRACE("DateTime::Set", __FILE__, __LINE__);
    this->Set(tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
        tm.tm_hour, tm.tm_min, tm.tm_sec, 0);
}


/*
 * vislib::sys::DateTime::Set
 */
void vislib::sys::DateTime::Set(const time_t time) {
    VLSTACKTRACE("DateTime::Set", __FILE__, __LINE__);
#if (defined(_MSC_VER) && (_MSC_VER >= 1400))
    struct tm tm;
    if (::localtime_s(&tm, &time) == 0) {
        this->Set(tm);
    } else {
        throw SystemException(ERROR_INVALID_DATA, __FILE__, __LINE__);
    }

#elif defined(_WIN32)
    struct tm *tm = ::localtime(&time);
    if (tm != NULL) {
        this->Set(*tm);
    } else {
        throw SystemException(ERROR_INVALID_DATA, __FILE__, __LINE__);
    }

#else  /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */
    struct tm tm;
    if (::localtime_r(&time, &tm) != NULL) {
        this->Set(tm);
    } else {
        throw SystemException(EINVAL, __FILE__, __LINE__);
    }
#endif /* (defined(_MSC_VER) && (_MSC_VER >= 1400)) */
}


#ifdef _WIN32
/*
 * vislib::sys::DateTime::Set
 */
void vislib::sys::DateTime::Set(const FILETIME& fileTime) {
    VLSTACKTRACE("DateTime::Set", __FILE__, __LINE__);
    SYSTEMTIME systemTime;
    if (!::FileTimeToSystemTime(&fileTime, &systemTime)) {
        throw SystemException(__FILE__, __LINE__);
    }
    this->Set(systemTime, true);
}


/*
 * vislib::sys::DateTime::Set
 */
void vislib::sys::DateTime::Set(const SYSTEMTIME& systemTime, 
        const bool isUTC) {
    VLSTACKTRACE("DateTime::Set", __FILE__, __LINE__);
    SYSTEMTIME lSystemTime;

    if (isUTC) {
        if (!::SystemTimeToTzSpecificLocalTime(NULL, 
                const_cast<SYSTEMTIME *>(&systemTime), &lSystemTime)) {
            throw SystemException(__FILE__, __LINE__);
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
void vislib::sys::DateTime::SetDate(const INT year, 
                                    const INT month, 
                                    const INT day) {
    VLSTACKTRACE("DateTime::SetDate", __FILE__, __LINE__);
    INT64 time = this->value % ONE_DAY;
    this->Set(year, month, day, 0, 0, 0);
    this->value += time;
}


/*
 * vislib::sys::DateTime::SetTime
 */
void vislib::sys::DateTime::SetTime(const INT hours, 
                                    const INT minutes, 
                                    const INT seconds,
                                    const INT milliseconds) {
    VLSTACKTRACE("DateTime::SetDate", __FILE__, __LINE__);
    this->value -= this->value % ONE_DAY;
    this->value += hours * ONE_HOUR
        + minutes * ONE_MINUTE 
        + seconds * ONE_SECOND
        + milliseconds;
}


/*
 * vislib::sys::DateTime::operator =
 */
vislib::sys::DateTime& vislib::sys::DateTime::operator =(const DateTime& rhs) {
    VLSTACKTRACE("DateTime::operator =", __FILE__, __LINE__);
    if (this != &rhs) {
        this->value = rhs.value;
    }
    return *this;
}


/*
 * vislib::sys::DateTime::operator +=
 */
vislib::sys::DateTime& vislib::sys::DateTime::operator +=(
        const DateTimeSpan& rhs) {
    VLSTACKTRACE("DateTime::operator +=", __FILE__, __LINE__);
    DateTimeSpan tmp(this->value);
    tmp += rhs;
    this->value = static_cast<INT64>(tmp);
    return *this;
}


/*
 * vislib::sys::DateTime::operator -=
 */
vislib::sys::DateTime& vislib::sys::DateTime::operator -=(
        const DateTimeSpan& rhs) {
    VLSTACKTRACE("DateTime::operator -=", __FILE__, __LINE__);
    DateTimeSpan tmp(this->value);
    tmp -= rhs;
    this->value = static_cast<INT64>(tmp);
    return *this;
}


/*
 * vislib::sys::DateTime::operator -
 */
vislib::sys::DateTimeSpan vislib::sys::DateTime::operator -(
        const DateTime& rhs) const {
    VLSTACKTRACE("DateTime::operator -", __FILE__, __LINE__);
    DateTimeSpan retval(this->value);
    DateTimeSpan r(rhs.value);
    retval -= r;
    return retval;
}


/*
 * vislib::sys::DateTime::operator struct tm
 */
vislib::sys::DateTime::operator struct tm(void) const {
    struct tm retval;
    INT year, month, day, hour, minute, second;
    
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
    return ::mktime(&static_cast<struct tm>(*this));
}


#ifdef _WIN32
/* 
 * vislib::sys::DateTime::operator FILETIME
 */
vislib::sys::DateTime::operator FILETIME(void) const {
    FILETIME retval;
    SYSTEMTIME systemTime = static_cast<SYSTEMTIME>(*this);
    
    if (!::SystemTimeToFileTime(&systemTime, &retval)) {
        throw SystemException(__FILE__, __LINE__);
    }

    return retval;
}


/*
 * vislib::sys::DateTime::operator SYSTEMTIME
 */
vislib::sys::DateTime::operator SYSTEMTIME(void) const {
    SYSTEMTIME localTime, retval;
    INT year, month, day, hours, minutes, seconds, millis;
    
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
const INT64 vislib::sys::DateTime::DAYS_AFTER_MONTH[13] = {
    0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365 
};


/*
 * vislib::sys::DateTime::ONE_DAY
 */
const INT64 vislib::sys::DateTime::ONE_DAY 
    = vislib::sys::DateTimeSpan::MILLISECONDS_PER_DAY;


/*
 * vislib::sys::DateTime::ONE_HOUR
 */
const INT64 vislib::sys::DateTime::ONE_HOUR 
    = vislib::sys::DateTimeSpan::MILLISECONDS_PER_HOUR;


/*
 * vislib::sys::DateTime::ONE_MINUTE
 */
const INT64 vislib::sys::DateTime::ONE_MINUTE 
    = vislib::sys::DateTimeSpan::MILLISECONDS_PER_MINUTE;
  

/*
 * vislib::sys::DateTime::ONE_SECOND
 */
const INT64 vislib::sys::DateTime::ONE_SECOND 
    = vislib::sys::DateTimeSpan::MILLISECONDS_PER_SECOND;


/*
 * vislib::sys::DateTime::ONE_YEAR
 */
const INT64 vislib::sys::DateTime::ONE_YEAR = static_cast<INT64>(365);
