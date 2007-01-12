/*
 * DateTime.cpp
 *
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
    return ((year % 4) == 0) && (((year % 100) != 0) || ((year % 400) == 0));
}


/*
 * vislib::sys::DateTime::DateTime
 */
vislib::sys::DateTime::DateTime(void) {
#ifdef _WIN32
    SYSTEMTIME systemTime;
    ::GetLocalTime(&systemTime);
    this->Set(systemTime, true);

#else /* _WIN32 */
    time_t utc = ::time(NULL);
    this->Set(utc, false);
#endif /* _WIN32 */
}


/*
 * vislib::sys::DateTime::~DateTime
 */
vislib::sys::DateTime::~DateTime(void) {
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
    INT64 days = this->value / ONE_DAY;     // Full days.
    INT64 cnt400Years = 0;                  // # of 400 year blocks.
    INT64 cnt100Years = 0;                  // # of 100 year blocks.
    INT64 cnt4Years = 0;                    // # of 4 year blocks.
    INT64 cntYears = 0;                     // # of remaining years.
    bool containsLeapYear = true;           // Contains 4 year block leap year?

    /* Ensure 'days' to be positive, B. C. dates are handled later. */
    //if (days < 0) {
    //    days = -days;
    //}
    if (days >= 0) {
        days += 366;
    //} else {
    //    days++;
    }


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
         * handle special case, that a year that is divisible by 4 and 100 is
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
    //if (outYear == 0) {
    //    outYear++;
    //}
    //if (this->value < 0) {
    //    outYear = -outYear;
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
void vislib::sys::DateTime::GetTime(INT& outHour, 
                                    INT& outMinute, 
                                    INT& outSecond) const {
    INT64 time = this->value % ONE_DAY;
    if (time < 0) {
        time += ONE_DAY;
    }
    time /= ONE_SECOND;

    outSecond = static_cast<INT>(time % 60L);
    time /= 60L;
    outMinute = static_cast<INT>(time % 60L);
    outHour = static_cast<INT>(time / 60L);
}


/*
 * vislib::sys::DateTime::Set
 */
void vislib::sys::DateTime::Set(const INT year, const INT month, const INT day,
        const INT hour, const INT minute, const INT second) {
    INT64 y = (year != 0) ? year : 1;       // The possibly corrected year.
    INT64 m = (month != 0) ? month : 1;     // The possibly corrected month.
    INT64 d = (day != 0) ? day : 1;         // The possibly corrected day.
    INT64 t = 0;                            // Auxiliary variable.

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
    ASSERT(m >= 1);
    ASSERT(m <= 12);
    ASSERT(y != 0);

    /* Compute the days since 01.01.0001. */
    this->value = ((y > 0) ? (y - 1) : y) * ONE_YEAR 
        + (y / 4) - (y / 100) + (y / 400)
        + DAYS_AFTER_MONTH[m - 1] 
        + ((d > 0) ? (d - 1) : d);

    /* If we are in a leap year and before March, we must subtract 1 day. */
    if ((m <= 2) && DateTime::IsLeapYear(static_cast<INT>(y))) {
        this->value--;
    }

    /* Convert the days to milliseconds. */
    this->value *= ONE_DAY;

    /* Add the time now. */
    this->value += hour * ONE_HOUR + minute * ONE_MINUTE + second * ONE_SECOND;
}


/*
 * vislib::sys::DateTime::Set
 */
void vislib::sys::DateTime::Set(const struct tm& tm) {
    this->Set(tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
        tm.tm_hour, tm.tm_min, tm.tm_sec);
}


/*
 * vislib::sys::DateTime::Set
 */
void vislib::sys::DateTime::Set(const time_t time, const bool isLocalTime) {

    if (isLocalTime) {
#if (_MSC_VER >= 1400)
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

#else  /*(_MSC_VER >= 1400) */
        struct tm tm;
        if (::localtime_r(&time, &tm) != NULL) {
            this->Set(tm);
        } else {
            throw SystemException(EINVAL, __FILE__, __LINE__);
        }
#endif /*(_MSC_VER >= 1400) */

    } else {
#if (_MSC_VER >= 1400)
        struct tm tm;
        if (::gmtime_s(&tm, &time) == 0) {
            this->Set(tm);
        } else {
            throw SystemException(ERROR_INVALID_DATA, __FILE__, __LINE__);
        }

#elif defined(_WIN32)
        struct tm *tm = ::gmtime(&time);
        if (tm != NULL) {
            this->Set(*tm);
        } else {
            throw SystemException(ERROR_INVALID_DATA, __FILE__, __LINE__);
        }

#else  /*(_MSC_VER >= 1400) */
        struct tm tm;
        if (::gmtime_r(&time, &tm) != NULL) {
            this->Set(tm);
        } else {
            throw SystemException(EINVAL, __FILE__, __LINE__);
        }
#endif /*(_MSC_VER >= 1400) */
    }
}


#ifdef _WIN32
/*
 * vislib::sys::DateTime::Set
 */
void vislib::sys::DateTime::Set(const FILETIME& fileTime, 
                                const bool isLocalTime) {
    SYSTEMTIME systemTime;
    if (!::FileTimeToSystemTime(&fileTime, &systemTime)) {
        throw SystemException(__FILE__, __LINE__);
    }
    
    this->Set(systemTime, isLocalTime);
}


/*
 * vislib::sys::DateTime::Set
 */
void vislib::sys::DateTime::Set(const SYSTEMTIME& systemTime,
                                const bool isLocalTime) {
    SYSTEMTIME lSystemTime;

    if (isLocalTime) {
        this->Set(systemTime.wYear, systemTime.wMonth, systemTime.wDay,
            systemTime.wHour, systemTime.wMinute, systemTime.wSecond);
    } else {
        if (!::SystemTimeToTzSpecificLocalTime(NULL, 
                const_cast<SYSTEMTIME *>(&systemTime), &lSystemTime)) {
            throw SystemException(__FILE__, __LINE__);
        }

        this->Set(lSystemTime.wYear, lSystemTime.wMonth, lSystemTime.wDay,
            lSystemTime.wHour, lSystemTime.wMinute, lSystemTime.wSecond);
    }
}
#endif /* _WIN32 */


/*
 * vislib::sys::DateTime::SetDate
 */
void vislib::sys::DateTime::SetDate(const INT year, 
                                    const INT month, 
                                    const INT day) {
    INT64 time = this->value % ONE_DAY;
    this->Set(year, month, day, 0, 0, 0);
    this->value += time;
}


/*
 * vislib::sys::DateTime::SetTime
 */
void vislib::sys::DateTime::SetTime(const INT hour, 
                                    const INT minute, 
                                    const INT second) {
    this->value -= this->value % ONE_DAY;
    this->value += hour * ONE_HOUR + minute * ONE_MINUTE + second * ONE_SECOND;
}


/*
 * vislib::sys::DateTime::operator =
 */
vislib::sys::DateTime& vislib::sys::DateTime::operator =(const DateTime& rhs) {
    if (this != &rhs) {
        this->value = rhs.value;
    }

    return *this;
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
    retval.tm_isdst = 0;    // TODO
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
    INT year, month, day, hour, minute, second;
    
    this->Get(year, month, day, hour, minute, second);

    localTime.wYear = year;
    localTime.wMonth = month;
    localTime.wDay = day;
    localTime.wHour = hour;
    localTime.wMinute = minute;
    localTime.wSecond = second;
    localTime.wMilliseconds = 0;       // TODO
    localTime.wDayOfWeek = 0;          // TODO

    ::TzSpecificLocalTimeToSystemTime(NULL, &localTime, &retval);

    return retval;
}
#endif /* _WIN32 */


/*
 * vislib::sys::DateTime::DAYS_AFTER_MONTH
 */
const INT64 vislib::sys::DateTime::DAYS_AFTER_MONTH[13] = {
    0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365 };


/*
 * vislib::sys::DateTime::ONE_DAY
 */
const INT64 vislib::sys::DateTime::ONE_DAY 
    = static_cast<INT64>(24) * 60 * 60 * ONE_SECOND;
    // Note: Bug when using ONE_HOUR constant here.


/*
 * vislib::sys::DateTime::ONE_HOUR
 */
const INT64 vislib::sys::DateTime::ONE_HOUR 
    = static_cast<INT64>(60) * 60 * ONE_SECOND;
    // Note: Bug when using ONE_MINUTE constant here.


/*
 * vislib::sys::DateTime::ONE_MINUTE
 */
const INT64 vislib::sys::DateTime::ONE_MINUTE 
    = static_cast<INT64>(60) * ONE_SECOND;
  

/*
 * vislib::sys::DateTime::ONE_SECOND
 */
const INT64 vislib::sys::DateTime::ONE_SECOND = static_cast<INT64>(1000);


/*
 * vislib::sys::DateTime::ONE_YEAR
 */
const INT64 vislib::sys::DateTime::ONE_YEAR = static_cast<INT64>(365);
