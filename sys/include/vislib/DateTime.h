/*
 * DateTime.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_DATETIME_H_INCLUDED
#define VISLIB_DATETIME_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <ctime>

#include "vislib/DateTimeSpan.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {


    /**
     * This class implements a combined date and time. The class uses a 
     * proleptic Gregorian calendar, i. e. the rules of the Gregorian
     * calendar are extended before 24.02.1582. Note, that the Gregorian
     * calendar does not have a year 0.
     *
     * @author Christoph Mueller
     */
    class DateTime {

    public:

        enum DayOfWeek {
            MONDAY = 0,
            TUESDAY = 1,
            WEDNESDAY = 2,
            THURSDAY = 3,
            FRIDAY = 4,
            SATURDAY = 5,
            SUNDAY = 6
        };

        /**
         * Answer whether 'year' is a leap year assuming a proleptic
         * Gregorian calendar.
         *
         * @param year The year to test. The result is undefined for the 
         *             non-existing year 0.
         *
         * @return true, if 'year' is a leap year, false otherwise.
         */
        static bool IsLeapYear(const INT year);

        /**
         * Create a new instance representing the creation time of the
         * object in local time.
         */
        DateTime(void);

        inline DateTime(const INT year, const INT month, const INT day, 
                const INT hour, const INT minute, const INT second) {
            this->Set(year, month, day, hour, minute, second);
        }

        inline DateTime(const struct tm& tm) {
            this->Set(tm);
        }

        inline DateTime(const time_t time, const bool isLocalTime = false) {
            this->Set(time, isLocalTime);
        }

#ifdef _WIN32
        inline DateTime(const FILETIME& fileTime, 
                        const bool isLocalTime = false) {
            this->Set(fileTime, isLocalTime);
        }

        inline DateTime(const SYSTEMTIME& systemTime, 
                        const bool isLocalTime = false) {
            this->Set(systemTime, isLocalTime);
        }
#endif /* _WIN32 */

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline DateTime(const DateTime& rhs) : value(rhs.value) {}

        /** Dtor. */
        ~DateTime(void);

        /**
         * Add the specified number of days to the current date.
         *
         * @param days The number of days to add.
         */
        inline void AddDays(const INT days) {
            this->value += days * ONE_DAY;
        }

        /**
         * Add the specified number of hours to the current date.
         *
         * @param hours The number of hours to add.
         */
        inline void AddHours(const INT hours) {
            this->value += hours * ONE_HOUR;
        }

        /**
         * Add the specified number of minutes to the current date.
         *
         * @param minutes The number of minutes to add.
         */
        inline void AddMinutes(const INT minutes) {
            this->value += minutes * ONE_MINUTE;
        }

        ///**
        // * Add the specified number of months to the current date.
        // *
        // * @param months The number of months to add.
        // */
        //void AddMonths(const INT months);

        /**
         * Add the specified number of seconds to the current date.
         *
         * @param seconds The number of seconds to add.
         */
        inline void AddSeconds(const INT seconds) {
            this->value += seconds * ONE_SECOND;
        }

        ///**
        // * Add the specified number of years to the current date.
        // *
        // * @param years The number of years to add.
        // */
        //void AddYears(const INT years);

        /**
         * Get the date and time.
         *
         * @param outYear   Receives the year. Negative years represent a 
         *                  date B. C.
         * @param outMonth  Receives the month. The value is within [1, 12].
         * @param outDay    Recevies the day in the month.
         * @param outHour   Receives the hours.
         * @param outMinute Receives the minutes.
         * @param outSecond Receives the seconds.
         */
        inline void Get(INT& outYear, INT& outMonth, INT& outDay,
                INT& outHour, INT& outMinute, INT& outSecond) const {
            this->GetDate(outYear, outMonth, outDay);
            this->GetTime(outHour, outMinute, outSecond);
        }

        /**
         * Get the date.
         *
         * @param outYear   Receives the year. Negative years represent a 
         *                  date B. C.
         * @param outMonth  Receives the month. The value is within [1, 12].
         * @param outDay    Recevies the day in the month.
         */
        void GetDate(INT& outYear, INT& outMonth, INT& outDay) const;

        /**
         * Get the time.
         *
         * @param outHour   Receives the hours.
         * @param outMinute Receives the minutes.
         * @param outSecond Receives the seconds.
         */
        void GetTime(INT& outHour, INT& outMinute, INT& outSecond) const;

        /** 
         * Get the value of the date, i. e. the date in milliseconds since
         * midnight 01.01.0001.
         *
         * @return The internal value of the date.
         */
        inline INT64 GetValue(void) const {
            return this->value;
        }

        /**
         * Set a new date and time.
         *
         * @param year  The year, positive for years A. D., negative for B. C.
         *              Zero is not a valid year.
         * @param month The month within [1, 31]. Invalid months are corrected
         *              by changing the year.
         * @param day   The day. Invalid dates are corrected by changing the
         *              month and/or year.
         * @param hour   The hour within [0, 24]. Invalid values are corrected.
         * @param minute The minute within [0, 59]. Invalid values are 
         *               corrected.
         * @param second The second within [0, 59]. Invalid values are 
         *               corrected.
         */
        void Set(const INT year, const INT month, const INT day, 
            const INT hour, const INT minute, const INT second);

        /**
         * Set the date and time from the specified struct tm.
         *
         * @param tm The struct tm to use the value of. The time is assumed to 
         *           be local time.
         */
        void Set(const struct tm& tm);

        /**
         * Set the date and time from a time_t. If not specified otherwise,
         * 'time' is assumed to be UTC.
         *
         * @param time        The time to set.
         * @param isLocalTime If set true, 'time' is assumed to be local time
         *                    instead of UTC and not converted.
         *
         * @throws SystemException If 'time' has an invalid value.
         */
        void Set(const time_t time, const bool isLocalTime = false);

#ifdef _WIN32
        /**
         * Change the value of this object to the value of 'fileTime'. 
         * 'fileTime' is regarded to be UTC, unless 'isLocalTime' is set true.
         *
         * @param fileTime    The FILETIME to set.
         * @param isLocalTime If set true, 'fileTime' is assumed to be local 
         *                    time instead of UTC and not converted.
         *
         * @throws SystemException If the conversion of the FILETIME into a
         *                         SYSTEMTIME structure failed.
         */
        void Set(const FILETIME& fileTime, const bool isLocalTime = false);

        /**
         * Change the value of this object to the value of 'systemTime'.
         * 'systemTime' is regarded to be UTC, uncless 'isLocalTime' is set 
         * true.
         *
         * @param systemTime  The SYSTEMTIME to set.
         * @param isLocalTime If set true, 'systemTime' is assumed to be local
         *                    time instead of UTC and not converted.
         *
         * @throws SystemException If the conversion from UTC to local time
         *                         failed.
         */
        void Set(const SYSTEMTIME& systemTime, const bool isLocalTime = false);
#endif /* _WIN32 */

        /**
         * Set the date part without modifying the time.
         *
         * @param year  The year, positive for years A. D., negative for B. C.
         *              Zero is not a valid year.
         * @param month The month within [1, 31]. Invalid months are corrected
         *              by changing the year.
         * @param day   The day. Invalid dates are corrected by changing the
         *              month and/or year.
         */
        void SetDate(const INT year, const INT month, const INT day); 

        /**
         * Se the time part without modifying the date.
         *
         * @param hour   The hour within [0, 24]. Invalid values are corrected.
         * @param minute The minute within [0, 59]. Invalid values are 
         *               corrected.
         * @param second The second within [0, 59]. Invalid values are 
         *               corrected.
         */
        void SetTime(const INT hour, const INT minute, const INT second);

        /**
         * Subtract the specified number of days from the current date.
         *
         * @param days The number of days to subtract.
         */
        inline void SubtractDays(const INT days) {
            this->value -= days * ONE_DAY;
        }

        /**
         * Subtract the specified number of hours from the current date.
         *
         * @param hours The number of hours to subtract.
         */
        inline void SubtractHours(const INT hours) {
            this->value -= hours * ONE_HOUR;
        }

        /**
         * Subtract the specified number of minutes from the current date.
         *
         * @param minutes The number of minutes to subtract.
         */
        inline void SubtractMinutes(const INT minutes) {
            this->value -= minutes * ONE_MINUTE;
        }

        /**
         * Subtract the specified number of seconds from the current date.
         *
         * @param seconds The number of seconds to subtract.
         */
        inline void SubtractSeconds(const INT seconds) {
            this->value -= seconds * ONE_SECOND;
        }

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        DateTime& operator =(const DateTime& rhs);

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if 'rhs' and this object are equal, false otherwise.
         */
        inline bool operator ==(const DateTime& rhs) const {
            return (this->value == rhs.value);
        }

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if 'rhs' and this object are not equal, 
         *         false otherwise.
         */
        inline bool operator !=(const DateTime& rhs) const {
            return (this->value != rhs.value);
        }

        /**
         * Answer whether this DateTime lies before 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if this DateTime lies before 'rhs', false otherwise.
         */
        inline bool operator <(const DateTime& rhs) const {
            return (this->value < rhs.value);
        }

        /**
         * Answer wether this DateTime lies before or on 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if this DateTime lies before or on 'rhs', 
         *         false otherwise.
         */
        inline bool operator <=(const DateTime& rhs) const {
            return ((*this < rhs) || (*this == rhs));
        }

        /**
         * Answer whether this DateTime lies after 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if this DateTime lies after 'rhs', false otherwise.
         */
        inline bool operator >(const DateTime& rhs) const {
            return !(*this <= rhs);
        }

        /**
         * Answer wether this DateTime lies after or on 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if this DateTime lies after or on 'rhs', 
         *         false otherwise.
         */
        inline bool operator >=(const DateTime& rhs) const {
            return !(*this < rhs);
        }

        /**
         * Conversion to struct tm.
         *
         * Note that this cast is slow as it must copy a structure.
         *
         * @return A struct tm holding the local time represented by this
         *         object.
         */
        operator struct tm(void) const;

        /**
         * Conversion to a time_t.
         *
         * Note that this cast is slow as it must copy a structure.
         *
         * @return The time_t representing the time of this object. Note that
         *         the value returned is UTC.
         */
        operator time_t(void) const;

#ifdef _WIN32
        /**
         * Conversion to FILETIME.
         *
         * Note that this cast is slow as it must copy a structure.
         *
         * @return A FILETIME representing the time of this object. Note that
         *         the value returned is UTC.
         */
        operator FILETIME(void) const;

        /**
         * Conversion to SYSTEMTIME.
         *
         * Note that this cast is slow as it must copy a structure.
         *
         * @return A SYSTEMTIME representing the time of this object. Note that
         *         the value returned is UTC.
         */
        operator SYSTEMTIME(void) const;
#endif /* _WIN32 */

    protected:

        /**
         * This array holds the days after the end of a month with january
         * being at index 1. Element 0 holds a zero, element 13 holds the 
         * total number of days of a non-leap year.
         */
        static const INT64 DAYS_AFTER_MONTH[13];

        /** One day in milliseconds. */
        static const INT64 ONE_DAY;

        /** One hour in milliseconds. */
        static const INT64 ONE_HOUR;

        /** One minute in milliseconds. */
        static const INT64 ONE_MINUTE;
        
        /** One second in milliseconds. */
        static const INT64 ONE_SECOND;

        /** The days in a normal, i. e. non-leap, year. */
        static const INT64 ONE_YEAR;

        /** The date value in milliseconds since 01.01.0001. */
        INT64 value;
    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_DATETIME_H_INCLUDED */
