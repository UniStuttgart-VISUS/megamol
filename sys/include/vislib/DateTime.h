/*
 * DateTime.h
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_DATETIME_H_INCLUDED
#define VISLIB_DATETIME_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include <ctime>

#include "vislib/DateTimeSpan.h"
#include "vislib/StackTrace.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {


    /**
     * This class implements a combined date and time. 
     *
     * The class basically implements a proleptic Gregorian calendar, i. e. the 
     * rules of the Gregorian calendar are extended indefinitely into the past,
     * even beyond 24.02.1582. 
     *
     * Dates are stored as ticks with a resolution of 100 ns (this is compatible
     * with .NET) since 01.01.01 00:00:00. The date range ends at approx. 
     * 29247 A.D.
     *
     * A DateTime is always assumed to be local time. Input methods, ctors and
     * conversion operators that accept or return point in time in other time
     * zones are converted accordingly.
     *
     * @author Christoph Mueller
     */
    class DateTime {

    public:

        /** Enumeration of week days. */
        enum DayOfWeek {
            SUNDAY = 0,     ///< Symbolic name of sunday.
            MONDAY = 1,     ///< Symbolic name of monday.
            TUESDAY = 2,    ///< Symbolic name of tueday.
            WEDNESDAY = 3,  ///< Symbolic name of wednesday.
            THURSDAY = 4,   ///< Symbolic name of thursday.
            FRIDAY = 5,     ///< Symbolic name of friday.
            SATURDAY = 6    ///< Symbolic name of saturday.
        };

        /**
         * Answer whether 'year' is a leap year assuming a proleptic
         * Gregorian calendar.
         *
         * The implementation follows the rule that leap years must be
         * divisable by 4, but not by 100 except for if they are divisable
         * by 400, too.
         *
         * @param year The year to test.
         *
         * @return true, if 'year' is a leap year, false otherwise.
         */
        static bool IsLeapYear(const INT year);

        /**
         * Get the current local time.
         *
         * @return The current date and time.
         */
        static DateTime Now(void);

        /**
         * Convert a time span to 100 ns ticks.
         *
         * @param hours        The full hours.
         * @param minutes      The full minutes.
         * @param seconds      The full seconds.
         * @param milliseconds The full milliseconds. This defaults to 0.
         * @param ticks        The remaining ticks. This defaults to 0.
         *
         * @return The time span in 100 ns ticks.
         *
         * @throws IllegalParamException If the conversion fails due to numeric
         *                               overflows.
         */
        inline static INT64 TimeToTicks(const INT hours, const INT minutes, 
                const INT seconds, const INT milliseconds = 0,
                const INT ticks = 0) {
            VLSTACKTRACE("DateTime::TimeToTicks", __FILE__, __LINE__);
            return DateTimeSpan::TimeToTicks(hours, minutes, seconds,
                milliseconds, ticks);
        }

        /**
         * Get the current local day with the time set zero.
         *
         * @return The current date.
         */
        static DateTime Today(void);

        /**
         * A constant empty time (at zero point 01.01.0001).
         */
        static DateTime EMPTY;

        /**
         * Creates a new instance representing the current point int time.
         */
        DateTime(void);

        /** 
         * Create a new DateTime with the specified initial value.
         *
         * The input of this ctor is assumed to be local time. No time zone
         * conversion will be performed.
         *
         * @param year         The year, positive for years A. D., negative 
         *                     for B. C. A value of 0 will be corrected to 1.
         * @param month        The month within [1, 12]. Invalid months are 
         *                     corrected anging the year.
         * @param day          The day. Invalid dates are corrected by changing 
         *                     the month and/or year.
         * @param hours        The hours within [0, 24]. Invalid values are 
         *                     corrected. This parameter defaults to 0.
         * @param minutes      The minutes within [0, 60[. Invalid values are 
         *                     corrected. This parameter defaults to 0.
         * @param seconds      The second within [0, 60[. Invalid values are 
         *                     corrected. This parameter defaults to 0.
         * @param milliseconds The milliseconds within [0, 1000[. Invalid values
         *                     are corrected. This parameter defaults to 0.
         *
         * @throws TODO should implement overflow check.
         */
        DateTime(const INT year, const INT month, const INT day, 
            const INT hours = 0, const INT minutes = 0, const INT seconds = 0,
            const INT milliseconds = 0);

        /**
         * Create a new instance from struct tm. This ctor allows for implicit
         * conversion to DateTime.
         *
         * The input is assumed to be local time. No time zone conversion will
         * be performed.
         *
         * @param tm The time that should be used for initialisation.
         */
        inline DateTime(const struct tm& tm) {
            VLSTACKTRACE("DateTime::DateTime", __FILE__, __LINE__);
            this->Set(tm);
        }

        /**
         * Create a new instance from time_t. This ctor allows for implicit 
         * conversion to DateTime.
         *
         * The input is assumed to be UTC and will converted to local time.
         *
         * This ctor is explicit, because time_t is an integral type (typedef
         * or define). This would allow implicit integer to time conversions,
         * which we consider dangerous.
         *
         * @param time The time that should be used for initialisation.
         */
        explicit inline DateTime(const time_t time) {
            VLSTACKTRACE("DateTime::DateTime", __FILE__, __LINE__);
            this->Set(time);
        }

#ifdef _WIN32
        /**
         * Create a new instance from FILETIME. This ctor allows for implicit 
         * conversion to DateTime.
         *
         * The input is assumed to be UTC and will converted to local time.
         *
         * @param fileTime The time that should be used for initialisation.
         */
        inline DateTime(const FILETIME& fileTime) {
            VLSTACKTRACE("DateTime::DateTime", __FILE__, __LINE__);
            this->Set(fileTime);
        }

        inline DateTime(const SYSTEMTIME& systemTime, const bool isUTC = true) {
            VLSTACKTRACE("DateTime::DateTime", __FILE__, __LINE__);
            this->Set(systemTime, isUTC);
        }
#endif /* _WIN32 */

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline DateTime(const DateTime& rhs) : ticks(rhs.ticks) {
            VLSTACKTRACE("DateTime::DateTime", __FILE__, __LINE__);
        }

        /** Dtor. */
        ~DateTime(void);

        ///**
        // * Add the specified number of days to the current date.
        // *
        // * @param days The number of days to add.
        // */
        //inline void AddDays(const INT days) {
        //    this->value += days * ONE_DAY;
        //}

        ///**
        // * Add the specified number of hours to the current date.
        // *
        // * @param hours The number of hours to add.
        // */
        //inline void AddHours(const INT hours) {
        //    this->value += hours * ONE_HOUR;
        //}

        ///**
        // * Add the specified number of minutes to the current date.
        // *
        // * @param minutes The number of minutes to add.
        // */
        //inline void AddMinutes(const INT minutes) {
        //    this->value += minutes * ONE_MINUTE;
        //}

        /////**
        //// * Add the specified number of months to the current date.
        //// *
        //// * @param months The number of months to add.
        //// */
        ////void AddMonths(const INT months);

        ///**
        // * Add the specified number of seconds to the current date.
        // *
        // * @param seconds The number of seconds to add.
        // */
        //inline void AddSeconds(const INT seconds) {
        //    this->value += seconds * ONE_SECOND;
        //}

        /////**
        //// * Add the specified number of years to the current date.
        //// *
        //// * @param years The number of years to add.
        //// */
        ////void AddYears(const INT years);

        /**
         * Get the date and time.
         *
         * @param outYear         Receives the year. Negative years represent a 
         *                        date B. C.
         * @param outMonth        Receives the month. The value is within 
         *                        [1, 12].
         * @param outDay          Recevies the day in the month.
         * @param outHours        Receives the hours.
         * @param outMinutes      Receives the minutes.
         * @param outSeconds      Receives the seconds.
         * @param outMilliseconds Receives the milliseconds.
         */
        inline void Get(INT& outYear, INT& outMonth, INT& outDay,
                INT& outHours, INT& outMinutes, INT& outSeconds, 
                INT& outMilliseconds) const {
            VLSTACKTRACE("DateTime::Get", __FILE__, __LINE__);
            this->GetDate(outYear, outMonth, outDay);
            this->GetTime(outHours, outMinutes, outSeconds, outMilliseconds);
        }

        /**
         * Get the date and time.
         *
         * @param outYear    Receives the year. Negative years represent a 
         *                   date B. C.
         * @param outMonth   Receives the month. The value is within 
         *                   [1, 12].
         * @param outDay     Recevies the day in the month.
         * @param outHours   Receives the hours.
         * @param outMinutes Receives the minutes.
         * @param outSeconds Receives the seconds.
         */
        inline void Get(INT& outYear, INT& outMonth, INT& outDay,
                INT& outHours, INT& outMinutes, INT& outSeconds) const {
            VLSTACKTRACE("DateTime::Get", __FILE__, __LINE__);
            this->GetDate(outYear, outMonth, outDay);
            this->GetTime(outHours, outMinutes, outSeconds);
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
         * @param outHours        Receives the hours.
         * @param outMinutes      Receives the minutes.
         * @param outSeconds      Receives the seconds.
         * @param outMilliseconds Receives the milliseconds.
         */
        void GetTime(INT& outHours, INT& outMinutes, INT& outSeconds, 
            INT& outMilliseconds) const;

        /**
         * Get the time.
         *
         * @param outHours   Receives the hours.
         * @param outMinutes Receives the minutes.
         * @param outSeconds Receives the seconds.
         */
        inline void GetTime(INT& outHours, INT& outMinutes, 
                INT& outSeconds) const {
            VLSTACKTRACE("DateTime::GetTime", __FILE__, __LINE__);
            INT tmp;
            this->GetTime(outHours, outMinutes, outSeconds, tmp);
        }

        /**
         * Get the 100 ns ticks since 01.01.0001.
         *
         * @return The total ticks that represent the time.
         */
        inline INT64 GetTotalTicks(void) const {
            VLSTACKTRACE("DateTime::GetTotalTicks", __FILE__, __LINE__);
            return this->ticks;
        }

        /**
         * Set a new date and time.
         *
         * The input of this method is assumed to be local time. No time zone
         * conversion will be performed.
         *
         * @param year         The year, positive for years A. D., negative 
         *                     for B. C. A value of 0 will be corrected to 1.
         * @param month        The month within [1, 12]. Invalid months are 
         *                     corrected anging the year.
         * @param day          The day. Invalid dates are corrected by changing 
         *                     the month and/or year.
         * @param hours        The hours within [0, 24]. Invalid values are 
         *                     corrected.
         * @param minutes      The minutes within [0, 60[. Invalid values are 
         *                     corrected.
         * @param seconds      The second within [0, 60[. Invalid values are 
         *                     corrected.
         * @param milliseconds The milliseconds within [0, 1000[. Invalid values
         *                     are corrected. This parameter defaults to 0.
         * @param ticks        The 100 ns ticks within [0, 
         *                     TICKS_PER_MILLISECOND[. This parameter defaults 
         *                     to 0.
         *
         * @throws TODO should implement overflow check.
         */
        void Set(const INT year, const INT month, const INT day, 
            const INT hours, const INT minutes, const INT seconds, 
            const INT milliseconds = 0, const INT ticks = 0);

        /**
         * Set the date and time from the specified struct tm. 'tm' is assumed 
         * to be local time and will not be converted. Only the following 
         * adjustments are performed:
         *
         * - tm_year starts at 1900, wherefore 1900 will be added to the year.
         * - tm_mon is zero-based, wherefore 1 will be added to the month.
         * - struct tm does not contain milliseconds, wherefore the milliseconds
         *   will be set 0.
         *
         * @param tm The struct tm to use the value of.
         */
        void Set(const struct tm& tm);

        /**
         * Set the date and time from a time_t. time_t is assumed to be UTC and
         * will be converted to local time.
         *
         * @param time  The time to set.
         *
         * @throws SystemException If 'time' has an invalid value.
         */
        void Set(const time_t time);

#ifdef _WIN32
        /**
         * Change the value of this object to the value of 'fileTime'. 
         * 'fileTime' is assumed to be UTC and will be converted to local
         * time by this method. That is because the FILETIME structure
         * is defined to be UTC.
         *
         * @param fileTime The FILETIME to set.
         *
         * @throws SystemException If the conversion of the FILETIME into a
         *                         SYSTEMTIME structure failed.
         */
        void Set(const FILETIME& fileTime);

        /**
         * Change the value of this object to the value of 'systemTime'.
         * If not specified differently, 'systemTime' is assumet to be UTC.
         *
         * @param systemTime The SYSTEMTIME to set.
         * @param isUTC      If set true, 'systemTime' is assumed to be UTC. 
         *                   This will trigger a conversion to local time by set
         *                   method. If the flag is false, the input treated as 
         *                   local time  and will not be converted. This 
         *                   parameter defaults true.
         *
         * @throws SystemException If the conversion from UTC to local time
         *                         failed.
         */
        void Set(const SYSTEMTIME& systemTime, const bool isUTC = true);
#endif /* _WIN32 */

        /**
         * Set the date part without modifying the time.
         *
         * The input of this method is assumed to be local time. No time zone
         * conversion will be performed.
         *
         * @param year  The year, positive for years A. D., negative for B. C.
         * @param month The month within [1, 31]. Invalid months are corrected
         *              by changing the year.
         * @param day   The day. Invalid dates are corrected by changing the
         *              month and/or year.
         */
        void SetDate(const INT year, const INT month, const INT day); 

        /**
         * Se the time part without modifying the date.
         *
         * The input of this method is assumed to be local time. No time zone
         * conversion will be performed.
         *
         * @param hours        The hours within [0, 24]. Invalid values are 
         *                     corrected.
         * @param minutes      The minutes within [0, 60[. Invalid values are 
         *                     corrected. 
         * @param seconds      The second within [0, 60[. Invalid values are 
         *                     corrected.
         * @param milliseconds The milliseconds within [0, 1000[. Invalid values
         *                     are corrected. This parameter defaults to 0.
         * @param ticks        The 100 ns ticks within [0, 
         *                     TICKS_PER_MILLISECOND[. This parameter defaults 
         *                     to 0.
         */
        void SetTime(const INT hour, const INT minute, const INT second,
            const INT milliseconds = 0, const INT ticks = 0);

        ///**
        // * Subtract the specified number of days from the current date.
        // *
        // * @param days The number of days to subtract.
        // */
        //inline void SubtractDays(const INT days) {
        //    this->value -= days * ONE_DAY;
        //}

        ///**
        // * Subtract the specified number of hours from the current date.
        // *
        // * @param hours The number of hours to subtract.
        // */
        //inline void SubtractHours(const INT hours) {
        //    this->value -= hours * ONE_HOUR;
        //}

        ///**
        // * Subtract the specified number of minutes from the current date.
        // *
        // * @param minutes The number of minutes to subtract.
        // */
        //inline void SubtractMinutes(const INT minutes) {
        //    this->value -= minutes * ONE_MINUTE;
        //}

        ///**
        // * Subtract the specified number of seconds from the current date.
        // *
        // * @param seconds The number of seconds to subtract.
        // */
        //inline void SubtractSeconds(const INT seconds) {
        //    this->value -= seconds * ONE_SECOND;
        //}

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
            VLSTACKTRACE("DateTime::operator ==", __FILE__, __LINE__);
            return (this->ticks == rhs.ticks);
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
            VLSTACKTRACE("DateTime::operator !=", __FILE__, __LINE__);
            return (this->ticks != rhs.ticks);
        }

        /**
         * Answer whether this DateTime lies before 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if this DateTime lies before 'rhs', false otherwise.
         */
        inline bool operator <(const DateTime& rhs) const {
            VLSTACKTRACE("DateTime::operator <", __FILE__, __LINE__);
            return (this->ticks < rhs.ticks);
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
            VLSTACKTRACE("DateTime::operator <=", __FILE__, __LINE__);
            return (this->ticks <= rhs.ticks);
        }

        /**
         * Answer whether this DateTime lies after 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if this DateTime lies after 'rhs', false otherwise.
         */
        inline bool operator >(const DateTime& rhs) const {
            VLSTACKTRACE("DateTime::operator >", __FILE__, __LINE__);
            return (this->ticks > rhs.ticks);
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
            VLSTACKTRACE("DateTime::operator >=", __FILE__, __LINE__);
            return (this->ticks >= rhs.ticks);
        }

        /**
         * Add this date and the given time span.
         *
         * @param rhs The right hand side operand.
         *
         * @return A point in time that is 'rhs' in the future from this point
         *         in time.
         *
         * @throws IllegalParamException If 'rhs' has such a value that the
         *                               result would overflow.
         */
        inline DateTime operator +(const DateTimeSpan& rhs) const {
            VLSTACKTRACE("DateTime::operator +", __FILE__, __LINE__);
            DateTime retval(*this);
            retval += rhs;
            return retval;
        }
	    
        /**
         * Subtract the given time span from this date.
         *
         * @param rhs The right hand side operand.
         *
         * @return A point in time that is 'rhs' before this point in time.
         *
         * @throws IllegalParamException If 'rhs' has such a value that the
         *                               result would overflow.
         */
        inline DateTime operator -(const DateTimeSpan& rhs) const {
            VLSTACKTRACE("DateTime::operator -", __FILE__, __LINE__);
            DateTime retval(*this);
            retval -= rhs;
            return retval;
        }

        /**
         * Move this point in time into the future by 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If 'rhs' has such a value that the
         *                               result would overflow.
         */
	    DateTime& operator +=(const DateTimeSpan& rhs);

        /**
         * Move this point in time into the past by 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If 'rhs' has such a value that the
         *                               result would overflow.
         */
	    DateTime& operator -=(const DateTimeSpan& rhs);

        /**
         * Compute the time span between this point in time and 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return The time span between the two points in time.
         *
         * @throws IllegalParamException If 'rhs' has such a value that the
         *                               result would overflow.
         */
        DateTimeSpan operator -(const DateTime& rhs) const;

        /**
         * Conversion to struct tm. The returned struct tm is local time.
         *
         * Note that this cast is slow as it must copy a structure.
         *
         * @return A struct tm holding the local time represented by this
         *         object.
         */
        operator struct tm(void) const;

        /**
         * Conversion to a time_t. The returned time_t is UTC.
         *
         * Note that this cast is slow as it must copy a structure.
         *
         * @return The time_t representing the time of this object. Note that
         *         the value returned is UTC.
         */
        operator time_t(void) const;

#ifdef _WIN32
        /**
         * Conversion to FILETIME. The returned FILETIME is UTC.
         *
         * Note that this cast is slow as it must copy a structure.
         *
         * @return A FILETIME representing the time of this object. Note that
         *         the value returned is UTC.
         */
        operator FILETIME(void) const;

        /**
         * Conversion to SYSTEMTIME. The returned SYSTEMTIME is UTC.
         *
         * Note that this cast is slow as it must copy a structure.
         *
         * @return A SYSTEMTIME representing the time of this object. Note that
         *         the value returned is UTC.
         */
        operator SYSTEMTIME(void) const;
#endif /* _WIN32 */

    private:

        /** Possible parts that our private getter can extract. */
        enum DatePart {
            DATE_PART_YEAR = 0x00000001,
            DATE_PART_DAY_OF_YEAR = 0x00000002,
            DATE_PART_MONTH = 0x00000004,
            DATE_PART_DAY = 0x00000008,
        };

        /**
         * This array holds the days after the end of a month with January
         * being at index 1. Element 0 holds a zero, element 13 holds the 
         * total number of days of a non-leap year.
         */
        static const INT64 DAYS_AFTER_MONTH[13];

        /**
         * The same as DAYS_AFTER_MONTH, but for leap years. 
         */
        static const INT64 DAYS_AFTER_MONTH_LY[13];

        /** The days in a normal, i. e. non-leap, year. */
        static const INT64 DAYS_PER_YEAR;

        /** The days in a four year period, including leap years. */
        static const INT64 DAYS_PER_4YEARS;

        /** The days in a 100 year period, including leap years. */
        static const INT64 DAYS_PER_100YEARS;

        /** The days in a 400 year period, including leap years. */
        static const INT64 DAYS_PER_400YEARS;

        /**
         * Create a new instance with the given initial value.
         *
         * This ctor is required for creating the EMPTY constant.
         *
         * @param ticks The ticks since 01.01.0001.
         * @param dowel Ignore this. Any data is acceptable
         */
        inline DateTime(const INT64 value, const INT dowel) : ticks(ticks) {
            VLSTACKTRACE("DateTime::DateTime", __FILE__, __LINE__);
        }

        /**
         * Get a specific part of this date.
         *
         * @param datePart The part to be retrieved.
         *
         * @return The value of the specified part.
         */
        INT64 get(const DatePart datePart) const;

        /** The date value in 100 ns ticks since 01.01.0001. */
        INT64 ticks;
    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_DATETIME_H_INCLUDED */
