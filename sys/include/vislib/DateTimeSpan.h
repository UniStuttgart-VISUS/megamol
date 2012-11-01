/*
 * DateTimeSpan.h
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_DATETIMESPAN_H_INCLUDED
#define VISLIB_DATETIMESPAN_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/StackTrace.h"
#include "vislib/String.h"
#include "vislib/types.h"



namespace vislib {
namespace sys {


    /**
     * This class represents a time span. The DateTimeSpan class has a 
     * resolution of 100 nanoseconds (which we call a tick here). This 
     * resolution is the same that .NET uses for its DateTime.
     */
    class DateTimeSpan {

    public:

        /**
         * Convert an amount of days to 100 ns ticks.
         *
         * @param days The days to convert.
         *
         * @return The days in 100 ns ticks.
         *
         * @throws TODO: range check!
         */
        static INT64 DaysToTicks(const INT64 days);

        /** 
         * Create a positive or negative DateTimeSpan of one day.
         *
         * @param isPositive Determines whether the time span is positive
         *                   or not. This parameter defaults to true.
         *
         * @return A time span of one day.
         */
        static inline DateTimeSpan OneDay(const bool isPositive = true) {
            return DateTimeSpan(isPositive ? TICKS_PER_DAY : -TICKS_PER_DAY);
        }

        /** 
         * Create a positive or negative DateTimeSpan of one hour.
         *
         * @param isPositive Determines whether the time span is positive
         *                   or not. This parameter defaults to true.
         *
         * @return A time span of one hour.
         */
        static inline DateTimeSpan OneHour(const bool isPositive = true) {
            return DateTimeSpan(isPositive ? TICKS_PER_HOUR : -TICKS_PER_HOUR);
        }

        /** 
         * Create a positive or negative DateTimeSpan of one millisecond.
         *
         * @param isPositive Determines whether the time span is positive
         *                   or not. This parameter defaults to true.
         *
         * @return A time span of one millisecond.
         */
        static inline DateTimeSpan OneMillisecond(
                const bool isPositive = true) {
            return DateTimeSpan(isPositive ? TICKS_PER_MILLISECOND
                : -TICKS_PER_MILLISECOND);
        }

        /** 
         * Create a positive or negative DateTimeSpan of one minute.
         *
         * @param isPositive Determines whether the time span is positive
         *                   or not. This parameter defaults to true.
         *
         * @return A time span of one minute.
         */
        static inline DateTimeSpan OneMinute(const bool isPositive = true) {
            return DateTimeSpan(isPositive ? TICKS_PER_MINUTE
                : -TICKS_PER_MINUTE);
        }

        /** 
         * Create a positive or negative DateTimeSpan of one second.
         *
         * @param isPositive Determines whether the time span is positive
         *                   or not. This parameter defaults to true.
         *
         * @return A time span of one second.
         */
        static inline DateTimeSpan OneSecond(const bool isPositive = true) {
            return DateTimeSpan(isPositive ? TICKS_PER_SECOND
                : -TICKS_PER_SECOND);
        }

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
         * @throws TODO: range check!
         */
        static INT64 TimeToTicks(const INT hours, const INT minutes, 
            const INT seconds, const INT milliseconds = 0,
            const INT ticks = 0);

        /** A constant empty time span. */
        static const DateTimeSpan EMPTY;

        /** One day in milliseconds. */
        static const INT64 MILLISECONDS_PER_DAY;

        /** One hour in milliseconds. */
        static const INT64 MILLISECONDS_PER_HOUR;

        /** One minute in milliseconds. */
        static const INT64 MILLISECONDS_PER_MINUTE;
        
        /** One second in milliseconds. */
        static const INT64 MILLISECONDS_PER_SECOND;

        /** The largest possible positive time span. */
        static const DateTimeSpan MAXIMUM;

        /** The largest possible negative time span. */
        static const DateTimeSpan MINIMUM;

        /** One day in milliseconds. */
        static const INT64 TICKS_PER_DAY;

        /** One hour in milliseconds. */
        static const INT64 TICKS_PER_HOUR;

        /** Ticks per millisecond. */
        static const INT64 TICKS_PER_MILLISECOND;

        /** One minute in milliseconds. */
        static const INT64 TICKS_PER_MINUTE;
        
        /** One second in milliseconds. */
        static const INT64 TICKS_PER_SECOND;

        /**
         * Create a time span of the given amount of ticks.
         *
         * @param ticks The total number of ticks of the timespan. This 
         *              parameter defaults to zero.
         */
        explicit inline DateTimeSpan(const INT64 ticks = 0L) throw()
            : ticks(ticks) {}

        /**
         * Create a new time span using the specified length.
         *
         * @param days         The day part of the time span.
         * @param hours        The hour part of the time span.
         * @param minutes      The minute part of the time span.
         * @param seconds      The second part of the time span.
         * @param milliseconds The millisecond part of the time span. This 
         *                     defaults to zero.
         * @param ticks        The 100 ns part of the time span. This defaults
         *                     to zero.
         *
         * @throws IllegalParamException If the parameters would cause a numeric
         *                               overflow.
         */
        DateTimeSpan(const INT days, const INT hours, const INT minutes,
            const INT seconds, const INT milliseconds = 0, const INT ticks = 0);

        /**
         * Copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        inline DateTimeSpan(const DateTimeSpan& rhs) throw() 
                : ticks(rhs.ticks) {
            VLSTACKTRACE("DateTimeSpan::DateTimeSpan", __FILE__, __LINE__);
        }

        /** Dtor. */
        ~DateTimeSpan(void);

        /**
         * Answer the complete days of the time span.
         *
         * @return The days part of the time span.
         */
        inline INT GetDays(void) const {
            VLSTACKTRACE("DateTimeSpan::GetDays", __FILE__, __LINE__);
            return static_cast<INT>(this->ticks / TICKS_PER_DAY);
        }

        /**
         * Answer the number of hours in the current day. 
         * This value lies within [0, 24[.
         *
         * @return The hours part of the time span.
         */
        inline INT GetHours(void) const {
            VLSTACKTRACE("DateTimeSpan::GetHours", __FILE__, __LINE__);
            return static_cast<INT>((this->ticks / TICKS_PER_HOUR) % 24);
        }

        /**
         * Answer the number of milliseconds in the current second.
         * This value lies within [0, 1000[.
         *
         * @return The milliseconds part of the time span.
         */
        inline INT GetMilliseconds(void) const {
            VLSTACKTRACE("DateTimeSpan::GetMilliseconds", __FILE__, __LINE__);
            return static_cast<INT>((this->ticks / TICKS_PER_MILLISECOND) 
                % 1000);
        }

        /**
         * Answer the number of minutes in the current hour.
         * This values lies within [0, 60[.
         *
         * @return The minutes part of the time span.
         */
        inline INT GetMinutes(void) const {
            VLSTACKTRACE("DateTimeSpan::GetMinutes", __FILE__, __LINE__);
            return static_cast<INT>((this->ticks / TICKS_PER_MINUTE) % 60);
        }

        /**
         * Answer the number of seconds in the current minute.
         * This value lies within [0, 60[.
         *
         * @return The seconds part of the time span.
         */
        inline INT GetSeconds(void) const {
            VLSTACKTRACE("DateTimeSpan::GetSeconds", __FILE__, __LINE__);
            return static_cast<INT>((this->ticks / TICKS_PER_SECOND) % 60);
        }

        /**
         * Answer the number of remaining ticks.
         * This value lies within [0, TICKS_PER_MILLISECOND[.
         *
         * @return The ticks part of the time span.
         */
        inline INT GetTicks(void) const {
            VLSTACKTRACE("DateTimeSpan::GetTicks", __FILE__, __LINE__);
            return static_cast<INT>(this->ticks % TICKS_PER_MILLISECOND);
        }

        /**
         * Answer the total number of ticks representing the time span.
         *
         * @return The total number of ticks.
         */
        inline INT64 GetTotalTicks(void) const {
            VLSTACKTRACE("DateTimeSpan::GetTotalTicks", __FILE__, __LINE__);
            return this->ticks;
        }

        /**
         * Set a new value for the time span. 
         *
         * @param days         The day part of the time span.
         * @param hours        The hour part of the time span.
         * @param minutes      The minute part of the time span.
         * @param seconds      The second part of the time span.
         * @param milliseconds The millisecond part of the time span. This 
         *                     defaults to zero.
         * @param ticks        The 100 ns part of the time span. This defaults
         *                     to zero.
         *
         * @throws IllegalParamException If the parameters would cause a numeric
         *                               overflow.
         */
        void Set(const INT days, const INT hours, const INT minutes,
            const INT seconds, const INT milliseconds = 0, const INT ticks = 0);

        /**
         * Answer a string representation of the time span.
         *
         * @return A string representation.
         */
        StringA ToStringA(void) const;

        /**
         * Answer a string representation of the time span.
         *
         * @return A string representation.
         */
        StringW ToStringW(void) const;

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this object and 'rhs' are equal, false otherwise.
         */
        inline bool operator ==(const DateTimeSpan& rhs) const throw() {
            VLSTACKTRACE("DateTimeSpan::operator ==", __FILE__, __LINE__);
            return (this->ticks == rhs.ticks);
        }
	
        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true if this object and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const DateTimeSpan& rhs) const throw() {
            VLSTACKTRACE("DateTimeSpan::operator !=", __FILE__, __LINE__);
            return (this->ticks != rhs.ticks);
        }

        /** 
         * Test whether this time span is shorter than 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return true of this time span is shorter than 'rhs', 
         *         false otherwise.
         */
        inline bool operator <(const DateTimeSpan& rhs) const throw() {
            VLSTACKTRACE("DateTimeSpan::operator <", __FILE__, __LINE__);
            return (this->ticks < rhs.ticks);
        }

        /** 
         * Test whether this time span is longer than 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return true of this time span is longer than 'rhs', 
         *         false otherwise.
         */
	    inline bool operator >(const DateTimeSpan& rhs) const throw() {
            VLSTACKTRACE("DateTimeSpan::operator <", __FILE__, __LINE__);
            return (this->ticks > rhs.ticks);
        }
	    
        /** 
         * Test whether this time span is shorter than or equal to 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return true of this time span is shorter than or equal to 'rhs', 
         *         false otherwise.
         */
        inline bool operator <=(const DateTimeSpan& rhs) const throw() {
            VLSTACKTRACE("DateTimeSpan::operator <=", __FILE__, __LINE__);
            return (this->ticks <= rhs.ticks);
        }

        /** 
         * Test whether this time span is longer than or equal to 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return true of this time span is longer than or equal to 'rhs', 
         *         false otherwise.
         */
	    inline bool operator >=(const DateTimeSpan& rhs) const throw() {
            VLSTACKTRACE("DateTimeSpan::operator >=", __FILE__, __LINE__);
            return (this->ticks >= rhs.ticks);
        }

        /**
         * Compute the sum of this time span and 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return The sum of this time span and 'rhs'.
         *
         * @throws IllegalParamException If 'rhs' has a value that would cause 
         *                               an overflow of the result.
         */
	    inline DateTimeSpan operator +(const DateTimeSpan& rhs) const {
            VLSTACKTRACE("DateTimeSpan::operator +", __FILE__, __LINE__);
            DateTimeSpan retval(*this);
            retval += rhs;
            return retval;
        }
	
        /**
         * Compute the difference between this time span and 'rhs'.
         *
         * @param rhs The right hand side operand.
         *
         * @return The difference between this time span and 'rhs'.
         *
         * @throws IllegalParamException If 'rhs' has a value that would cause 
         *                               an overflow of the result.
         */
        inline DateTimeSpan operator -(const DateTimeSpan& rhs) const {
            VLSTACKTRACE("DateTimeSpan::operator -", __FILE__, __LINE__);
            DateTimeSpan retval(*this);
            retval -= rhs;
            return retval;
        }

        /**
         * Add 'rhs' to this time span.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this, which is the sum of the prior value of this time span
         *         and 'rhs'.
         *
         * @throws IllegalParamException If 'rhs' has a value that would cause 
         *                               an overflow of the result.
         */
        inline DateTimeSpan& operator +=(const DateTimeSpan& rhs) {
            VLSTACKTRACE("DateTimeSpan::operator +=", __FILE__, __LINE__);
            this->add(rhs.ticks);
            return *this;
        }
	
        /**
         * Subtract 'rhs' from this time span.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this, which is the difference between the prior value of 
         *         this time span and 'rhs'.
         *
         * @throws IllegalParamException If 'rhs' has a value that would cause 
         *                               an overflow of the result.
         */
        DateTimeSpan& operator -=(const DateTimeSpan& rhs);
	
        /**
         * Negate this time span.
         *
         * @return This time span multiplied with -1.
         *
         * @throws IllegalStateException If 'rhs' the time span is the smallest
         *                               possible negative value, because this
         *                               would cause an overflow of the result.
         */
        DateTimeSpan operator -(void) const;

        /**
         * Assigment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
	    DateTimeSpan& operator =(const DateTimeSpan& rhs) throw();

        /**
         * Answer the total number of ticks representing the time span.
         *
         * @return The total number of ticks.
         */
        inline operator INT64(void) const throw() {
            VLSTACKTRACE("DateTimeSpan::operator INT64", __FILE__, __LINE__);
            return this->ticks;
        }

    private:

        /**
         * Add the specified amount of ticks to the value of this time 
         * span. The result will be assigned to the value of this time span.
         *
         * This method checks the range of the result of the addition before
         * perfoming the actual computation. 
         *
         * @param ticks The ticks to be added.
         *
         * @throws IllegalParamException If 'ticks' has a value that would 
         *                               cause an overflow of the result.
         */
        void add(const INT64 ticks);

        /** The value in 100 ns ticks. */
        INT64 ticks;

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_DATETIMESPAN_H_INCLUDED */
