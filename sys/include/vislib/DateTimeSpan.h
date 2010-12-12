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
     * resolution of one millisecond.
     */
    class DateTimeSpan {

    public:

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

        /** A constant time span of one day (positive). */
        static const DateTimeSpan ONE_DAY;

        /** A constant time span of one hour (positive). */
        static const DateTimeSpan ONE_HOUR;

        /** A constant time span of one millisecond (positive). */
        static const DateTimeSpan ONE_MILLISECOND;

        /** A constant time span of one minute (positive). */
        static const DateTimeSpan ONE_MINUTE;

        /** A constant time span of one second (positive). */
        static const DateTimeSpan ONE_SECOND;

        /**
         * Create a time span of the given amount of milliseconds
         *
         * @param value The total milliseconds of the timespan. This defaults to
         *              zero.
         */
        explicit inline DateTimeSpan(const INT64 value = 0L) throw()
            : value(value) {}

        /**
         * Create a new time span using the specified length.
         *
         * @param days         The day part of the time span.
         * @param hours        The hour part of the time span.
         * @param minutes      The minute part of the time span.
         * @param seconds      The second part of the time span.
         * @param milliseconds The millisecond part of the time span. This 
         *                     defaults to zero.
         *
         * @throws IllegalParamException TODO
         */
        DateTimeSpan(const INT days, const INT hours, const INT minutes,
                const INT seconds, const INT milliseconds = 0L);

        /**
         * Copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        inline DateTimeSpan(const DateTimeSpan& rhs) throw() 
            : value(rhs.value) {}

        /** Dtor. */
        ~DateTimeSpan(void);

        /**
         * Answer the complete days of the time span.
         *
         * @return The days part of the time span.
         */
        inline INT64 GetDays(void) const {
            return this->value / MILLISECONDS_PER_DAY;
        }

        /**
         * Answer the number of hours in the current day.
         *
         * @return The hours part of the time span.
         */
        inline INT64 GetHours(void) const {
            return this->value % MILLISECONDS_PER_DAY / MILLISECONDS_PER_HOUR;
        }

        /**
         * Answer the number of milliseconds in the current second.
         *
         * @return The milliseconds part of the time span.
         */
        inline INT64 GetMilliseconds(void) const {
            return this->value % MILLISECONDS_PER_SECOND;
        }

        /**
         * Answer the number of minutes in the current hour.
         *
         * @return The minutes part of the time span.
         */
        inline INT64 GetMinutes(void) const {
            return this->value % MILLISECONDS_PER_DAY % MILLISECONDS_PER_HOUR 
                / MILLISECONDS_PER_MINUTE;
        }

        /**
         * Answer the number of seconds in the current minute.
         *
         * @return The seconds part of the time span.
         */
        inline INT64 GetSeconds(void) const {
            return this->value % MILLISECONDS_PER_DAY % MILLISECONDS_PER_HOUR 
                % MILLISECONDS_PER_MINUTE / MILLISECONDS_PER_SECOND;
        }

        /**
         * Answer a value that represents the time span in terms of
         * days.
         *
         * Please be advised that the value returned might be truncated due
         * to the resolution of the return type.
         *
         * @returns The complete and fractional days that represent the
         *          time span.
         */
        inline double GetTotalDays(void) const {
            return static_cast<double>(this->value) 
                / static_cast<double>(MILLISECONDS_PER_DAY);
        }

        /**
         * Answer a value that represents the time span in terms of
         * hours.
         *
         * Please be advised that the value returned might be truncated due
         * to the resolution of the return type.
         *
         * @returns The complete and fractional hours that represent the
         *          time span.
         */
        inline double GetTotalHours(void) const {
            return static_cast<double>(this->value) 
                / static_cast<double>(MILLISECONDS_PER_HOUR);
        }

        /**
         * Answer a value that represents the time span in terms of
         * milliseconds.
         *
         * @returns The complete milliseconds that represent the time span.
         */
        inline INT64 GetTotalMilliseconds(void) const {
            return this->value;
        }

        /**
         * Answer a value that represents the time span in terms of
         * minutes.
         *
         * Please be advised that the value returned might be truncated due
         * to the resolution of the return type.
         *
         * @returns The complete and fractional minutes that represent the
         *          time span.
         */
        inline double GetTotalMinutes(void) const {
            return static_cast<double>(this->value) 
                / static_cast<double>(MILLISECONDS_PER_MINUTE);
        }

        /**
         * Answer a value that represents the time span in terms of
         * seconds.
         *
         * Please be advised that the value returned might be truncated due
         * to the resolution of the return type.
         *
         * @returns The complete and fractional seconds that represent the
         *          time span.
         */
        inline double GetTotalSeconds(void) const {
            return static_cast<double>(this->value) 
                / static_cast<double>(MILLISECONDS_PER_SECOND);
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
         *
         * @throws IllegalParamException TODO
         */
        void Set(const INT days, const INT hours, const INT minutes,
            const INT seconds, const INT milliseconds = 0L);

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
            return (this->value == rhs.value);
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
            return (this->value != rhs.value);
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
            return (this->value < rhs.value);
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
            return (this->value > rhs.value);
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
            return (this->value <= rhs.value);
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
            return (this->value >= rhs.value);
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
            this->add(rhs.value);
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
         * Convert the time span into an integer that represents the time span
         * in milliseconds.
         *
         * @return The total milliseconds that represent the time span.
         */
        inline operator INT64(void) const throw() {
            VLSTACKTRACE("DateTimeSpan::operator INT64", __FILE__, __LINE__);
            return this->value;
        }

    private:

        /**
         * Add the specified amount of milliseconds to the value of this time 
         * span. The result will be assigned to the value of this time span.
         *
         * This method checks the range of the result of the addition before
         * perfoming the actual computation. 
         *
         * @throws IllegalParamException If 'millis' has a value that would 
         *                               cause an overflow of the result.
         */
        void add(const INT64 millis);

        /** The value in milliseconds. */
        INT64 value;

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_DATETIMESPAN_H_INCLUDED */
