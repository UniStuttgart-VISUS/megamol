/*
 * DateTimeSpan.h
 *
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


#include "vislib/types.h"


namespace vislib {
namespace sys {


    /**
     * This class represents a time span.
     */
    class DateTimeSpan {

    public:

        /**
         * Create a zero time span.
         */
        inline DateTimeSpan(void) : value(0) {}

        /**
         * Create a new time span using the specified length.
         *
         * @param days
         * @param hours
         * @param minutes
         * @param seconds
         */
        inline DateTimeSpan(const INT days, const INT hours, const INT minutes,
                const INT seconds) {
            this->Set(days, hours, minutes, seconds);
        }

        /** Dtor. */
        ~DateTimeSpan(void);

        /**
         * Answer the day part of the time span.
         *
         * @return The day part.
         */
        inline INT64 GetDays(void) const {
            return this->value / ONE_DAY;
        }

        /**
         * Answer the hour part of the time span.
         *
         * @return The hour part.
         */
        inline INT64 GetHours(void) const {
            return this->value / ONE_HOUR;
        }

        /**
         * Answer the the millisecond part of the time span.
         *
         * @return The millisecond part.
         */
        inline INT64 GetMilliseconds(void) const {
            return this->value % ONE_SECOND;
        }

        inline INT64 GetMinutes(void) const {
            return this->value / ONE_MINUTE;
        }

        inline INT64 GetSeconds(void) const {
            return this->value / ONE_SECOND;
        }

        inline double GetTotalDays(void) const {
            return static_cast<double>(this->value) 
                / static_cast<double>(ONE_DAY);
        }

        inline double GetTotalHours(void) const {
            return static_cast<double>(this->value) 
                / static_cast<double>(ONE_HOUR);
        }

        inline INT64 GetTotalMilliseconds(void) const {
            return this->value;
        }

        inline double GetTotalMinutes(void) const {
            return static_cast<double>(this->value) 
                / static_cast<double>(ONE_MINUTE);
        }

        inline double GetTotalSeconds(void) const {
            return static_cast<double>(this->value) 
                / static_cast<double>(ONE_SECOND);
        }

        /**
         * 
         * @param days
         * @param hours
         * @param minutes
         * @param seconds
         */
        void Set(const INT days, const INT hours, const INT minutes,
            const INT seconds);

    private:

        /** One day in milliseconds. */
        static const INT64 ONE_DAY;

        /** One hour in milliseconds. */
        static const INT64 ONE_HOUR;

        /** One minute in milliseconds. */
        static const INT64 ONE_MINUTE;
        
        /** One second in milliseconds. */
        static const INT64 ONE_SECOND;

        /** The value in milliseconds. */
        INT64 value;

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_DATETIMESPAN_H_INCLUDED */
