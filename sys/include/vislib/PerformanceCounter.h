/*
 * PerformanceCounter.h  10.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_PERFORMANCECOUNTER_H_INCLUDED
#define VISLIB_PERFORMANCECOUNTER_H_INCLUDED
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
     * This class provides a system independent performance counter. The 
     * resolution is milliseconds. However, the zero point of the counter
     * is undefined.
     *
     * The performance counter can also be used with the full available
     * precision of the underlying system, but this functionality must be
     * requested explicitly. The default value for the full precision flags
     * is set false for backwards compatibility.
     *
     * @author Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de)
     */
    class PerformanceCounter {

    public:

        /**
         * Query the performance counter for its current value.
         *
         * @param useFullPrecision Set this true to retrieve the native full
         *                         precision value of the performance counter. 
         *                         If false, the return value are milliseconds
         *                         regardless of the hardware and system the
         *                         software is running on.
         *
         * @return The current performance counter value. If 'useFullPrecision'
         *         is true, the returned value has the full precision the
         *         current hardware allows. Otherwise, the value is normalised
         *         to milliseconds.
         *
         * @throws SystemException If the performance counter could not be 
         *                         queried.
         */
        static UINT64 Query(const bool useFullPrecision = false);

        /**
         * Query the performance counter for its current value in milliseconds.
         * Depending on the capabilities of the underlying hardware, the value
         * might have a fractional part.
         *
         * Using this method is equivalent to the following code:
         *
         * UINT64 value = PerformanceCounter::Query(true);
         * UINT64 frequency = PerformanceCounter::QueryFrequency();
         * double result = static_cast<double>(value) * 1000.0 
         *     / static_cast<double>(frequency);
         *
         * @return The current perfomance counter value in milliseconds.
         *
         * @throws SystemException If the performance counter could not be 
         *                         queried.
         */
        inline static double QueryMillis(void) {
            return ToMillis(Query(true));
        }

        /**
         * Query the performance counter's native frequency in counts per second.
         *
         * @return The performance counter frequency in counts per second.
         *
         * @throws SystemException If the frequency could not be queried.
         */
        static UINT64 QueryFrequency(void);

        /**
         * Convert a full resolution performance counter value to milliseconds.
         *
         * @param value The performance counter value with the full resolution 
         *              of the underlying system.
         * 
         * @return The milliseconds that 'value' represents.
         *
         * @throws SystemException If the frequency could not be queried.
         */
        inline static double ToMillis(const UINT64 value) {
            return (static_cast<double>(value) * 1000.0)
                / static_cast<double>(QueryFrequency());
        }

        /**
         * Create a new performance counter that is capable of setting marks for
         * computing differences. The mark is initially set to the creation 
         * time.
         *
         * @param isUsingFullPrecisionMark If this flag is set, all marks will 
         *                                 have the full precision of the 
         *                                 underlying machine. Otherwise, the 
         *                                 values are normalised to represent
         *                                 milliseconds.
         */
        explicit inline PerformanceCounter(const bool isUsingFullPrecisionMark 
                = false) : isUsingFullPrecisionMark(isUsingFullPrecisionMark),
                mark(Query(isUsingFullPrecisionMark)) {}

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline PerformanceCounter(const PerformanceCounter& rhs) 
                : isUsingFullPrecisionMark(rhs.isUsingFullPrecisionMark),
                mark(rhs.mark) {}

        /** Dtor. */
        inline ~PerformanceCounter(void) {}

        /**
         * Answer the difference between the current performance counter value
         * and the mark.
         *
         * The semantics of the difference depends on whether the counter is in 
         * full precision mode or not. If IsUsingFullPrecisionMark() returns
         * true, the value has the finest resolution possible and must be
         * divided by the counter frequency to obtain seconds. Otherwise, the
         * value returned represents milliseconds.
         *
         * @return The difference between now and the mark.
         */
        inline INT64 Difference(void) const {
            return (static_cast<INT64>(PerformanceCounter::Query(
                this->isUsingFullPrecisionMark)) 
                - static_cast<INT64>(this->mark));
        }

        /**
         * Answer the last mark.
         *
         * The semantics of the mark depends on whether the counter is in 
         * full precision mode or not. If IsUsingFullPrecisionMark() returns
         * true, the value has the finest resolution possible and must be
         * divided by the counter frequency to obtain seconds. Otherwise, the
         * value returned represents milliseconds.
         *
         * @return The last mark.
         */
        inline UINT64 GetMark(void) const {
            return this->mark;
        }

        /**
         * Answer whether the mark has the full counter precision.
         *
         * @return True if the mark has full counter precision, false if the
         *         mark represents milliseconds.
         */
        inline bool IsUsingFullPrecisionMark(void) const {
            return this->isUsingFullPrecisionMark;
        }

        /**
         * Set the mark to the current performance counter value.
         *
         * @return The new value of the mark, i. e. the current performance 
         *         counter value.
         */
        inline UINT64 SetMark(void) {
            return (this->mark = PerformanceCounter::Query(
                this->isUsingFullPrecisionMark));
        }

        /** 
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        PerformanceCounter& operator =(const PerformanceCounter& rhs);

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are equal, false otherwise.
         */
        inline bool operator ==(const PerformanceCounter& rhs) const {
            return (this->mark == rhs.mark) && (this->isUsingFullPrecisionMark 
                == rhs.isUsingFullPrecisionMark);
        }

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         *
         * @return true, if *this and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const PerformanceCounter& rhs) const {
            return !(*this == rhs);
        }

    private:

        /** Enables use of full precision marks. */
        bool isUsingFullPrecisionMark;

        /**
         * The mark, i. e. the performance counter value when the last
         * mark was set.
         */
        UINT64 mark;
        
    };

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_PERFORMANCECOUNTER_H_INCLUDED */
