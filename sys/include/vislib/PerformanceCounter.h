/*
 * PerformanceCounter.h  10.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_PERFORMANCECOUNTER_H_INCLUDED
#define VISLIB_PERFORMANCECOUNTER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#include "vislib/types.h"


namespace vislib {
namespace sys {

    /**
     * This class provides a system independent high-resolution performance
     * counter.
     *
     * @author Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de)
     */
    class PerformanceCounter {

    public:

        /**
         * Query the performance counter for its current value.
         *
         * @return The current performance counter value.
         *
         * @throws SystemException If the performance counter could not be 
         *                         queried.
         */
        static UINT64 Query(void);

        /**
         * Create a new performance counter that is capable of setting marks for
         * computing differences. The mark is initially set to the creation 
         * time.
         */
        inline PerformanceCounter(void) : mark(Query()) {}

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline PerformanceCounter(const PerformanceCounter& rhs) 
                : mark(rhs.mark) {}

        /** Dtor. */
        inline ~PerformanceCounter(void) {}

        /**
         * Answer the difference between the current performance counter value
         * and the mark.
         *
         * @return The difference between now and the mark.
         */
        inline INT64 Difference(void) const {
            return (static_cast<INT64>(PerformanceCounter::Query()) 
                - static_cast<INT64>(this->mark));
        }

        /**
         * Answer the last mark.
         *
         * @return The last mark.
         */
        inline UINT64 GetMark(void) const {
            return this->mark;
        }

        /**
         * Set the mark to the current performance counter value.
         *
         * @return The new value of the mark, i. e. the current performance 
         *         counter value.
         */
        inline UINT64 SetMark(void) {
            return (this->mark = PerformanceCounter::Query());
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
            return (this->mark == rhs.mark);
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

        /**
         * The mark, i. e. the performance counter value when the last
         * mark was set.
         */
        UINT64 mark;
		
	};

} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_PERFORMANCECOUNTER_H_INCLUDED */
