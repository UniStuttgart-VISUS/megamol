/*
 * LamportClock.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_LAMPORTCLOCK_H_INCLUDED
#define VISLIB_LAMPORTCLOCK_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AutoLock.h"
#include "vislib/CriticalSection.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {


    /**
     * This class implements a Lamport clock (Leslie Lamport (1978): "Time, 
     * clocks, and the ordering of events in a distributed system") which can 
     * be used to implement a happened-before ordering. 
     *
     * The class implements a monotonically incrementing counter that has to
     * be incremented for local operations. On receiving a remote message, the 
     * maximum of the local value and the message timestamp plus one is used as
     * new value. It is up to the user to increment the counter, especially when
     * a message is sent.
     *
     * The counter used is a 64 bit unsigned integer. No measures against an
     * arithmetic overflow are taken.
     *
     * The class is thread-safe.
     */
    class LamportClock {

    public:

        /** Ctor. */
        LamportClock(void);

        /** Dtor. */
        ~LamportClock(void);

        /**
         * Answer the current value of the counter.
         *
         * @return The current value of the counter.
         */
        inline UINT64 GetValue(void) const {
            AutoLock l(this->lock);
            return this->value;
        }

        /**
         * Tells the clock to make a local step, i. e. increment the counter 
         * by 1.
         *
         * @return The new value of the counter.
         */
        UINT64 StepLocal(void);

        /**
         * Tells the clock that a message with timestamp 'timestamp' was 
         * received. The counter is adjusted appropriately.
         *
         * @param timestamp The timestamp of the message received.
         *
         * @return The new value of the counter.
         */
        UINT64 StepReceive(UINT64 timestamp);

        /**
         * Tells the clock to make a local step, i. e. increment the counter
         * by 1, but return the old value of the counter.
         *
         * @return The old value of the counter.
         */
        UINT64 operator ++(int);

        /**
         * Tells the clock to make a local step, i. e. increment the counter 
         * by 1.
         *
         * @return The new value of the counter.
         */
        inline UINT64 operator ++(void) {
            return this->StepLocal();
        }

        /**
         * Answer the current value of the counter.
         *
         * @return The current value of the counter.
         */
        UINT64 operator *(void) const {
            return this->GetValue();
        }

    private:

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Always.
         */
        LamportClock(const LamportClock& rhs);

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @throws IllegalParamException if &rhs != this.
         */
        LamportClock& operator =(const LamportClock& rhs);

        /** The critical section protecting the counter. */
        mutable CriticalSection lock;

        /** The value of counter. */
        UINT64 value;

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_LAMPORTCLOCK_H_INCLUDED */

