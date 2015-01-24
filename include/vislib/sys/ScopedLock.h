/*
 * ScopedLock.h
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SCOPEDLOCK_H_INCLUDED
#define VISLIB_SCOPEDLOCK_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/IllegalParamException.h"
#include "vislib/StackTrace.h"
#include "vislib/UnsupportedOperationException.h"


namespace vislib {
namespace sys {

    /**
     * This class provides a simplifies mechanism for using synchronisation
     * objects. It acquires the lock passed in the constructor on construction
     * and releases it in the dtor.
     */
    template<class T> class ScopedLock {

    public:

        /**
         * Acquires 'lock' and stores it for releasing it in dtor.
         *
         * WARNING: This ctor might throw an exception, if the lock cannot be
         * acquired even when waiting infinitely. You should not go one if this
         * happens.
         *
         * @throws SystemException If the lock could not be acquired.
         */
        inline ScopedLock(T& lock) : lock(lock) {
            VLSTACKTRACE("ScopedLock::ScopedLock", __FILE__, __LINE__);
            this->lock.Lock();
        }

        /**
         * The dtor releases 'lock'.
         */
        inline ~ScopedLock(void) {
            VLSTACKTRACE("ScopedLock::~ScopedLock", __FILE__, __LINE__);
            this->lock.Unlock();
        }

    private:

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        inline ScopedLock(const ScopedLock& rhs) : lock(rhs.lock) {
            throw UnsupportedOperationException(
                "vislib::sys::ScopedLock::ScopedLock",
                __FILE__, __LINE__);
        }

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @throws IllegalParamException if &rhs != this.
         */
        inline ScopedLock& operator =(const ScopedLock& rhs) {
            if (this != &rhs) {
                throw IllegalParamException("rhs", __FILE__, __LINE__);
            }
            return *this;
        }

        /** The actual lock object. */
        T& lock;
    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SCOPEDLOCK_H_INCLUDED */

