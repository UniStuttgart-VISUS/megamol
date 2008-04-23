/*
 * AutoLock.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_AUTOLOCK_H_INCLUDED
#define VISLIB_AUTOLOCK_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/SyncObject.h"


namespace vislib {
namespace sys {


    /**
     * This class provides a simplifies mechanism for using synchronisation
     * objects. It acquires the lock passed in the constructor on construction
     * and releases it in the dtor.
     */
    class AutoLock {

    public:

        /**
         * Acqures 'lock' and stores it for releasing it in dtor.
         *
         * WARNING: This ctor might throw an exception, if the lock cannot be
         * acquired even when waiting infinitely. You should not go one if this
         * happens.
         *
         * @throws SystemException If the lock could not be acquired.
         */
        AutoLock(SyncObject& lock);

        /**
         * The dtor releases 'lock'.
         */
        ~AutoLock(void);

    private:

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        AutoLock(const AutoLock& rhs);

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @throws IllegalParamException if &rhs != this.
         */
        AutoLock& operator =(const AutoLock& rhs);

        /** The synchronisation object that is hold by the lock. */
        SyncObject& lock;

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_AUTOLOCK_H_INCLUDED */
