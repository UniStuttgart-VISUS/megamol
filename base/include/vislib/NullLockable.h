/*
 * NullLockable.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_NULLLOCKABLE_H_INCLUDED
#define VISLIB_NULLLOCKABLE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/forceinline.h"


namespace vislib {

    /**
     * This class implements a 'Lockable' interface without any locking
     * mechanism. This should be used on all Objects requireing a 'Lockable'
     * but do not have to be threadsafe for performance reasons.
     */
    class NullLockable {

    public:

        /** Ctor. */
        NullLockable(void);

        /** Dtor. */
        virtual ~NullLockable(void);

        /**
         * Aquires the lock of this lockable.
         * Details on the behaviour depend on the 'SyncObject' used.
         *
         * This implementation does nothing and therefore object using this
         * lockable are not threadsafe.
         */
        VISLIB_FORCEINLINE void Lock(void) {
            // intentionally empty
        }

        /**
         * Releases the lock of this lockable.
         * Details on the behaviour depend on the 'SyncObject' used.
         *
         * This implementation does nothing and therefore object using this
         * lockable are not threadsafe.
         */
        VISLIB_FORCEINLINE void Unlock(void) {
            // intentionally empty
        }

    };

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_NULLLOCKABLE_H_INCLUDED */
