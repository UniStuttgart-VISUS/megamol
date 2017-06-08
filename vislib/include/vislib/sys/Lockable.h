/*
 * Lockable.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_LOCKABLE_H_INCLUDED
#define VISLIB_LOCKABLE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


namespace vislib {
namespace sys {


    /**
     * This class implements a 'Lockable' interface using a 'SyncObject' for
     * locking. You may use any class derived from 'SyncObject' as template
     * parameter 'T'.
     */
    template<class T> class Lockable {
    public:

        /** Ctor. */
        Lockable(void);

        /** Dtor. */
        virtual ~Lockable(void);

        /**
         * Aquires the lock of this lockable.
         * Details on the behaviour depend on the 'SyncObject' used.
         */
        inline void Lock(void) const;

        /**
         * Releases the lock of this lockable.
         * Details on the behaviour depend on the 'SyncObject' used.
         */
        inline void Unlock(void) const;

    private:

        /** The syncObj used for locking */
        mutable T syncObj;

    };


    /*
     * Lockable::Lockable
     */
    template<class T> Lockable<T>::Lockable(void) : syncObj() {
        // intentionally empty
    }


    /*
     * Lockable::~Lockable
     */
    template<class T> Lockable<T>::~Lockable(void) {
        // intentionally empty
    }


    /*
     * Lockable::Lock
     */
    template<class T> void Lockable<T>::Lock(void) const {
        this->syncObj.Lock();
    }


    /*
     * Lockable::Unlock
     */
    template<class T> void Lockable<T>::Unlock(void) const {
        this->syncObj.Unlock();
    }


} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_LOCKABLE_H_INCLUDED */
