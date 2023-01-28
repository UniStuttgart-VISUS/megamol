/*
 * Lockable.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


namespace vislib::sys {


/**
 * This class implements a 'Lockable' interface using a 'SyncObject' for
 * locking. You may use any class derived from 'SyncObject' as template
 * parameter 'T'.
 */
template<class T>
class Lockable {
public:
    /** Ctor. */
    Lockable();

    /** Dtor. */
    virtual ~Lockable();

    /**
     * Aquires the lock of this lockable.
     * Details on the behaviour depend on the 'SyncObject' used.
     */
    inline void Lock() const;

    /**
     * Releases the lock of this lockable.
     * Details on the behaviour depend on the 'SyncObject' used.
     */
    inline void Unlock() const;

private:
    /** The syncObj used for locking */
    mutable T syncObj;
};


/*
 * Lockable::Lockable
 */
template<class T>
Lockable<T>::Lockable() : syncObj() {
    // intentionally empty
}


/*
 * Lockable::~Lockable
 */
template<class T>
Lockable<T>::~Lockable() {
    // intentionally empty
}


/*
 * Lockable::Lock
 */
template<class T>
void Lockable<T>::Lock() const {
    this->syncObj.Lock();
}


/*
 * Lockable::Unlock
 */
template<class T>
void Lockable<T>::Unlock() const {
    this->syncObj.Unlock();
}


} // namespace vislib::sys

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
