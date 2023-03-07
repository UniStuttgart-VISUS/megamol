/*
 * ArrayElementDftCtor.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/forceinline.h"


namespace vislib {


/**
 * The ArrayElementDftCtor class is a functor that allows the vislib::Array
 * class constructing objects of the template type T that have been
 * allocated typelessly.
 */
template<class T>
class ArrayElementDftCtor {

public:
    /**
     * Call the default constructor of T on 'inOutAddress'.
     *
     * @param inOutAddress The address of the object to be constructed.
     */
    VISLIB_FORCEINLINE static void Ctor(T* inOutAddress) {
        new (inOutAddress) T;
    }

    /**
     * Call the destructor of T on the object at 'inOutAddress'.
     *
     * @param inOutAddress The address of the object to be destructed.
     */
    VISLIB_FORCEINLINE static void Dtor(T* inOutAddress) {
        inOutAddress->~T();
    }

    /** Dtor. */
    ~ArrayElementDftCtor();

private:
    /**
     * Disallow instances.
     */
    ArrayElementDftCtor();
};


/*
 * ArrayElementDftCtor<T>::~ArrayElementDftCtor
 */
template<class T>
ArrayElementDftCtor<T>::~ArrayElementDftCtor() {
    // Nothing to do.
}


/*
 * ArrayElementDftCtor<T>::ArrayElementDftCtor
 */
template<class T>
ArrayElementDftCtor<T>::ArrayElementDftCtor() {
    // Nothing to do.
}

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
