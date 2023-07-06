/*
 * ArrayElementPtrCtor.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/SingleAllocator.h"
#include "vislib/forceinline.h"


namespace vislib {


/**
 * The ArrayElementPtrCtor class is a functor that allows the vislib::Array
 * class initialising arrays of pointers to T and deallocating these objects
 * using the specified allocator A.
 */
template<class T, class A = SingleAllocator<T>>
class ArrayElementPtrCtor {

public:
    /**
     * Call the default constructor of T on 'inOutAddress'.
     *
     * @param inOutAddress The address of the object to be constructed.
     */
    VISLIB_FORCEINLINE static void Ctor(T** inOutAddress) {
        *inOutAddress = NULL;
    }

    /**
     * Deallocate the object that the pointer at address 'inOutAddress'
     * designates using the allocator A and reset the pointer to NULL.
     *
     * @param inOutAddress The address of the object to be destructed.
     */
    VISLIB_FORCEINLINE static void Dtor(T** inOutAddress) {
        A::Deallocate(*inOutAddress);
        *inOutAddress = NULL;
    }

    /** Dtor. */
    ~ArrayElementPtrCtor();

private:
    /**
     * Disallow instances.
     */
    ArrayElementPtrCtor();
};


/*
 * ArrayElementPtrCtor<T, A>::~ArrayElementPtrCtor
 */
template<class T, class A>
ArrayElementPtrCtor<T, A>::~ArrayElementPtrCtor() {
    // Nothing to do.
}


/*
 * ArrayElementPtrCtor<T, A>::ArrayElementPtrCtor
 */
template<class T, class A>
ArrayElementPtrCtor<T, A>::ArrayElementPtrCtor() {
    // Nothing to do.
}

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
