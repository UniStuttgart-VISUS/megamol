/*
 * PtrArray.h
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2007 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_PTRARRAY_H_INCLUDED
#define VISLIB_PTRARRAY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Array.h"
#include "vislib/ArrayElementPtrCtor.h"


namespace vislib {

    /**
     * This is a special array that holds pointers to objects of type T and 
     * takes ownership, i.e. if an array element is deleted the array will 
     * deallocate the object designated by this array element. The object will
     * be deleted using the allocator A. THE USER IS RESPONSIBLE FOR ONLY
     * STORING OBJECTS ON THE HEAP THAT CAN BE DEALLOCATED BY THE GIVEN
     * ALLOCATOR!
     *
     * For synchronisation of the PtrArray using the L template parameter,
     * read the documentation of vislib::Array.
     *
     * If you want to store pointers to single objects in the array, use 
     * SingleAllocator for A and allocate the objects using new. This is the 
     * default.
     *
     * If you want to store pointers to arrays of objects in the array, use
     * ArrayAllocator for A and allocate the array using new[].
     */
    template<class T, class L = NullLockable, class A = SingleAllocator<T> >
    class PtrArray : public Array<T *, L, ArrayElementPtrCtor<T, A> > {

    protected:

        /** Immediate superclass. */
        typedef Array<T *, L, ArrayElementPtrCtor<T, A> > Super;

    public:

        /** 
         * Create an array with the specified initial capacity.
         *
         * @param capacity The initial capacity of the array.
         */
        PtrArray(const SIZE_T capacity = Super::DEFAULT_CAPACITY);

        /**
         * Create a new array with the specified initial capacity and
         * use 'element' as default value for all elements.
         *
         * @param capacity The initial capacity of the array.
         * @param element  The default value to set.
         */
        inline PtrArray(const SIZE_T capacity, const T& element)
                : Super(capacity, element) {}

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        PtrArray(const PtrArray& rhs) : Super(rhs) {}

        /** Dtor. */
        virtual ~PtrArray(void);
    };


    /*
     * vislib::PtrArray<T, L, A>::PtrArray
     */
    template<class T, class L , class A>
    PtrArray<T, L, A>::PtrArray(const SIZE_T capacity) : Super(capacity) {
        // Nothing to do. The constructor/destructor functor also works in
        // ctor and dtor as it is not dependent on the virtual table.
    }


    /*
     * vislib::PtrArray<T, L, A>::~PtrArray
     */
    template<class T, class L , class A> PtrArray<T, L, A>::~PtrArray(void) {
        // Nothing to do. The constructor/destructor functor also works in
        // ctor and dtor as it is not dependent on the virtual table.
    }

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_PTRARRAY_H_INCLUDED */
