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
#include "vislib/SingleAllocator.h"


namespace vislib {

    /**
     * This is a special array that holds pointers to objects of type T and 
     * takes ownership, i.e. if an array element is deleted the array will 
     * deallocate the object designated by this array element. The object will
     * be deleted using the allocator A. THE USER IS RESPONSIBLE FOR ONLY
     * STORING OBJECTS ON THE HEAP THAT CAN BE DEALLOCATED BY THE GIVEN
     * ALLOCATOR!
     *
     * If you want to store pointers to single objects in the array, use 
     * SingleAllocator for A and allocate the objects using new. This is the 
     * default.
     *
     * If you want to store pointers to arrays of objects in the array, use
     * ArrayAllocator for A and allocate the array using new[].
     */
    template<class T, class A = SingleAllocator<T> > class PtrArray 
            : public Array<T *> {

    protected:

        /** Immediate superclass. */
        typedef Array<T *> Super;

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

    protected:

        /**
         * Construct element at 'inOutAddress'.
         *
         * This implementation sets the pointer designated by 'inOutAddress' 
         * NULL.
         *
         * @param inOutAddress Pointer to the object to construct.
         */
        virtual void ctor(T **inOutAddress) const;

        /**
         * Destruct element at 'inOutAddress'.
         *
         * This implementation calls A::Deallocate on the pointer designated
         * by 'inOutAddress'.
         *
         * @param inOutAddress Pointer to the object to destruct.
         */
        virtual void dtor(T **inOutAddress) const;
    };


    /*
     * PtrArray<T, A>::PtrArray
     */
    template<class T, class A> PtrArray<T, A>::PtrArray(const SIZE_T capacity) 
            : Super(capacity) {
        // No dynamic binding within ctor, so initialise all elements with
        // NULL pointer by hand.
        ::memset(this->elements, 0, this->capacity * sizeof(T *));
    }


    /*
     * vislib::PtrArray<T, A>::~PtrArray
     */
    template<class T, class A> PtrArray<T, A>::~PtrArray(void) {
        // dtor() method does not work in dtor, because there is no dynamic
        // binding within the dtor. Therefore, we deallocate the elements
        // ourselves before the parent dtor deallocates the element array.
        for (SIZE_T i = 0; i < this->count; i++) {
            this->dtor(this->elements + i);
        }
        this->count = 0;
    }


    /*
     * vislib::PtrArray<T, A>::ctor
     */
    template<class T, class A> 
    void PtrArray<T, A>::ctor(T **inOutAddress) const {
        inOutAddress = NULL;
    }


    /*
     * vislib::PtrArray<T, A>::dtor
     */
    template<class T, class A> 
    void PtrArray<T, A>::dtor(T **inOutAddress) const {
        A::Deallocate(*inOutAddress);
        inOutAddress = NULL;
    }

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_PTRARRAY_H_INCLUDED */
