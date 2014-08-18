/*
 * SynchronisedArray.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SYNCHRONISEDARRAY_H_INCLUDED
#define VISLIB_SYNCHRONISEDARRAY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Array.h"
#include "vislib/CriticalSection.h"
#include "vislib/Lockable.h"


namespace vislib {
namespace sys {


    /**
     * This is a specialisation of an array that includes inherits from Lockable
     * to provide a certain amount of thread-safety. Read the documentation of
     * vislib::Array for more information about which operations are 
     * thread-safe and which are not.
     *
     * The template parameter S must be a SyncObject-derived class. The default
     * is vislib::sys::CriticalSection.
     */
    template<class T, class S = CriticalSection,
            class C = ArrayElementDftCtor<T> > class SynchronisedArray
            : Array<T, Lockable<S>, C> {

    protected:

        /** Immediate superclass. */
        typedef Array<T, Lockable<S>, C> Super;

    public:

        /** 
         * Create an array with the specified initial capacity.
         *
         * @param capacity The initial capacity of the array.
         */
        SynchronisedArray(const SIZE_T capacity = Super::DEFAULT_CAPACITY);

        /**
         * Create a new array with the specified initial capacity and
         * use 'element' as default value for all elements.
         *
         * @param capacity The initial capacity of the array.
         * @param element  The default value to set.
         */
        inline SynchronisedArray(const SIZE_T capacity, const T& element)
                : Super(capacity, element) {}

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        SynchronisedArray(const SynchronisedArray& rhs) : Super(rhs) {}

        /** Dtor. */
        virtual ~SynchronisedArray(void);
    };


    /*
     * vislib::sys::SynchronisedArray<T, S, C>::SynchronisedArray
     */
    template<class T, class S , class C>
    SynchronisedArray<T, S, C>::SynchronisedArray(const SIZE_T capacity)
            : Super(capacity) {
        // Nothing to do.
    }


    /*
     * vislib::sys::SynchronisedArray<T, S, C>::~SynchronisedArray
     */
    template<class T, class S , class C>
    SynchronisedArray<T, S, C>::~SynchronisedArray(void) {
        // Nothing to do.
    }

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SYNCHRONISEDARRAY_H_INCLUDED */
