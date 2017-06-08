/*
 * ArrayAllocator.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ARRAYALLOCATOR_H_INCLUDED
#define VISLIB_ARRAYALLOCATOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/memutils.h"
#include "vislib/types.h"


namespace vislib {


    /**
     * Allocates arrays of dynamic memory. 
     *
     * It uses the C++ array new and delete operators and therefore guarantees
     * that the default ctors and the dtors of the objects allocated are called
     * when necessary.
     */
    template<class T> class ArrayAllocator {

    public:

        /** The type of the object handled by the allocator.*/
        typedef T TargetType;

        /** The pointer type that is handled by the allocator. */
        typedef T *TargetPtrType;


        /**
         * Allocates 'cnt' contiguous objects of type T.
         *
         * @param cnt The number of elements to allocate.
         *
         * @return The pointer to the memory.
         *
         * @throws std::bad_alloc If the memory could not be allocated.
         */
        static inline TargetPtrType Allocate(const SIZE_T cnt) {
            return new T[cnt];
        }

        /**
         * Deallocate the memory designated by 'inOutPtr'.
         *
         * @param inOutPtr The pointer to the memory. The pointer will be set 
         *                 NULL when the method returns.
         */
        static inline void Deallocate(TargetPtrType& inOutPtr) {
            delete[] inOutPtr;
            inOutPtr = NULL;
        }

        ///**
        // * Reallocate 'ptr' to be an array having 'cntNew' elements. The 
        // * overlapping part of the old array will remain in the new array, too.
        // *
        // * @param ptr    The pointer to be reallocated.
        // * @param cntOld The current size of the array designated by 'ptr'.
        // * @param cntNew The new size of the array.
        // *
        // * @return A pointer to the reallocated memory.
        // *
        // * @throws std::bad_alloc If the new memory could not be allocated.
        // */
        //static inline T *Reallocate(T *ptr, const SIZE_T cntOld, 
        //		const SIZE_T cntNew) {
        //	T *retval = new T[cnt];
        //	::memcpy(retval, ptr, (cntOld < cntNew) ? cntOld : cntNew);
        //	delete[] ptr;
        //	return retval;
        //}

    private:

        /** Disallow instances. */
        ArrayAllocator(void);

        /** Dtor. */
        ~ArrayAllocator(void);

    };
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ARRAYALLOCATOR_H_INCLUDED */

