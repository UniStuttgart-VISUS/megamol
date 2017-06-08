/*
 * SingleAllocator.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SINGLEALLOCATOR_H_INCLUDED
#define VISLIB_SINGLEALLOCATOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/memutils.h"


namespace vislib {


    /**
     * This class creates typed memory for a single object of the template type.
     * It therefore cannot be used for allocating continuous arrays.
     *
     * The allocator uses the C++ allocation and deallocation mechanisms and 
     * therefore guarantees that the default ctor is called on the newly
     * allocated object and that the dtor is called before deallocating an
     * object.
     */
    template<class T> class SingleAllocator {

    public:

        /** The type of the object handled by the allocator.*/
        typedef T TargetType;

        /** The pointer type that is handled by the allocator. */
        typedef T *TargetPtrType;

        /**
         * Allocate an object of type T.
         *
         * @return A pointer to the newly allocated object.
         *
         * @throws std::bad_alloc If there was not enough memory to allocate the
         *                        object.
         */
        static inline TargetPtrType Allocate(void) {
            return new T;
        }

        /**
         * Deallocate 'ptr' and set it NULL.
         *
         * @param inOutPtr The pointer to be deallocated. The pointer will be 
         *                 set NULL before the method returns.
         */
        static inline void Deallocate(TargetPtrType& inOutPtr) {
            delete inOutPtr;
            inOutPtr = NULL;
        }

    private:

        /** Disallow instances. */
        SingleAllocator(void);

        /** Dtor. */
        ~SingleAllocator(void);

    };
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SINGLEALLOCATOR_H_INCLUDED */
