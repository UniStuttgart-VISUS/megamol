/*
 * NullAllocator.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_NULLALLOCATOR_H_INCLUDED
#define VISLIB_NULLALLOCATOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/types.h"


namespace vislib {


    /**
     * This is an allocator class to be used in collections or SmartPtr classes
     * which does not allocate or deallocate any memory. Instead the allocation
     * methods always return 'NULL' and the deallocation methods only set the
     * incoming pointers to 'NULL'. This class is fully compatible with the
     * 'SingleAllocator' and the 'ArrayAllocator'.
     *
     * One use of this class is a 'SmartPtr' pointing to an object placed on
     * the global application stack.
     */
    template<class T> class NullAllocator {

    public:

        /** The type of the object handled by the allocator.*/
        typedef T TargetType;

        /** The pointer type that is handled by the allocator. */
        typedef T *TargetPtrType;

        /**
         * Allways returns 'NULL'.
         *
         * @param cnt Ignored.
         *
         * @return 'NULL'
         */
        static inline TargetPtrType Allocate(const SIZE_T cnt = 0) {
            return NULL;
        }

        /**
         * Sets the pointer 'inOutPtr' to 'NULL'.
         *
         * @param inOutPtr The pointer will be set NULL.
         */
        static inline void Deallocate(TargetPtrType& inOutPtr) {
            inOutPtr = NULL;
        }

    private:

        /** Disallow instances. */
        NullAllocator(void) {
            // intentionally empty
        }

        /** Dtor. */
        ~NullAllocator(void) {
            // intentionally empty
        }

    };

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_NULLALLOCATOR_H_INCLUDED */
