/*
 * SingleAllocator.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SINGLEALLOCATOR_H_INCLUDED
#define VISLIB_SINGLEALLOCATOR_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#include "vislib/memutils.h"


namespace vislib {


    /**
     * This class creates typed memory for a single object of the template type.
	 * It therefore cannot be used for allocating arrays and syntactially 
	 * differs from an array allocator.
	 *
	 * The allocator guarantees that the default ctor is called on the newly
	 * allocated object and that the dtor is called before deallocating an
	 * object.
     */
    template<class T> class SingleAllocator {

    public:

		/**
		 * Allocate an object of type T.
		 *
		 * @return A pointer to the newly allocated object.
		 *
		 * @throws std::bad_alloc If there was not enough memory to allocate the
		 *                        object.
		 */
		static inline T *Allocate(void) {
			return new T;
		}

		/**
		 * Deallocate 'ptr' and set it NULL.
		 *
		 * @param inOutPtr The pointer to be deallocated.
		 */
		static inline void Deallocate(T *& inOutPtr) {
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

#endif /* VISLIB_SINGLEALLOCATOR_H_INCLUDED */
