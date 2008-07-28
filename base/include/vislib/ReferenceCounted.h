/*
 * ReferenceCounted.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_REFERENCECOUNTED_H_INCLUDED
#define VISLIB_REFERENCECOUNTED_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/assert.h"
#include "vislib/SingleAllocator.h"
#include "vislib/types.h"


namespace vislib {


    /**
     * This class adds a reference counting mechanism to a class.
     *
     * Objects that inherit from ReferenceCounted must always be allocated on
     * the heap.
     *
     * The template parameter A is the allocator used for releasing the
     * object once its reference count falls to zero. The allocator specified 
     * here must match for all instances of the object and all derived objects.
     * The default allocator is a SingleAllocator for ReferenceCounted that will
     * release the object using delete (Note: Instantiation requires <> template
     * argument list for default parameter).
     *
     * The reference count of a newly created object is 1.
     */
    //template<class A = SingleAllocator<ReferenceCounted<A> > > 
    template<class A> class ReferenceCounted {

    public:

        /**
         * Increment the reference count.
         *
         * @return The new value of the reference counter.
         */
        inline UINT32 AddRef(void) {
            return ++this->cntRefs;
        }

        /**
         * Decrement the reference count. If the reference count reaches zero,
         * the object is released using the allocator A.
         *
         * @return The new value of the reference counter.
         */
        UINT32 Release(void);

    protected:

        /** Ctor. */
        ReferenceCounted(void);

        /**
         * Copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        ReferenceCounted(const ReferenceCounted& rhs);

        /** 
         * Dtor. 
         *
         * Making the dtor protected prevents explicit deletion of objects using
         * delete and creation of objects on the stack to a certain extent.
         */
        virtual ~ReferenceCounted(void);

        /**
         * Assignment.
         *
         * Note that assignment does not change the reference count of an
         * object.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        ReferenceCounted& operator =(const ReferenceCounted& rhs);

    private:
        
        /** The current reference count. */
        UINT32 cntRefs;

        /* Allow the allocator deleting the object. */
        typedef A _A;
        typedef typename A::TargetPtrType _P;
        friend void _A::Deallocate(_P& inOutPtr);
        friend typename A;

    };


    /*
     * vislib::ReferenceCounted<A>::Release
     */
    template<class A> UINT32 ReferenceCounted<A>::Release(void) {
        ASSERT(this->cntRefs > 0);
        UINT32 retval = --this->cntRefs;
        if (this->cntRefs == 0) {
            A::TargetPtrType r = reinterpret_cast<A::TargetPtrType>(this);
            A::Deallocate(r);
        }
        return retval;
    }


    /*
     * vislib::ReferenceCounted<A>::ReferenceCounted
     */
    template<class A> ReferenceCounted<A>::ReferenceCounted(void) : cntRefs(1) {
    }


    /*
     * ReferenceCounted<A>::ReferenceCounted
     */
    template<class A> ReferenceCounted<A>::ReferenceCounted(
            const ReferenceCounted& rhs) : cntRefs(1) {
    }


    /*
     * vislib::ReferenceCounted<A>::~ReferenceCounted
     */
    template<class A> ReferenceCounted<A>::~ReferenceCounted(void) {
        ASSERT(this->cntRefs == 0);
    }


    /*
     * vislib::ReferenceCounted<A>::operator =
     */
    template<class A> ReferenceCounted<A>& ReferenceCounted<A>::operator =(
            const ReferenceCounted& rhs) {
        // Nothing to be done! No not modify the reference count!
        return *this;
    }

    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_REFERENCECOUNTED_H_INCLUDED */
